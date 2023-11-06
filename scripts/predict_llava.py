import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import hydra
from omegaconf import OmegaConf
from wandb import Image as wandb_Image
from wandb import Table

import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path


def get_default_prompt_ids(model, config, tokenizer):

    if model.config.mm_use_im_start_end:
        query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + config.generation.prompt
    else:
        query = DEFAULT_IMAGE_TOKEN + '\n' + config.generation.prompt

    conv = conv_templates[config.generation.conversation_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    prompt_input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    
    return prompt_input_ids

class ImagesDataset(Dataset):
    def __init__(self, list_of_dirs, image_processor, default_prompt_ids):
        
        self.image_processor = image_processor
        
        self.default_prompt_ids = default_prompt_ids
        
        self.image_paths = []
        
        for directory in list_of_dirs:
            for filename in os.listdir(directory):
                try:
                    path_to_img = os.path.join(directory, filename)
                    image = Image.open(path_to_img)
                    image.close()
                    self.image_paths.append(path_to_img)
                except IOError:
                    continue
                except Exception as exc:
                    raise

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path)
                
        except Exception as exc:
            raise
            
        image_processed = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half()
     
        image.close()
        
        return {"pixel_values": image_processed.squeeze(), 
                "image_path": image_path,
                "input_ids": self.default_prompt_ids
               }

def collate_fn(batch):
    processed_batch = {}
    processed_batch["pixel_values"] = torch.stack([example["pixel_values"] for example in batch])
    processed_batch["input_ids"] = torch.stack([example["input_ids"] for example in batch])
    
    processed_batch["image_path"] = []
    for example in batch:
        processed_batch["image_path"].extend([example["image_path"]])

    return processed_batch

def make_accelerator(config):
    
    if config.wandb.enabled:
        accelerator = Accelerator(log_with="wandb")

        accelerator.init_trackers(
            project_name=config.wandb.project, 
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        )
    else:
        accelerator = Accelerator()            
        
    return accelerator

def load_model(config):
    
    model_name = get_model_name_from_path(config.checkpoint.base_model)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=config.checkpoint.base_model,
                                                                           model_name=model_name,
                                                                           model_base=config.checkpoint.base_model,
                                                                           device_map='auto'
                                                                          )
    
    return tokenizer, model, image_processor

def make_csv_and_log(accelerator, paths, captions, config):
    df = pd.DataFrame(columns=["paths", "caption"])
    df.paths = paths
    df.caption = captions
    df.drop_duplicates(subset="paths", inplace=True, ignore_index=True)
    
    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
    csv_name = f"{wandb_tracker.id}_{wandb_tracker.name}__" if config.wandb.enabled else ""
    csv_name = csv_name + "++++".join(os.path.basename(x) for x in config.image_folders_list) + "__"
    csv_name = csv_name + config.resulting_csv.base_name
    df.to_csv(os.path.join(config.resulting_csv.directory, csv_name))
    
    if config.wandb.enabled and config.wandb.log_visual_examples:
        
        table_data = []
        for img_id, (caption, path_to_img) in enumerate(zip(captions, paths)):
            if img_id == config.wandb.n_logged_visual_examples:
                break
                
            img = Image.open(path_to_img).resize(tuple(config.wandb.img_loging_shape))
            table_data.append([caption, wandb_Image(img)])
           
        
        table = Table(data=table_data, columns=["generated_caption", "image"])

        wandb_tracker.log({'table': table})

@hydra.main(version_base=None, config_path="../configs", config_name="predict_llava_config")
def main(config):
    disable_torch_init()
    
    accelerator = make_accelerator(config)
    
    tokenizer, model, image_processor = load_model(config)
    
    default_prompt_ids = get_default_prompt_ids(model, config, tokenizer)
    
    dataset = ImagesDataset(config.image_folders_list, image_processor, default_prompt_ids)
    
    dataloader = DataLoader(dataset,
                          shuffle=False,
                          batch_size=config.dataloader.batch_size_per_gpu,
                          num_workers=config.dataloader.n_workers,
                          collate_fn=collate_fn)
    
    model, dataloader = accelerator.prepare(model, dataloader)
    model = accelerator.unwrap_model(model)
    
    collected_paths = []
    collected_captions = []
    
    progress_bar = tqdm(dataloader, desc="Generating captions...", leave=True,
                            disable=not accelerator.is_main_process)
    
    for batch_id, batch in enumerate(progress_bar):
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=batch["input_ids"],
                images=batch["pixel_values"],
                do_sample=config.generation.temperature > 0,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                num_beams=config.generation.num_beams,
                max_new_tokens=config.generation.max_new_tokens,
                use_cache=True)
        
        
        # generated_captions = list(processor.batch_decode(generated_ids, skip_special_tokens=True))
        input_token_len = batch["input_ids"].shape[1]
        n_diff_input_output = (batch["input_ids"] != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
        generated_captions = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        
        collected_paths.extend(batch["image_path"])
        collected_captions.extend(generated_captions)
    
    collected_captions = accelerator.gather_for_metrics(collected_captions)
    collected_paths = accelerator.gather_for_metrics(collected_paths)
    
    if accelerator.is_main_process:
        make_csv_and_log(accelerator, collected_paths, collected_captions, config)


if __name__ == '__main__':
    main()
