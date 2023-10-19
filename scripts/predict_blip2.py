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
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from accelerate import Accelerator


class ImagesDataset(Dataset):
    def __init__(self, list_of_dirs, processor):
        
        self.processor = processor
        
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
        
        image_processed = self.processor(images=image, return_tensors="pt")       
        image.close()
        
        
        return {"pixel_values": image_processed.pixel_values.squeeze(), 
                "image_path": image_path
               }

def collate_fn(batch):
    processed_batch = {}
    processed_batch["pixel_values"] = torch.stack([example["pixel_values"] for example in batch])
    
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
    
    processor = Blip2Processor.from_pretrained(config.checkpoint.processor, device_map="auto")
    
    model = Blip2ForConditionalGeneration.from_pretrained(config.checkpoint.base_model,
                                                              device_map="auto")
    if config.checkpoint.load_adapter:
        model.load_adapter(config.checkpoint.adapter)
        
    return model, processor

def make_csv_and_log(accelerator, paths, captions, config):
    df = pd.DataFrame(columns=["paths", "caption"])
    df.paths = paths
    df.caption = captions
    df.drop_duplicates(subset="paths", inplace=True, ignore_index=True)
    
    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
    csv_name = f"{wandb_tracker.id}_{wandb_tracker.name}_" if config.wandb.enabled else ""
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
                
@hydra.main(version_base=None, config_path="../configs", config_name="predict_blip2_config")
def main(config):
    
    accelerator = make_accelerator(config)
    
    model, processor = load_model(config)
    
    dataset = ImagesDataset(config.image_folders_list, processor)
    
    dataloader = DataLoader(dataset,
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
        
        generated_ids = model.generate(pixel_values=batch["pixel_values"],
                                                                 max_length=config.generation.max_length)
        generated_captions = list(processor.batch_decode(generated_ids, skip_special_tokens=True))
        
        collected_paths.extend(batch["image_path"])
        collected_captions.extend(generated_captions)
    
    collected_captions = accelerator.gather_for_metrics(collected_captions)
    collected_paths = accelerator.gather_for_metrics(collected_paths)
    
    if accelerator.is_main_process:
        make_csv_and_log(accelerator, collected_paths, collected_captions, config)


if __name__ == '__main__':
    main()