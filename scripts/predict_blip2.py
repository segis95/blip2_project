import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import hydra
import wandb
from omegaconf import OmegaConf

import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from train_blip2 import ImageCaptioningDataset

def get_images_stream(list_of_dirs):
    for directory in list_of_dirs:
        for filename in os.listdir(directory):
            try:
                path_to_img = os.path.join(directory, filename)
                image = Image.open(path_to_img)
                yield path_to_img, image
            except:
                continue

def make_csv_and_save(paths, captions, config):
    df = pd.DataFrame(columns=['paths', 'caption'])
    df.paths = paths
    df.caption = captions
    
    csv_name = f"{wandb.run.id}_{wandb.run.name}_" if config.wandb.enabled else ""
    csv_name = csv_name + config.resulting_csv.base_name
    df.to_csv(os.path.join(config.resulting_csv.directory, csv_name))
                
@hydra.main(version_base=None, config_path="../configs", config_name="predict_blip2_config")
def main(cfg):
    
    if cfg.wandb.enabled:
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        
        wandb.init(project=cfg.wandb.project)
    
    with torch.cuda.device(cfg.gpu):
        processor = AutoProcessor.from_pretrained(cfg.base_model_checkpoint)
        model = Blip2ForConditionalGeneration.from_pretrained(cfg.base_model_checkpoint, device_map="auto")
        model.load_adapter(cfg.peft_checkpoint)
    
    collected_paths = []
    collected_captions = []
    data_img_log = []
    
    for img_id, (img_path, image) in tqdm(enumerate(get_images_stream(cfg.image_folders_list))):
        
        img_processed = processor(images=image, return_tensors="pt").to(f"cuda:{cfg.gpu}", torch.float16)
        generated_ids = model.generate(pixel_values=img_processed.pixel_values, max_length=100)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        collected_paths.append(img_path)
        collected_captions.append(generated_caption)
        
        if img_id < cfg.wandb.n_logged_visual_examples:
            data_img_log.append([generated_caption, image.resize(tuple(cfg.wandb.img_loging_shape))])
        
        image.close()
        
    
    make_csv_and_save(collected_paths, collected_captions, cfg)
    
    if cfg.wandb.enabled and cfg.wandb.log_visual_examples:
        
        table = wandb.Table(data=list(map(lambda x: [x[0], wandb.Image(x[1])], data_img_log)), 
                            columns=["generated_caption", "image"])
        
        wandb.run.log({"table": table})


if __name__ == '__main__':
    main()        
    
    
    
    
        
    
        
    
        
    