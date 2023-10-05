import os
import pandas
from tqdm import tqdm
from PIL import Image
import gc

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

import hydra
from omegaconf import OmegaConf


class ImageCaptioningDataset(Dataset):
    def __init__(self, images_root, csvs_list, processor, crop_box=None):

        self.crop_box = crop_box
        self.img_path_caption_pairs = []
        for csv in csvs_list:
            df = pandas.read_csv(csv, index_col=0)
            assert all(c1 == c2 for c1, c2 in zip(df.columns, ['paths', 'caption']))
            
            self.img_path_caption_pairs.extend(
                [(os.path.join(images_root, p), c) for p, c in zip(df.paths,  df.caption)]
            )
            
        self.processor = processor

    def __len__(self):
        return len(self.img_path_caption_pairs)

    def __getitem__(self, idx):
        img_path, caption = self.img_path_caption_pairs[idx]
        
        try:
            image = Image.open(img_path)
            
            if self.crop_box is not None:
                image = image.crop(box=self.crop_box)
                
        except Exception as exc:
            raise
        
        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["prompt"] = caption
        
        image.close()
        
        return encoding


def make_collate_fn(processor):
    def collate_fn(batch):
        processed_batch = {}
        for key in batch[0].keys():
            if key != "prompt":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["prompt"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
                
        return processed_batch

    return collate_fn


def build_model(pretrained, lora_config):
    processor = AutoProcessor.from_pretrained(pretrained, device_map="cpu")
    model = Blip2ForConditionalGeneration.from_pretrained(pretrained, device_map="cpu")

    model = get_peft_model(model, lora_config)
    model.train()
    
    model.print_trainable_parameters()
    
    return model, processor


def make_dataloader(processor, config):
    
    train_dataset = ImageCaptioningDataset(images_root=config.data_path,
                                           csvs_list=config.csv_list,
                                           processor=processor,
                                           crop_box=tuple(config.crop_box))
    
    collate_fn = make_collate_fn(processor)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=config.training.batch_size,
                                  collate_fn=collate_fn,
                                  num_workers=config.training.dataloader_workers)
    
    return train_dataloader

def make_accelerator(config):
    
    if config.wandb.enabled:
        accelerator = Accelerator(log_with="wandb", 
                    gradient_accumulation_steps=config.training.gradient_accumulation_steps)

        accelerator.init_trackers(
            project_name=config.wandb.project, 
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        )

        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
        
        if accelerator.is_main_process:
            peft_checkpoint_path = os.path.join(
                                    config.peft.checkpoint_root,
                                    f"{wandb_tracker.id}_{wandb_tracker.name}") 
    else:
        accelerator = Accelerator(gradient_accumulation_steps=config.training.gradient_accumulation_steps)
        peft_checkpoint_path = os.path.join(config.peft.checkpoint_root, "TEST")
            
      
    if accelerator.is_main_process:
        os.makedirs(peft_checkpoint_path, exist_ok=True)
        config.peft.checkpoint_path = peft_checkpoint_path
        
    return accelerator
    

def train_model(accelerator, model, optimizer, train_dataloader, config):
    
    for epoch in range(config.training.n_epochs):
        print("Epoch:", epoch)
        progress_bar = tqdm(train_dataloader, desc="Training", leave=True,
                            disable=not accelerator.is_main_process)
        
        for idx, batch in enumerate(progress_bar):
            
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch['input_ids'],
                            pixel_values=batch['pixel_values'],
                            labels=batch['input_ids'])
            
                loss = outputs.loss
                if config.wandb.enabled:
                    accelerator.log({"loss": loss.item()})

                progress_bar.set_postfix({'loss': loss.item()})

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        
        if accelerator.is_main_process:
            epoch_saving_path = os.path.join(config.peft.checkpoint_path, f'epoch_{epoch}')
            os.makedirs(epoch_saving_path, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(epoch_saving_path)
    
    if config.wandb.enabled:
        accelerator.end_training()
        
@hydra.main(version_base=None, config_path="../configs", config_name="train_blip2_config")
def main(cfg):
    
    lora_config = LoraConfig(
        r=cfg.peft.lora.r,
        lora_alpha=cfg.peft.lora.lora_alpha,
        lora_dropout=cfg.peft.lora.lora_dropout,
        bias=cfg.peft.lora.bias,
        target_modules=list(cfg.peft.lora.target_modules)
    )
    

    model, processor = build_model(pretrained=cfg.model.pretrained,
                                  lora_config=lora_config
                                  )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    train_dataloader = make_dataloader(processor=processor,
                                       config=cfg
                                     )
    
    
    accelerator = make_accelerator(cfg)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    train_model(accelerator=accelerator,
               model=model,
               optimizer=optimizer,
               train_dataloader=train_dataloader,
               config=cfg
               )


if __name__ == '__main__':
    main()