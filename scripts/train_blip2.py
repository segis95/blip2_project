import pandas
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

class ImageCaptioningDataset(Dataset):
    def __init__(self, images_root, csvs_list, processor, crop_box):

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
            image = Image.open(img_path).crop(box=self.crop_box)
        except Exception as exc:
            raise
        
        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        encoding["prompt"] = caption
        
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
    processor = AutoProcessor.from_pretrained(pretrained)
    model = Blip2ForConditionalGeneration.from_pretrained(pretrained, device_map="auto")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor

def make_dataloader(images_root_folder, csvs, processor, batch_size, crop_box):
    
    train_dataset = ImageCaptioningDataset(images_root_folder, csvs, processor, crop_box=crop_box)
    collate_fn = make_collate_fn(processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)
    
    return train_dataloader
    
def train_model(model, train_dataloader, learning_rate, n_epochs, wandb_log=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    model.train()
    
    device = "cuda:0"
    for epoch in range(n_epochs):
        print("Epoch:", epoch)
        progress_bar = tqdm(train_dataloader, desc="Training", leave=True)
        for idx, x in enumerate(progress_bar):
         
            outputs = model(input_ids=x[1].to(device),
                            pixel_values=x[0].to(device),
                            labels=x[1].to(device))
            
            loss = outputs.loss
            if wandb_log:
                wandb.log({"loss": loss.item()})
                
            progress_bar.set_postfix({'loss': loss.item()})
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

@hydra.main(version_base=None, config_path="../configs", config_name="train_blip2_config")
def main(cfg):
    
    if cfg.wandb:
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        
        wandb.init(project=cfg.wandb.project)
        
    lora_config = LoraConfig(
        r=cfg.peft.lora.r,
        lora_alpha=cfg.peft.lora.lora_alpha,
        lora_dropout=cfg.peft.lora.lora_dropout,
        bias="none",
        target_modules=cfg.peft.target_modules
    )

    model, processor = build_model(pretrained=cfg.model.pretrained,
                                  lora_config=lora_config
                                  )
    
    
    train_dataloader = make_dataloader(images_root_folder=cfg.data_path,
                                     csvs=cfg.csv_list,
                                     processor=processor,
                                     batch_size=cfg.training.batch_size,
                                     crop_box=tuple(cfg.crop_box)
                                     )
    
    train_model(model=model,
               learning_rate=cfg.training.learning_rate,
               n_epochs=cfg.training.n_epochs,
               wandb_log=cfg.wandb
               )


if __name__ == '__main__':
    main()