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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param}"
    )

    
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


def build_model(config):
    
    processor = AutoProcessor.from_pretrained(config.model.pretrained_checkpoint, device_map="cpu")
    model = Blip2ForConditionalGeneration.from_pretrained(config.model.pretrained_checkpoint, device_map="cpu")
    
    if config.peft.add_adapter:
        
        lora_config = LoraConfig(
            r=config.peft.lora.r,
            lora_alpha=config.peft.lora.lora_alpha,
            lora_dropout=config.peft.lora.lora_dropout,
            bias=config.peft.lora.bias,
            target_modules=list(config.peft.lora.target_modules)
        )
        
        model = get_peft_model(model, lora_config)
        
    model.train()
    
    if config.peft.add_adapter: 
        model.print_trainable_parameters()
        
    print_trainable_parameters(model)
    
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
    
    checkpoint_root = config.peft.checkpoint_root if config.peft.add_adapter else config.model.checkpoint_root
                                                
    
    if config.logging.wandb.enabled:
        accelerator = Accelerator(log_with="wandb", 
                    gradient_accumulation_steps=config.training.gradient_accumulation_steps)

        accelerator.init_trackers(
            project_name=config.logging.wandb.project, 
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        )
        
        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
        
        if accelerator.is_main_process:
            
            checkpoint_path = os.path.join(
                                    checkpoint_root,
                                    f"{wandb_tracker.id}_{wandb_tracker.name}") 
    else:
        accelerator = Accelerator(gradient_accumulation_steps=config.training.gradient_accumulation_steps)
        checkpoint_path = os.path.join(checkpoint_root, "TEST")
            
    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)
        config.logging.checkpoint_path = checkpoint_path
        
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
                if config.logging.wandb.enabled:
                    accelerator.log({"loss": loss.item()})

                progress_bar.set_postfix({'loss': loss.item()})

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        
        if accelerator.is_main_process and (epoch + 1) % config.logging.checkpoint_every_nth_epoch == 0:
            epoch_saving_path = os.path.join(config.logging.checkpoint_path, f'epoch_{epoch}')
            os.makedirs(epoch_saving_path, exist_ok=True)
            
            accelerator.unwrap_model(model).save_pretrained(epoch_saving_path)
            if not config.peft.add_adapter:
                train_dataloader.processor.save_pretrained(epoch_saving_path)
    
    if config.logging.wandb.enabled:
        accelerator.end_training()
        
@hydra.main(version_base=None, config_path="../configs", config_name="train_blip2_config")
def main(cfg):    

    model, processor = build_model(config=cfg)
    
    train_dataloader = make_dataloader(processor=processor,
                                       config=cfg
                                     )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
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