import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf, DictConfig
import os
import json
import sys
from model import SimVLMModel, LLMConfig, VisualConfig, MultiModelConfig
from transformers import AutoTokenizer, AutoProcessor, default_data_collator
from utils import SimVLMDataset, save_model, ColoredLogger
from torch.utils.data import DataLoader, random_split
from datasets import Dataset, load_dataset
from tqdm import tqdm
import wandb


def main():
    # load config
    conf = OmegaConf.from_cli()
    visual_config = VisualConfig(file_path=conf.visual_model.file_path, model_name=conf.visual_model.name)
    llm_config = LLMConfig(file_path=conf.llm.file_path, model_name=conf.llm.name)
    multi_model_config = MultiModelConfig()


    # load wandb
    ColoredLogger.info(f"wandb mode: {conf.wandb.mode}")
    if conf.wandb.mode == 'online':
        wandb.init(
            project=conf.wandb.project,
            name=conf.wandb.name,
            config={
                "visual_model": conf.visual_model.name,
                'llm': conf.llm.name,
                "learning_rate": conf.train.learning_rate,
                "train_batch_size": conf.train.batch_size,
            },
            mode=conf.wandb.mode,
        )

    # load and processor
    processor = AutoProcessor.from_pretrained(os.path.join(visual_config.file_path, visual_config.model_name), use_fast=True)
    
    for model_path in os.listdir(conf.evaluate.model.check_path):
        if model_path != 'checkpoint-22327':
            continue
        llm_config.file_path = os.path.join(conf.evaluate.model.check_path, model_path)
        # load model
        simple_vlm = SimVLMModel.from_specific_checkpoint(checkpoint_path=os.path.join(conf.evaluate.model.check_path, model_path), llm_config=llm_config, visual_config=visual_config, multi_model_config=multi_model_config)
        tokenizer = simple_vlm.tokenizer
        
        images_path = conf.evaluate.data.image_path
        prompt = "详细描述一下图中的内容，越详细越好。<image>"
        
        conversations = [
            {
                'role': 'user',
                'content': prompt
            }
        ]
        formatted_input = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        tokenized_input = tokenizer(formatted_input, return_tensors="pt")
        
        for image in os.listdir(images_path):
            image_path = os.path.join(images_path, image)
            image_data = Image.open(image_path).convert('RGB')
            pixel_values = processor(images=image_data, return_tensors="pt").pixel_values.to('cuda')

            input_ids = tokenized_input['input_ids'].to('cuda')
            attention_mask = tokenized_input['attention_mask'].to('cuda')

            with torch.no_grad():
                outputs = simple_vlm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=conf.evaluate.generation.max_new_tokens,
                    num_beams=conf.evaluate.generation.num_beams,
                    early_stopping=True,
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"model_ckpt: {model_path}   ;Photo: {image} ;Generated Description: {generated_text}\n")
            
        del simple_vlm
        

if __name__ == "__main__":
    main()