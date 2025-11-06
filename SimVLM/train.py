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

    # load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(llm_config.file_path, llm_config.model_name))
    processor = AutoProcessor.from_pretrained(os.path.join(visual_config.file_path, visual_config.model_name), use_fast=True)

    # load model
    simple_vlm = SimVLMModel.from_pretrained(projector_weights_path=conf.train.model_save_path, llm_config=llm_config, visual_config=visual_config, multi_model_config=multi_model_config)

    # load dataset
    # simVLM_dataset = SimVLMDataset(qa_file_path=conf.train.data.questions_path, image_file_path=conf.train.data.image_path)
    simVLM_dataset = load_dataset('json', data_files=conf.train.data.questions_path)['train']
    # only use 1000 of all dataset
    if conf.dev.dataset.select != -1:
        simVLM_dataset = simVLM_dataset.select(range(conf.dev.dataset.select))

    def process_image_qa(example):
        r"""
        process conversation and image to input id and pixel values.
        get label and attention mask.
        """
        # process image
        image = Image.open(os.path.join(conf.train.data.image_path, example['image'])).convert('RGB')
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        formatted_input = tokenizer.apply_chat_template(example['conversations'], tokenize=False)

        part1, part2 = formatted_input.split("assistant\n")
        part2, part3 = part2.split("<|im_end|>\n")
        part1 = part1 + 'assistant\n'
        part3 = '<|im_end|>\n'

        max_length = conf.train.max_length
        
        # get the input_ids, attention_mask, labels
        tokenized_part1 = tokenizer(part1, return_tensors="pt")
        tokenized_part2 = tokenizer(part2, return_tensors="pt")
        tokenized_part3 = tokenizer(part3, return_tensors="pt")
        input_ids = torch.cat([tokenized_part1.input_ids, tokenized_part2.input_ids, tokenized_part3.input_ids], dim=-1).squeeze(0)
        prompt_len = tokenized_part1.input_ids.size(-1)
        answer_len_with_special_token = tokenized_part2.input_ids.size(-1) + tokenized_part3.input_ids.size(-1)
        answer_len = tokenized_part2.input_ids.size(-1)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([-100 for _ in range(prompt_len)] + input_ids[prompt_len:prompt_len+answer_len].tolist() + [-100 for _ in range(tokenized_part3.input_ids.size(-1))])

        # padding
        if input_ids.size(-1) < max_length:
            pad_len = max_length - input_ids.size(-1)
            input_ids = torch.cat([input_ids, torch.tensor([tokenizer.pad_token_id for _ in range(pad_len)])], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)], dim=-1)
            labels = torch.cat([labels, torch.tensor([-100 for _ in range(pad_len)])], dim=-1)
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values.squeeze(0),
            'labels': labels
        }
    
    simVLM_dataset = simVLM_dataset.map(process_image_qa, batched=False, remove_columns=simVLM_dataset.column_names, load_from_cache_file=False, cache_file_name=None)
    val_ratio = conf.val.ratio
    train_ratio = 1 - val_ratio
    split_dataset = simVLM_dataset.train_test_split(test_size=val_ratio)
    train_dataset, val_dataset = split_dataset['train'], split_dataset['test']
    
    train_dataloader = DataLoader(train_dataset, batch_size=conf.train.batch_size, shuffle=True, collate_fn=default_data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=conf.val.batch_size, shuffle=False, collate_fn=default_data_collator)

    num_epochs = conf.train.num_epochs
    learning_rate = conf.train.learning_rate
    optimizer = torch.optim.AdamW(simple_vlm.parameters(), lr=learning_rate)

    # get the checkpoint saving step
    saving_path = conf.train.model_save_path
    weight_paths = os.listdir(saving_path)
    def find_the_last_ckpt(weight_paths: list) -> str:
        saved_steps = [int(item.split('-')[-1]) for item in weight_paths]
        saved_steps.sort()
        ckpt_name = 'checkpoint-' + str(saved_steps[-1])
        return ckpt_name

    if len(weight_paths) == 0:
        last_stop_step = 1
    else:
        ckpt_name = find_the_last_ckpt(weight_paths)
        last_stop_step = int(ckpt_name.split('-')[-1]) + 1

    # start training
    step = 1
    total_steps = len(train_dataloader) * num_epochs
    training_bar = tqdm(total=total_steps-last_stop_step+1, desc="Training")
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            if step < last_stop_step:
                step += 1
                continue
            batch = {k: v.to(conf.train.device_map) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = simple_vlm(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(simple_vlm.parameters(), 1.0)
            optimizer.step()

            step += 1
            training_bar.update(1)
            if conf.wandb.mode == 'online':
                wandb.log({'train/loss': loss.item(), 'train/epoch': epoch+1, 'train/step': step})
            training_bar.set_postfix({'epoch': epoch+1, 'loss': loss.item()})

            if step % conf.train.saving_steps == 0:
                save_model(simple_vlm, saving_path, step)
    # validation
    ColoredLogger.info("Starting validation...")
    simple_vlm.eval()
    validation_bar = tqdm(total=len(val_dataloader), desc="Validation")
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(conf.train.device_map) for k, v in batch.items()}
            outputs = simple_vlm(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            if conf.wandb.mode == 'online':
                wandb.log({'val/loss': loss.item()})
            total_val_loss += loss.item()
            validation_bar.update(1)
    
    ColoredLogger.info(f"Validation Loss: {total_val_loss / len(val_dataloader)}")


if __name__ == "__main__":
    main()