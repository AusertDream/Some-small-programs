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
    simple_vlm.llm.resize_token_embeddings(len(simple_vlm.tokenizer))
    # load dataset
    # simVLM_dataset = SimVLMDataset(qa_file_path=conf.train.data.questions_path, image_file_path=conf.train.data.image_path)
    simVLM_dataset = load_dataset('json', data_files=conf.train.data.questions_path)['train']
    # only use part of all dataset
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

        max_length = conf.train.max_length

        input_ids = tokenizer(formatted_input, padding="max_length", max_length=max_length, truncation=True).input_ids
        attention_mask = tokenizer(formatted_input, padding="max_length", max_length=max_length, truncation=True).attention_mask
        labels = torch.full_like(torch.tensor(input_ids), -100).tolist()
        assisstant_str = "assistant\n"
        assistant_start = formatted_input.find(assisstant_str) + len(assisstant_str)
        prompt_tokens = tokenizer(formatted_input[:assistant_start], add_special_tokens=False).input_ids
        prompt_len = len(prompt_tokens)
        assistant_end = input_ids.index(tokenizer.eos_token_id, prompt_len) + 1 if tokenizer.eos_token_id in input_ids[prompt_len:] else len(input_ids)
        labels[prompt_len:assistant_end] = input_ids[prompt_len:assistant_end]

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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataloader), eta_min=0.0)

    # get the checkpoint saving step
    saving_path = conf.train.model_save_path
    weight_paths = os.listdir(saving_path)
    def find_the_last_ckpt(weight_paths: list) -> str:
        saved_steps = [int(item.split('-')[-1]) for item in weight_paths]
        saved_steps.sort()
        ckpt_name = 'checkpoint-' + str(saved_steps[-1])
        return ckpt_name

    if len(weight_paths) == 0:
        last_stop_step = 0
    else:
        ckpt_name = find_the_last_ckpt(weight_paths)
        last_stop_step = int(ckpt_name.split('-')[-1]) + 1

    # start training
    step = 0
    total_steps = len(train_dataloader) * num_epochs
    training_bar = tqdm(total=total_steps-last_stop_step+1, desc="Training")
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            if step < last_stop_step:
                lr_scheduler.step()
                step += 1
                training_bar.update(1)
                continue
            batch = {k: v.to(conf.train.device_map) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs, answers = simple_vlm(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            # print(answers)
            loss = outputs.loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(simple_vlm.parameters(), 1.0)
            # log the projector's gradients
            if conf.wandb.mode == 'online':
                wandb.log({'train/loss': loss.item(), 'train/epoch': epoch+1, 'train/step': step, 'train/lr': lr_scheduler.get_last_lr()[0]})
                for name, p in simple_vlm.named_parameters():
                    wandb.log({f"train_debug/grad_norm/{name}": torch.norm(p.grad).item(), f"train_debug/param_norm/{name}": torch.norm(p).item(), f"train_debug/grad_param_ratio/{name}": torch.norm(p.grad).item() / (torch.norm(p).item() + 1e-12)})
                
                if step % 500 == 0:
                    # answers: {
                    #     "predicted_answer": answer,
                    #     "ground_truth": ground_truth
                    # }
                    wandb.log({"train_debug/sample_answer": wandb.Table(data=[[answers[0]["predicted_answer"], answers[0]["ground_truth"]]], columns=["predicted_answer", "ground_truth"])})
            optimizer.step()
            lr_scheduler.step()
            step += 1
            training_bar.update(1)
            training_bar.set_postfix({'epoch': epoch+1, 'loss': loss.item()})
            print(step)
            if step % conf.train.saving_steps == 0 or step == total_steps:
                print(step)
                save_model(simple_vlm, saving_path, step)
    # validation
    ColoredLogger.info("Starting validation...")
    simple_vlm.eval()
    validation_bar = tqdm(total=len(val_dataloader), desc="Validation")
    total_val_loss = 0.0
    with torch.no_grad():
        step = 0
        for batch in val_dataloader:
            batch = {k: v.to(conf.train.device_map) for k, v in batch.items()}
            outputs, answers = simple_vlm(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            if conf.wandb.mode == 'online':
                wandb.log({'val/loss': loss.item(), 'val/step': step})
            step+=1
            total_val_loss += loss.item()
            validation_bar.update(1)
    
    ColoredLogger.info(f"Validation Loss: {total_val_loss / len(val_dataloader)}")


if __name__ == "__main__":
    main()