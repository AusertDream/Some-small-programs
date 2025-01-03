import json
import time

import fontTools.ttLib.tables.sbixStrike
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as nn
import numpy as np
import torch.nn.functional as F
from myTransformer import myTransformerClass
import datasets
from datasets import load_dataset_builder, load_dataset
from torch.nn.parallel import DataParallel
import os
import sys
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import sacrebleu



def getMask(target):
    batch_size = target.shape[0]
    seq_len = target.shape[1]
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    # (batch_size, seq_len, seq_len)
    return mask
    

if __name__ == "__main__":
    with open("./model_config.json","r") as f:
        config = json.load(f)
    max_len, d_model, drop_prob, block_nums, head_nums, d_ff=config["max_len"], config["d_model"], config["dropout_rate"], config["block_nums"], config["head_nums"], config["d_ff"]
    debug_mode, data_size_debug, mode = config["debug_mode"], config["data_size_debug"], config["mode"]
    version_name, SMS_toggle = config["version_name"], config["SMS_toggle"]
    root_path = config["root_path"]
    SMSlog_path = config["SMS_path"]
    data_path = os.path.join(root_path, "data")
    model_path = os.path.join(root_path, "model")
    log_path = os.path.join(root_path, "log")

    data_de_en = os.path.join(data_path, "de-en")
    print("load config done")
    
    ds = load_dataset("parquet", data_dir=data_de_en)

    train_dataset = ds["train"]
    validation_dataset = ds["validation"]
    test_dataset = ds["test"]

    train_dataset = train_dataset.shuffle(seed=42)
    validation_dataset = validation_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)

    if debug_mode == 1:
        train_dataset = train_dataset.select(range(data_size_debug))


    print("load dataset done")
    # expriment
    log_path_origin = os.path.join(log_path, "origin")
    log_path_addRelu_addXeInit = os.path.join(log_path, "addRelu_addXeInit")
    log_path_debug = os.path.join(log_path, "debug")
    model_path_origin = os.path.join(model_path, "original")

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    tokenizer.model_max_length = max_len
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def tokenize_function(examples):
        de_input_ids = []
        en_input_ids = []
        for translation in examples["translation"]:
            de_tokenized = tokenizer(translation["de"], padding="max_length", 
                                    truncation=True, max_length=tokenizer.model_max_length)
            en_tokenized = tokenizer(translation["en"], padding="max_length", 
                                    truncation=True, max_length=tokenizer.model_max_length)
            de_input_ids.append(de_tokenized["input_ids"])
            en_input_ids.append(en_tokenized["input_ids"])
        return {
            "de_input_ids": de_input_ids,
            "en_input_ids": en_input_ids,
        }

    
    def get_attention_mask(examples):
        de_attention_mask = []
        en_attention_mask = []
        for translation in examples["translation"]:
            de_tokenized = tokenizer(translation["de"], padding="max_length", 
                                    truncation=True, max_length=tokenizer.model_max_length)
            en_tokenized = tokenizer(translation["en"], padding="max_length", 
                                    truncation=True, max_length=tokenizer.model_max_length)
            de_attention_mask.append(de_tokenized["attention_mask"])
            en_attention_mask.append(en_tokenized["attention_mask"])
        return {
            "de_attention_mask": de_attention_mask,
            "en_attention_mask": en_attention_mask,
        }



    print("start tokenize")
    #get attention mask
    train_attention_mask = train_dataset.map(get_attention_mask, batched=True, remove_columns=["translation"])
    validation_attention_mask = validation_dataset.map(get_attention_mask, batched=True, remove_columns=["translation"])
    test_attention_mask = test_dataset.map(get_attention_mask, batched=True, remove_columns=["translation"])

    #tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["translation"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["translation"])


    # 设置vocab_size, vocabsize是128000，但是实际上pad id都跑到128256去了
    d_output = tokenizer.pad_token_id + 1
    vocab_size = tokenizer.pad_token_id + 1
    device = "cuda:1"
    model = myTransformerClass(vocab_size, d_model, max_len, 
                               drop_prob, block_nums, head_nums, 
                               d_ff, d_output, pad_idx=tokenizer.pad_token_id).to(device)
    
    # 初始化模型参数
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    model = DataParallel(model, device_ids=[1, 2])

    # setting
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.1)

    if mode == "train":
        print("start training")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            cnt = 0
            for i in tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch = train_dataset[i:min(i+batch_size, len(train_dataset))]
                batch_attention_mask = train_attention_mask[i:min(i+batch_size, len(train_dataset))]
                source_batch = batch["en_input_ids"]
                target_batch = batch["de_input_ids"]
                source_attention_mask = torch.tensor(batch_attention_mask["en_attention_mask"]).to(device)
                targets = torch.tensor(target_batch).to(device)
                source = torch.tensor(source_batch).to(device)
                mask = getMask(targets).to(device)
                optimizer.zero_grad()
                outputs = model(source, targets, mask, source_attention_mask)
                a = outputs.view(-1, outputs.size(-1))
                b = targets.view(-1)
                loss_value = loss(a, b)
                loss_value = (loss_value * source_attention_mask.view(-1)).sum() / source_attention_mask.sum()
                loss_value.backward()
                optimizer.step()
                lr_scheduler.step(loss_value)
                # 每1000个batch记录一次平均loss，存放在txt文件中
                total_loss += loss_value.item()
                cnt += 1
                if cnt % 1000 == 0:
                    file_name = "train_loss"+".txt"
                    with open(os.path.join(log_path_origin, file_name), "a") as f:
                        f.write(str(total_loss / cnt))
                        f.write("\n")
                    cnt = 0
                    total_loss = 0
                # 最后一个batch，如果cnt没有正好凑到10000，还是要记录一下平均值。
                if epoch == num_epochs-1 and i + batch_size >= len(train_dataset):
                    if cnt !=0:
                        file_name = "train_loss"+".txt"
                        with open(os.path.join(log_path_origin, file_name), "a") as f:
                            f.write(str(total_loss / cnt))
                            f.write("\n")
                        cnt = 0
                        total_loss = 0

        print("training done")
        # Save the model
        file_name = version_name+".pth"
        if not os.path.exists(model_path_origin):
            os.makedirs(model_path_origin)
        torch.save(model, os.path.join(model_path_origin, file_name))
        print("model saved")


    if mode == "validation":
        # load params
        file_name = os.path.join(model_path_origin, version_name + ".pth")
        model = torch.load(file_name)
        print("start validation")
        # Validation
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(validation_dataset), batch_size), desc="Validation:"):
                batch = validation_dataset[i:min(len(validation_dataset), i+batch_size)]
                batch_attention_mask = validation_attention_mask[i:min(len(validation_dataset), i+batch_size)]
                targets = torch.tensor(batch["de_input_ids"]).to(device)
                source = torch.tensor(batch["en_input_ids"]).to(device)
                source_attention_mask = torch.tensor(batch_attention_mask["en_attention_mask"]).to(device)
                mask = getMask(targets).to(device)

                outputs = model(source, targets, mask, source_attention_mask)
                a = outputs.view(-1, outputs.size(-1))
                b = targets.view(-1)
                val_loss_value = loss(a, b)
                val_loss_value = (val_loss_value * source_attention_mask.view(-1)).sum() / source_attention_mask.sum()
                # print("loss:", val_loss_value.item())
                # 每1个batch记录一次平均loss，存放在txt文件中
                file_name = "valid_loss"+".txt"
                with open(os.path.join(log_path_origin, file_name), "a") as f:
                    f.write(str(val_loss_value.item()))
                    f.write("\n")
    
    
    if mode == "test":
        # load params
        file_name = os.path.join(model_path_origin, version_name + ".pth")
        model = torch.load(file_name)
        print("start test")
        model.eval()
        totel_bleu = 0
        with torch.no_grad():
            for i in tqdm(range(0, len(test_dataset), batch_size), desc="Test:"):
                batch = test_dataset[i:min(len(test_dataset), i+batch_size)]
                batch_attention_mask = test_attention_mask[i:min(len(test_dataset), i+batch_size)]
                targets = torch.tensor(batch["de_input_ids"]).to(device)
                source = torch.tensor(batch["en_input_ids"]).to(device)
                source_attention_mask = torch.tensor(batch_attention_mask["en_attention_mask"]).to(device)
                mask = getMask(targets).to(device)

                outputs = model(source, targets, mask, source_attention_mask)
                # decode the outputs
                decoded_outputs = tokenizer.batch_decode(outputs.argmax(dim=-1), skip_special_tokens=True)
                decoded_label = tokenizer.batch_decode(targets, skip_special_tokens=True)
                a = outputs.view(-1, outputs.size(-1))
                b = targets.view(-1)
                test_loss_value = loss(a, b)
                test_loss_value = (test_loss_value * source_attention_mask.view(-1)).sum() / source_attention_mask.sum()

                
                # 计算bleu
                bleu = sacrebleu.corpus_bleu(decoded_outputs, [decoded_label])
                totel_bleu += bleu.score


                # # 每1个batch记录一次平均loss，存放在txt文件中
                # file_name = "test_loss"+".txt"
                # with open(os.path.join(log_path_origin, file_name), "a") as f:
                #     f.write(str(test_loss_value.item()))
                #     f.write("\n")

                # outputs_name = "outputs"+".txt"
                # with open(os.path.join(log_path_origin, outputs_name), "a") as f:
                #     f.write(str(decoded_outputs))
                #     f.write("\n")


        # output the final bleu
        count = int(np.ceil(len(test_dataset) / batch_size))
        print(f"Average BLEU score: {totel_bleu / count}")
        




        
    
    

        
                

    


