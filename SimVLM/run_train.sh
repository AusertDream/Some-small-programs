set -x

export CUDA_VISIBLE_DEVICES=1

python train.py \
    visual_model.name=clip-vit-base-patch32 \
    visual_model.file_path=/data/jty/models \
    llm.name=base_model \
    llm.file_path=/data/jty/models/mySimVLM \
    train.data.questions_path=/data/jty/datasets/minimind-v/pretrain_data.jsonl \
    train.data.image_path=/data/jty/datasets/minimind-v/pretrain_images/ \
    train.max_length=128 \
    train.batch_size=24 \
    train.num_epochs=1 \
    train.learning_rate=1e-4 \
    train.model_save_path=/data/jty/models/mySimVLM/expr7 \
    train.saving_steps=4000 \
    train.device_map='cuda' \
    val.ratio=0.1 \
    val.batch_size=24 \
    wandb.mode='online' \
    wandb.project='mySimVLM' \
    wandb.name='simVLM_train' \
    dev.dataset.select=-1 \
