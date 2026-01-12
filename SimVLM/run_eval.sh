set -x

export CUDA_VISIBLE_DEVICES=2

python evaluate.py \
    visual_model.name=clip-vit-base-patch32 \
    visual_model.file_path=/data/jty/models \
    evaluate.data.image_path=/data/jty/datasets/minimind-v/dataset/eval_images \
    evaluate.model.check_path=/data/jty/models/mySimVLM/expr7 \
    evaluate.generation.max_length=1024 \
    evaluate.generation.max_new_tokens=256 \
    evaluate.generation.num_beams=4 \
    llm.name=llm \
    llm.file_path=/data/jty/models/mySimVLM/expr7 \
    wandb.mode='offline' \
    wandb.project='mySimVLM' \
    wandb.name='simVLM_train' \
    dev.dataset.select=-1 \
