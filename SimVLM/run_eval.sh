set -x

export CUDA_VISIBLE_DEVICES=7

python evaluate.py \
    visual_model.name=clip-vit-base-patch32 \
    visual_model.file_path=/data/jty/models \
    llm.name=qwen_3B \
    llm.file_path=/data/jty/models \
    evaluate.data.image_path=/data/jty/datasets/minimind-v/dataset/eval_images \
    evaluate.model.check_path=/data/jty/models/mySimVLM/expr1/checkpoint-23000/projector_weights.pth \
    evaluate.generation.max_length=1024 \
    evaluate.generation.num_beams=4 \
    wandb.mode='offline' \
    wandb.project='mySimVLM' \
    wandb.name='simVLM_train' \
    dev.dataset.select=-1 \
