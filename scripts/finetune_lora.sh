#!/bin/bash

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1.5-7b"
################## VICUNA ##################

 deepspeed --master_port=$((RANDOM + 10000)) --include localhost:0 /content/GeoChat/geochat/train/train_mem.py \
    --deepspeed /content/GeoChat/scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /content/GeoChat/llava-v1.5-7b \
    --version $PROMPT_VERSION \
    --data_path /content/GeoChat/population_instruct_all_prompts.json \
    --image_folder /content/GeoChat/prompt_prep/images  \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /content/GeoChat/llava-v1.5-7b/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /content/output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 7000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 16 \
    --report_to wandb
