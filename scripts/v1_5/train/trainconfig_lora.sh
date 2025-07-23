#!/bin/bash


# 默认参数
: ${DEVICE:=localhost:0,1,2,3}
: ${MASTER_PORT:=29600}
: ${LORA_RANK:=32}
: ${LORA_ALPHA:=64}
: ${PER_DEVICE_TRAIN_BATCH_SIZE:=4}
: ${NUM_TRAIN_EPOCHS:=3}
: ${TRAINER_NAME:="LoRASculpt"}
: ${DATASET_NAME:=""}
: ${LEARNING_RATE:=2e-4}
: ${OUTPUT_DIR:=""}

: ${GRADIENT_ACCUMULATION_STEPS:=1}
: ${MODEL_NAME_OR_PATH:="your_path_to_base_model/llava-v1.5-7b-ft"}
: ${DEEPSPEED_ZEROFILE:="LoRASculpt/scripts/zero2.json"}


if [ "$DATASET_NAME" == "iconqa_txt" ]; then
    data_path="your_path_to_train_json_scripts/iconqa_txt-train.json"
    image_folder="your_path_to_image_folder/iconqa"
elif [ "$DATASET_NAME" == "coco" ]; then
    data_path="your_path_to_train_json_scripts/coco-train.json"
    image_folder="your_path_to_image_folder/coco"
else
    echo "Unsupported DATASET_NAME: $DATASET_NAME"
    exit 1
fi



deepspeed --include $DEVICE --master_port $MASTER_PORT llava/train/train_mem.py \
    --lora_enable True --lora_r $LORA_RANK --lora_alpha $LORA_ALPHA --mm_projector_lr 2e-5 \
    --deepspeed $DEEPSPEED_ZEROFILE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --version v1 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower your_path_to_vision_tower/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 15 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --trainer_name $TRAINER_NAME
