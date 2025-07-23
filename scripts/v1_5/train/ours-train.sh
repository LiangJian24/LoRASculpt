#!/bin/bash


export DEVICE=localhost:0,1,2,3
OUTPUT_DIR_PREFIX="your_path_to_checkpoints/llava-v1.5-7b"


export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export DEEPSPEED_ZEROFILE="./scripts/zero2.json"
export NUM_TRAIN_EPOCHS=3
export LORA_RANK=32
export LORA_ALPHA=64
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LAMBDA=1e-3



#####################################################################################
export DATASET_NAME='iconqa_txt'    # 'iconqa_txt', 'coco'
export TRAINER_NAME="LoRASculpt"

HYPERPARAMS="lora-r${LORA_RANK}-a${LORA_ALPHA}-e${NUM_TRAIN_EPOCHS}-CMRLAMBDA${CMR_LAMBDA}-OMEGA${OMEGA}-RATIO${AB_PRESERVE_RATIO}"
export OUTPUT_DIR="${OUTPUT_DIR_PREFIX}-${DATASET_NAME}-${TRAINER_NAME}-${HYPERPARAMS}"
bash ./scripts/v1_5/train/trainconfig_lora.sh
#####################################################################################
