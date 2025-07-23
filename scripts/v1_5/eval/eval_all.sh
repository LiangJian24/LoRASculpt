#!/bin/bash


OUTPUT_DIR_PREFIX="your_path_to_checkpoints/llava-v1.5-7b"




##########################################################################################
DATASET_NAME="iconqa_txt"       # 'iconqa_txt', 'coco'
TRAINER_NAME="LoRASculpt"

LORA_RANK=32
LORA_ALPHA=64
NUM_TRAIN_EPOCHS=3
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LAMBDA=1e-3

HYPERPARAMS="lora-r${LORA_RANK}-a${LORA_ALPHA}-e${NUM_TRAIN_EPOCHS}-CMRLAMBDA${CMR_LAMBDA}-OMEGA${OMEGA}-RATIO${AB_PRESERVE_RATIO}"
MODEL_LORA="${OUTPUT_DIR_PREFIX}-${DATASET_NAME}-${TRAINER_NAME}-${HYPERPARAMS}"
RESULT_DIR="${MODEL_LORA}/eval_results"
SUMMARY_OUTPUT_DIR="${MODEL_LORA}/SummaryResult_${DATASET_NAME}-${TRAINER_NAME}-${HYPERPARAMS}"
##########################################################################################









##########################################################################################
##### create path of summary file #####
mkdir -p $RESULT_DIR
current_time=$(date '+%Y-%m-%d %H:%M:%S')
summary_file=$SUMMARY_OUTPUT_DIR
> "$summary_file"    # Clear out the output file if it exists.
echo "Current Time: $current_time" >> "$summary_file"
echo "" >> "$summary_file"
##########################################################################################



EVAL_ON_ICONQA=./scripts/v1_5/eval/downstream/eval_gqa.sh/eval_iconqa_txt.sh
# EVAL_ON_COCO=./scripts/v1_5/eval/downstream/eval_gqa.sh/eval_coco.sh

EVAL_ON_GQA=./scripts/v1_5/eval/upstream/eval_gqa.sh/eval_gqa.sh
EVAL_ON_OKVQA=./scripts/v1_5/eval/upstream/eval_gqa.sh/eval_okvqa.sh
EVAL_ON_OCRVQA=./scripts/v1_5/eval/upstream/eval_gqa.sh/eval_ocrvqa.sh
EVAL_ON_TEXTVQA=./scripts/v1_5/eval/upstream/eval_gqa.sh/eval_textvqa.sh




bash $EVAL_ON_ICONQA $MODEL_LORA $RESULT_DIR $SUMMARY_OUTPUT_DIR
# bash $EVAL_ON_COCO $MODEL_LORA $RESULT_DIR $SUMMARY_OUTPUT_DIR

bash $EVAL_ON_OKVQA $MODEL_LORA $RESULT_DIR $SUMMARY_OUTPUT_DIR
bash $EVAL_ON_OCRVQA $MODEL_LORA $RESULT_DIR $SUMMARY_OUTPUT_DIR
bash $EVAL_ON_GQA $MODEL_LORA $RESULT_DIR $SUMMARY_OUTPUT_DIR
bash $EVAL_ON_TEXTVQA $MODEL_LORA $RESULT_DIR $SUMMARY_OUTPUT_DIR
