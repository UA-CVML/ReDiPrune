#!/bin/bash
set -xeuo pipefail

LOG_DIR=./logs
RUN_NAME=radiprune_llava_videochatgpt
mkdir -p "$LOG_DIR/$RUN_NAME"
LOG_NAME=rediprune_llava_videochatgpt.log

CUDA_VISIBLE_DEVICES=0,1 EVAL_TIME=TRUE LAYER_INDEX=0 SUBSET_RATIO=0.1 python -m accelerate.commands.launch \
  --num_processes=2 \
  -m lmms_eval \
    --model llava_vid \
    --model_args "pretrained=lmms-lab/LLaVA-NeXT-Video-7B-DPO,attn_implementation=sdpa" \
    --tasks videochatgpt_temporal \
    --include_path lmms_eval/tasks/videochatgpt \
    --batch_size 1 \
    --verbosity INFO \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME | tee $LOG_DIR/$RUN_NAME/$LOG_NAME

echo "===== Summary memory/latency ====="
python3 ./extract_time.py --path $LOG_DIR/$RUN_NAME/$LOG_NAME