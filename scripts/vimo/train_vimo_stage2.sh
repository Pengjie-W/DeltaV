#!/usr/bin/env bash
# ViMo Stage 2 (SFT): all params updated except TSIM-Tok and the understanding MLP.
# Basic recipe — DeepSpeed ZeRO-1 (configs/sft.yaml), no gradient_checkpointing, no packing.
# For ZeRO-3 / gradient_checkpointing / packing, see docs/advanced_zero3_gc.md and docs/packing.md.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NODE_COUNT=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
export MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TOTAL_GPUS=$(($NODE_COUNT * $PROC_PER_NODE))

CONFIG_FILE=${CONFIG_FILE:-configs/sft.yaml}
TRAIN_SCRIPT=train/train_vimo.py
VISION_CFG=${VISION_CFG:-configs/vimo_cfg.json}
# Stage 2 starts from the Stage 1 alignment checkpoint:
STAGE1_MODEL_PATH=${STAGE1_MODEL_PATH:-./Checkpoints_MLLM/vimo_stage1/vimo/run_1/checkpoint-step-xxxxx/tfmr}
DATA_PATH=${DATA_PATH:-data/vimo_sft_sample.json}
OUTPUT_DIR=${OUTPUT_DIR:-./Checkpoints_MLLM/vimo_stage2/}
SHUFFLE=${SHUFFLE:-False}

COMMON_ARGS="
    --config_file ${CONFIG_FILE}
    --num_processes ${TOTAL_GPUS}
    --num_machines ${NODE_COUNT}
    --machine_rank ${NODE_RANK}
    --main_process_ip ${MASTER_ADDR}
    --main_process_port ${MASTER_PORT}
    --deepspeed_multinode_launcher standard
"

accelerate launch ${COMMON_ARGS} ${TRAIN_SCRIPT} \
    --model_path ${STAGE1_MODEL_PATH} \
    --mixture_config ${DATA_PATH} \
    --n_epochs 1 \
    --train_bsz_per_gpu 1 \
    --learning_rate 4e-5 \
    --gradient_accumulation_steps 4 \
    --output_dir ${OUTPUT_DIR} \
    --extra_vision_cfg ${VISION_CFG} \
    --max_ckpts 100 \
    --stage 2 \
    --loss_norm_mode separate \
    --image_loss_weight 1 \
    --use_json_num_tokens \
    --min_pixels 65536 \
    --max_pixels 1048576 \
    --num_workers 8 \
    --save_steps 1250 \
    --max_seq_len 8192 \
    --min_lr_ratio 0.1 \
    --warmup_rates 0.01 \
    --shuffle ${SHUFFLE} \
    --resume_from_checkpoint latest
