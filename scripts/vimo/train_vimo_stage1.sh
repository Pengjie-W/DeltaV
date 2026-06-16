#!/usr/bin/env bash
# ViMo Stage 1 (alignment): only the generation MLP + visual head are trained.
# Basic recipe — DeepSpeed ZeRO-1 (configs/sft.yaml), no gradient_checkpointing, no packing.
# For ZeRO-3 / gradient_checkpointing / packing, see docs/advanced_zero3_gc.md and docs/packing.md.
set -e

# ---- repo root (this script lives in scripts/) ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ---- distributed env (single node by default) ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NODE_COUNT=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
export MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TOTAL_GPUS=$(($NODE_COUNT * $PROC_PER_NODE))

# ---- paths (override via env) ----
CONFIG_FILE=${CONFIG_FILE:-configs/sft.yaml}
TRAIN_SCRIPT=train/train_vimo.py
VISION_CFG=${VISION_CFG:-configs/vimo_cfg.json}
GEN_WEIGHTS_PATH=${GEN_WEIGHTS_PATH:-weights/tsim_tok/tsim_tok.pt}   # frozen TSIM-Tok
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/path/to/Qwen3-VL-2B-Instruct}    # backbone init
DATA_PATH=${DATA_PATH:-data/vimo_sft_sample.json}                    # mixture config json
OUTPUT_DIR=${OUTPUT_DIR:-./Checkpoints_MLLM/vimo_stage1/}
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
    --model_path ${BASE_MODEL_PATH} \
    --mixture_config ${DATA_PATH} \
    --n_epochs 1 \
    --train_bsz_per_gpu 2 \
    --learning_rate 4e-3 \
    --gradient_accumulation_steps 4 \
    --output_dir ${OUTPUT_DIR} \
    --max_ckpts 100 \
    --extra_vision_cfg ${VISION_CFG} \
    --gen_weights_path ${GEN_WEIGHTS_PATH} \
    --stage 1 \
    --loss_norm_mode separate \
    --image_loss_weight 1 \
    --use_json_num_tokens \
    --min_pixels 65536 \
    --max_pixels 1048576 \
    --num_workers 8 \
    --save_steps 1250 \
    --min_lr_ratio 1 \
    --warmup_rates 0.01 \
    --max_seq_len 8192 \
    --shuffle ${SHUFFLE} \
    --resume_from_checkpoint latest
