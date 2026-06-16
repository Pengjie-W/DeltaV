#!/usr/bin/env bash
# TSIM-Tok Stage 1 — single-image training. Visual backbone stays frozen.
# Unified trainer: reuses train/train_tsim_tok.py (single data source = --data_ratios 1).
# Difference from Stage 2: single source, no --finetune, --return_num_tokens 0.
# Note: in Stage 1 each sample has only 1 image (num_images=1), so
# total_len = sum(num_images) - len(num_images) = 0 and num_tokens is always the empty list [].
# A single image has no inter-frame delta, so no random sampling from n_delta actually
# happens (random sampling only occurs in Stage 2 with multiple images).
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NODE_COUNT=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
export MASTER_PORT=${MASTER_PORT:-5678}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-/path/to/Qwen3-VL-2B-Instruct}    # vision_config + visual init weights
EXTRA_CFG=${EXTRA_CFG:-configs/vimo_cfg.json}
DATA_PATH=${DATA_PATH:-data/tsim_tok_stage1_sample.json}
RESULTS_DIR=${RESULTS_DIR:-checkpoints/tsim_tok_stage1}
VAL_DATA=${VAL_DATA:-data/tsim_tok_stage1_sample.json}
VQ_CKPT=${VQ_CKPT:-}                       # optional resume .pt
RESUME=""; [ -n "${VQ_CKPT}" ] && RESUME="--vq-ckpt ${VQ_CKPT}"

scripts/torchrun.sh train/train_tsim_tok.py \
    --image-size 256 --results-dir "${RESULTS_DIR}" --mixed-precision none \
    --base_model_path "${BASE_MODEL_PATH}" \
    --data_paths "${DATA_PATH}" \
    --data_ratios 1 \
    --global-batch-size 32 --bucket_max_counts 32 --bucket_widths 4 --num-workers 32 \
    --ckpt-every 5000 --eval-every 5000 --epochs 7 \
    --transformer-config-file configs/vit_transformer.yaml --log-every 50 --lr 16e-4 --ema --disc-start 75000 \
    ${RESUME} \
    --warming-up-steps 5000 --extra_cfg "${EXTRA_CFG}" \
    --val-data-paths "${VAL_DATA}" \
    --use_pad 0 --train-stage 1 --val_n_delta 144 --val_bs 64 \
    --return_num_tokens 0 \
    --disc-weight 0.1 --disc-lr 4e-4