#!/usr/bin/env bash
# TSIM-Tok evaluation — reconstruct images under per-sample token budgets and report SSIM.
# Uses the variable-token (--use_token) path; val data carries num_tokens per sample.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONBREAKPOINT=0
export NODE_COUNT=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
export MASTER_PORT=${MASTER_PORT:-5678}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-/path/to/Qwen3-VL-2B-Instruct}    # vision_config + visual init weights
EXTRA_CFG=${EXTRA_CFG:-configs/vimo_cfg.json}
RESULTS_DIR=${RESULTS_DIR:-checkpoints/tsim_tok_eval}
VQ_CKPT=${VQ_CKPT:-weights/tsim_tok/tsim_tok.pt}
VAL_DATA=${VAL_DATA:-data/tsim_tok_stage2_sample.json}

scripts/torchrun.sh inference/eval_tsim_tok.py \
    --image-size 256 --results-dir "${RESULTS_DIR}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --num-workers 32 \
    --transformer-config configs/vit_transformer.yaml \
    --vq-ckpt "${VQ_CKPT}" \
    --extra_cfg "${EXTRA_CFG}" --val-data "${VAL_DATA}" \
    --val_n_delta 144 --val_bs 64 --batch-size 1 \
    --metric_image_select exclude_first --use_token
