#!/usr/bin/env bash
# TSIM-Tok Stage 2 — multi-image, variable-token-budget training (--finetune).
# Multiple data sources are mixed with --data_ratios; each carries per-sample num_tokens.
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
RESULTS_DIR=${RESULTS_DIR:-checkpoints/tsim_tok_stage2}
# Space-separated source list (override via env). Defaults to the bundled sample.
DATA_PATHS=${DATA_PATHS:-data/tsim_tok_stage2_sample.json}
DATA_RATIOS=${DATA_RATIOS:-1}
GBS=${GBS:-32}
BMC=${BMC:-32}
BW=${BW:-4}
VAL_DATA_PATHS=${VAL_DATA_PATHS:-data/tsim_tok_stage2_sample.json}
VQ_CKPT=${VQ_CKPT:-}                       # resume from Stage 1 .pt
RESUME=""; [ -n "${VQ_CKPT}" ] && RESUME="--vq-ckpt ${VQ_CKPT}"

scripts/torchrun.sh train/train_tsim_tok.py \
    --image-size 256 --results-dir "${RESULTS_DIR}" --mixed-precision none \
    --base_model_path "${BASE_MODEL_PATH}" \
    --data_paths ${DATA_PATHS} \
    --data_ratios ${DATA_RATIOS} \
    --global-batch-size ${GBS} --bucket_max_counts ${BMC} --bucket_widths ${BW} \
    --num-workers 32 --ckpt-every 5000 --eval-every 5000 --epochs 1 \
    --transformer-config-file configs/vit_transformer.yaml --log-every 50 \
    --lr 16e-4 --min-lr 1e-4 --ema --disc-start 10000 \
    ${RESUME} \
    --warming-up-steps 5000 --extra_cfg "${EXTRA_CFG}" \
    --val-data-paths ${VAL_DATA_PATHS} \
    --use_pad 0 --train-stage 1 --val_n_delta 144 --val_bs 64 \
    --disc-weight 0.1 --disc-lr 4e-4 --disc-min-lr 1e-5 --finetune
