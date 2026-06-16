#!/usr/bin/env bash
# ViMo interleaved inference (Zebra-CoT / StructCoT share one entrypoint; switch with JSON_PATH).
# If samples carry precomputed `num_tokens`, pass --use_json_num_tokens. Otherwise enable the
# TSIM Router (--use_tsim_router) to allocate incremental token budgets from temporal similarity.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export NODE_COUNT=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
export MASTER_PORT=${MASTER_PORT:-29501}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TOTAL_GPUS=$(($NODE_COUNT * $PROC_PER_NODE))

MODEL_PATH=${MODEL_PATH:-weights/vimo_2b}
EXTRA_CFG=${EXTRA_CFG:-configs/vimo_cfg.json}
JSON_PATH=${JSON_PATH:-data/zebra_test_sample.json}   # or data/struct_test_sample.json
EXP_NAME=${EXP_NAME:-vimo_infer}
OUT_DIR=${OUT_DIR:-./test_json/${EXP_NAME}}

# Token source: precomputed json tokens (default) OR TSIM Router fallback.
TOKEN_ARGS=${TOKEN_ARGS:---use_json_num_tokens --strict_num_tokens}
# To use the router instead, set e.g.:
#   TOKEN_ARGS="--use_tsim_router --visual_extractor_repo /path/to/dinov2 --visual_extractor_ckpt weights/dinov2/dinov2_vitb14.pth"

# Visualization: leave empty for pure inference/eval (text-only .json output, then
# merge + extract). To also decode and save the generated images, set e.g.:
#   VIS_ARGS="--decode_and_save_image --concat_gt_images"
# In that mode results are streamed to .jsonl and the merge/extract steps are skipped.
VIS_ARGS=${VIS_ARGS:-}

# Dataset order: leave empty to keep JSON order. To shuffle before distributed sharding,
# set e.g. SHUFFLE_ARGS="--shuffle_dataset --shuffle_seed 42" (use a fixed seed to reproduce).
SHUFFLE_ARGS=${SHUFFLE_ARGS:-}

torchrun \
  --nnodes=${NODE_COUNT} --node_rank=${NODE_RANK} --nproc_per_node=${PROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
  inference/infer_vimo.py \
  --model_path ${MODEL_PATH} \
  --extra_cfg ${EXTRA_CFG} \
  --json_path ${JSON_PATH} \
  --save_dir ${OUT_DIR}/images \
  --result_dir ${OUT_DIR}/json_parts \
  --final_json ${OUT_DIR}/image_output.json \
  --is_incremental_encoding \
  ${TOKEN_ARGS} \
  --image_token_num_per_image 145 \
  --min_pixels 65536 --max_pixels 1048576 \
  ${VIS_ARGS} \
  ${SHUFFLE_ARGS}

# Pure inference (no image saving): merge per-rank .json parts and extract answers.
# When saving images (VIS_ARGS set), outputs are PNGs + .jsonl; skip these steps.
if [ -z "${VIS_ARGS}" ]; then
  python tools/inference/merge_results.py \
    --result_dir ${OUT_DIR}/json_parts \
    --final_json ${OUT_DIR}/image_output.json \
    --world_size ${TOTAL_GPUS}

  python tools/inference/extract.py --final_json ${OUT_DIR}/image_output.json
fi
