#!/usr/bin/env bash
# Step 0 of packing: add a per-sample token_length to every sample, and emit a rewritten
# mixture config that points at the length-augmented jsonl files. Feed OUT_CONFIG (DATA_PATH)
# to either packing recipe: scripts/train_vimo_packing.sh (normal, ZeRO-2) or
# scripts/train_vimo_zero3_gc_packing.sh (ZeRO-3 + gradient checkpointing). See docs/packing.md.
#
# token_length = text tokens (image placeholders stripped) + input-image vision tokens
# (smart_resize, patch 16, merge 2) + input/output incremental num_tokens (each plus a
# small per-image wrapper). A sqlite cache makes re-runs incremental.
set -e

# ---- repo root (this script lives in scripts/) ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ---- paths (override via env) ----
DATA_PATH=${DATA_PATH:-data/vimo_sft_sample.json}                    # source mixture config json
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/path/to/Qwen3-VL-2B-Instruct}    # tokenizer (Qwen3-VL-2B)
OUT_CONFIG=${OUT_CONFIG:-data/vimo_sft_sample.with_length.json}      # rewritten config for training
CACHE_DB=${CACHE_DB:-./add_lengths_cache.sqlite}
NUM_WORKERS=${NUM_WORKERS:-8}

python tools/data_processing/add_packing_lengths.py \
    --config-path ${DATA_PATH} \
    --tokenizer-path ${BASE_MODEL_PATH} \
    --rewritten-config-path ${OUT_CONFIG} \
    --cache-db ${CACHE_DB} \
    --output-suffix with_total_length \
    --min-pixels 65536 \
    --max-pixels 1048576 \
    --patch-size 16 \
    --merge-size 2 \
    --num-workers ${NUM_WORKERS}
