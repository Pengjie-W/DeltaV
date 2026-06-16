#!/bin/bash
set -e

# Locate the repo root (this script lives in tools/data_processing/similarity/, 3 levels deep)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

INPUT_JSON=$1
OUTPUT_JSON=$2
CKPT_PATH=${3:-weights/dinov2/dinov2_vitb14.pth}
NUM_WORKERS=${4:-8}

# DINOv2 loading: by default (left empty) torch.hub auto-downloads the code + weights;
# for offline/intranet, set VISUAL_EXTRACTOR_REPO=/path/to/dinov2 (local clone) + a local CKPT_PATH.
VISUAL_EXTRACTOR_REPO=${VISUAL_EXTRACTOR_REPO:-}

# Distributed args: default to a single node (single/multi GPU)
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  tools/data_processing/similarity/compute_similarity.py \
  --input_json "${INPUT_JSON}" \
  --output_json "${OUTPUT_JSON}" \
  --visual_extractor_ckpt "${CKPT_PATH}" \
  --visual_extractor_repo "${VISUAL_EXTRACTOR_REPO}" \
  --num_workers "${NUM_WORKERS}"
