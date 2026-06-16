#!/usr/bin/env bash
# ViMo advanced training: DeepSpeed ZeRO + gradient checkpointing (no packing).
# Advanced entry point train/train_vimo_zero3_gc.py — a ZeRO-3-symmetric variant of the
# basic recipe. Defaults to Stage 2 (SFT); set STAGE=1 / GEN_WEIGHTS_PATH for Stage 1.
# Stage 1 uses ZeRO-2 + gradient checkpointing; Stage 2 uses ZeRO-3 + gradient checkpointing.
# For packing on top of this, see scripts/train_vimo_packing.sh (normal) /
# scripts/train_vimo_zero3_gc_packing.sh (ZeRO-3) and docs/packing.md.
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
STAGE=${STAGE:-2}
# Stage 1 (alignment) -> ZeRO-2 + GC; Stage 2 (SFT) -> ZeRO-3 + GC. Override via CONFIG_FILE.
if [ "${STAGE}" = "1" ]; then
    CONFIG_FILE=${CONFIG_FILE:-configs/sft_zero2.yaml}              # ZeRO-2 accelerate config
else
    CONFIG_FILE=${CONFIG_FILE:-configs/sft_zero3.yaml}              # ZeRO-3 accelerate config
fi
TRAIN_SCRIPT=train/train_vimo_zero3_gc.py
VISION_CFG=${VISION_CFG:-configs/vimo_cfg.json}
# Stage 2 starts from the Stage 1 alignment checkpoint:
STAGE1_MODEL_PATH=${STAGE1_MODEL_PATH:-./Checkpoints_MLLM/vimo_stage1/vimo/run_1/checkpoint-step-xxxxx/tfmr}
MODEL_PATH=${MODEL_PATH:-${STAGE1_MODEL_PATH}}
DATA_PATH=${DATA_PATH:-data/vimo_sft_sample.json}                    # mixture config json
OUTPUT_DIR=${OUTPUT_DIR:-./Checkpoints_MLLM/vimo_zero3_gc/}
GEN_WEIGHTS_PATH=${GEN_WEIGHTS_PATH:-}                               # frozen TSIM-Tok weights, required for stage 1

# Stage 1 requires the generation weights; only when set are they appended as an argument
# (leaving it unset for Stage 2 has no effect).
STAGE_ARGS=""
if [ -n "${GEN_WEIGHTS_PATH}" ]; then
    STAGE_ARGS="--gen_weights_path ${GEN_WEIGHTS_PATH}"
fi

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
    --model_path ${MODEL_PATH} \
    --mixture_config ${DATA_PATH} \
    --n_epochs 1 \
    --train_bsz_per_gpu 1 \
    --learning_rate 4e-5 \
    --gradient_accumulation_steps 4 \
    --output_dir ${OUTPUT_DIR} \
    --extra_vision_cfg ${VISION_CFG} \
    --max_ckpts 100 \
    --stage ${STAGE} \
    ${STAGE_ARGS} \
    --gradient_checkpointing \
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
    --resume_from_checkpoint latest
