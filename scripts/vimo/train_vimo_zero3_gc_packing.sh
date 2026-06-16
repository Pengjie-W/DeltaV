#!/usr/bin/env bash
# ViMo ADVANCED training with SEQUENCE PACKING (ZeRO-3 + gradient checkpointing + packing).
# Same entry point train/train_vimo_zero3_gc.py as scripts/train_vimo_zero3_gc.sh, with the
# packing machinery enabled. Packing needs a precomputed per-sample token_length:
#   1) run  bash scripts/add_packing_lengths.sh   to produce a *.with_length config, then
#   2) point DATA_PATH at that config and run this script.
# Defaults to Stage 2 (SFT); set STAGE=1 for the alignment stage (needs GEN_WEIGHTS_PATH).
# For the normal packing recipe (ZeRO-2, no gradient checkpointing) see
# scripts/train_vimo_packing.sh. See docs/packing.md for details.
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
CONFIG_FILE=${CONFIG_FILE:-configs/sft_zero3.yaml}                   # ZeRO-3 accelerate config
TRAIN_SCRIPT=train/train_vimo_zero3_gc.py
VISION_CFG=${VISION_CFG:-configs/vimo_cfg.json}
STAGE=${STAGE:-2}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/path/to/Qwen3-VL-2B-Instruct}    # backbone init (stage 1)
STAGE1_MODEL_PATH=${STAGE1_MODEL_PATH:-./Checkpoints_MLLM/vimo_stage1/vimo/run_1/checkpoint-step-xxxxx/tfmr}
GEN_WEIGHTS_PATH=${GEN_WEIGHTS_PATH:-weights/tsim_tok/tsim_tok.pt}   # frozen TSIM-Tok (stage 1)
# Config WITH per-sample token_length (output of scripts/add_packing_lengths.sh):
DATA_PATH=${DATA_PATH:-data/vimo_sft_sample.with_length.json}
OUTPUT_DIR=${OUTPUT_DIR:-./Checkpoints_MLLM/vimo_zero3_gc_packing/}

# ---- packing knobs (defaults reproduce the released recipe) ----
PACK_TOTAL_LENGTH=${PACK_TOTAL_LENGTH:-11264}                        # also used as --max_seq_len
THEORETICAL_GLOBAL_SAMPLE_NUM=${THEORETICAL_GLOBAL_SAMPLE_NUM:-512}

# ---- stage-conditional defaults ----
# Stage 1 (alignment): init from the base model, higher LR, needs the frozen TSIM-Tok weights.
# Stage 2 (SFT): resume from the Stage 1 checkpoint, lower LR.
if [ "${STAGE}" = "1" ]; then
    MODEL_PATH=${MODEL_PATH:-${BASE_MODEL_PATH}}
    LR=${LR:-4e-3}
    TRAIN_BSZ=${TRAIN_BSZ:-2}
    GRAD_ACCUM=${GRAD_ACCUM:-4}
    STAGE_ARGS="--gen_weights_path ${GEN_WEIGHTS_PATH}"
else
    MODEL_PATH=${MODEL_PATH:-${STAGE1_MODEL_PATH}}
    LR=${LR:-4e-5}
    TRAIN_BSZ=${TRAIN_BSZ:-1}
    GRAD_ACCUM=${GRAD_ACCUM:-1}
    STAGE_ARGS=""
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
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
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
    --max_seq_len ${PACK_TOTAL_LENGTH} \
    --min_lr_ratio 1 \
    --warmup_rates 0.01 \
    --enable_packing \
    --pack_total_length ${PACK_TOTAL_LENGTH} \
    --pack_total_length_threshold_ratio 0.9 \
    --enable_global_sample_mean_loss \
    --theoretical_global_sample_num ${THEORETICAL_GLOBAL_SAMPLE_NUM} \
    --resume_from_checkpoint latest
