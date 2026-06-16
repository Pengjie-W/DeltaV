# Advanced ViMo training: ZeRO-3 + gradient checkpointing

The basic recipe (`train/train_vimo.py`, README §1) uses DeepSpeed ZeRO-1 without gradient
checkpointing or packing — simplest to read and run. For larger-scale / memory-constrained
training, use the advanced entry point which adds **gradient checkpointing**, a higher ZeRO
stage, per-task batch shuffling, and an OOM-skip mechanism.

## Files

- Training: `train/train_vimo_zero3_gc.py`
- Model: `vimo/modeling_vimo_zero3_gc.py` — a ZeRO-3-symmetric variant of
  `vimo/modeling_vimo.py`. It adds:
  - symmetric `n_delta` padding so all ranks build identical parameter shapes under ZeRO-3
    partitioning (avoids deadlocks from divergent module construction), and
  - gradient-checkpointing plumbing.
  The token-budget configuration logic is identical to the basic model.
- Module: `vimo/tsim_tok/modules_tsim_tok_zero3_modsym.py` — the matching ZeRO-3 tokenizer module.
- Accelerate configs: `configs/sft_zero2.yaml` (Stage 1) and `configs/sft_zero3.yaml` (Stage 2).
- Run wrapper: `scripts/vimo/train_vimo_zero3_gc.sh` (mirrors `scripts/vimo/train_vimo_stage2.sh`).

## Run

`scripts/vimo/train_vimo_zero3_gc.sh` mirrors `scripts/vimo/train_vimo_stage2.sh` but swaps in the
advanced entry point, `--gradient_checkpointing`, and a ZeRO accelerate config chosen by
stage: **Stage 1 (alignment) → ZeRO-2 + gradient checkpointing** (`configs/sft_zero2.yaml`),
**Stage 2 (SFT) → ZeRO-3 + gradient checkpointing** (`configs/sft_zero3.yaml`). Override paths
(or `CONFIG_FILE`) via env vars:

```bash
# Stage 2 (default) — resume from a Stage 1 alignment checkpoint (ZeRO-3 + GC)
STAGE1_MODEL_PATH=./Checkpoints_MLLM/vimo_stage1/.../tfmr \
DATA_PATH=data/vimo_sft_sample.json \
bash scripts/vimo/train_vimo_zero3_gc.sh

# Stage 1 (alignment) — needs the frozen TSIM-Tok weights (ZeRO-2 + GC)
STAGE=1 MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
GEN_WEIGHTS_PATH=weights/tsim_tok/tsim_tok.pt \
DATA_PATH=data/vimo_sft_sample.json \
bash scripts/vimo/train_vimo_zero3_gc.sh
```

The ZeRO config is the only stage-conditional part (Stage 1 → ZeRO-2, Stage 2 → ZeRO-3, both
with gradient checkpointing); Stage 1 additionally needs the frozen TSIM-Tok weights via
`GEN_WEIGHTS_PATH`. The remaining hyperparameters (learning rate, batch size, sequence length,
`min_lr_ratio`) are fixed in the script — read it for the exact values, or override `CONFIG_FILE`
and the path env vars as needed.

Note `--gradient_checkpointing` is a boolean flag (`store_true`) — pass it bare, with no
value.

Optional advanced flags (defaults shown) live on the same entry point:

- `--random_task_plan_per_rank 0` — set to `1` to give each rank a different per-epoch
  task-mixing plan (default `0`: all ranks share one shuffled plan).
- `--oom_skip_radius 1` — on resume, auto-skip steps within this radius of the last
  recorded OOM `step_in_epoch`.
- `--target_global_batch_size` / `--target_global_batch_size_tolerance 0` — constrain the
  per-step global sample count (mainly used together with packing).

## Notes

- The advanced model imports the same `vimo` package (`vimo.backbone_vimo`,
  `vimo.configuration_vimo`) and the ZeRO-3 tokenizer module; it does not change ViMo's
  parameters, so checkpoints remain compatible with the basic model for inference.
- For packing on top of this, see [packing.md](packing.md).
