# Sequence packing

Packing concatenates multiple short samples into one fixed-length sequence to reduce
padding waste. It needs a precomputed per-sample token length, then a packing-aware trainer.
Two recipes are provided — they share the same packing engine and differ only in the
DeepSpeed ZeRO stage and whether gradient checkpointing is on:

- **Normal** — `train/train_vimo_packing.py` + `scripts/vimo/train_vimo_packing.sh`
  (basic ViMo model, ZeRO-2, no gradient checkpointing). The released 2B recipe.
- **Advanced** — `train/train_vimo_zero3_gc.py` + `scripts/vimo/train_vimo_zero3_gc_packing.sh`
  (ZeRO-3 + gradient checkpointing on top of packing, for larger-scale / memory-constrained runs).

Both support Stage 1 (alignment) and Stage 2 (SFT), selected with `STAGE`.

## 1. Compute per-sample lengths

`tools/data_processing/add_packing_lengths.py` adds a `token_length` field to each sample.
The wrapper script fills in the correct flag names and the 2B pixel budget:

```bash
DATA_PATH=data/vimo_sft_sample.json \
BASE_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
OUT_CONFIG=data/vimo_sft_sample.with_length.json \
bash scripts/data/add_packing_lengths.sh
```

`token_length` = text tokens (image placeholders stripped) + input-image vision tokens
(via `smart_resize`, patch 16, merge 2) + input/output incremental `num_tokens`
(each plus a small per-image wrapper). Per-jsonl results are written as
`*.with_total_length.jsonl` next to the source (with a sqlite cache to make re-runs
incremental), and `--rewritten-config-path` emits a new mixture config pointing at them —
that file (`OUT_CONFIG`) is what training consumes in step 2.

## 2. Train with packing — two recipes

Both recipes take the length-augmented config from step 1 as `DATA_PATH` and reproduce the
released packing knobs by default. Override paths and packing budget via env vars.

### 2.1 Normal packing (ZeRO-2, no gradient checkpointing)

Entry point `train/train_vimo_packing.py` (basic `vimo/modeling_vimo.py` model),
launcher `scripts/vimo/train_vimo_packing.sh`, accelerate config `configs/sft_zero2.yaml`.

```bash
# Stage 2 (default) — SFT, resume from a Stage 1 alignment checkpoint
STAGE1_MODEL_PATH=./Checkpoints_MLLM/vimo_stage1/.../tfmr \
DATA_PATH=data/vimo_sft_sample.with_length.json \
bash scripts/vimo/train_vimo_packing.sh

# Stage 1 (alignment) — init from the base model, needs the frozen TSIM-Tok weights
STAGE=1 BASE_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
GEN_WEIGHTS_PATH=weights/tsim_tok/tsim_tok.pt \
DATA_PATH=data/vimo_sft_sample.with_length.json \
bash scripts/vimo/train_vimo_packing.sh
```

The script picks stage-appropriate hyperparameters automatically: Stage 1 uses
`learning_rate 4e-3`, `train_bsz_per_gpu 2`, `gradient_accumulation_steps 4` and requires
`GEN_WEIGHTS_PATH`; Stage 2 uses `learning_rate 4e-5`, `train_bsz_per_gpu 1`,
`gradient_accumulation_steps 1`.

### 2.2 Advanced packing (ZeRO-3 + gradient checkpointing)

Entry point `train/train_vimo_zero3_gc.py` (the ZeRO-3-symmetric model variant, see
[advanced_zero3_gc.md](advanced_zero3_gc.md)), launcher
`scripts/vimo/train_vimo_zero3_gc_packing.sh`, accelerate config `configs/sft_zero3.yaml`.
Same invocations as above, just a different script:

```bash
# Stage 2 (default)
STAGE1_MODEL_PATH=./Checkpoints_MLLM/vimo_stage1/.../tfmr \
DATA_PATH=data/vimo_sft_sample.with_length.json \
bash scripts/vimo/train_vimo_zero3_gc_packing.sh

# Stage 1 (alignment)
STAGE=1 BASE_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
GEN_WEIGHTS_PATH=weights/tsim_tok/tsim_tok.pt \
DATA_PATH=data/vimo_sft_sample.with_length.json \
bash scripts/vimo/train_vimo_zero3_gc_packing.sh
```

It adds `--gradient_checkpointing` (a bare `store_true` flag) and runs under ZeRO-3;
the stage-appropriate hyperparameters are the same as the normal recipe.

### Packing flags (shared)

Both scripts set the same packing knobs, which reproduce the released recipe:

- `--enable_packing` — turns packing on.
- `--pack_total_length` (= `--max_seq_len`, default `11264`) — the packed-sequence budget.
- `--pack_total_length_threshold_ratio 0.9` — when a pack is "full enough" to emit.
- `--enable_global_sample_mean_loss` + `--theoretical_global_sample_num` (default `512`) —
  keep the loss/LR scale comparable to non-packed training despite variable per-pack sample
  counts.

Override the budget or target with `PACK_TOTAL_LENGTH=...` / `THEORETICAL_GLOBAL_SAMPLE_NUM=...`.

## Notes

- Packing changes throughput and effective batch composition, not model semantics.
- Keep the same `--extra_vision_cfg` and token budgets as non-packed training.
- The normal and advanced recipes differ only in the ZeRO stage and gradient checkpointing;
  the packing sampler, the per-sample mean loss, and the resulting model are identical.
