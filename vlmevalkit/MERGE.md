# Integrating ViMo into VLMEvalKit

VLMEvalKit is kept **independent** from this repo. To evaluate ViMo on VLMEvalKit
benchmarks you only need to (1) drop in one new model file, (2) merge two tiny edits
into upstream files, and (3) make the `vimo` package importable.

## 0. Dependency on the `vimo` package

`vimo_evalkit.py` imports the model from this repo's `vimo` package
(`vimo.modeling_vimo`, `vimo.processing_vimo`, `vimo.configuration_vimo`). It keeps its
**own embedded `TSIMRouter`** (it does NOT import `vimo.tsim_router`), so the two repos
stay independent. Make `vimo` importable in one of two ways:

- `pip install -e /path/to/this-repo` (so `import vimo` works), **or**
- `export VIMO_REPO=/path/to/this-repo`  (the file prepends it to `sys.path`).

## 1. REQUIRED — add the model (the only genuinely new file)

Copy `vlmevalkit/vlm/vimo_evalkit.py` (from this repo) into your VLMEvalKit checkout:

```
cp vlmevalkit/vlm/vimo_evalkit.py  <VLMEvalKit>/vlmeval/vlm/vimo_evalkit.py
```

## 2. REQUIRED — register the model (merge 2 edits, do NOT overwrite)

**`vlmeval/vlm/__init__.py`** — add inside the CUDA-guarded import block:

```python
from .vimo_evalkit import ViMo
```

**`vlmeval/config.py`** — add a registration entry (mirrors the old `SeqImg-VL-Token`,
with renamed kwargs/env vars). Env vars: `VIMO_MODEL_PATH`, `VIMO_EXTRA_CFG_PATH`,
`VIMO_TSIM_INTERVALS_PATH`, `VISUAL_EXTRACTOR_CKPT`, `VISUAL_EXTRACTOR_REPO`:

```python
        qwen3vl_series["ViMo"] = partial(
            ViMo,
            model_path=os.environ.get("VIMO_MODEL_PATH"),
            extra_cfg_path=os.environ.get("VIMO_EXTRA_CFG_PATH", "configs/vimo_cfg.json"),
            tsim_intervals_path=os.environ.get("VIMO_TSIM_INTERVALS_PATH", "tools/data_processing/tsim_intervals.json"),
            visual_extractor_ckpt_path=os.environ.get("VISUAL_EXTRACTOR_CKPT"),
            visual_extractor_repo_path=os.environ.get("VISUAL_EXTRACTOR_REPO"),
            budget_key="budget",
            token_alpha=0.8,
            n_base=144,
            exclude_base_in_output=True,
            is_incremental_encoding=True,
            use_custom_prompt=False,
            use_vllm=False,
            temperature=0.7,
            max_new_tokens=16384,
            image_token_num_per_image=145,
        )
        supported_VLM["ViMo"] = qwen3vl_series["ViMo"]
```

> The `visual_extractor_*` kwargs are only needed when a sample has **no** precomputed
> `num_tokens` and the TSIM Router must allocate budgets. For benchmarks that don't
> require visual updates you can omit them.

## 3. Run

```bash
export VIMO_MODEL_PATH=/path/to/weights/vimo_2b
export VIMO_REPO=/path/to/this-repo            # or pip install -e the repo
# inference
torchrun --nproc-per-node=8 run.py --mode infer --data MMBench_DEV_EN MME ChartQA_TEST --model ViMo
# scoring only (after inference)
python run.py --mode eval --data MMBench_DEV_EN --model ViMo
```

## Naming map (old → new)

| old (SeqImg) | new (ViMo) |
|---|---|
| class `SeqImgVL_Token_Vis` | `ViMo` |
| class `DinoTokenSampler` | `TSIMRouter` |
| registered name `SeqImg-VL-Token` | `ViMo` |
| env `SEQIMGVL_MODEL_PATH` | `VIMO_MODEL_PATH` |
| kwarg `dino_ckpt_path` | `visual_extractor_ckpt_path` |
| kwarg `token_bins_path` | `tsim_intervals_path` |
| kwarg `token_key` / value | `budget_key` / `"budget"` |
| method `get_sample_num_tokens` | `allocate_token_budgets` |
| method `text_image_to_text_image_generate` | `interleaved_generate` |
