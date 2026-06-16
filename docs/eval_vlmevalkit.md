# Understanding-benchmark evaluation via VLMEvalKit

ViMo's multimodal *understanding* benchmarks (BLINK, ChartQA, MME, MMBench, MathVista,
MMVet, MMVP, LogicVista, …) are evaluated through [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

VLMEvalKit is kept **independent** from this repo. The integration files live in
`vlmevalkit/`:

```
vlmevalkit/
  vlm/vimo_evalkit.py      # the ViMo model wrapper (registered as "ViMo")
  eval_only.py             # optional scoring-only helper
  MERGE.md                 # exactly which files to add / merge into a fresh checkout
```

## Setup

Follow `vlmevalkit/MERGE.md`. In short:

1. Copy `vlmevalkit/vlm/vimo_evalkit.py` into `<VLMEvalKit>/vlmeval/vlm/`.
2. Add `from .vimo_evalkit import ViMo` to `vlmeval/vlm/__init__.py`.
3. Add the `ViMo` registration block to `vlmeval/config.py` (snippet in MERGE.md).
4. Make the `vimo` package importable: `pip install -e .` here, or
   `export VIMO_REPO=/path/to/this-repo`.

`vimo_evalkit.py` imports the model from this repo's `vimo` package but keeps its **own**
embedded `TSIMRouter` (independent of `vimo/tsim_tok/tsim_router.py`).

## Run

```bash
export VIMO_MODEL_PATH=weights/vimo_2b
export VIMO_REPO=/path/to/this-repo
torchrun --nproc-per-node=8 run.py --mode infer \
  --data BLINK MMBench_DEV_EN MME ChartQA_TEST MathVista_MINI MMVet --model ViMo
python run.py --mode eval --data MMBench_DEV_EN --model ViMo
```
