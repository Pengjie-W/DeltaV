# Understanding-benchmark evaluation via VLMEvalKit

DeltaV's multimodal *understanding* benchmarks (BLINK, ChartQA, MME, MMBench, MathVista,
MMVet, MMVP, LogicVista, …) are evaluated through [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

VLMEvalKit is kept **independent** from this repo. The integration files live in
`vlmevalkit/`:

```
vlmevalkit/
  vlm/deltav_evalkit.py      # the DeltaV model wrapper (registered as "DeltaV")
  eval_only.py             # optional scoring-only helper
  MERGE.md                 # exactly which files to add / merge into a fresh checkout
```

## Setup

Follow `vlmevalkit/MERGE.md`. In short:

1. Copy `vlmevalkit/vlm/deltav_evalkit.py` into `<VLMEvalKit>/vlmeval/vlm/`.
2. Add `from .deltav_evalkit import DeltaV` to `vlmeval/vlm/__init__.py`.
3. Add the `DeltaV` registration block to `vlmeval/config.py` (snippet in MERGE.md).
4. Make the `deltav` package importable: `pip install -e .` here, or
   `export DELTAV_REPO=/path/to/this-repo`.

`deltav_evalkit.py` imports the model from this repo's `deltav` package but keeps its **own**
embedded `TSIMRouter` (independent of `deltav/tsim_tok/tsim_router.py`).

## Run

```bash
export DELTAV_MODEL_PATH=weights/deltav_2b
export DELTAV_REPO=/path/to/this-repo
torchrun --nproc-per-node=8 run.py --mode infer \
  --data BLINK MMBench_DEV_EN MME ChartQA_TEST MathVista_MINI MMVet --model DeltaV
python run.py --mode eval --data MMBench_DEV_EN --model DeltaV
```
