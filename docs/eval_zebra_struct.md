# Zebra-CoT / StructCoT evaluation

Two steps: (1) run ViMo inference to produce predictions, (2) score them with an LLM API.

## 1. Inference

```bash
MODEL_PATH=weights/vimo_2b \
JSON_PATH=data/zebra_test_sample.json \
EXP_NAME=vimo_zebra \
bash scripts/vimo/infer_vimo.sh        # -> ./test_json/vimo_zebra/image_output.json
```
```bash
MODEL_PATH=weights/vimo_2b \
JSON_PATH=data/struct_test_sample.json \
EXP_NAME=vimo_struct \
bash scripts/vimo/infer_vimo.sh        # -> ./test_json/vimo_struct/image_output.json
```
Swap `JSON_PATH=data/struct_test_sample.json` for StructCoT. The script runs distributed
inference (`inference/infer_vimo.py`), merges shards (`tools/inference/merge_results.py`), then extracts
text (`tools/inference/extract.py`).

## 2. Scoring (LLM-API based)

The scorers call an OpenAI-compatible API (DashScope by default). Export your key first:

```bash
export DASHSCOPE_API_KEY=<your key>
# optional: export DASHSCOPE_BASE_URL=<endpoint>
# optional: export JUDGE_MODEL=qwen2.5-72b-instruct   # the LLM judge used for scoring
```

`JUDGE_MODEL` selects the judge model for both the prediction-extraction and the
by-category grading steps (default `qwen2.5-72b-instruct`); the scorer scripts thread it
through as `--model`.

```bash
# Zebra-CoT (optionally override the judge model via JUDGE_MODEL)
# JUDGE_MODEL=qwen3-235b-a22b-instruct-2507 \
INPUT_FILE=./test_json/vimo_zebra/image_output.json bash tools/eval/score_zebra.sh
# StructCoT
# JUDGE_MODEL=qwen3-235b-a22b-instruct-2507 \
INPUT_FILE=./test_json/vimo_struct/image_output.json bash tools/eval/score_struct.sh
```

Each scorer runs three steps (with resume): convert → extract prediction against the GT
cache → evaluate by category. GT caches ship under `tools/eval/gt_cache/`:

- `zebra_gt_final_answer.json` — Zebra-CoT final answers.
- `struct_gt_800_each.json` — StructCoT, 800 samples per major category.

Output: `..._evaluated_by_category.json` with per-category accuracy.

> Security note: the original scripts hardcoded an API key; the released copies read it from
> `DASHSCOPE_API_KEY` instead. Never commit real keys.


