#!/usr/bin/env bash
# StructCoT scoring (LLM-API based). Consumes the image_output.json produced by infer_vimo.sh.
# Requires:  export DASHSCOPE_API_KEY=<your key>   (and optionally DASHSCOPE_BASE_URL)
# Judge model: export JUDGE_MODEL=<model>          (default qwen2.5-72b-instruct)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

INPUT_FILE=${INPUT_FILE:-./test_json/vimo_infer/image_output.json}
GT_CACHE=${GT_CACHE:-tools/eval/gt_cache/struct_gt_800_each.json}
JUDGE_MODEL=${JUDGE_MODEL:-qwen2.5-72b-instruct}
OUT_DIR="$(dirname "${INPUT_FILE}")/eval_outputs_v2"; mkdir -p "${OUT_DIR}"
BASE="$(basename "${INPUT_FILE}" .json)"

python tools/eval/struct/convert_to_extract_format.py \
  --input "${INPUT_FILE}" --output "${OUT_DIR}/${BASE}_converted.jsonl" --id_mode sample_png
python tools/eval/struct/extract_pred.py \
  --input "${OUT_DIR}/${BASE}_converted.jsonl" --gt_cache "${GT_CACHE}" \
  --output "${OUT_DIR}/${BASE}_subset5600_extract.json" --max_workers 8 --overwrite \
  --model "${JUDGE_MODEL}"
python tools/eval/struct/eval_by_category.py \
  --input_json_path "${OUT_DIR}/${BASE}_subset5600_extract.json" \
  --output_json_path "${OUT_DIR}/${BASE}_subset5600_evaluated_by_category.json" --max_workers 8 --overwrite \
  --model "${JUDGE_MODEL}"
echo "[DONE] -> ${OUT_DIR}/${BASE}_subset5600_evaluated_by_category.json"
