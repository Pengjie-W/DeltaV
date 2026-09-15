#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
INPUT_FILE=${INPUT_FILE:-./test_json/deltav_infer/image_output.json}
OUT_DIR=${OUT_DIR:-"$(dirname "${INPUT_FILE}")/eval_structcot"}
python tools/eval/evaluate.py --benchmark structcot \
  --predictions "${INPUT_FILE}" --output_dir "${OUT_DIR}" \
  --judge_model "${JUDGE_MODEL:-qwen2.5-72b-instruct}" \
  --max_workers "${MAX_WORKERS:-8}" "$@"
