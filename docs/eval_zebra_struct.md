# StructCoT / Zebra-CoT benchmark evaluation

Evaluate **any model's predictions**, not just DeltaV. Scoring uses the public test
questions and reference answers bundled in this repository and a text-only LLM judge.
It measures final-answer accuracy, **not generated-image quality**.

| Benchmark | Public test file in `wpj2003/StructCoT` | Samples | Categories |
| --- | --- | ---: | --- |
| StructCoT | `test/structcot_test.json` | 5,600 | 7 major categories, 800 each |
| Zebra-CoT | `test/zebracot_test.json` | 4,000 | 4 major categories, 1,000 each; 18 fine-grained tasks |

The StructCoT test split excludes Zebra-CoT samples. The two scores are reported
separately. Download the dataset from [ModelScope](https://www.modelscope.cn/datasets/wpj2003/StructCoT).

## 1. Choose your workflow

- **Already have predictions?** Skip dataset and checkpoint downloads. Scoring needs
  only this repository, Python **3.10+**, your prediction file, and a judge API key.
  It uses the Python standard library: no `pip install -r requirements.txt`, GPU,
  PyTorch, DeltaV weights, test images, or training data are required. Continue at
  [prediction format](#3-prediction-format).
- **Need to run inference first?** Download only the public **test JSONs and test
  image archive** below. Use your own model, or the DeltaV example. Do not download
  `train/` or `dist_train/` for evaluation.

Run repository commands from the code repository root, unless stated otherwise.

## 2. Download only the evaluation data

Install the ModelScope downloader in the environment used for data preparation:

```bash
pip install modelscope
export DATASET_ROOT="$PWD/datasets/StructCoT"

python -m modelscope.cli.cli download \
  --dataset wpj2003/StructCoT \
  test/structcot_test.json \
  test/zebracot_test.json \
  dist_test/StructCoT_test_images.tar \
  --local_dir "$DATASET_ROOT"
```

The command explicitly selects **three files**. Do not omit the filenames: an
unfiltered dataset download may also fetch the much larger training release.
The single test image archive serves both test splits. Even when evaluating only
StructCoT, no training image archive is needed.

Extract from the **dataset root**, not from inside `dist_test/`:

```bash
tar -xf "$DATASET_ROOT/dist_test/StructCoT_test_images.tar" -C "$DATASET_ROOT"
```

Expected layout:

```text
datasets/StructCoT/
  test/structcot_test.json
  test/zebracot_test.json
  dist_test/StructCoT_test_images.tar
  images_test/<major category>/<task>/...
```

Keep space for both the archive and extracted images. The archive already contains
the `images_test/` prefix, matching paths in both test JSONs.

If your installed CLI does not recognize the download syntax, inspect
`python -m modelscope.cli.cli download --help` and the dataset page's download
instructions. You can also download these same three files from the dataset's
file browser and preserve the directory layout. Do **not** fall back to downloading
the entire dataset. The remote download has not been exercised in the offline
release-preparation environment.

For this public dataset, first try the download without a token. If ModelScope
requires authentication for your account/environment, follow its download/login
instructions. A **ModelScope token is not a DashScope judge API key**; do not share
or commit either credential. The `upload` commands used by dataset maintainers
are not needed by benchmark users.

### Prepare DeltaV inference inputs

The public test files have relative image paths and do not include explicit IDs.
This helper validates the release and order, adds public IDs, checks extracted
images, and resolves image paths without modifying the downloaded files:

```bash
python tools/eval/prepare_inputs.py \
  --benchmark structcot --dataset_root "$DATASET_ROOT" \
  --output outputs/structcot_test_absolute.json

MODEL_PATH=weights/deltav_2b \
JSON_PATH="$PWD/outputs/structcot_test_absolute.json" \
EXP_NAME=deltav_structcot \
bash scripts/deltav/infer_deltav.sh
```

Install DeltaV's inference environment and download its weights according to
[INSTALL.md](../INSTALL.md) before running inference. Output:
`test_json/deltav_structcot/image_output.json`. Score this file, **not** the
`image_output_extracted.json` convenience output.

For Zebra-CoT, use `--benchmark zebracot`, an output such as
`outputs/zebracot_test_absolute.json`, and `EXP_NAME=deltav_zebracot`.
Leave `VIS_ARGS` unset for the normal merged-JSON workflow. Other models can read
the public test JSON directly; retain the **original test-array index** as each
prediction's ID. Only feed the model the question and input images, not the
reference `output_text` or `output_image` fields.

The tiny `data/*_infer_sample.json` examples are for inference smoke checks, not
complete benchmark submissions. Do not use their local row numbers as public IDs.

## 3. Prediction format

You provide **only model outputs**. Questions, ground truth, and categories come
from the bundled evaluation cache; no generated images or reference answers need
to be submitted.

### Recommended: ID + prediction (JSON or JSONL)

```json
[
  {"id": 0, "prediction": "My reasoning... Final Answer: 17."},
  {"id": 1, "prediction": "The answer is C."}
]
```

This is a **two-record format example**, not a complete submission. Supply all
5,600/4,000 records for a full score. `prediction` can be the final answer or the
full model response. `null` or an empty string counts as an incorrect answer.

- `id` is the **zero-based index in the original published test JSON array**.
  Integers and digit strings such as `"000001"` are accepted.
- Rows with IDs can be shuffled. For distributed inference, preserve IDs before
  sharding; do not renumber each shard or a filtered subset.
- Duplicate IDs, out-of-range IDs, and non-text predictions are rejected before
  any API calls.

### No IDs: a full, ordered list of output strings

```json
["First sample's response", "Second sample's response"]
```

Again, expand this example to the **entire** test set in its original order.
The evaluator uses array positions as IDs. It cannot detect reordered plain text:
if results may be shuffled or incomplete, use explicit IDs instead.

### Existing DeltaV output

`image_output.json` is accepted directly through `model_output_text`.
The evaluator uses `id`, then `_global_idx`, then `sample_NNNNNN.png` output names
when available; otherwise only a full ordered file is accepted. The preparation
helper's explicit public IDs take priority over run-local indices. Supplied
question/input-image metadata is checked against the public cache. Do not submit
both `prediction` and `model_output_text` in the same record.

Legacy StructCoT scoring IDs referred to the pre-filtered 7,000-sample source and
are **not** public test IDs. The released cache uses 0–5599, retains `legacy_id`
for provenance, and uses the published prompts. During migration, all 5,600
reference responses and input/output image names were checked; the 508 prompts
with removed `<image>` placeholders were normalized. No reference answers were
regenerated. Do not mix old cache IDs with the public release.

## 4. Obtain and configure a judge API key

The default platform is **Alibaba Cloud Model Studio / 百炼 (DashScope)**, using
its OpenAI-compatible chat-completions endpoint. No OpenAI account or API key is
required for this default platform.

1. Sign in to Alibaba Cloud and open **Model Studio / 百炼**. Activate the service
   and complete any account verification requested by the console.
2. Select the region in which you will call the judge, open the **API Key**
   management page, and create a key with permission to call the model.
3. Check model availability, balance/quota, and the endpoint for that region.
   Copy the key locally; never put it in a prediction file or commit it.
4. Follow the official guides for current console labels, regions, and access:
   [Get an API key](https://help.aliyun.com/zh/model-studio/get-api-key) and
   [OpenAI-compatible API](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope).

In Bash, read the key without displaying it or placing its literal value in history:

```bash
read -rsp "DashScope API key: " DASHSCOPE_API_KEY
echo
export DASHSCOPE_API_KEY

export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export JUDGE_MODEL="qwen2.5-72b-instruct"
```

The endpoint above is the legacy mainland-China/Beijing configuration. Use the
endpoint shown by your console if your key belongs to another region. **Key,
region, endpoint, and model access must match.** The release preserves the old
judge name for continuity; its current availability in your account has not been
verified. If you choose another available model, record that change and do not
assume scores are directly comparable. There is no silent model fallback.

Other OpenAI-compatible services can be used by setting `DASHSCOPE_BASE_URL` to
their API base URL and `DASHSCOPE_API_KEY` to **that service's key**. These variable
names are kept for compatibility. The base URL must stop before `/chat/completions`.
Temperature is 0; DashScope requests also disable thinking, as in the old scripts.

Scoring sends questions, model responses, and reference answers to the configured
provider. Review its data policy and pricing. StructCoT performs rule-based
extraction first, with API fallback; Zebra-CoT uses API extraction for nonempty
responses, preserving its original protocol. A nonempty response can therefore
require both an extraction call and a grading call. No quota or free tier is assumed.

## 5. Score predictions

Validate the format and alignment **without a key, API calls, or output writes**:

```bash
python tools/eval/evaluate.py \
  --benchmark structcot --predictions predictions.json \
  --output_dir outputs/my_model_structcot --validate_only
```

Run a full evaluation:

```bash
python tools/eval/evaluate.py \
  --benchmark structcot --predictions predictions.json \
  --output_dir outputs/my_model_structcot --model_name "My model"
```

For Zebra-CoT, change `--benchmark` to `zebracot` and use a separate output
directory. Existing DeltaV-style launch commands also work:

```bash
INPUT_FILE=./test_json/deltav_structcot/image_output.json \
bash tools/eval/score_struct.sh --model_name DeltaV-2B

INPUT_FILE=./test_json/deltav_zebracot/image_output.json \
bash tools/eval/score_zebra.sh --model_name DeltaV-2B
```

The wrappers accept `INPUT_FILE`, `OUT_DIR`, `JUDGE_MODEL`, and `MAX_WORKERS`;
extra CLI arguments are forwarded to the unified evaluator. Unlike the legacy
scripts, they do not accept arbitrary `GT_CACHE` overrides: the versioned public
cache defines the benchmark.

For an explicitly non-leaderboard smoke run, submit a few records **with original
IDs** and add `--allow_partial`. This reports subset accuracy and coverage, not a
full benchmark score. It still makes billable API calls unless `--validate_only`
is also set. Start with a small authorized smoke run before spending on a full run.

### Resume and errors

- Re-run the identical command to resume extraction and grading. Output records
  are flushed as they complete. Use one output directory per benchmark/model run.
- Input, cache, evaluator/protocol, judge, and run settings are fingerprinted.
  Changing them requires a new output directory or explicit `--reset_progress`,
  which removes previous results and progress. The key itself is not persisted,
  so a replacement key for the same service can resume a run.
- Default concurrency is 8, timeout is 60 seconds, and maximum attempts per API
  call is 5. Configure `--max_workers`, `--timeout`, and `--max_attempts` as needed.
  Authentication, permissions, and other non-retryable HTTP errors stop the run;
  rate limits, server errors, and transient network failures use bounded retries.
- Unfinished API requests do not silently become wrong answers. Failed runs exit
  nonzero, keep progress, and do not publish a completed score. An unrecognized
  judge label is reported as `Invalid` and remains in the accuracy denominator,
  matching the historical scorers.
- Concurrent writes to one output directory are refused. After an unclean kill,
  remove `.eval.lock` only after confirming that no scorer still uses that directory.
- `unset DASHSCOPE_API_KEY` when finished. Keys and raw HTTP error bodies are not
  written to evaluation logs or metadata.

## 6. Results and leaderboard protocol

Each output directory contains:

| File | Contents |
| --- | --- |
| `run.json` | Benchmark version, input/cache/protocol hashes, model and judge configuration |
| `progress.jsonl` | Resumable per-sample extraction and grading records |
| `results.json` | Metadata, `summary`, `category_summary`, and per-sample `results` |
| `leaderboard.md` | Markdown table in README column order, with a partial-run warning if applicable |

Accuracy in JSON is a fraction in `[0, 1]`; the leaderboard displays percentages
rounded to one decimal place. **Overall = total correct / total evaluated**.
For complete published splits, major categories are equally sized, so this also
equals their macro-average before rounding. A partial run uses submitted samples
only, reports coverage, and is not eligible for the full leaderboard. Unobserved
categories display `-`, not fabricated zero scores.

### README category mapping

| Benchmark | README column | Cache / dataset major category |
| --- | --- | --- |
| StructCoT | Strategy Planning | Strategic Game and Planning |
| StructCoT | Spatial Planning | Spatial Perception and Embodied Planning |
| StructCoT | Logic | Logical and Abstract Reasoning |
| StructCoT | Math | Mathematical and Algorithmic Reasoning |
| StructCoT | Science | Natural Science and Domain-Specific Reasoning |
| StructCoT | Visual Search | Visual Search |
| StructCoT | Jigsaw Restoration | Image Restoration and Jigsaw Reasoning |
| Zebra-CoT | 2D | 2D Visual Reasoning |
| Zebra-CoT | 3D | 3D Visual Reasoning |
| Zebra-CoT | Science | Scientific Reasoning |
| Zebra-CoT | Strategy | Visual Logic & Strategic Games |

For Zebra-CoT, scoring categories come from the original `config` prefix, **not**
the seven-category image-directory layout shared by the dataset release.

The existing [README leaderboards](../README.md#benchmark) retain historical
reported scores, not measurements rerun during this code release. For a new
result, retain the predictions and share the evaluated model/version, benchmark
version, inference settings, judge configuration, and `results.json` together
with the table row. A full-coverage run is not proof that a changed judge exactly
reproduces historical numbers. External understanding benchmarks remain covered
by [VLMEvalKit](eval_vlmevalkit.md), not this scorer.

`tools/eval/gt_cache/manifest.json` records the public test-file hashes, original
cache hashes, category counts, and the checksums of the bundled, reindexed caches.
The evaluator verifies bundled cache integrity before making API requests.
