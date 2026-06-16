# Dataset format & TSIM Router token allocation

## Interleaved sample format (ViMo training)

Each MLLM training sample is one JSON line (`.jsonl`) with interleaved text and images:

```json
{
  "id": "structcot_checkers_20952",
  "task_type": ["interleaved"],
  "dataset": "StructCoT",
  "messages": [
    {"role": "user", "content": "<input_image_1>\n<input_image_2>\nQuestion ..."},
    {"role": "assistant", "content": "<output_image_1> Step 1 ... <output_image_2> Step 2 ..."}
  ],
  "input_images":  ["/abs/path/start.jpg", "/abs/path/end.jpg"],
  "output_images": ["/abs/path/step_1.jpg", "/abs/path/step_2.jpg"],
  "num_tokens": [144, 81]
}
```

- `<input_image_N>` / `<output_image_N>` placeholders in `messages` are resolved against
  the ordered `input_images` / `output_images` path lists.
- `num_tokens` is the per-image **incremental token budget** produced by the TSIM Router
  (the first image always uses the base budget `n_base`; subsequent images use the routed
  budgets). Training reads it with `--use_json_num_tokens`.

Datasets are mixed via a **mixture config** (passed as `--mixture_config`):

```json
{
  "multimodal_reasoning": {
    "image_loss_weight": 0.25,
    "datasets": [
      {"name": "StructCoT", "output_path": "data/vimo_mllm_sample.jsonl", "ratio": 10}
    ]
  }
}
```

See `data/vimo_sft_sample.json` (+ `data/vimo_mllm_sample.jsonl`) for a runnable example.

## Inference dataset format (Zebra-CoT / StructCoT)

```json
{
  "config": "Visual Logic & Strategic Games - Tetris",
  "input_prompt": "Fill the entire grid EXCEPT ...",
  "input_image":  ["/abs/path/problem.jpg"],
  "output_image": ["/abs/path/reasoning_01.jpg", "..."],
  "num_tokens": [144, 81, ...]
}
```

See `data/zebra_test_sample.json`, `data/struct_test_sample.json`.

## TSIM Router: similarity → token budget

The router (`vimo/tsim_tok/tsim_router.py`, paper Sec. 3.2) maps temporal visual change to an
incremental token budget. Pipeline (scripts under `tools/data_processing/`):

1. **Extract images** — `extract_images.py`: each sample → an ordered list of image paths.
2. **Compute similarity** — `similarity/run_sim.sh` → `compute_similarity.py`: adjacent-image
   SSIM/PSNR → `*_sim.json`.
3. **Assign budgets** — `assign_token_budget.py` (`--first_token 144 --include_first_token`):
   maps similarity to budgets via `tools/data_processing/tsim_intervals.json` (reads each interval's
   `budget`); writes `<out_dir>/tokens.json`.
4. **Merge into dataset** — `merge_tokens_into_dataset.py`: writes `num_tokens` back into the
   MLLM samples keyed by `tuple(image_paths)` → `*_with_tokens.{json,jsonl}`.

### Worked example (real `checker_move` sample)

All commands run from the repo root. We use the checker sample shipped in
`data/vimo_mllm_sample.jsonl` (its 6 images — `start, end, step_1..4` — live under
`data/image/checker_move/`). Outputs go to a scratch `out/`.

**Step 1 — Extract images.** Resolve each sample's `<input_image_N>` / `<output_image_N>`
placeholders into an ordered path list:

```bash
python tools/data_processing/extract_images.py \
  --input data/vimo_mllm_sample.jsonl \
  --output out/image_lists.json
```

`out/image_lists.json` — one inner list per sample, in message order:

```json
[
  ["data/image/checker_move/checker_20952_start.jpg",
   "data/image/checker_move/checker_20952_end.jpg",
   "data/image/checker_move/checker_20952_step_1.jpg",
   "data/image/checker_move/checker_20952_step_2.jpg",
   "data/image/checker_move/checker_20952_step_3.jpg",
   "data/image/checker_move/checker_20952_step_4.jpg"]
]
```

**Step 2 — Compute similarity.** DINOv2 features → adjacent-image similarity:

```bash
bash tools/data_processing/similarity/run_sim.sh \
  out/image_lists.json out/sim.json
```

> The wrapper is self-contained: it `cd`s to the repo root and runs the sibling
> `compute_similarity.py` under `torchrun` (GPU required). On first run it **downloads the
> DINOv2 code + pretrained weights from GitHub** and caches them under `~/.cache/torch/hub/`,
> so the command above needs no extra arguments.
>
> **Offline / intranet:** clone DINOv2 and fetch the weights once on a networked machine,
> then pass the local clone (via env) and the local weights (3rd positional arg). Use
> **absolute paths** — the wrapper `cd`s to the repo root first:
> ```bash
> git clone https://github.com/facebookresearch/dinov2.git facebookresearch_dinov2_main
> wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
> VISUAL_EXTRACTOR_REPO=/abs/facebookresearch_dinov2_main \
>   bash tools/data_processing/similarity/run_sim.sh \
>   out/image_lists.json out/sim.json /abs/dinov2_vitb14_pretrain.pth
> ```
> This loads from the local clone (`source="local"`) and the local weights instead of
> downloading.

`out/sim.json` — a dict keyed by the Python `str()` of the path list (note the single
quotes), each value a lower-triangular block (`row 0 = [1.0]`; `row t` = similarity of image
`t` to each earlier image):

```json
{
  "['data/image/checker_move/checker_20952_start.jpg', '...end.jpg', '...step_1.jpg', '...step_2.jpg', '...step_3.jpg', '...step_4.jpg']": [
    [1.0],
    [0.83],
    [0.62, 0.71],
    [0.55, 0.66, 0.94],
    [0.51, 0.63, 0.91, 0.96],
    [0.49, 0.60, 0.74, 0.78, 0.83]
  ]
}
```

**Step 3 — Assign token budgets.** Map similarity → per-image budget via
`tools/data_processing/tsim_intervals.json` (`alpha=0.8` exponentially-weighted history):

```bash
python tools/data_processing/assign_token_budget.py \
  --train_sim_json out/sim.json \
  --token_bins_jsons tools/data_processing/tsim_intervals.json \
  --train_out_dirs out/struct \
  --first_token 144 --include_first_token
```

`out/struct/tokens.json` — the first image takes the base budget `144`; the rest are routed.
For this sample the budgets come out as `[144, 100, 81, 49, 49, 100]`, all drawn from the
interval `budget` pool `{144, 100, 81, 49, 9}` in `tsim_intervals.json`:

```json
[
  {
    "img_paths": ["data/image/checker_move/checker_20952_start.jpg", "...", "...step_4.jpg"],
    "num_tokens": [144, 100, 81, 49, 49, 100]
  }
]
```

**Step 4 — Merge into dataset.** Write a tiny driving config, then inject `num_tokens` back
into the samples (joined on `tuple(img_paths)`):

```bash
cat > out/merge_config.json <<'JSON'
{ "StructCoT": [ {"path": "data/vimo_mllm_sample.jsonl", "token": "out/struct/tokens.json"} ] }
JSON

python tools/data_processing/merge_tokens_into_dataset.py \
  --config out/merge_config.json \
  --out_dir out/merged
```

Produces `out/merged/vimo_mllm_sample_with_tokens.jsonl` — each original sample with its
routed `num_tokens` field added — ready for training (`--use_json_num_tokens`).

### The mapping rule (`tools/data_processing/tsim_intervals.json`)

The offline-calibrated intervals file (keys renamed for the release):

```json
{
  "config": {"alpha": 0.8},
  "tsim_intervals": [
    {"tsim_left": 0.0147, "tsim_right": 0.5141, "budget": 100.0},
    {"...": "..."}
  ]
}
```

- Intervals are sorted by `tsim_left`; a similarity value falls in `[tsim_left, tsim_right)`
  and takes that interval's `budget`. Budgets are **monotonic non-increasing** (more similar
  → fewer tokens).
- For step *t*, history similarities are aggregated into a single TSIM value with an
  exponentially-weighted mean (`alpha**(n-1-j)`), then mapped to a budget.

> The original calibration file used keys `bins` / `sim_left` / `sim_right` /
> `ssim_tok_pick_nearest`. The released copy renames them to `tsim_intervals` / `tsim_left`
> / `tsim_right` / `budget`. The routers read the new names.

### Programmatic use

```python
from vimo.tsim_tok.tsim_router import TSIMRouter
router = TSIMRouter(
    visual_extractor_ckpt_path="/abs/dinov2_vitb14_pretrain.pth",  # only used by the offline branch
    tsim_intervals_path="tools/data_processing/tsim_intervals.json",
    visual_extractor_repo_path="",  # leave empty -> torch.hub auto-downloads facebookresearch/dinov2
    n_base=144,
)
budgets = router.allocate_token_budgets(["img0.jpg", "img1.jpg", "img2.jpg"])  # -> [b1, b2]
```
