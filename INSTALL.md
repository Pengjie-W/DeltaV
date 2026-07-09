# Installation

```bash
# 1. Create an environment (Python 3.10 recommended)
conda create -n deltav python=3.10 -y
conda activate deltav

# 2. Install PyTorch matching your CUDA, then the rest
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 3. (optional, recommended) make the packages importable from anywhere
pip install -e .          # if a setup is added; otherwise run scripts from the repo root
```

All scripts add the repo root to `sys.path`, so running them from the repo root is enough
even without `pip install -e`.

## Optional: compiled deformable-attention op

TSIM-Tok's encoder uses multi-scale deformable attention. A CUDA op lives under
`tsim_tok/modules/ops`. Building it is optional (a pure-PyTorch path is used otherwise):

```bash
cd deltav/tsim_tok/modules/ops
bash make.sh
cd -
```
## Optional: flash-attention

```bash
pip install flash-attn --no-build-isolation 
```

## Weights

Place (or symlink) the released checkpoint under `weights/`:

```
weights/
  deltav_2b/                 # DeltaV 2B (HF-style dir: config.json + *.safetensors + tokenizer)
```

The DINOv2 ViT-B/14 used by the TSIM Router is **not** in this list — it is downloaded
automatically (see below), so you don't need to place it manually.

## Optional: DINOv2 (TSIM Router visual extractor)

The TSIM Router and the similarity data step use a frozen **DINOv2 ViT-B/14** to measure
temporal similarity. **By default nothing needs to be configured** — `torch.hub` fetches
it on first use.

- **Default (online):** `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`
  downloads the model code **and** pretrained weights on first run and caches them under
  `~/.cache/torch/hub/`. The first run needs network access to GitHub. (Set `TORCH_HOME`
  to change the cache location.) If you never set the variables below, this is what
  happens automatically.
- **Offline / intranet (optional):** if the automatic download cannot reach GitHub, clone
  the repo and fetch the weights once on a networked machine:
  ```bash
  git clone https://github.com/facebookresearch/dinov2.git facebookresearch_dinov2_main
  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
  ```
  Then point every entry point at the local clone (repo) plus the local weight file. Use
  **absolute paths** — `run_sim.sh` `cd`s to the repo root first, so relative paths resolve
  from there:
  - similarity data step: `VISUAL_EXTRACTOR_REPO=/abs/facebookresearch_dinov2_main bash tools/data_processing/similarity/run_sim.sh out/image_lists.json out/sim.json /abs/dinov2_vitb14_pretrain.pth`
  - native inference / visualization: `--visual_extractor_repo /abs/facebookresearch_dinov2_main --visual_extractor_ckpt /abs/dinov2_vitb14_pretrain.pth`
  - VLMEvalKit: `VISUAL_EXTRACTOR_REPO=/abs/facebookresearch_dinov2_main` (weights via the model kwarg `visual_extractor_ckpt_path`)
