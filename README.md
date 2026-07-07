<div align="center">

# ViMo: Thinking with Visual Updates in Unified Multimodal Models

**An unified multimodal model that thinks with visual updates — modeling only the sparse visual changes across reasoning steps instead of regenerating full images.**

[![arXiv](https://img.shields.io/badge/Arxiv-ViMo-b31b1b.svg?logo=arXiv)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ViMo--2B-yellow.svg?logo=HuggingFace)](https://huggingface.co/dle666/ViMo-2B/tree/main)
[![ModelScope](https://img.shields.io/badge/ModelScope-ViMo--2B-green.svg)](https://www.modelscope.cn/models/wpj2003/ViMo-2B)
[![Dataset](https://img.shields.io/badge/Dataset-StructCoT-orange.svg)](https://www.modelscope.cn/datasets/wpj2003/StructCoT)
[![Demo](https://img.shields.io/badge/Demo-blue)](http://vlrlabmonkey.xyz:10088/)
[![Website](https://img.shields.io/badge/Website-ViMo-blue.svg)]()

</div>

---

## News

* ```2026.06.22 ``` 🚀 We release [ViMo-2B](https://huggingface.co/dle666/ViMo-2B/tree/main), a unified multimodal model for interleaved multimodal reasoning.

## Introduction

ViMo is a unified multimodal model (UMM) that integrates multimodal understanding and generation for interleaved multimodal reasoning. It represents evolving visual states through compact **incremental visual tokens** that focus on sparse but reasoning-relevant changes across reasoning steps, reducing redundant modeling of largely unchanged visual content. Token budgets are allocated by the **TSIM Router** with temporal-similarity routing, and visual states are encoded by the **TSIM-Tok** tokenizer.

This repository releases the **ViMo-2B UMM** together with the **TSIM-Tok tokenizer**, training scripts, inference scripts, evaluation utilities, and tiny samples.

## ViMo Workflow

https://github.com/user-attachments/assets/abad70d5-1a9a-41e8-b1a0-f8f46ab89f08

## Repository Layout

```text
vimo/        ViMo model code: modeling, processing, configuration, backbone
  tsim_tok/  TSIM-Tok visual tokenizer and TSIM Router
train/       Training entry points
inference/   Inference and TSIM-Tok evaluation
scripts/     Ready-to-run scripts for ViMo, TSIM-Tok, and data utilities
configs/     Model and acceleration configs
data/        Tiny samples
docs/        Extended tutorials and README media assets
tools/       Data processing, evaluation, and inference post-processing
```

## Installation

See [INSTALL.md](INSTALL.md) for the full setup guide. Quick version:

```bash
conda create -n vimo python=3.10 -y && conda activate vimo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Download Checkpoints

Download our models from Huggingface.

```bash
pip install huggingface_hub

python tools/download_model.py -n ViMo-2B      # or TSIM-Tok
```

You can also download our models from ModelScope.

```bash
pip install modelscope

python tools/download_model.py -t modelscope -n ViMo-2B   # or TSIM-Tok
```

The released checkpoints are placed under `weights/`:

```text
weights/
  vimo_2b/
  tsim_tok/
    tsim_tok.pt
```

## Training and Inference

### 1. ViMo UMM

Training has two stages on top of a frozen TSIM-Tok. The basic recipe below is the simplest path for reproducing ViMo UMM training.

```bash
# Stage 1: alignment. Train the generation MLP and visual head.
# GEN_WEIGHTS_PATH defaults to weights/tsim_tok/tsim_tok.pt.
BASE_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
DATA_PATH=data/vimo_sft_sample.json \
bash scripts/vimo/train_vimo_stage1.sh

# Stage 2: SFT. Update all params except TSIM-Tok and the understanding MLP.
STAGE1_MODEL_PATH=./Checkpoints_MLLM/vimo_stage1/.../tfmr \
DATA_PATH=data/vimo_sft_sample.json \
bash scripts/vimo/train_vimo_stage2.sh
```

### ViMo Inference

```bash
# Pure inference for evaluation. Outputs text-only .json, then merge + extract answers.
# Zebra-CoT and StructCoT share one entrypoint; switch with JSON_PATH.
MODEL_PATH=weights/vimo_2b \
JSON_PATH=data/zebra_test_sample.json \
bash scripts/vimo/infer_vimo.sh
```

To also decode and save generated images, set `VIS_ARGS`. This streams results to `.jsonl` and skips merge/extract steps. Add `--concat_gt_images` to dump a ground-truth montage alongside each prediction.

```bash
MODEL_PATH=weights/vimo_2b \
JSON_PATH=data/zebra_test_sample.json \
VIS_ARGS="--decode_and_save_image --concat_gt_images" \
bash scripts/vimo/infer_vimo.sh
```

If samples carry precomputed `num_tokens`, the script uses them via `--use_json_num_tokens`. Otherwise set `TOKEN_ARGS="--use_tsim_router ..."` and the TSIM Router allocates incremental token budgets from temporal similarity. See [docs/data_and_token.md](docs/data_and_token.md).

### 2. TSIM-Tok Visual Tokenizer

TSIM-Tok training has two stages: single-image training, then multi-image training with variable token budgets. The visual backbone stays frozen throughout.

`BASE_MODEL_PATH` points at the Qwen3-VL-2B base directory. Its `config.json` supplies the visual `vision_config`, and its `model.safetensors` initializes the frozen visual backbone.

```bash
# Stage 1: single image.
BASE_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
DATA_PATH=data/tsim_tok_stage1_sample.json \
bash scripts/tsim_tok/train_tsim_tok_stage1.sh

# Stage 2: multi-image, variable-token-budget training.
BASE_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
VQ_CKPT=checkpoints/tsim_tok_stage1/model_dump/<ckpt>.pt \
DATA_PATHS=data/tsim_tok_stage2_sample.json \
bash scripts/tsim_tok/train_tsim_tok_stage2.sh
```

### TSIM-Tok Evaluation

Reconstruction SSIM under per-sample token budgets:

```bash
BASE_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
VQ_CKPT=weights/tsim_tok/tsim_tok.pt \
VAL_DATA=data/tsim_tok_stage2_sample.json \
bash scripts/tsim_tok/eval_tsim_tok.sh
```

## Documentation

- [docs/data_and_token.md](docs/data_and_token.md): dataset format and how the TSIM Router turns image similarity into token budgets.
- [docs/eval_zebra_struct.md](docs/eval_zebra_struct.md): Zebra-CoT and StructCoT scoring.
- [docs/advanced_zero3_gc.md](docs/advanced_zero3_gc.md): ZeRO-3 and gradient checkpointing.
- [docs/packing.md](docs/packing.md): sequence packing, length computation, and packing training.
- [docs/eval_vlmevalkit.md](docs/eval_vlmevalkit.md): understanding benchmarks via VLMEvalKit. This guide is still being refined.

## Qualitative Examples

<p align="center">
  <img src="docs/assets/und_example.png" alt="Qualitative comparison of multimodal reasoning" width="90%">
</p>

<p align="center">
  <em>Qualitative comparison of multimodal reasoning. Full-image modeling (Base) exhibits inconsistent intermediate visual states, while ViMo maintains consistent visual representations through visual updates.</em>
</p>

We further examine how different visual modeling paradigms affect multimodal reasoning. In the chess example, full-image modeling omits the piece at position g6, altering the perceived board configuration and leading to an incorrect strategic judgment. In the Polybius-square example, reconstructing the letter P as E directly causes an incorrect encoding result.

These examples show that visual inconsistency is not limited to reconstruction quality: local errors in intermediate images can alter the semantic evidence used for reasoning and corrupt the subsequent decision process. In contrast, ViMo grounds each reconstruction on previously established visual information, reducing error accumulation across reasoning steps and producing more reliable visual evidence.

## Benchmark

### External Multimodal Reasoning and Understanding Evaluation

| Model | #Param | VStar | EMMA | M3CoT | MathVista | VisuLogic | MMBench | MME-P | MMVP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **General UMMs** | | | | | | | | | |
| Chameleon | 7B | 32.5 | 8.6 | 16.1 | 21.7 | 4.5 | 6.0 | 530 | 4.7 |
| Anole | 7B | 34.0 | 6.6 | 15.8 | 22.5 | 3.7 | 6.2 | 508 | 6.7 |
| Janus-pro | 1B | 43.5 | 18.9 | 45.9 | 37.6 | 25.0 | 60.2 | 1398 | 39.3 |
| Janus-pro | 7B | 39.3 | 21.5 | 49.1 | 42.7 | 17.5 | 66.7 | 1509 | 34.7 |
| OmniGen2 | 7B | 41.4 | 14.7 | 50.3 | 60.2 | 0.1 | 76.1 | 1588 | 35.3 |
| Bagel | 7B | 70.1 | 28.7 | 31.4 | 72.5 | 28.9 | 83.7 | 1665 | 69.3 |
| EMU3.5 | 34B | - | - | - | 28.3 | 11.4 | 13.7 | 791 | 16.7 |
| **Understanding-centric MLLMs** | | | | | | | | | |
| Qwen3-VL | 2B | 71.7 | 22.2 | 53.0 | 61.1 | 11.5 | 77.1 | 1482 | 45.0 |
| Qwen3-VL | 8B | 83.7 | 30.6 | 61.2 | 77.6 | 22.5 | 85.2 | 1729 | 59.3 |
| InternVL3.5 | 2B | 68.1 | 12.7 | 51.3 | 60.8 | 26.0 | 78.2 | 1552 | 48.7 |
| InternVL3.5 | 8B | 69.1 | 16.6 | 59.9 | 74.1 | 29.7 | 82.7 | 1688 | 57.3 |
| **Latent Interleaved Reasoning Models** | | | | | | | | | |
| Monet | 7B | 79.1 | 22.1 | 44.2 | 62.5 | 10.6 | 75.3 | 1636 | 48.7 |
| Mirage | 8B | 13.6 | 13.9 | 1.08 | 29.9 | 0.4 | 12.3 | 549 | 0.0 |
| VPT-Det | 2B | 43.5 | 20.1 | 44.4 | 41.8 | 25.6 | 73.3 | 1516 | 34.0 |
| **Explicit Interleaved Reasoning UMMs** | | | | | | | | | |
| Bagel-Zebra-CoT | 7B | 64.9 | 20.6 | 62.6 | 72.1 | 0 | 55.6 | 1647 | 22.0 |
| ThinkMorph | 7B | 64.4 | 22.4 | 48.8 | 67.8 | 6.5 | 78.2 | 1478 | 8.6 |
| **ViMo** [[Weight]](https://huggingface.co/dle666/ViMo-2B) | **2B** | 75.9 | 28.6 | 54.5 | 69.3 | 23.5 | 82.3 | 1555 | 51.3 |

VStar, EMMA, M3CoT, MathVista, and VisuLogic are grouped as multimodal reasoning benchmarks, while MMBench, MME-P, and MMVP are grouped as multimodal understanding benchmarks.

### In-domain Multimodal Reasoning Evaluation

| Model | #Param | Zebra 2D | Zebra 3D | Zebra Science | Zebra Strategy | Zebra Overall | Struct Strategy Planning | Struct Spatial Planning | Struct Logic | Struct Math | Struct Science | Struct Visual Search | Struct Jigsaw Restoration | Struct Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Understanding-centric MLLMs** | | | | | | | | | | | | | | |
| GPT-5.2 | - | 67.6 | 19.3 | 73.3 | 54.4 | 53.7 | 43.1 | 33.8 | 42.1 | 76.3 | 50.4 | 87.0 | 57.1 | 55.7 |
| Gemini-3.1 Pro | - | 68.7 | 19.0 | 83.3 | 60.4 | 57.9 | 71.6 | 28.2 | 50.2 | 78.3 | 55.0 | 79.4 | 65.3 | 61.1 |
| Gemini 3.0 Flash | - | 66.5 | 19.4 | 78.4 | 54.5 | 54.7 | 55.0 | 33.3 | 44.8 | 74.8 | 48.4 | 83.6 | 64.9 | 57.8 |
| Qwen3-VL | 2B | 44.3 | 13.2 | 30.3 | 9.2 | 24.3 | 3.4 | 31.4 | 4.6 | 41.4 | 29.4 | 80.8 | 39.3 | 32.9 |
| Qwen3-VL | 8B | 50.7 | 16.9 | **56.0** | 22.7 | 36.6 | **21.6** | 25.4 | 13.1 | **59.3** | 39.3 | 83.8 | 46.5 | 41.3 |
| InternVL3.5 | 8B | 29.7 | 11.4 | 48.9 | 19.8 | 27.5 | 6.9 | 36.3 | 17.5 | 36.1 | 32.0 | 75.8 | 41.0 | 35.1 |
| Qwen2.5-VL | 72B | 43.2 | 17.3 | 50.1 | 25.8 | 34.1 | 14.8 | 34.4 | 31.4 | 48.0 | 36.5 | **84.9** | 47.0 | 42.4 |
| **General UMMs** | | | | | | | | | | | | | | |
| Chameleon | 7B | 13.3 | 3.0 | 5.2 | 9.9 | 7.9 | 5.6 | 12.5 | 4.1 | 9.1 | 13.1 | 23.5 | 14.4 | 11.8 |
| Anole | 7B | 10.8 | 2.8 | 4.8 | 8.5 | 6.7 | 5.4 | 0.1 | 3.8 | 8.9 | 12.8 | 16.8 | 11.4 | 9.9 |
| Janus-pro | 7B | 31.7 | 7.7 | 11.5 | 18.0 | 17.2 | 4.3 | 24.4 | 13.4 | 16.6 | 12.0 | 74.6 | 33.9 | 25.6 |
| OmniGen2 | 7B | 26.5 | 1.3 | 9.6 | 9.7 | 11.8 | 0.6 | 25.3 | 1.5 | 8.4 | 10.1 | 78.1 | 28.5 | 21.8 |
| Bagel | 7B | 43.3 | 14.7 | 44.5 | 16.3 | 29.7 | 16.4 | 24.9 | 12.8 | 49.0 | 35.5 | 84.6 | 49.0 | 38.9 |
| EMU3.5 | 34B | 10.1 | 3.6 | 8.6 | 11.8 | 8.5 | 2.8 | 29.1 | 4.6 | 19.3 | 15.6 | 21.1 | 18.8 | 15.9 |
| **Latent Interleaved Reasoning Models** | | | | | | | | | | | | | | |
| Monet | 7B | 37.5 | 12.0 | 15.1 | 23.0 | 21.9 | 2.3 | 19.9 | 21.9 | 33.8 | 25.8 | 59.6 | 33.8 | 28.1 |
| Mirage | 8B | 2.2 | 2.5 | 10.7 | 12.4 | 7.0 | 0.9 | 14.3 | 12.4 | 35.8 | 22.5 | 11.0 | 30.4 | 18.2 |
| VPT-Det | 2B | 32.3 | 3.5 | 6.5 | 18.7 | 15.3 | 7.5 | 26.5 | 8.8 | 14.5 | 15.1 | 73.1 | 35.9 | 25.9 |
| **Explicit Interleaved Reasoning UMMs** | | | | | | | | | | | | | | |
| Bagel-Zebra-CoT | 7B | - | - | - | - | - | 7.0 | 24.6 | 22.8 | 33.3 | 27.3 | 81.0 | 41.9 | 34.0 |
| ThinkMorph | 7B | 43.0 | 11.6 | 31.4 | 22.9 | 27.2 | 21.4 | 19.5 | 26.4 | 43.4 | 26.0 | 84.1 | 49.9 | 38.7 |
| **ViMo** [[Weight]](https://huggingface.co/dle666/ViMo-2B) | **2B** | **78.9** | **20.0** | 41.1 | **38.3** | **44.6** | 16.4 | **53.0** | **66.0** | 30.1 | **45.6** | 84.3 | **62.6** | **51.1** |

The StructCoT test set excludes all samples originating from the Zebra-CoT dataset.

## Acknowledgements

We would like to thank [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and [VFMTok](https://github.com/CVMI-Lab/VFMTok) for providing base models and code, as well as their contributions to this field. We also thank [Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT) for providing a valuable interleaved multimodal reasoning dataset. We also thank everyone who contributed to this open-source effort.

## Copyright

Please do not hesitate to share your valuable feedback—it is a key motivation that drives us to continuously improve our framework.

**Note:** Our model is intended for academic research and non-commercial use only. If you are interested in a faster (smaller) or stronger model, please contact us at [xbai@hust.edu.cn](mailto:xbai@hust.edu.cn) or [ylliu@hust.edu.cn](mailto:ylliu@hust.edu.cn).
