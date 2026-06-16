import os
import json
import torch
import logging
import argparse
import random
import shutil
from typing import List, Dict, Any
from dataclasses import dataclass
import re

import wandb
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import (
    set_seed,
)

from transformers import AutoModelForCausalLM
import PIL.Image
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False
from typing import Optional
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import sys, os
# repo root (parent of train/) so `vimo` and `train` packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vimo.processing_vimo import ViMoProcessor
from vimo.modeling_vimo import ViMoModel, Qwen3VLTextAttention
from vimo.configuration_vimo import ViMoConfig
import transformers
from vimo.rope2d import get_rope_index_3

from train.flash_attn_varlen import qwen3vl_forward
Qwen3VLTextAttention.forward = (qwen3vl_forward)
from vimo.modeling_vimo import TSIMTokExtraCfg

from safetensors.torch import load_file

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

IMAGE_VOCAB_SIZE = 16384
IMAGE_EOS_ID = IMAGE_VOCAB_SIZE
IMAGE_VOCAB_SIZE_WITH_EOS = IMAGE_VOCAB_SIZE + 1

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    text_label: torch.Tensor

    def __len__(self):
        return len(self.input_ids)

def get_custom_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    min_lr_ratio=0.0, 
    num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_learning_rate(step, initial_lr, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles=0.5):
    if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps)) * initial_lr
    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
    scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
    return scaled_factor * initial_lr


class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.tensor([0.0], device=device)
        self.total = torch.tensor([0.0], device=device)
        self.total_loss = torch.tensor([0.0], device=device)

        self.text_right = torch.tensor([0.0], device=device)
        self.text_total = torch.tensor([0.0], device=device)
        self.text_loss = torch.tensor([0.0], device=device)

        self.image_right = torch.tensor([0.0], device=device)
        self.image_total = torch.tensor([0.0], device=device)
        self.image_loss = torch.tensor([0.0], device=device)

        self.world_size = dist.get_world_size()

    def __call__(self, image_logits, image_labels, text_logits, text_labels, loss, image_loss, text_loss):
        return self.update(image_logits, image_labels, text_logits, text_labels, loss, image_loss, text_loss)

    def update(self, image_logits, image_labels, text_logits, text_labels, loss, image_loss, text_loss):
        self.n_step += 1
        with torch.no_grad():
            if image_logits.shape[0] != 0:
                image_preds = image_logits.argmax(dim=-1)
                self.right += (image_preds == image_labels).masked_fill(image_labels.eq(-100), 0).sum().item()
                self.total += (image_labels != -100).sum().item()
                self.image_right += (image_preds == image_labels).masked_fill(image_labels.eq(-100), 0).sum().item()
                self.image_total += (image_labels != -100).sum().item()

            if text_logits.shape[0] != 0:
                text_shift_preds = text_logits[..., :-1, :].argmax(dim=-1)
                text_shift_labels = text_labels[..., 1:]

                self.right += (text_shift_preds == text_shift_labels).masked_fill(text_shift_labels.eq(-100), 0).sum().item()
                self.total += (text_shift_labels != -100).sum().item()

                self.text_right += (text_shift_preds == text_shift_labels).masked_fill(text_shift_labels.eq(-100), 0).sum().item()
                self.text_total += (text_shift_labels != -100).sum().item()

            self.image_loss += image_loss.item()
            self.text_loss += text_loss.item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        dist.all_reduce(self.image_right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.image_total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.image_loss, op=torch.distributed.ReduceOp.SUM)

        dist.all_reduce(self.text_right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.text_total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.text_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total.clamp_min(1)).item()
        image_acc = (self.image_right / self.image_total.clamp_min(1)).item()
        text_acc = (self.text_right / self.text_total.clamp_min(1)).item()

        loss = self.total_loss.item() / (self.world_size * max(self.n_step, 1))
        image_loss = self.image_loss.item() / (self.world_size * max(self.n_step, 1))
        text_loss = self.text_loss.item() / (self.world_size * max(self.n_step, 1))

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)

            self.image_right.fill_(0)
            self.image_total.fill_(0)
            self.image_loss.fill_(0)

            self.text_right.fill_(0)
            self.text_total.fill_(0)
            self.text_loss.fill_(0)

        return acc, loss, image_acc, image_loss, text_acc, text_loss

    
def load_mixture_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    task_configs = []
    for task_name, task_info in cfg.items():
        image_loss_weight = float(task_info["image_loss_weight"])
        datasets = task_info["datasets"]

        cur = {
            "task_name": task_name,
            "image_loss_weight": image_loss_weight,
            "datasets": []
        }

        for ds in datasets:
            output_path = ds["output_path"]
            ratio = float(ds.get("ratio", 1.0))
            cur["datasets"].append({
                "name": ds["name"],
                "path": output_path,   # training uses only output_path
                "ratio": ratio,
            })
        task_configs.append(cur)

    return task_configs


import mmap
import numpy as np
from torch.utils.data import Dataset

class JsonlOffsetDataset(Dataset):
    def __init__(self, path: str, build_index: bool = True):
        self.path = path
        self.idx_path = path + ".idx.npy"

        if os.path.exists(self.idx_path):
            self.offsets = np.load(self.idx_path, mmap_mode="r")
        else:
            if not build_index:
                raise FileNotFoundError(f"Index not found: {self.idx_path}")
            self.offsets = self._build_offsets()
            np.save(self.idx_path, self.offsets)

        self._fp = None

    def _build_offsets(self):
        logger.info(f"Building offsets for {self.path}")
        offsets = []
        offset = 0
        with open(self.path, "rb") as f:
            for line in f:
                offsets.append(offset)
                offset += len(line)
        offsets = np.asarray(offsets, dtype=np.int64)
        logger.info(f"Built {len(offsets)} offsets for {self.path}")
        return offsets

    def __len__(self):
        return len(self.offsets)

    def _ensure_open(self):
        if self._fp is None:
            self._fp = open(self.path, "rb")
        return self._fp

    def __getitem__(self, idx):
        fp = self._ensure_open()
        fp.seek(int(self.offsets[idx]))
        line = fp.readline()
        if not line:
            raise IndexError(f"Failed to read line at idx={idx} from {self.path}")
        return json.loads(line.decode("utf-8"))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def __del__(self):
        if getattr(self, "_fp", None) is not None:
            try:
                self._fp.close()
            except Exception:
                pass

import math
import random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset

class RatioDataset(Dataset):
    """
    Performs ratio-based sampling/repetition over a base dataset; supports
    multi-process epoch synchronization with persistent workers.
    """
    def __init__(self, base_dataset: Dataset, ratio: float, seed: int = 42, name: str = ""):
        self.base_dataset = base_dataset
        self.ratio = float(ratio)
        self.seed = seed
        self.name = name

        # Use a multiprocessing shared-memory value to synchronize the epoch,
        # so that updates from the main process are visible in real time to all
        # persistent workers.
        self.shared_epoch = mp.Value('i', 0)

        # Worker-local state
        self._local_epoch = -1
        self.indices = np.array([], dtype=np.int64)

        # The main process needs the total length when building the DataLoader,
        # so it is computed up front.
        self._len = self._calc_len()

    def set_epoch(self, epoch: int):
        # The main process only updates the shared-memory value here.
        self.shared_epoch.value = epoch

    def _calc_len(self):
        n = len(self.base_dataset)
        if n == 0 or self.ratio <= 0:
            return 0
        if self.ratio < 1:
            return int(n * self.ratio)
        full_repeat = int(math.floor(self.ratio))
        frac = self.ratio - full_repeat
        return full_repeat * n + int(n * frac)

    def _build_indices(self, current_epoch: int):
        n = len(self.base_dataset)
        if n == 0 or self.ratio <= 0:
            return np.array([], dtype=np.int64)

        rng = random.Random(self.seed + current_epoch)

        if self.ratio < 1:
            k = int(n * self.ratio)
            perm = np.arange(n, dtype=np.int64)
            rng.shuffle(perm)
            return perm[:k]

        full_repeat = int(math.floor(self.ratio))
        frac = self.ratio - full_repeat

        # Pre-allocate a numpy array to avoid the memory blow-up of a Python list.
        total_len = self._len
        out = np.empty(total_len, dtype=np.int64)
        
        offset = 0
        for i in range(full_repeat):
            perm = np.arange(n, dtype=np.int64)
            rng_i = random.Random(self.seed + current_epoch * 100003 + i)
            rng_i.shuffle(perm)
            out[offset : offset + n] = perm
            offset += n

        if frac > 0:
            k = int(n * frac)
            perm = np.arange(n, dtype=np.int64)
            rng_f = random.Random(self.seed + current_epoch * 100003 + full_repeat)
            rng_f.shuffle(perm)
            out[offset : offset + k] = perm[:k]

        return out

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Lazy update: before fetching each item, the worker checks whether the
        # shared epoch has changed. On entering a new epoch the worker rebuilds
        # its numpy index.
        current_epoch = self.shared_epoch.value
        if current_epoch != self._local_epoch:
            self.indices = self._build_indices(current_epoch)
            self._local_epoch = current_epoch
            
        return self.base_dataset[self.indices[idx]]


class SftCollator:
    def __init__(self, args, extra_vision_cfg, processor, image_loss_weight: float):
        self.config = extra_vision_cfg
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.args = args
        self.image_loss_weight = image_loss_weight

        self.processor.image_start_tag = "<|vision_start|>"
        self.processor.image_tag = "<|image_pad|>"
        self.processor.image_id = self.tokenizer.vocab.get(self.processor.image_tag)
        self.processor.image_end_tag = "<|vision_end|>"
        self.processor.pad_tag = "<|vision_pad|>"
        self.processor.image_end_id = self.tokenizer.vocab.get(self.processor.image_end_tag)
        self.processor.pad_id = self.tokenizer.vocab.get(self.processor.pad_tag)
        self.processor.first_gen_num_image_tokens = self.config.gen_cfg.n_base
    def _safe_open_image(self, image_path):
        try:
            with PIL.Image.open(image_path) as img:
                img = img.convert("RGB")
            return img, None
        except Exception as e:
            return None, f"{image_path}: {repr(e)}"
    def _slice_grid_by_image_indices(self, grid_thw, keep_image_indices):
        if grid_thw is None:
            return None
        if len(keep_image_indices) == 0:
            return None

        idx = torch.as_tensor(keep_image_indices, dtype=torch.long, device=grid_thw.device)
        return grid_thw.index_select(0, idx)
    def _slice_pixel_values_by_image_indices(self, pixel_values, grid_thw, keep_image_indices):
        """
        pixel_values: [sum_i (t_i*h_i*w_i), dim]
        grid_thw:     [num_images, 3]
        keep_image_indices: which images to keep (along the image dimension)
        """
        if pixel_values is None:
            return None
        if grid_thw is None:
            raise ValueError("grid_thw is required when slicing pixel_values by image indices")

        if len(keep_image_indices) == 0:
            return None

        device = pixel_values.device
        grid_thw_cpu = grid_thw.detach().cpu()

        # number of visual tokens occupied by each image
        per_image_lens = []
        for i in range(grid_thw_cpu.shape[0]):
            t, h, w = grid_thw_cpu[i].tolist()
            per_image_lens.append(int(t * h * w))

        # prefix sum to locate the start/end position of each image within pixel_values
        offsets = [0]
        for n in per_image_lens:
            offsets.append(offsets[-1] + n)

        keep_rows = []
        for img_idx in keep_image_indices:
            start = offsets[img_idx]
            end = offsets[img_idx + 1]
            keep_rows.append(torch.arange(start, end, device=device, dtype=torch.long))

        if len(keep_rows) == 0:
            return None

        keep_rows = torch.cat(keep_rows, dim=0)
        return pixel_values.index_select(0, keep_rows)

    def _truncate_segments_right(self, segments, max_len: int):
        """
        Right-side truncation: keep the first max_len tokens from the left.
        Rules:
        - atomic segments (image blocks, prefix, suffix) cannot be split; they
          are either fully kept or dropped entirely
        - text segments may be partially truncated at the token level
        """
        kept = []
        cur_len = 0

        for seg in segments:
            seg_len = len(seg["ids"])
            remain = max_len - cur_len
            if remain <= 0:
                break

            if seg_len <= remain:
                kept.append(seg)
                cur_len += seg_len
                continue

            # the current segment does not fit
            if seg.get("atomic", False):
                break

            # only text segments allow partial truncation
            kept.append({
                **seg,
                "ids": seg["ids"][:remain],
                "loss_mask": seg["loss_mask"][:remain],
            })
            cur_len += remain
            break

        return kept
    def _make_skip_batch(self, reason: str):
        logger.warning(f"[Collator] skip batch: {reason}")
        return {
            "skip_batch": torch.tensor([1], dtype=torch.long),
            "skip_reason": reason,
        }
    def get_code_book(self, image_paths):
        images = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
        images_outputs = self.processor.image_processor(images, return_tensors="pt")
        return images_outputs['pixel_values'].to(torch.bfloat16)

    def process_image(self, image_paths, to_und_token):
        images = []
        bad_files = []

        for image_path in image_paths:
            img, err = self._safe_open_image(image_path)
            if err is not None:
                bad_files.append(err)
            else:
                images.append(img)

        if bad_files:
            return None, None, bad_files

        try:
            images_outputs = self.processor.image_processor(
                images,
                to_und_token=to_und_token,
                min_pixels=self.args.min_pixels,
                max_pixels=self.args.max_pixels,
                return_tensors="pt",
            )
            return images_outputs["pixel_values"].to(torch.bfloat16), images_outputs["image_grid_thw"], None
        except Exception as e:
            return None, None, [f"image_processor failed: {repr(e)}"]

    def validate_sample(self, sample):
        msg_input_ids = []
        msg_output_ids = []

        for msg in sample["messages"]:
            msg_input_ids.extend(re.findall(r"<(input_image_\d+)>", msg["content"]))
            msg_output_ids.extend(re.findall(r"<(output_image_\d+)>", msg["content"]))

        declared_input_ids = [img["id"] for img in sample.get("input_images", [])]
        declared_output_ids = [img["id"] for img in sample.get("output_images", [])]

        if msg_input_ids != declared_input_ids:
            raise ValueError(f"input image ids mismatch: {msg_input_ids} vs {declared_input_ids}")
        if msg_output_ids != declared_output_ids:
            raise ValueError(f"output image ids mismatch: {msg_output_ids} vs {declared_output_ids}")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            PLACEHOLDER_PATTERN = re.compile(r"<(input_image_\d+|output_image_\d+)>")

            try:
                for x in batch:
                    self.validate_sample(x)
            except Exception as e:
                return self._make_skip_batch(f"validate_sample failed: {repr(e)}")

            all_input_images = []
            all_output_images = []
            per_sample_input_paths = []
            per_sample_output_paths = []

            for x in batch:
                cur_input_images = [img["path"] for img in x.get("input_images", [])]
                cur_output_images = [img["path"] for img in x.get("output_images", [])]
                per_sample_input_paths.append(cur_input_images)
                per_sample_output_paths.append(cur_output_images)
                all_input_images.extend(cur_input_images)
                all_output_images.extend(cur_output_images)

            # process input images
            if len(all_input_images) > 0:
                input_pixel_values, input_pixel_values_grid_thw, err1 = self.process_image(
                    all_input_images, to_und_token=False
                )
                if err1 is not None:
                    return self._make_skip_batch("process input_image failed: " + " | ".join(err1))

                und_input_pixel_values, und_input_pixel_values_grid_thw, err2 = self.process_image(
                    all_input_images, to_und_token=True
                )
                if err2 is not None:
                    return self._make_skip_batch("process und_input_image failed: " + " | ".join(err2))
            else:
                input_pixel_values, input_pixel_values_grid_thw = None, None
                und_input_pixel_values, und_input_pixel_values_grid_thw = None, None

            # process output images
            if len(all_output_images) > 0:
                pixel_values, pixel_values_grid_thw, err3 = self.process_image(
                    all_output_images, to_und_token=False
                )
                if err3 is not None:
                    return self._make_skip_batch("process output_image failed: " + " | ".join(err3))
            else:
                pixel_values, pixel_values_grid_thw = None, None

            # ----------------------------------------------------------
            # generate num_tokens per sample first
            # ----------------------------------------------------------
            batch_num_tokens_raw = []
            for x in batch:
                if getattr(self.args, "use_json_num_tokens", False):
                    cur_tokens = x.get("num_tokens", [])

                    n_img = len(x.get("input_images", [])) + len(x.get("output_images", []))
                    # images present but num_tokens is empty: raise explicitly to fail fast
                    if n_img > 0 and cur_tokens is None:
                        raise ValueError(
                            f"sample has {n_img} images but num_tokens is empty, sample={x}"
                        )
                    if cur_tokens is None:
                        cur_tokens = []

                    if not isinstance(cur_tokens, list):
                        raise ValueError(f"sample 'num_tokens' must be list, got {type(cur_tokens)}")



                    batch_num_tokens_raw.append(cur_tokens[1:])
                else:
                    n_img = len(x.get("input_images", [])) + len(x.get("output_images", []))
                    batch_num_tokens_raw.append([
                        random.choice(self.config.gen_cfg.n_delta)
                        for _ in range(max(0, n_img - 1))
                    ])

            pre_data = []
            batch_position_ids = []
            batch_image_num = []
            batch_num_input_image = []

            # global image-cropping indices
            global_keep_input_indices = []
            global_keep_output_indices = []

            # num_tokens retained after truncation
            batch_num_tokens_kept = []

            global_input_base = 0
            global_output_base = 0

            for sample_idx, x in enumerate(batch):
                messages = x["messages"]
                sample_num_tokens = batch_num_tokens_raw[sample_idx]
                sample_token_ptr = 0
                sample_first_image = True

                segments = []

                local_input_ptr = 0
                local_output_ptr = 0

                # ------------------------------------------------------
                # build segments first, do not concatenate into the final
                # input_ids directly
                # ------------------------------------------------------
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]

                    prefix = f"<|im_start|>{role}\n"
                    prefix_ids = self.processor.tokenizer.encode(prefix, add_special_tokens=False)
                    segments.append({
                        "type": "prefix",
                        "atomic": True,
                        "ids": prefix_ids,
                        "loss_mask": [0] * len(prefix_ids),
                    })

                    last = 0
                    for m in PLACEHOLDER_PATTERN.finditer(content):
                        if m.start() > last:
                            text_part = content[last:m.start()]
                            text_ids = self.processor.tokenizer.encode(text_part, add_special_tokens=False)
                            segments.append({
                                "type": "text",
                                "atomic": False,
                                "ids": text_ids,
                                "loss_mask": ([1] if role == "assistant" else [0]) * len(text_ids),
                            })

                        image_key = m.group(1)

                        if image_key.startswith("input_image_"):
                            # compute the number of und tokens from the und grid
                            if und_input_pixel_values_grid_thw is None:
                                raise ValueError("Found input image placeholder but und_input_pixel_values_grid_thw is None")

                            t, h, w = und_input_pixel_values_grid_thw[global_input_base + local_input_ptr]
                            und_num_img_tokens = int((h * w).item() // 4) if isinstance(h, torch.Tensor) else (h * w) // 4

                            consumed_num_token = None
                            if sample_first_image:
                                cur_len = self.processor.first_gen_num_image_tokens
                                sample_first_image = False
                            else:
                                if sample_token_ptr >= len(sample_num_tokens):
                                    raise ValueError(
                                        f"sample {sample_idx} num_tokens not enough, need more for {image_key}, current={sample_num_tokens}"
                                    )
                                cur_len = sample_num_tokens[sample_token_ptr]
                                consumed_num_token = cur_len
                                sample_token_ptr += 1

                            img_str = (
                                self.processor.image_start_tag
                                + self.processor.image_tag * und_num_img_tokens
                                + self.processor.image_end_tag
                                + self.processor.image_start_tag
                                + self.processor.pad_tag * cur_len
                                + self.processor.image_end_tag
                            )
                            img_ids = self.processor.tokenizer.encode(img_str, add_special_tokens=False)
                            segments.append({
                                "type": "input_image",
                                "atomic": True,
                                "ids": img_ids,
                                "loss_mask": [0] * len(img_ids),
                                "local_input_idx": local_input_ptr,
                                "consumed_num_token": consumed_num_token,  # None for the first image
                            })

                            local_input_ptr += 1

                        else:
                            consumed_num_token = None
                            if sample_first_image:
                                cur_len = self.processor.first_gen_num_image_tokens
                                sample_first_image = False
                            else:
                                if sample_token_ptr >= len(sample_num_tokens):
                                    raise ValueError(
                                        f"sample {sample_idx} num_tokens not enough, need more for {image_key}, current={sample_num_tokens}"
                                    )
                                cur_len = sample_num_tokens[sample_token_ptr]
                                consumed_num_token = cur_len
                                sample_token_ptr += 1

                            img_str = (
                                self.processor.image_start_tag
                                + self.processor.pad_tag * cur_len
                                + self.processor.image_end_tag
                            )
                            img_ids = self.processor.tokenizer.encode(img_str, add_special_tokens=False)
                            segments.append({
                                "type": "output_image",
                                "atomic": True,
                                "ids": img_ids,
                                "loss_mask": ([1] if role == "assistant" else [0]) * len(img_ids),
                                "local_output_idx": local_output_ptr,
                                "consumed_num_token": consumed_num_token,  # None for the first image
                            })

                            local_output_ptr += 1

                        last = m.end()

                    if last < len(content):
                        text_part = content[last:]
                        text_ids = self.processor.tokenizer.encode(text_part, add_special_tokens=False)
                        segments.append({
                            "type": "text",
                            "atomic": False,
                            "ids": text_ids,
                            "loss_mask": ([1] if role == "assistant" else [0]) * len(text_ids),
                        })

                    suffix = "<|im_end|>\n"
                    suffix_ids = self.processor.tokenizer.encode(suffix, add_special_tokens=False)
                    segments.append({
                        "type": "suffix",
                        "atomic": True,
                        "ids": suffix_ids,
                        "loss_mask": ([1] if role == "assistant" else [0]) * len(suffix_ids),
                    })

                # ------------------------------------------------------
                # right-side truncation: keep the first max_seq_len tokens
                # ------------------------------------------------------
                kept_segments = self._truncate_segments_right(segments, self.args.max_seq_len)

                # ------------------------------------------------------
                # rebuild the sample based on the truncation result
                # ------------------------------------------------------
                sample_input_ids = []
                sample_loss_mask = []
                kept_input_local_indices = []
                kept_output_local_indices = []
                kept_num_tokens = []

                for seg in kept_segments:
                    sample_input_ids.extend(seg["ids"])
                    sample_loss_mask.extend(seg["loss_mask"])

                    if seg["type"] == "input_image":
                        kept_input_local_indices.append(seg["local_input_idx"])
                        if seg.get("consumed_num_token") is not None:
                            kept_num_tokens.append(seg["consumed_num_token"])

                    elif seg["type"] == "output_image":
                        kept_output_local_indices.append(seg["local_output_idx"])
                        if seg.get("consumed_num_token") is not None:
                            kept_num_tokens.append(seg["consumed_num_token"])

                # optional: deduplicate while preserving the original order
                kept_input_local_indices = sorted(set(kept_input_local_indices))
                kept_output_local_indices = sorted(set(kept_output_local_indices))

                # build sft_format (for logging only)
                sample_sft_format = self.processor.tokenizer.decode(
                    sample_input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )

                input_ids = torch.LongTensor(sample_input_ids)
                text_label = input_ids.clone()
                loss_mask = torch.tensor(sample_loss_mask, dtype=torch.bool)

                text_label[~loss_mask] = -100
                text_label[text_label == self.processor.pad_id] = -100
                text_label[text_label == self.processor.image_end_id] = -100

                # ------------------------------------------------------
                # position_ids only consume the und grid of input images that
                # remain after truncation
                # ------------------------------------------------------
                if len(kept_input_local_indices) == 0:
                    position_ids, _ = get_rope_index_3(2, input_ids.unsqueeze(0))
                else:
                    cur_keep_global_input = [global_input_base + i for i in kept_input_local_indices]
                    cur_keep_global_input_t = torch.tensor(
                        cur_keep_global_input,
                        dtype=torch.long,
                        device=und_input_pixel_values_grid_thw.device
                    )
                    cur_und_grid = und_input_pixel_values_grid_thw.index_select(0, cur_keep_global_input_t)
                    position_ids, _ = get_rope_index_3(2, input_ids.unsqueeze(0), cur_und_grid)

                batch_position_ids.append(position_ids)
                batch_image_num.append(len(kept_output_local_indices))
                batch_num_input_image.append(len(kept_input_local_indices))
                batch_num_tokens_kept.append(kept_num_tokens)

                # record the global indices of kept images
                global_keep_input_indices.extend([global_input_base + i for i in kept_input_local_indices])
                global_keep_output_indices.extend([global_output_base + i for i in kept_output_local_indices])

                pre_data.append(
                    VLChatProcessorOutput(
                        sft_format=sample_sft_format,
                        input_ids=input_ids,
                        text_label=text_label,
                    )
                )

                global_input_base += len(x.get("input_images", []))
                global_output_base += len(x.get("output_images", []))

            # ----------------------------------------------------------
            # apply synchronized cropping to the image tensors / grids
            # ----------------------------------------------------------
            input_pixel_values = self._slice_pixel_values_by_image_indices(
                input_pixel_values,
                input_pixel_values_grid_thw,
                global_keep_input_indices,
            )
            und_input_pixel_values = self._slice_pixel_values_by_image_indices(
                und_input_pixel_values,
                und_input_pixel_values_grid_thw,
                global_keep_input_indices,
            )
            pixel_values = self._slice_pixel_values_by_image_indices(
                pixel_values,
                pixel_values_grid_thw,
                global_keep_output_indices,
            )

            input_pixel_values_grid_thw = self._slice_grid_by_image_indices(
                input_pixel_values_grid_thw,
                global_keep_input_indices,
            )
            und_input_pixel_values_grid_thw = self._slice_grid_by_image_indices(
                und_input_pixel_values_grid_thw,
                global_keep_input_indices,
            )
            pixel_values_grid_thw = self._slice_grid_by_image_indices(
                pixel_values_grid_thw,
                global_keep_output_indices,
            )

            prepare_inputs = self.processor.batchify(pre_data)
            batch_position_ids = torch.cat(batch_position_ids, dim=2)
            batch_image_num = torch.tensor(batch_image_num)
            batch_num_input_image = torch.tensor(batch_num_input_image)

            flat_num_tokens = []
            for toks in batch_num_tokens_kept:
                flat_num_tokens.extend(toks)

            return {
                "input_ids": prepare_inputs.input_ids,
                "text_label": prepare_inputs.text_label,
                "pixel_values": pixel_values,
                "input_pixel_values": input_pixel_values,
                "und_input_pixel_values": und_input_pixel_values,
                "attention_mask": prepare_inputs.attention_mask,
                "image_grid_thw": input_pixel_values_grid_thw,
                "und_image_grid_thw": und_input_pixel_values_grid_thw,
                "output_image_grid_thw": pixel_values_grid_thw,
                "position_ids": batch_position_ids,
                "image_num": batch_image_num,
                "num_input_image": batch_num_input_image,
                "num_tokens": flat_num_tokens,
                "image_loss_weight": torch.tensor(self.image_loss_weight, dtype=torch.float32),
                "skip_batch": torch.tensor([0], dtype=torch.long),
                "skip_reason": "",
            }
        except Exception as e:
            return self._make_skip_batch(f"collator failed: {repr(e)}")

from torch.utils.data import ConcatDataset, DataLoader

def build_task_dataloaders(args, processor, extra_vision_cfg):
    task_cfgs = load_mixture_config(args.mixture_config)
    task_loaders = {}
    task_num_batches = {}

    for task_cfg in task_cfgs:
        task_name = task_cfg["task_name"]
        image_loss_weight = task_cfg["image_loss_weight"]

        sub_datasets = []
        for ds in task_cfg["datasets"]:
            base_ds = JsonlOffsetDataset(ds["path"])
            ratio_ds = RatioDataset(
                base_dataset=base_ds,
                ratio=ds["ratio"],
                seed=args.seed,
                name=ds["name"]
            )
            sub_datasets.append(ratio_ds)

        if len(sub_datasets) == 1:
            merged_ds = sub_datasets[0]
        else:
            merged_ds = ConcatDataset(sub_datasets)

        collator = SftCollator(
            args=args,
            extra_vision_cfg=extra_vision_cfg,
            processor=processor,
            image_loss_weight=image_loss_weight
        )

        loader = DataLoader(
            merged_ds,
            batch_size=args.train_bsz_per_gpu,
            shuffle=args.shuffle,
            drop_last=True,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )

        task_loaders[task_name] = loader
        task_num_batches[task_name] = len(loader)

        logger.info(
            f"[Task={task_name}] datasets={len(sub_datasets)}, "
            f"samples={len(merged_ds)}, batches={len(loader)}, "
            f"image_loss_weight={image_loss_weight}"
        )

    return task_loaders, task_num_batches

def set_epoch_for_dataset(ds, epoch: int):
    if isinstance(ds, RatioDataset):
        ds.set_epoch(epoch)
    elif isinstance(ds, ConcatDataset):
        for sub_ds in ds.datasets:
            set_epoch_for_dataset(sub_ds, epoch)




def ForCausalLMLoss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    vocab_size: int,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    reduction: str = "sum",
) -> torch.Tensor:

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Invalid reduction: {reduction}")

    # avoid fp16/bf16 precision issues
    logits = logits.float()

    # if shift_labels is not provided, construct it from labels
    if shift_labels is None:
        if labels is None:
            raise ValueError("Either labels or shift_labels must be provided")

        # causal LM shift
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # flatten
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device)

    # use sum directly
    loss = F.cross_entropy(
        logits,
        shift_labels,
        ignore_index=ignore_index,
        reduction=reduction,
    )

    return loss

def count_text_valid_tokens(text_labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    text_labels: [B, L]
    For a causal LM, the labels that actually contribute to the loss are the
    shifted labels, i.e. text_labels[..., 1:].
    """
    shift_labels = text_labels[..., 1:]
    return (shift_labels != ignore_index).sum()


def count_image_valid_tokens(output_image_tokens: Optional[torch.Tensor]) -> torch.Tensor:
    """
    output_image_tokens: flattened image-token labels, or any tensor that can be
    view(-1)'d. -100 is treated as the ignore index by default.
    """
    if output_image_tokens is None:
        return torch.tensor(0, device='cpu')

    if not isinstance(output_image_tokens, torch.Tensor):
        output_image_tokens = torch.as_tensor(output_image_tokens)

    return (output_image_tokens.view(-1) != -100).sum()
def unwrap_if_accelerated(model):
    return model.module if hasattr(model, "module") else model


def resolve_hf_model_dir(model_path: str) -> str:
    """
    Supports the following input forms:
    1) An HF model directory directly: xxx/tfmr or xxx/hf_model
    2) A checkpoint root directory: xxx/checkpoint-step-1000
       (tfmr / hf_model inside it is preferred automatically)
    3) An original model directory: returned as-is
    """
    if model_path is None:
        return model_path

    candidates = [
        os.path.join(model_path, "tfmr"),
        os.path.join(model_path, "hf_model"),
        model_path,
    ]

    for c in candidates:
        if os.path.isdir(c):
            has_config = os.path.exists(os.path.join(c, "config.json"))
            has_processor = (
                os.path.exists(os.path.join(c, "preprocessor_config.json"))
                or os.path.exists(os.path.join(c, "processor_config.json"))
            )
            has_weights = (
                os.path.exists(os.path.join(c, "model.safetensors"))
                or os.path.exists(os.path.join(c, "pytorch_model.bin"))
                or any(x.startswith("model-") and x.endswith(".safetensors") for x in os.listdir(c))
            )
            if has_config and (has_weights or has_processor):
                return c

    return model_path


def find_latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None

    ckpts = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if not os.path.isdir(full):
            continue
        m = re.match(r"checkpoint-step-(\d+)", name)
        if m:
            ckpts.append((int(m.group(1)), full))

    if not ckpts:
        return None

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]

def save_checkpoint(
    model,
    processor,
    optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    update_step: int,
    is_last: bool = False
) -> None:
    save_dir = os.path.join(args.output_dir, f"checkpoint-step-{update_step}")

    if accelerator.is_main_process:
        checkpoint_dirs = [
            f for f in os.listdir(args.output_dir)
            if f.startswith("checkpoint-step-") and os.path.isdir(os.path.join(args.output_dir, f))
        ]

        if args.max_ckpts > 0 and len(checkpoint_dirs) >= args.max_ckpts:
            def _extract_step(x):
                m = re.match(r"checkpoint-step-(\d+)", x)
                return int(m.group(1)) if m else -1

            checkpoint_dirs = sorted(checkpoint_dirs, key=_extract_step)
            num_to_delete = len(checkpoint_dirs) - args.max_ckpts + 1
            for ckpt in checkpoint_dirs[:num_to_delete]:
                shutil.rmtree(os.path.join(args.output_dir, ckpt), ignore_errors=True)

        os.makedirs(save_dir, exist_ok=True)

        hf_save_dir = os.path.join(save_dir, "tfmr")

        # save processor
        processor.save_pretrained(hf_save_dir)

        # save HF model
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            hf_save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=True,
        )

        # save training progress
        train_state = {
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,   # micro step
            "update_step": update_step,   # optimizer step
            "model_path": args.model_path,
            "stage": args.stage,
        }
        with open(os.path.join(save_dir, "training_state.json"), "w", encoding="utf-8") as f:
            json.dump(train_state, f, indent=2, ensure_ascii=False)

    accelerator.wait_for_everyone()

    # save optimizer / scheduler / scaler / rng / distributed state
    accelerator.save_state(save_dir)

    logger.info(f"Checkpoint saved successfully at {save_dir}")

def gather_image_logits(hidden_states, start_indices, token_counts, model):
    device = hidden_states.device
    token_counts = torch.as_tensor(token_counts, device=device)

    # which image each token belongs to
    image_ids = torch.repeat_interleave(
        torch.arange(len(token_counts), device=device),
        token_counts
    )

    # offset of each token within its image
    offsets = torch.cat([
        torch.arange(c, device=device) for c in token_counts
    ])

    # gather index
    gather_indices = start_indices[image_ids] + offsets

    # gather hidden
    hidden = torch.index_select(hidden_states, 1, gather_indices).squeeze(0)
    hidden = model.norm(hidden)

    logits = model.vision_head(hidden)

    return logits, gather_indices
def should_skip_globally(batch, accelerator):
    local_skip = batch.get("skip_batch", 0)
    if not isinstance(local_skip, torch.Tensor):
        local_skip = torch.tensor([int(bool(local_skip))], device=accelerator.device)
    else:
        local_skip = local_skip.to(accelerator.device)

    if local_skip.ndim == 0:
        local_skip = local_skip.unsqueeze(0)

    dist.all_reduce(local_skip, op=dist.ReduceOp.MAX)
    return local_skip.item() > 0
def build_batch_plan(task_num_batches: Dict[str, int], seed: int, epoch: int):
    plan = []
    for task_name, n_batches in task_num_batches.items():
        plan.extend([task_name] * n_batches)

    rng = random.Random(seed + epoch)
    rng.shuffle(plan)
    return plan
from accelerate.utils import DistributedDataParallelKwargs
def train(args: argparse.Namespace) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            config=vars(args),
            dir=args.log_dir,
            mode="offline"
        )
    from accelerate.utils import DistributedType

    logger.info(f"accelerator.distributed_type = {accelerator.distributed_type}")
    logger.info(f"accelerator.state = {accelerator.state}")

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        logger.info("DeepSpeed is ENABLED")
        logger.info(f"DeepSpeed plugin: {accelerator.state.deepspeed_plugin}")
    else:
        logger.info("DeepSpeed is NOT enabled")
    
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == "latest":
        resume_ckpt = find_latest_checkpoint(args.output_dir)

    if resume_ckpt is None:
        logger.info(f"No checkpoint found in {args.output_dir}, fallback to base model: {args.model_path}")
    else:
        logger.info(f"Found checkpoint for resume: {resume_ckpt}")

    # source for processor/config
    processor_source = resolve_hf_model_dir(resume_ckpt) if resume_ckpt is not None else resolve_hf_model_dir(args.model_path)

    processor = ViMoProcessor.from_pretrained(processor_source)
    if "to_und_token" not in processor.image_processor._valid_kwargs_names:
        processor.image_processor._valid_kwargs_names.append("to_und_token")
    config = ViMoConfig.from_pretrained(processor_source)
    extra_vision_cfg = TSIMTokExtraCfg.load(args.extra_vision_cfg) if args.extra_vision_cfg else TSIMTokExtraCfg()

    model = build_model(args, config, extra_vision_cfg, resume_ckpt=resume_ckpt)

    freeze_params(model, args.stage)
    print_trainable_params(model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    task_loaders, _ = build_task_dataloaders(args, processor, extra_vision_cfg)

    model.language_model.loss_function = ForCausalLMLoss

    # prepare model / optimizer / dataloaders first
    prepare_items = [model, optimizer]
    prepare_items.extend(task_loaders.values())
    prepared = accelerator.prepare(*prepare_items)

    model = prepared[0]
    optimizer = prepared[1]
    prepared_loaders = prepared[2:]

    task_names = list(task_loaders.keys())
    task_loaders = {k: v for k, v in zip(task_names, prepared_loaders)}

    # the actual per-rank dataloader length must be counted after prepare
    task_num_batches = {task_name: len(loader) for task_name, loader in task_loaders.items()}
    total_batches_per_epoch = sum(task_num_batches.values())
    num_training_steps = (total_batches_per_epoch * args.n_epochs) // accelerator.gradient_accumulation_steps
    num_training_steps = max(num_training_steps, 1)

    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio
    )

    # prepare the scheduler separately
    lr_scheduler = accelerator.prepare(lr_scheduler)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    model.train()

    global_step = 0          # micro step
    update_step = 0          # optimizer step
    start_epoch = 1
    resume_step_in_epoch = 0

    if resume_ckpt is not None:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")
        accelerator.load_state(resume_ckpt)

        state_path = os.path.join(resume_ckpt, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                train_state = json.load(f)

            start_epoch = int(train_state.get("epoch", 1))
            global_step = int(train_state.get("global_step", 0))
            update_step = int(train_state.get("update_step", 0))
            resume_step_in_epoch = int(train_state.get("step_in_epoch", -1)) + 1

            logger.info(
                f"Resume loaded: start_epoch={start_epoch}, "
                f"resume_step_in_epoch={resume_step_in_epoch}, "
                f"global_step={global_step}, update_step={update_step}"
            )
        else:
            logger.warning("training_state.json not found, fallback to epoch=1 / step=0")

    logger.info(f"Per-rank task_num_batches: {task_num_batches}")
    logger.info(f"Per-rank total_batches_per_epoch: {total_batches_per_epoch}")
    logger.info(f"num_training_steps: {num_training_steps}")
    logger.info(f"model class after prepare: {type(model)}")
    for epoch in range(start_epoch, args.n_epochs + 1):
        for task_name, loader in task_loaders.items():
            try:
                ds = loader.dataset
                set_epoch_for_dataset(ds, epoch)
            except Exception as e:
                logger.warning(f"set_epoch failed for {task_name}: {e}")

        task_iters = {}
        batch_plan = build_batch_plan(task_num_batches, seed=args.seed, epoch=epoch)
        train_iter = tqdm(batch_plan, total=len(batch_plan)) if accelerator.is_main_process else batch_plan

        for step_in_epoch, task_name in enumerate(train_iter):
            if task_name not in task_iters:
                task_iters[task_name] = iter(task_loaders[task_name])

            batch = next(task_iters[task_name])

            if resume_ckpt is not None and epoch == start_epoch and step_in_epoch < resume_step_in_epoch:
                global_step += 1
                continue

            global_skip = should_skip_globally(batch, accelerator)
            if global_skip:
                if accelerator.is_main_process:
                    reason = batch.get("skip_reason", "")
                    logger.warning(
                        f"[Epoch {epoch}] skip synchronized batch on all ranks. "
                        f"task={task_name}, reason={reason}"
                    )
                global_step += 1
                continue

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs, output_image_tokens, output_image_index, output_token_counts = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    und_pixel_values=batch['und_input_pixel_values'],
                    pixel_values=batch['input_pixel_values'],
                    output_pixel_values=batch['pixel_values'],
                    und_image_grid_thw=batch['und_image_grid_thw'],
                    image_grid_thw=batch['image_grid_thw'],
                    output_image_grid_thw=batch['output_image_grid_thw'],
                    position_ids=batch['position_ids'],
                    output_image_num=batch["image_num"],
                    num_input_image=batch["num_input_image"],
                    num_tokens=batch["num_tokens"],
                    incremental_encoding=extra_vision_cfg.gen_cfg.incremental_encoding
                )

                hidden_states = outputs[0]

                image_loss_sum, output_image_logits, image_token_count = compute_image_loss(
                    model=model,
                    hidden_states=hidden_states,
                    batch=batch,
                    output_image_index=output_image_index,
                    output_token_counts=output_token_counts,
                    output_image_tokens=output_image_tokens,
                )

                text_label = batch['text_label']
                text_logits = model.lm_head(hidden_states)
                text_loss_sum = model.language_model.loss_function(
                    logits=text_logits,
                    labels=text_label,
                    vocab_size=151936,
                )

                text_token_count = count_text_valid_tokens(text_label).to(hidden_states.device)
                image_token_count = image_token_count.to(hidden_states.device)

                if args.loss_norm_mode == "separate":
                    text_loss = text_loss_sum / text_token_count.clamp_min(1)
                    image_loss = image_loss_sum / image_token_count.clamp_min(1)
                elif args.loss_norm_mode == "total":
                    total_token_count = (text_token_count + image_token_count).clamp_min(1)
                    text_loss = text_loss_sum / total_token_count
                    image_loss = image_loss_sum / total_token_count
                else:
                    raise ValueError(f"Unsupported loss_norm_mode: {args.loss_norm_mode}")

                cur_image_loss_weight = batch["image_loss_weight"]
                if isinstance(cur_image_loss_weight, torch.Tensor):
                    cur_image_loss_weight = cur_image_loss_weight.to(hidden_states.device).item()

                loss = text_loss + cur_image_loss_weight * image_loss

            metric(
                output_image_logits,
                output_image_tokens,
                text_logits,
                text_label,
                loss,
                image_loss,
                text_loss
            )

            accelerator.backward(loss)

            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                update_step += 1

                if (global_step + 1) % (accelerator.gradient_accumulation_steps * 2) == 0:
                    torch.cuda.empty_cache()

                acc, train_loss, image_acc, image_loss_v, text_acc, text_loss_v = metric.get_metric()

                if accelerator.is_main_process:
                    if hasattr(train_iter, "set_postfix"):
                        train_iter.set_postfix(
                            epoch=epoch,
                            step=update_step,
                            task=task_name,
                            loss=f"{train_loss:.3f} (img:{image_loss_v:.3f}, txt:{text_loss_v:.3f})",
                            acc=f"{acc:.3f} (img:{image_acc:.3f}, txt:{text_acc:.3f})",
                            img_w=f"{cur_image_loss_weight:.2f}",
                            lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                        )

                    wandb.log({
                        'loss': train_loss,
                        'acc': acc,
                        'image_acc': image_acc,
                        'text_acc': text_acc,
                        'lr': lr_scheduler.get_last_lr()[0],
                        'task': task_name,
                        'image_loss_weight': cur_image_loss_weight,
                    }, step=update_step)

                if args.save_steps > 0 and update_step % args.save_steps == 0:
                    save_checkpoint(
                        model=model,
                        processor=processor,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        accelerator=accelerator,
                        args=args,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step+1,
                        update_step=update_step,
                        is_last=False
                    )

            global_step += 1

        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()

    if update_step > 0 and (args.save_steps <= 0 or update_step % args.save_steps != 0):
        save_checkpoint(
            model=model,
            processor=processor,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            args=args,
            epoch=epoch,
            step_in_epoch=step_in_epoch if 'step_in_epoch' in locals() else 0,
            global_step=global_step,
            update_step=update_step,
            is_last=True
        )


def init_tok_embeddings_from_vq(model, zero_init_rest: bool = False):
    """
    Initialize the first rows of model.tok_embeddings from
    visual.slot_quantize.embedding.weight.

    - if tok_embeddings has more rows than the vq embedding, only the first
      min_rows rows are copied
    - if tok_embeddings has fewer rows than the vq embedding, only the first
      min_rows rows are copied
    - the number of columns (embedding dim) must match
    - when zero_init_rest=True, the extra rows are zeroed out; otherwise the
      original initialization is kept
    """
    with torch.no_grad():
        src = model.visual.slot_quantize.embedding.weight.data  # [N_src, D]

        # tok_embeddings may be an nn.Embedding or an nn.Parameter
        if isinstance(model.tok_embeddings, torch.nn.Embedding):
            dst = model.tok_embeddings.weight
        else:
            dst = model.tok_embeddings

        if dst.ndim != 2 or src.ndim != 2:
            raise ValueError(
                f"Expected 2D weights, got dst.ndim={dst.ndim}, src.ndim={src.ndim}"
            )

        if dst.shape[1] != src.shape[1]:
            raise ValueError(
                f"Embedding dim mismatch: tok_embeddings shape={tuple(dst.shape)}, "
                f"vq embedding shape={tuple(src.shape)}"
            )

        n = min(dst.shape[0], src.shape[0])
        dst[:n].copy_(src[:n])

        if zero_init_rest and dst.shape[0] > n:
            dst[n:].zero_()

        logger.info(
            f"Initialized tok_embeddings from visual.slot_quantize.embedding: "
            f"copied first {n} rows, src_shape={tuple(src.shape)}, dst_shape={tuple(dst.shape)}, "
            f"zero_init_rest={zero_init_rest}"
        )

def build_model(args, config, extra_vision_cfg, resume_ckpt=None):
    # only restore HF weights from a checkpoint when one actually exists
    if resume_ckpt is not None:
        resolved_resume_path = resolve_hf_model_dir(resume_ckpt)
        if os.path.isdir(resolved_resume_path) and os.path.exists(os.path.join(resolved_resume_path, "config.json")):
            logger.info(f"Loading resume model from checkpoint: {resolved_resume_path}")
            return ViMoModel.from_pretrained(
                resolved_resume_path,
                config=config,
                extra_cfg=extra_vision_cfg,
                torch_dtype=torch.bfloat16,
            )

    # without a resume checkpoint, initialize normally according to the stage
    if args.stage == 1:
        model = ViMoModel(config=config, extra_cfg=extra_vision_cfg)

        state_dict = load_file(os.path.join(args.model_path, "model.safetensors"))
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.visual.") and not (
                k.startswith("model.visual.deepstack_merger_list") or
                k.startswith("model.visual.merger")
            ):
                k = k.replace("model.visual.", "visual.backbone.", 1)
            elif k.startswith("model.visual.deepstack_merger_list") or k.startswith("model.visual.merger"):
                k = k.replace("model.visual.", "", 1)
            else:
                k = k.replace("model.", "", 1)
            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"[Stage 1] load base model done. missing={len(missing)}, unexpected={len(unexpected)}")

        model = model.bfloat16()

        gen_weights = torch.load(args.gen_weights_path, map_location="cpu", weights_only=False)
        vision_ckpt = gen_weights["model"]
        missing_keys, unexpected_keys = model.visual.load_state_dict(vision_ckpt, strict=False)
        logger.info(f"[Stage 1] load gen weights done. missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

        init_tok_embeddings_from_vq(model)
        return model

    else:
        resolved_model_path = resolve_hf_model_dir(args.model_path)
        logger.info(f"[Stage 2] load base HF model from {resolved_model_path}")
        return ViMoModel.from_pretrained(
            resolved_model_path,
            config=config,
            extra_cfg=extra_vision_cfg,
            torch_dtype=torch.bfloat16,
        )

def freeze_params(model, stage):
    for name, p in model.named_parameters():
        if stage == 1:
            if 'visual' in name or 'language_model' in name or 'lm_head' in name or 'merger' in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
        else:  # stage 2
            if 'visual' in name or 'merger' in name:
                p.requires_grad = False
            else:
                p.requires_grad = True

def print_trainable_params(model):
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    logger.info(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def compute_image_loss(
        model,
        hidden_states,
        batch,
        output_image_index,
        output_token_counts,
        output_image_tokens
):
    device = hidden_states.device

    if int(batch["image_num"].sum().item()) <= 0:
        # dummy path: ensure vision-related parameters are "used" on all ranks
        dummy_hidden = hidden_states[:, :1, :]   # [B,1,H]
        dummy_hidden = model.norm(dummy_hidden)
        dummy_logits = model.vision_head(dummy_hidden.reshape(-1, dummy_hidden.shape[-1]))
        zero_loss = dummy_logits.sum() * 0.0

        return (
            zero_loss,
            torch.empty(0, IMAGE_VOCAB_SIZE_WITH_EOS, device=device),
            torch.tensor(0, device=device, dtype=torch.long),
        )


    output_image_index = torch.as_tensor(
        output_image_index,
        device=hidden_states.device,
        dtype=torch.long
    )

    start_indices = output_image_index[:, 0] - 1

    output_image_logits, _ = gather_image_logits(
        hidden_states=hidden_states,
        start_indices=start_indices,
        token_counts=output_token_counts,
        model=model
    )

    if not isinstance(output_image_tokens, torch.Tensor):
        output_image_tokens = torch.as_tensor(output_image_tokens, device=hidden_states.device)
    else:
        output_image_tokens = output_image_tokens.to(hidden_states.device)

    image_valid_tokens = (output_image_tokens.view(-1) != -100).sum()

    image_loss = model.language_model.loss_function(
        logits=output_image_logits,
        labels=None,
        shift_labels=output_image_tokens,
        vocab_size=IMAGE_VOCAB_SIZE_WITH_EOS,
        reduction="sum"
    )

    return image_loss, output_image_logits, image_valid_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training parameter configuration')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='vimo', help='Experiment name')
    parser.add_argument('--run_name', type=str, default='run_1', help='Run name')
    parser.add_argument('--model_path', type=str, default='', help='Pre-trained model path')

    # Data related
    parser.add_argument('--mixture_config', type=str, required=True, help='Mixture training config json path')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch_factor', type=int, default=2)

    parser.add_argument('--output_dir', type=str, default='./Interleaved_model/', help='Model save path')
    parser.add_argument('--max_ckpts', type=int, default=5, help='Maximum number of checkpoints to save')
    parser.add_argument('--log_dir', type=str, default='./train_logs', help='Log save path')

    # Training related
    parser.add_argument('--max_seq_len', type=int, default=8192, help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping threshold, set to 0 for no clipping')
    parser.add_argument('--train_bsz_per_gpu', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=1, help='Minimum learning rate ratio to peak learning rate')
    parser.add_argument('--warmup_rates', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')

    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--extra_vision_cfg", type=str, default="configs/vimo_cfg.json")
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True, help='Training stage: 1 or 2')
    def _str2bool(v):
        return v if isinstance(v, bool) else str(v).lower() in ('1', 'true', 'yes', 'y')
    parser.add_argument('--shuffle', type=_str2bool, default=False,
                        help='Whether the training DataLoader shuffles samples (default: False)')
    parser.add_argument(
        "--gen_weights_path",
        type=str,
        default=None,
        help="Path to VQGAN generation weights, required for stage 1"
    )
    parser.add_argument(
        '--loss_norm_mode',
        type=str,
        default='separate',
        choices=['separate', 'total'],
        help=(
            "Loss normalization mode: "
            "'separate' = text/image loss each divide by its own valid token count; "
            "'total' = text/image loss each divide by total valid token count."
        )
    )
    parser.add_argument(
        '--image_loss_weight',
        type=float,
        default=1.0,
        help='Weight multiplier for image loss after normalization'
    )
    parser.add_argument(
        '--use_json_num_tokens',
        action='store_true',
        help='If set, use num_tokens from json instead of random sampling'
    )
    parser.add_argument('--min_pixels', type=int, default=784, help='Minimum image pixels')
    parser.add_argument('--max_pixels', type=int, default=50176, help='Maximum image pixels')
    parser.add_argument(
        '--save_steps',
        type=int,
        default=1000,
        help='Save checkpoint every N optimizer update steps'
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory for resuming training. "
            "Use 'latest' to automatically resume from latest checkpoint in output_dir."
    )
    args = parser.parse_args()

    if args.stage == 1 and not args.gen_weights_path:
        raise ValueError("stage 1 requires --gen_weights_path")
    # Set paths
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    # Start training
    train(args)     
