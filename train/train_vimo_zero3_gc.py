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
from accelerate.utils import DistributedType
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
import bisect
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
from datetime import datetime
from contextlib import nullcontext
from collections import defaultdict

import deepspeed

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vimo.processing_vimo import ViMoProcessor
from vimo.modeling_vimo_zero3_gc import ViMoModel, Qwen3VLTextAttention
from vimo.configuration_vimo import ViMoConfig
import transformers
from vimo.rope2d import get_rope_index_3

from train.flash_attn_varlen import qwen3vl_forward
Qwen3VLTextAttention.forward = (qwen3vl_forward)
from vimo.modeling_vimo_zero3_gc import TSIMTokExtraCfg

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
    # pixel_values: torch.Tensor
    # num_image_tokens: torch.IntTensor
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
        self.total_loss_unscaled = torch.tensor([0.0], device=device)
        self.total_samples = torch.tensor([0.0], device=device)

        self.text_right = torch.tensor([0.0], device=device)
        self.text_total = torch.tensor([0.0], device=device)
        self.text_loss = torch.tensor([0.0], device=device)
        self.text_loss_unscaled = torch.tensor([0.0], device=device)

        self.image_right = torch.tensor([0.0], device=device)
        self.image_total = torch.tensor([0.0], device=device)
        self.image_loss = torch.tensor([0.0], device=device)
        self.image_loss_unscaled = torch.tensor([0.0], device=device)

        self.world_size = dist.get_world_size()

    def __call__(
        self,
        image_logits,
        image_labels,
        text_logits,
        text_labels,
        loss,
        image_loss,
        text_loss,
        sample_count,
        unscaled_loss,
        unscaled_image_loss,
        unscaled_text_loss,
    ):
        return self.update(
            image_logits,
            image_labels,
            text_logits,
            text_labels,
            loss,
            image_loss,
            text_loss,
            sample_count,
            unscaled_loss,
            unscaled_image_loss,
            unscaled_text_loss,
        )

    def update(
        self,
        image_logits,
        image_labels,
        text_logits,
        text_labels,
        loss,
        image_loss,
        text_loss,
        sample_count,
        unscaled_loss,
        unscaled_image_loss,
        unscaled_text_loss,
    ):
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
            self.total_loss_unscaled += unscaled_loss.item()
            self.image_loss_unscaled += unscaled_image_loss.item()
            self.text_loss_unscaled += unscaled_text_loss.item()
            self.total_samples += float(sample_count)

    def get_metric(self, reset=True, sync=True):
        if sync:
            dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.total_loss_unscaled, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.total_samples, op=torch.distributed.ReduceOp.SUM)

            dist.all_reduce(self.image_right, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.image_total, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.image_loss, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.image_loss_unscaled, op=torch.distributed.ReduceOp.SUM)

            dist.all_reduce(self.text_right, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.text_total, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.text_loss, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.text_loss_unscaled, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total.clamp_min(1)).item()
        image_acc = (self.image_right / self.image_total.clamp_min(1)).item()
        text_acc = (self.text_right / self.text_total.clamp_min(1)).item()

        metric_world = self.world_size if sync else 1
        loss = self.total_loss.item() / (metric_world * max(self.n_step, 1))
        unscaled_loss = self.total_loss_unscaled.item() / (metric_world * max(self.n_step, 1))
        image_loss = self.image_loss.item() / (metric_world * max(self.n_step, 1))
        image_loss_unscaled = self.image_loss_unscaled.item() / (metric_world * max(self.n_step, 1))
        text_loss = self.text_loss.item() / (metric_world * max(self.n_step, 1))
        text_loss_unscaled = self.text_loss_unscaled.item() / (metric_world * max(self.n_step, 1))
        total_samples = int(self.total_samples.item())

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
            self.total_loss_unscaled.fill_(0)
            self.total_samples.fill_(0)

            self.image_right.fill_(0)
            self.image_total.fill_(0)
            self.image_loss.fill_(0)
            self.image_loss_unscaled.fill_(0)

            self.text_right.fill_(0)
            self.text_total.fill_(0)
            self.text_loss.fill_(0)
            self.text_loss_unscaled.fill_(0)

        return (
            acc,
            loss,
            image_acc,
            image_loss,
            text_acc,
            text_loss,
            total_samples,
            unscaled_loss,
            image_loss_unscaled,
            text_loss_unscaled,
        )

    
def _parse_optional_bool(value, field_name: str):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean string for {field_name}: {value}")
    raise ValueError(f"Unsupported boolean type for {field_name}: {type(value)}")


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
            "packing_shuffle": _parse_optional_bool(task_info.get("shuffle", None), f"{task_name}.shuffle"),
            "pack_total_length": task_info.get("pack_total_length", None),
            "pack_max_batch_size": task_info.get("pack_max_batch_size", None),
            "pack_total_length_threshold": task_info.get("pack_total_length_threshold", None),
            "pack_total_length_threshold_ratio": task_info.get("pack_total_length_threshold_ratio", None),
            "datasets": []
        }

        for ds in datasets:
            output_path = ds["output_path"]
            ratio = float(ds.get("ratio", 1.0))
            cur["datasets"].append({
                "name": ds["name"],
                "path": output_path,   # training uses output_path only
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
        self.len_path = path + ".token_length.npy"
        self._token_lengths = None

        if os.path.exists(self.idx_path):
            self.offsets = np.load(self.idx_path, mmap_mode="r")
        else:
            if not build_index:
                raise FileNotFoundError(f"Index not found: {self.idx_path}")
            self.offsets = self._build_offsets()
            np.save(self.idx_path, self.offsets)

        if os.path.exists(self.len_path):
            self._token_lengths = np.load(self.len_path, mmap_mode="r")

        self._build_lock_path = self.len_path + ".lock"

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

    def _build_token_lengths(self):
        logger.info(f"Building token lengths for {self.path}")
        token_lengths = np.empty(len(self.offsets), dtype=np.int32)
        with open(self.path, "rb") as f:
            for i, line in enumerate(tqdm(f, total=len(self.offsets), desc=f"build token_length {os.path.basename(self.path)}", leave=False)):
                sample = json.loads(line.decode("utf-8"))
                total_len = sample.get("token_length", sample.get("total_length", sample.get("length", None)))
                if total_len is None:
                    raise KeyError(
                        f"Sample at idx={i} in {self.path} is missing 'token_length'/'total_length'/'length' for packing"
                    )
                token_lengths[i] = int(total_len)

        rank = int(os.environ.get("RANK", "0"))
        tmp_path = self.len_path + f".tmp.rank{rank}.npy"
        np.save(tmp_path, token_lengths)
        if not os.path.exists(tmp_path):
            raise FileNotFoundError(f"Temporary token length file not found after np.save: {tmp_path}")
        os.replace(tmp_path, self.len_path)

        logger.info(f"Built {len(token_lengths)} token lengths for {self.path}")
        self._token_lengths = np.load(self.len_path, mmap_mode="r")

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
    
    def get_packing_total_length(self, idx):
        if self._token_lengths is None:
            self.ensure_token_lengths_ready()

        return int(self._token_lengths[idx])

    def ensure_token_lengths_ready(self):
        if self._token_lengths is not None:
            return

        if os.path.exists(self.len_path):
            self._token_lengths = np.load(self.len_path, mmap_mode="r")
            return

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if rank == 0:
            self._build_token_lengths()
        else:
            logger.info(f"Rank {rank} waiting for rank0 to prepare token lengths: {self.len_path}")

        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

        # This function is called before accelerator.prepare(); in some launches
        # torch.distributed is not initialized yet, so ranks cannot barrier.
        # Non-zero ranks should poll for rank0 output instead of failing immediately.
        if rank != 0 and not os.path.exists(self.len_path):
            wait_timeout_sec = float(os.environ.get("TOKEN_LENGTH_WAIT_TIMEOUT_SEC", "900"))
            poll_interval_sec = float(os.environ.get("TOKEN_LENGTH_WAIT_POLL_SEC", "0.2"))
            start_ts = time.time()
            logger.info(
                f"Rank {rank} polling token length file for up to {wait_timeout_sec:.1f}s: {self.len_path}"
            )
            while (time.time() - start_ts) < wait_timeout_sec and not os.path.exists(self.len_path):
                time.sleep(poll_interval_sec)

        if not os.path.exists(self.len_path):
            raise FileNotFoundError(
                f"Token length file not found after rank0 build/wait: {self.len_path}"
            )

        self._token_lengths = np.load(self.len_path, mmap_mode="r")

    
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

# Helper function to ensure token lengths are ready for any dataset structure
def ensure_dataset_token_lengths_ready(ds):
    if isinstance(ds, JsonlOffsetDataset):
        ds.ensure_token_lengths_ready()
        return

    if isinstance(ds, RatioDataset):
        ensure_dataset_token_lengths_ready(ds.base_dataset)
        return

    if isinstance(ds, ConcatDataset):
        for sub_ds in ds.datasets:
            ensure_dataset_token_lengths_ready(sub_ds)
        return
import math
import random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset

class RatioDataset(Dataset):
    """
    Apply ratio-based sampling/repetition over a base dataset, with synchronized
    refresh across persistent worker processes.
    """
    def __init__(self, base_dataset: Dataset, ratio: float, seed: int = 42, name: str = ""):
        self.base_dataset = base_dataset
        self.ratio = float(ratio)
        self.seed = seed
        self.name = name

        # Use a multiprocessing shared-memory value to synchronize the epoch,
        # so changes made in the main process are visible to all persistent workers.
        self.shared_epoch = mp.Value('i', 0)

        # Worker-local state
        self._local_epoch = -1
        self.indices = np.array([], dtype=np.int64)

        # The main process needs the total length when building the DataLoader, so precompute it.
        self._len = self._calc_len()

    def set_epoch(self, epoch: int):
        # The main process only updates the shared-memory value here.
        self.shared_epoch.value = epoch
        
    def materialize_indices(self, epoch: Optional[int] = None):
        current_epoch = self.shared_epoch.value if epoch is None else int(epoch)
        if current_epoch != self._local_epoch:
            self.indices = self._build_indices(current_epoch)
            self._local_epoch = current_epoch
        return self.indices

    def map_index(self, idx: int, epoch: Optional[int] = None) -> int:
        indices = self.materialize_indices(epoch)
        return int(indices[idx])
    
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

        # Preallocate a numpy array to avoid the memory blow-up of a Python list.
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
        mapped_idx = self.map_index(idx)
        return self.base_dataset[mapped_idx]

OUTPUT_PLACEHOLDER_PATTERN = re.compile(r"<output_image_\d+>")


def sanitize_sample_inplace(x: dict):
    """
    Sanitize a single sample in place:
    1. Remove all <output_image_x> placeholders from messages
    2. Clear output_images
    3. Trim num_tokens (drop the corresponding number of trailing tokens)
    """
    output_images = x.get("output_images", [])
    n_out = len(output_images)

    for msg in x.get("messages", []):
        if "content" in msg and isinstance(msg["content"], str):
            msg["content"] = OUTPUT_PLACEHOLDER_PATTERN.sub("", msg["content"])

    x["output_images"] = []

    if "num_tokens" in x and isinstance(x["num_tokens"], list):
        if n_out > 0:
            if len(x["num_tokens"]) >= n_out:
                x["num_tokens"] = x["num_tokens"][:-n_out]
            else:
                x["num_tokens"] = []


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

        # Number of visual tokens occupied by each image
        per_image_lens = []
        for i in range(grid_thw_cpu.shape[0]):
            t, h, w = grid_thw_cpu[i].tolist()
            per_image_lens.append(int(t * h * w))

        # Prefix sums to locate each image's start/end position in pixel_values
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
        Right truncation: keep the first max_len tokens from the left.
        Rules:
        - atomic segments (image blocks, prefix, suffix) cannot be split; keep the whole
          segment or drop it entirely
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

            # Current segment does not fit
            if seg.get("atomic", False):
                break

            # Only text segments allow partial truncation
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
                    if getattr(self.args, "drop_output_images", False):
                        sanitize_sample_inplace(x)
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

            # Process input images
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

            # Process output images
            if len(all_output_images) > 0:
                pixel_values, pixel_values_grid_thw, err3 = self.process_image(
                    all_output_images, to_und_token=False
                )
                if err3 is not None:
                    return self._make_skip_batch("process output_image failed: " + " | ".join(err3))
            else:
                pixel_values, pixel_values_grid_thw = None, None

            # ----------------------------------------------------------
            # Generate num_tokens per sample first
            # ----------------------------------------------------------
            sampler_lengths = [get_sample_sampler_length(x, self.args.max_seq_len) for x in batch]
            raw_sample_lengths = [
                int(x.get("token_length", x.get("total_length", x.get("length", -1))))
                for x in batch
            ]
            batch_num_tokens_raw = []
            for x in batch:
                if getattr(self.args, "use_json_num_tokens", False):
                    cur_tokens = x.get("num_tokens", [])

                    n_img = len(x.get("input_images", [])) + len(x.get("output_images", []))
                    # Has images but num_tokens is empty: raise explicitly to fail fast
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

            # Global image-cropping indices
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
                # Build segments first, instead of directly concatenating into final input_ids
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
                            # Use the und grid to compute the number of und tokens
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
                                        f"sample {sample_idx} num_tokens insufficient, need more for {image_key}, current={sample_num_tokens}"
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
                                        f"sample {sample_idx} num_tokens insufficient, need more for {image_key}, current={sample_num_tokens}"
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
                # Right truncation: keep the first max_seq_len tokens from the left
                # ------------------------------------------------------
                kept_segments = self._truncate_segments_right(segments, self.args.max_seq_len)

                # ------------------------------------------------------
                # Rebuild the sample from the truncation result
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

                # Optional: deduplicate while preserving the original order
                kept_input_local_indices = sorted(set(kept_input_local_indices))
                kept_output_local_indices = sorted(set(kept_output_local_indices))

                # Build sft_format (for logging only)
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
                # position_ids only sees the und grid of the input images kept after truncation
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
            # slice the image tensors / grids in sync
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
                "sampler_lengths": torch.tensor(sampler_lengths, dtype=torch.long),
                "sampler_length_sum": torch.tensor(sum(sampler_lengths), dtype=torch.long),
                "sampler_length_max": torch.tensor(max(sampler_lengths) if len(sampler_lengths) > 0 else 0, dtype=torch.long),
                "raw_sample_lengths": torch.tensor(raw_sample_lengths, dtype=torch.long),
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
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    target_local_pack_bs = None
    target_local_pack_bs_tol = 0
    if getattr(args, "target_global_batch_size", None) is not None:
        target_local_pack_bs = max(1, int(round(float(args.target_global_batch_size) / float(world_size))))
        target_local_pack_bs_tol = int(math.ceil(float(args.target_global_batch_size_tolerance) / float(world_size)))
        target_local_pack_bs_tol = max(0, target_local_pack_bs_tol)
        logger.info(
            "Enable target global batch-size control for packing: "
            f"global_target={args.target_global_batch_size}, global_tol=±{args.target_global_batch_size_tolerance}, "
            f"world_size={world_size}, local_target={target_local_pack_bs}, local_tol=±{target_local_pack_bs_tol}"
        )

    for task_cfg in task_cfgs:
        task_name = task_cfg["task_name"]
        image_loss_weight = task_cfg["image_loss_weight"]
        task_pack_total_length = task_cfg.get("pack_total_length", None)
        task_pack_max_batch_size = task_cfg.get("pack_max_batch_size", None)
        task_pack_total_length_threshold = task_cfg.get("pack_total_length_threshold", None)
        task_pack_total_length_threshold_ratio = task_cfg.get("pack_total_length_threshold_ratio", None)
        task_packing_shuffle = task_cfg.get("packing_shuffle", None)

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

        use_packing = getattr(args, "enable_packing", False)
        if use_packing:
            pack_total_length = task_pack_total_length
            if pack_total_length is None:
                pack_total_length = getattr(args, "pack_total_length", None)
            if pack_total_length is None:
                pack_total_length = args.max_seq_len * args.train_bsz_per_gpu
            pack_total_length = int(pack_total_length)

            pack_total_length_threshold = task_pack_total_length_threshold
            if pack_total_length_threshold is None:
                pack_total_length_threshold = getattr(args, "pack_total_length_threshold", None)
            if pack_total_length_threshold is None:
                threshold_ratio = task_pack_total_length_threshold_ratio
                if threshold_ratio is None:
                    threshold_ratio = getattr(args, "pack_total_length_threshold_ratio", 0.9)
                pack_total_length_threshold = int(pack_total_length * threshold_ratio)

            pack_max_batch_size = task_pack_max_batch_size
            if pack_max_batch_size is None:
                pack_max_batch_size = args.pack_max_batch_size
            if pack_max_batch_size is not None:
                pack_max_batch_size = int(pack_max_batch_size)
                if pack_max_batch_size <= 0:
                    pack_max_batch_size = None
            if task_packing_shuffle is None:
                task_packing_shuffle = bool(getattr(args, "packing_shuffle", True))
            logger.info(f"[Task={task_name}] preparing token_length sidecar files on rank0 ...")
            ensure_dataset_token_lengths_ready(merged_ds)
            logger.info(f"[Task={task_name}] token_length sidecar files ready.")
            batch_sampler = ISFPackingBatchSampler(
                dataset=merged_ds,
                max_total_length=pack_total_length,
                max_batch_size=pack_max_batch_size,
                total_length_threshold=pack_total_length_threshold,
                sample_max_length=args.max_seq_len,
                target_batch_size=target_local_pack_bs,
                target_batch_size_tolerance=target_local_pack_bs_tol,
                shuffle=task_packing_shuffle,
                seed=args.seed,
                drop_last=True,
                max_rounds=args.max_rounds,
            )

            loader = DataLoader(
                merged_ds,
                batch_sampler=batch_sampler,
                collate_fn=collator,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )
        else:
            loader = DataLoader(
                merged_ds,
                batch_size=args.train_bsz_per_gpu,
                shuffle=True,
                drop_last=True,
                collate_fn=collator,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )

        task_loaders[task_name] = loader

        logger.info(
            f"[Task={task_name}] datasets={len(sub_datasets)}, "
            f"samples={len(merged_ds)}, "
            f"image_loss_weight={image_loss_weight}, "
            f"packing={getattr(args, 'enable_packing', False)}, "
            f"packing_shuffle={task_packing_shuffle if task_packing_shuffle is not None else bool(getattr(args, 'packing_shuffle', True))}, "
            f"target_local_pack_bs={target_local_pack_bs}, "
            f"target_local_pack_bs_tol=±{target_local_pack_bs_tol}, "
            f"pack_total_length={task_pack_total_length if task_pack_total_length is not None else getattr(args, 'pack_total_length', None)}, "
            f"pack_max_batch_size={task_pack_max_batch_size if task_pack_max_batch_size is not None else args.pack_max_batch_size}"
        )

    return task_loaders

def set_epoch_for_dataset(ds, epoch: int):
    if isinstance(ds, RatioDataset):
        ds.set_epoch(epoch)
    elif isinstance(ds, ConcatDataset):
        for sub_ds in ds.datasets:
            set_epoch_for_dataset(sub_ds, epoch)


def _get_sample_total_length(
    ds,
    idx: int,
    epoch: Optional[int] = None,
    max_length: Optional[int] = None,
):
    if isinstance(ds, JsonlOffsetDataset):
        total_len = ds.get_packing_total_length(idx)

    elif isinstance(ds, RatioDataset):
        mapped_idx = ds.map_index(idx, epoch)
        total_len = _get_sample_total_length(
            ds.base_dataset,
            mapped_idx,
            epoch,
            max_length=None,
        )

    elif isinstance(ds, ConcatDataset):
        if idx < 0:
            if -idx > len(ds):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(ds) + idx
        dataset_idx = bisect.bisect_right(ds.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - ds.cumulative_sizes[dataset_idx - 1]
        total_len = _get_sample_total_length(
            ds.datasets[dataset_idx],
            sample_idx,
            epoch,
            max_length=None,
        )

    else:
        sample = ds[idx]
        total_len = sample.get(
            "token_length",
            sample.get("total_length", sample.get("length", None))
        )
        if total_len is None:
            raise KeyError(
                f"Sample at idx={idx} is missing 'total_length' (or fallback 'length') for packing"
            )
        total_len = int(total_len)

    total_len = int(total_len)
    if max_length is not None:
        total_len = min(total_len, int(max_length))

    return total_len


class ISFPackingBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset,
        max_total_length: int,
        max_batch_size: int,
        total_length_threshold: Optional[int] = None,
        sample_max_length: Optional[int] = None,
        target_batch_size: Optional[int] = None,
        target_batch_size_tolerance: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        max_rounds: int = 32,
    ):
        self.dataset = dataset
        self.max_total_length = int(max_total_length)
        self.max_batch_size = int(max_batch_size) if max_batch_size is not None else None
        self.total_length_threshold = (
            int(total_length_threshold)
            if total_length_threshold is not None
            else int(self.max_total_length * 0.9)
        )
        self.sample_max_length = int(sample_max_length) if sample_max_length is not None else None
        self.target_batch_size = (
            int(target_batch_size)
            if target_batch_size is not None and int(target_batch_size) > 0
            else None
        )
        self.target_batch_size_tolerance = max(0, int(target_batch_size_tolerance))
        if self.target_batch_size is not None:
            self.target_batch_size_min = max(1, self.target_batch_size - self.target_batch_size_tolerance)
            self.target_batch_size_max = self.target_batch_size + self.target_batch_size_tolerance
        else:
            self.target_batch_size_min = None
            self.target_batch_size_max = None

        if self.max_batch_size is None:
            self._effective_max_batch_size = self.target_batch_size_max
        elif self.target_batch_size_max is None:
            self._effective_max_batch_size = self.max_batch_size
        else:
            self._effective_max_batch_size = min(self.max_batch_size, self.target_batch_size_max)

        if (
            self.target_batch_size_min is not None
            and self._effective_max_batch_size is not None
            and self._effective_max_batch_size < self.target_batch_size_min
        ):
            logger.warning(
                "Packing target batch-size constraints conflict: "
                f"target_range=[{self.target_batch_size_min},{self.target_batch_size_max}], "
                f"effective_max_batch_size={self._effective_max_batch_size}. "
                "Lower bound may not be reachable."
            )
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.max_rounds = int(max_rounds)
        self.epoch = 0

        self._cached_batches = None
        self._cached_epoch = None
        self._cached_lengths = None
        self._cached_length_epoch = None

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self._cached_batches = None
            self._cached_epoch = None
            self._cached_lengths = None
            self._cached_length_epoch = None
        else:
            self.epoch = epoch

    def _get_lengths(self):
        if self._cached_lengths is not None and self._cached_length_epoch == self.epoch:
            return self._cached_lengths

        #     self.dataset.set_epoch(self.epoch)
        set_epoch_for_dataset(self.dataset, self.epoch)

        lengths = np.empty(len(self.dataset), dtype=np.int32)
        print(f"dataset length: {len(self.dataset)}")
        for idx in tqdm(range(len(self.dataset)), total=len(self.dataset), desc=f"packing lengths epoch={self.epoch}", leave=False):
            # lengths[idx] = _get_sample_total_length(self.dataset, idx, self.epoch)
            lengths[idx] = _get_sample_total_length(
                self.dataset,
                idx,
                self.epoch,
                max_length=self.sample_max_length,
            )

        self._cached_lengths = lengths
        self._cached_length_epoch = self.epoch
        return lengths

    def _sampling(self, items):
        groups = []
        group = []
        total_len_sum = 0

        for item in items:
            idx, total_len = item

            # a single sample exceeding pack_total_length becomes its own batch
            if total_len > self.max_total_length:
                if group:
                    groups.append(group)
                    group = []
                    total_len_sum = 0
                groups.append([item])
                continue

            exceed_total = len(group) > 0 and total_len_sum + total_len > self.max_total_length
            exceed_bs = (self._effective_max_batch_size is not None) and (len(group) >= self._effective_max_batch_size)

            if exceed_total or exceed_bs:
                groups.append(group)
                group = [item]
                total_len_sum = total_len
            else:
                group.append(item)
                total_len_sum += total_len

        if group:
            groups.append(group)
        return groups

    def _filtering(self, groups):
        filter_groups = []
        need_resampling_data = []

        for group in groups:
            group_total_len = sum(x[1] for x in group)
            group_bs = len(group)

            total_ok = group_total_len >= self.total_length_threshold
            single_ok = group_bs == 1
            full_bs_ok = (self._effective_max_batch_size is not None) and (group_bs >= self._effective_max_batch_size)

            if self.target_batch_size_min is not None:
                # the target_bs lower bound is a soft constraint: if unmet, recirculate for more
                # resampling rounds, and only release once the final round still falls short.
                forced_single = single_ok and group[0][1] > self.max_total_length
                if group_bs < self.target_batch_size_min and not forced_single:
                    need_resampling_data.extend(group)
                    continue

            if total_ok or single_ok or full_bs_ok:
                filter_groups.append(group)
            else:
                need_resampling_data.extend(group)

        return filter_groups, need_resampling_data

    def _build_batches(self):
        if self._cached_batches is not None and self._cached_epoch == self.epoch:
            return self._cached_batches

        set_epoch_for_dataset(self.dataset, self.epoch)

        lengths = self._get_lengths()
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(indices)

        items = [(int(idx), int(lengths[idx])) for idx in indices]

        packed_groups = []
        remain = items
        max_rounds = self.max_rounds

        for _ in tqdm(range(max_rounds), total=max_rounds, desc=f"packing rounds epoch={self.epoch}", leave=False):
            if not remain:
                break
            sampled = self._sampling(remain)
            accepted, remain = self._filtering(sampled)
            packed_groups.extend(accepted)
            if not remain:
                break

        if remain:
            packed_groups.extend(self._sampling(remain))

        batches = [[x[0] for x in group] for group in packed_groups if len(group) > 0]
        if self.drop_last:
            batches = [b for b in batches if len(b) > 0]

        self._cached_batches = batches
        self._cached_epoch = self.epoch
        return batches

    def __iter__(self):
        for batch in self._build_batches():
            if batch:
                yield batch

    def __len__(self):
        return len(self._build_batches())

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

    # if shift_labels is not provided, build it from labels
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
    For a causal LM, the labels that actually contribute to the loss are the shifted ones,
    i.e. text_labels[..., 1:]
    """
    shift_labels = text_labels[..., 1:]
    return (shift_labels != ignore_index).sum()


def count_image_valid_tokens(output_image_tokens: Optional[torch.Tensor]) -> torch.Tensor:
    """
    output_image_tokens: flattened image-token labels, or any tensor that can be view(-1)-ed
    -100 is the ignore index by default
    """
    if output_image_tokens is None:
        return torch.tensor(0, device='cpu')

    if not isinstance(output_image_tokens, torch.Tensor):
        output_image_tokens = torch.as_tensor(output_image_tokens)

    return (output_image_tokens.view(-1) != -100).sum()


def get_distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_batch_sample_count(batch: Dict[str, Any]) -> int:
    attention_mask = batch.get("attention_mask", None)
    if isinstance(attention_mask, torch.Tensor):
        cu_seqlens = attention_mask.reshape(-1)
        if cu_seqlens.numel() >= 2:
            return int(cu_seqlens.numel() - 1)

    num_input_image = batch.get("num_input_image", None)
    if isinstance(num_input_image, torch.Tensor):
        if num_input_image.ndim == 0:
            return 1
        return int(num_input_image.shape[0])
    return 0


def get_global_sample_count_tensor(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    local_sample_count = torch.tensor(float(get_batch_sample_count(batch)), device=device, dtype=torch.float32)
    global_sample_count = local_sample_count.clone()
    world_size = get_distributed_world_size()
    if world_size > 1:
        dist.all_reduce(global_sample_count, op=dist.ReduceOp.SUM)
    return global_sample_count


def get_packed_cu_seqlens(attention_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError(f"attention_mask must be torch.Tensor, got {type(attention_mask)}")

    cu_seqlens = attention_mask.to(device=device, dtype=torch.long).reshape(-1)
    if cu_seqlens.numel() < 2:
        raise ValueError(f"Invalid packed attention_mask: expect len>=2, got {cu_seqlens.numel()}")
    return cu_seqlens


def compute_global_sample_mean_losses(
    model,
    text_logits: torch.Tensor,
    text_label: torch.Tensor,
    output_image_logits: torch.Tensor,
    output_image_tokens: torch.Tensor,
    output_token_counts,
    batch: Dict[str, Any],
    image_loss_weight: float,
    loss_norm_mode: str,
):
    """
    Under packing, normalize the loss per "sample":
    1) normalize within each sample first (separate/total)
    2) average across all ranks by the global sample count (rather than averaging per rank then averaging those)
    """
    device = text_logits.device
    # Keep anchors in graph to avoid rank-wise graph divergence on corner batches.
    text_zero_anchor = text_logits.sum() * 0.0
    image_zero_anchor = output_image_logits.sum() * 0.0
    cu_seqlens = get_packed_cu_seqlens(batch["attention_mask"], device=device)
    sample_count = int(cu_seqlens.numel() - 1)

    image_num = batch.get("image_num", None)
    if isinstance(image_num, torch.Tensor):
        image_num = image_num.to(device=device, dtype=torch.long).reshape(-1)
    elif image_num is None:
        image_num = torch.zeros(sample_count, device=device, dtype=torch.long)
    else:
        image_num = torch.as_tensor(image_num, device=device, dtype=torch.long).reshape(-1)

    if image_num.numel() != sample_count:
        raise ValueError(
            f"image_num length ({image_num.numel()}) != sample_count ({sample_count})"
        )

    if not isinstance(output_image_tokens, torch.Tensor):
        output_image_tokens = torch.as_tensor(output_image_tokens, device=device, dtype=torch.long)
    else:
        output_image_tokens = output_image_tokens.to(device=device)

    output_token_counts = torch.as_tensor(output_token_counts, device=device, dtype=torch.long)
    total_output_images = int(image_num.sum().item())
    if output_token_counts.numel() != total_output_images:
        raise ValueError(
            f"output_token_counts length ({output_token_counts.numel()}) != total output images ({total_output_images})"
        )

    sample_text_losses = []
    sample_image_losses = []
    zero = text_logits.new_zeros(())

    image_ptr = 0
    image_token_ptr = 0
    for i in range(sample_count):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())

        if end > start:
            sample_text_logits = text_logits[:, start:end, :]
            sample_text_label = text_label[:, start:end]
            sample_text_loss_sum = model.language_model.loss_function(
                logits=sample_text_logits,
                labels=sample_text_label,
                vocab_size=151936,
            )
            sample_text_token_count = count_text_valid_tokens(sample_text_label).to(
                device=device, dtype=torch.float32
            )
        else:
            sample_text_loss_sum = text_zero_anchor
            sample_text_token_count = zero

        n_image = int(image_num[i].item())
        if n_image > 0:
            sample_image_token_counts = output_token_counts[image_ptr:image_ptr + n_image]
            sample_image_token_len = int(sample_image_token_counts.sum().item())

            sample_output_image_tokens = output_image_tokens[
                image_token_ptr:image_token_ptr + sample_image_token_len
            ]
            sample_output_image_logits = output_image_logits[
                image_token_ptr:image_token_ptr + sample_image_token_len
            ]

            sample_image_loss_sum = model.language_model.loss_function(
                logits=sample_output_image_logits,
                labels=None,
                shift_labels=sample_output_image_tokens,
                vocab_size=IMAGE_VOCAB_SIZE_WITH_EOS,
                reduction="sum",
            )
            sample_image_token_count = (sample_output_image_tokens != -100).sum().to(
                device=device, dtype=torch.float32
            )

            image_ptr += n_image
            image_token_ptr += sample_image_token_len
        else:
            sample_image_loss_sum = image_zero_anchor
            sample_image_token_count = zero

        if loss_norm_mode == "separate":
            sample_text_loss = sample_text_loss_sum / sample_text_token_count.clamp_min(1.0)
            sample_image_loss = sample_image_loss_sum / sample_image_token_count.clamp_min(1.0)
        elif loss_norm_mode == "total":
            sample_total_token_count = (sample_text_token_count + sample_image_token_count).clamp_min(1.0)
            sample_text_loss = sample_text_loss_sum / sample_total_token_count
            sample_image_loss = sample_image_loss_sum / sample_total_token_count
        else:
            raise ValueError(f"Unsupported loss_norm_mode: {loss_norm_mode}")

        sample_text_losses.append(sample_text_loss)
        sample_image_losses.append(sample_image_loss)

    if image_ptr != total_output_images:
        raise ValueError(f"Image pointer mismatch: {image_ptr} vs {total_output_images}")
    if image_token_ptr != int(output_image_tokens.numel()):
        raise ValueError(
            f"Image token pointer mismatch: {image_token_ptr} vs {int(output_image_tokens.numel())}"
        )

    text_loss_sum_by_sample = torch.stack(sample_text_losses).sum() if sample_text_losses else zero
    image_loss_sum_by_sample = torch.stack(sample_image_losses).sum() if sample_image_losses else zero
    total_loss_sum_by_sample = text_loss_sum_by_sample + float(image_loss_weight) * image_loss_sum_by_sample

    local_sample_count = torch.tensor(float(sample_count), device=device, dtype=torch.float32)
    global_sample_count = local_sample_count.clone()
    world_size = get_distributed_world_size()
    if world_size > 1:
        dist.all_reduce(global_sample_count, op=dist.ReduceOp.SUM)

    # DDP/DeepSpeed averages gradients across ranks; multiply by world_size here to make it
    # equivalent to a global per-sample average
    loss_scale = (
        total_loss_sum_by_sample.new_tensor(float(world_size))
        / global_sample_count.clamp_min(1.0).to(dtype=total_loss_sum_by_sample.dtype)
    )

    total_loss = total_loss_sum_by_sample * loss_scale
    image_loss = image_loss_sum_by_sample * loss_scale
    text_loss = text_loss_sum_by_sample * loss_scale
    return total_loss, image_loss, text_loss, global_sample_count


def get_sample_sampler_length(sample: Dict[str, Any], max_length: Optional[int] = None) -> int:
    total_len = sample.get("token_length", sample.get("total_length", sample.get("length", None)))
    if total_len is None:
        # Fall back to a -1 placeholder when the length field is missing (no error). This value is
        # only used for the enable_batch_token_stats_logging display and never feeds loss / sampling /
        # any training computation; packing reads the real sample length via the separate path
        # JsonlOffsetDataset.get_packing_total_length + the .token_length.npy sidecar.
        return -1
    total_len = int(total_len)
    if max_length is not None:
        total_len = min(total_len, int(max_length))
    return total_len

def get_batch_token_stats(batch: Dict[str, Any]) -> Dict[str, int]:
    input_ids = batch.get("input_ids", None)
    attention_mask = batch.get("attention_mask", None)
    text_label = batch.get("text_label", None)
    output_image_tokens = batch.get("output_image_tokens", None)
    sampler_lengths = batch.get("sampler_lengths", None)
    sampler_length_sum = batch.get("sampler_length_sum", None)
    sampler_length_max = batch.get("sampler_length_max", None)
    raw_sample_lengths = batch.get("raw_sample_lengths", None)
    sample_count = get_batch_sample_count(batch)

    input_token_count = 0
    #     input_token_count = int(attention_mask.long().sum().item())
    # elif isinstance(input_ids, torch.Tensor):
    #     input_token_count = int(input_ids.numel())
    input_token_count = int(input_ids.numel())

    seq_len = 0
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2:
        seq_len = int(input_ids.shape[1])
    elif isinstance(input_ids, torch.Tensor) and input_ids.ndim == 1:
        seq_len = int(input_ids.shape[0])

    text_token_count = 0
    if isinstance(text_label, torch.Tensor):
        text_token_count = int(count_text_valid_tokens(text_label).item())

    image_token_count = 0
    if isinstance(output_image_tokens, torch.Tensor):
        image_token_count = int(count_image_valid_tokens(output_image_tokens).item())
    else:
        image_num = batch.get("image_num", None)
        num_tokens = batch.get("num_tokens", None)
        if isinstance(image_num, torch.Tensor):
            image_num_list = image_num.detach().cpu().tolist()
        elif image_num is None:
            image_num_list = []
        else:
            image_num_list = list(image_num)

        if isinstance(num_tokens, torch.Tensor):
            num_tokens_list = num_tokens.detach().cpu().tolist()
        elif num_tokens is None:
            num_tokens_list = []
        else:
            num_tokens_list = list(num_tokens)

        ptr = 0
        total_img_tokens = 0
        for n_img in image_num_list:
            n_img = int(n_img)
            if n_img <= 0:
                continue
            take = min(n_img, max(0, len(num_tokens_list) - ptr))
            if take > 0:
                total_img_tokens += sum(int(x) for x in num_tokens_list[ptr:ptr + take])
                ptr += take
        image_token_count = int(total_img_tokens)

    total_supervised_tokens = int(text_token_count + image_token_count)
    if isinstance(sampler_lengths, torch.Tensor):
        sampler_lengths_list = [int(x) for x in sampler_lengths.detach().cpu().tolist()]
    elif sampler_lengths is None:
        sampler_lengths_list = []
    else:
        sampler_lengths_list = [int(x) for x in sampler_lengths]

    if isinstance(raw_sample_lengths, torch.Tensor):
        raw_sample_lengths_list = [int(x) for x in raw_sample_lengths.detach().cpu().tolist()]
    elif raw_sample_lengths is None:
        raw_sample_lengths_list = []
    else:
        raw_sample_lengths_list = [int(x) for x in raw_sample_lengths]

    if isinstance(sampler_length_sum, torch.Tensor):
        sampler_length_sum_value = int(sampler_length_sum.item())
    else:
        sampler_length_sum_value = int(sum(sampler_lengths_list)) if sampler_length_sum is None else int(sampler_length_sum)

    if isinstance(sampler_length_max, torch.Tensor):
        sampler_length_max_value = int(sampler_length_max.item())
    else:
        sampler_length_max_value = int(max(sampler_lengths_list)) if sampler_length_max is None and sampler_lengths_list else int(sampler_length_max or 0)
    return {
        "samples": int(sample_count),
        "seq_len": int(seq_len),
        "input_tokens": int(input_token_count),
        "text_tokens": int(text_token_count),
        "image_tokens": int(image_token_count),
        "supervised_tokens": int(total_supervised_tokens),
        "sampler_lengths": sampler_lengths_list,
        "sampler_length_sum": int(sampler_length_sum_value),
        "sampler_length_max": int(sampler_length_max_value),
        "raw_sample_lengths": raw_sample_lengths_list,
    }
def unwrap_if_accelerated(model):
    return model.module if hasattr(model, "module") else model


def resolve_hf_model_dir(model_path: str) -> str:
    """
    Accepts several forms:
    1) a HF model dir directly: xxx/tfmr or xxx/hf_model
    2) a checkpoint root dir: xxx/checkpoint-step-1000
       (the inner tfmr / hf_model is preferred automatically)
    3) an original model dir: returned as-is
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


def _load_torch_state_dict(weight_path: str):
    state_obj = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(state_obj, dict):
        for field_name in ("state_dict", "model", "module"):
            nested = state_obj.get(field_name)
            if isinstance(nested, dict):
                logger.info(f"Extract nested state dict '{field_name}' from {weight_path}")
                return nested
        if all(isinstance(k, str) for k in state_obj.keys()):
            return state_obj
    raise ValueError(f"Unsupported torch checkpoint structure: {weight_path}")


def _load_sharded_state_dict(model_dir: str, shard_files: List[str], loader):
    merged_state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(model_dir, shard_file)
        shard_state = loader(shard_path)
        duplicate_keys = set(merged_state_dict.keys()).intersection(shard_state.keys())
        if duplicate_keys:
            first_key = next(iter(duplicate_keys))
            raise ValueError(
                f"Duplicate key '{first_key}' found while loading shard {shard_path}"
            )
        merged_state_dict.update(shard_state)
    logger.info(
        f"Loaded sharded checkpoint: dir={model_dir}, shards={len(shard_files)}, "
        f"num_tensors={len(merged_state_dict)}"
    )
    return merged_state_dict


def load_hf_state_dict_any_layout(model_path: str):
    model_dir = resolve_hf_model_dir(model_path)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    single_safetensors = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single_safetensors):
        logger.info(f"Loading single safetensors checkpoint: {single_safetensors}")
        return load_file(single_safetensors)

    single_bin = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(single_bin):
        logger.info(f"Loading single bin checkpoint: {single_bin}")
        return _load_torch_state_dict(single_bin)

    safetensors_index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(safetensors_index):
        with open(safetensors_index, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        if not shard_files:
            raise ValueError(f"No shard files found in {safetensors_index}")
        missing_files = [x for x in shard_files if not os.path.exists(os.path.join(model_dir, x))]
        if missing_files:
            raise FileNotFoundError(
                f"Missing shard files in {model_dir}: {missing_files[:3]}"
            )
        return _load_sharded_state_dict(model_dir, shard_files, load_file)

    bin_index = os.path.join(model_dir, "pytorch_model.bin.index.json")
    if os.path.exists(bin_index):
        with open(bin_index, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        if not shard_files:
            raise ValueError(f"No shard files found in {bin_index}")
        missing_files = [x for x in shard_files if not os.path.exists(os.path.join(model_dir, x))]
        if missing_files:
            raise FileNotFoundError(
                f"Missing shard files in {model_dir}: {missing_files[:3]}"
            )
        return _load_sharded_state_dict(model_dir, shard_files, _load_torch_state_dict)

    fallback_safetensors_shards = sorted(
        x for x in os.listdir(model_dir)
        if x.startswith("model-") and x.endswith(".safetensors")
    )
    if fallback_safetensors_shards:
        logger.warning(
            f"No safetensors index found in {model_dir}, fallback to sorted shard loading."
        )
        return _load_sharded_state_dict(model_dir, fallback_safetensors_shards, load_file)

    fallback_bin_shards = sorted(
        x for x in os.listdir(model_dir)
        if (
            (x.startswith("pytorch_model-") and x.endswith(".bin"))
            or (x.startswith("model-") and x.endswith(".bin"))
        )
    )
    if fallback_bin_shards:
        logger.warning(
            f"No bin index found in {model_dir}, fallback to sorted shard loading."
        )
        return _load_sharded_state_dict(model_dir, fallback_bin_shards, _load_torch_state_dict)

    raise FileNotFoundError(
        "Cannot find model weights. Expected one of: model.safetensors, pytorch_model.bin, "
        "model.safetensors.index.json + model-*.safetensors, "
        "pytorch_model.bin.index.json + pytorch_model-*.bin"
    )


def find_latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None

    def _is_complete_checkpoint(ckpt_dir: str) -> bool:
        # 1) training_state must exist (otherwise step/epoch cannot be restored correctly)
        if not os.path.exists(os.path.join(ckpt_dir, "training_state.json")):
            return False

        # 2) the HF model dir must be available (at least for a weights-only resume)
        hf_dir = resolve_hf_model_dir(ckpt_dir)
        if not os.path.isdir(hf_dir):
            return False
        if not os.path.exists(os.path.join(hf_dir, "config.json")):
            return False

        # non-sharded weights
        if (
            os.path.exists(os.path.join(hf_dir, "model.safetensors"))
            or os.path.exists(os.path.join(hf_dir, "pytorch_model.bin"))
        ):
            return True

        # sharded safetensors: require the index + at least one shard file
        index_file = os.path.join(hf_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            has_shard = any(
                x.startswith("model-") and x.endswith(".safetensors")
                for x in os.listdir(hf_dir)
            )
            return has_shard
        # sharded pytorch bin: require the index + at least one shard file (the save format when safe_serialization=False)
        bin_index_file = os.path.join(hf_dir, "pytorch_model.bin.index.json")
        if os.path.exists(bin_index_file):
            has_bin_shard = any(
                x.startswith("pytorch_model-") and x.endswith(".bin")
                for x in os.listdir(hf_dir)
            )
            return has_bin_shard
        return False

    ckpts = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if not os.path.isdir(full):
            continue
        m = re.match(r"checkpoint-step-(\d+)", name)
        if m:
            step = int(m.group(1))
            if _is_complete_checkpoint(full):
                ckpts.append((step, full))
            else:
                logger.warning(f"Skip incomplete checkpoint: {full}")

    if not ckpts:
        return None

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        isinstance(exc, RuntimeError)
        and (
            "out of memory" in msg
            or "cuda error: out of memory" in msg
            or "cuda out of memory" in msg
        )
    )


def get_oom_skip_record_path(output_dir: str, rank: Optional[int] = None) -> str:
    if rank is None:
        return os.path.join(output_dir, "oom_skip_record.json")
    return os.path.join(output_dir, f"oom_skip_record_rank{rank}.json")


def load_oom_skip_record(output_dir: str):
    candidate_paths = [get_oom_skip_record_path(output_dir)]

    try:
        if os.path.isdir(output_dir):
            rank_paths = sorted(
                os.path.join(output_dir, name)
                for name in os.listdir(output_dir)
                if name.startswith("oom_skip_record_rank") and name.endswith(".json")
            )
            candidate_paths.extend(rank_paths)
    except Exception as e:
        logger.warning(f"Failed to enumerate OOM record files under {output_dir}: {e}")

    for path in candidate_paths:
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                logger.warning(f"Loaded OOM skip record from {path}: {data}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load oom skip record from {path}: {e}")

    return None


def save_oom_skip_record(output_dir: str, record: Dict[str, Any], rank: Optional[int] = None):
    os.makedirs(output_dir, exist_ok=True)

    path = get_oom_skip_record_path(output_dir, rank=rank)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)
    logger.warning(f"Saved OOM skip record to {path}: {record}")

    if rank is not None:
        try:
            canonical_path = get_oom_skip_record_path(output_dir)
            canonical_tmp_path = canonical_path + ".tmp"
            with open(canonical_tmp_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            os.replace(canonical_tmp_path, canonical_path)
            logger.warning(f"Saved canonical OOM skip record to {canonical_path}: {record}")
        except Exception as e:
            logger.warning(f"Failed to save canonical OOM skip record: {e}")


def should_skip_due_to_oom_record(record, epoch: int, step_in_epoch: int, task_name: str) -> bool:
    if not record:
        return False

    record_epoch = int(record.get("epoch", -1))
    if record_epoch != epoch:
        return False

    # if record_task and record_task != task_name:
    #     return False

    skip_radius = int(record.get("skip_radius", 0))
    failed_step = int(record.get("step_in_epoch", -10**18))
    return abs(step_in_epoch - failed_step) <= skip_radius

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

    # ZeRO-3: get_state_dict triggers allgather across ALL ranks, must be outside is_main_process
    _state_dict = accelerator.get_state_dict(model)

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
            state_dict=_state_dict,
            safe_serialization=False,
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
    # NOTE:
    # With torch==2.6 + deepspeed, large checkpoint shards (model and optimizer state) can hit a
    # zip-offset anomaly that makes torch.load raise "not a ZIP archive". Force legacy serialization
    # for deepspeed-related state files here to avoid corrupt archives.
    original_torch_save = torch.save

    def _patched_torch_save(obj, f, *args, **kwargs):
        file_path = os.fspath(f) if isinstance(f, (str, os.PathLike)) else ""
        file_name = os.path.basename(file_path)
        if (
            re.search(r"mp_rank_\d+_model_states\.pt$", file_name)
            or file_name.endswith("_optim_states.pt")
        ):
            if kwargs.get("_use_new_zipfile_serialization", True):
                kwargs["_use_new_zipfile_serialization"] = False
                logger.warning(
                    f"Force legacy torch serialization for deepspeed state: {file_path}"
                )
        return original_torch_save(obj, f, *args, **kwargs)

    torch.save = _patched_torch_save
    try:
        accelerator.save_state(save_dir)
    finally:
        torch.save = original_torch_save

    logger.info(f"Checkpoint saved successfully at {save_dir}")

def gather_image_logits(hidden_states, start_indices, token_counts, model):
    device = hidden_states.device
    token_counts = torch.as_tensor(token_counts, device=device)

    # which image each token belongs to
    image_ids = torch.repeat_interleave(
        torch.arange(len(token_counts), device=device),
        token_counts
    )

    # offset of the token within its image
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
def build_batch_plan(
    task_num_batches: Dict[str, int],
    seed: int,
    epoch: int,
    random_task_plan_per_rank: bool = False,
    rank: int = 0,
):
    plan = []
    for task_name, n_batches in task_num_batches.items():
        plan.extend([task_name] * n_batches)

    plan_seed = int(seed) + int(epoch)
    if random_task_plan_per_rank:
        # Keep per-rank plan deterministic across resume, but different across ranks.
        plan_seed += int(rank) * 1_000_003

    rng = random.Random(plan_seed)
    rng.shuffle(plan)
    return plan


# Helper for resuming: build the remaining batch plan and dataloaders skipping already consumed batches
def build_resumed_epoch_plan_and_loaders(
    accelerator,
    task_loaders,
    task_num_batches: Dict[str, int],
    seed: int,
    epoch: int,
    resume_step_in_epoch: int,
    random_task_plan_per_rank: bool = False,
    rank: int = 0,
):
    full_plan = build_batch_plan(
        task_num_batches,
        seed=seed,
        epoch=epoch,
        random_task_plan_per_rank=random_task_plan_per_rank,
        rank=rank,
    )

    if resume_step_in_epoch <= 0:
        return full_plan, task_loaders, {}

    if resume_step_in_epoch > len(full_plan):
        raise ValueError(
            f"resume_step_in_epoch={resume_step_in_epoch} exceeds total batches per epoch={len(full_plan)}"
        )

    consumed_prefix = full_plan[:resume_step_in_epoch]
    skip_counts = defaultdict(int)
    for task_name in consumed_prefix:
        skip_counts[task_name] += 1

    resumed_task_loaders = {}
    for task_name, loader in task_loaders.items():
        num_skip = skip_counts.get(task_name, 0)
        if num_skip > 0:
            resumed_task_loaders[task_name] = accelerator.skip_first_batches(loader, num_skip)
        else:
            resumed_task_loaders[task_name] = loader

    resumed_plan = full_plan[resume_step_in_epoch:]
    return resumed_plan, resumed_task_loaders, dict(skip_counts)
from accelerate.utils import DistributedDataParallelKwargs
def set_epoch_for_batch_sampler(batch_sampler, epoch: int):
    if batch_sampler is None:
        return

    if hasattr(batch_sampler, "set_epoch"):
        try:
            batch_sampler.set_epoch(epoch)
        except TypeError:
            pass

    inner = getattr(batch_sampler, "batch_sampler", None)
    if inner is not None and inner is not batch_sampler:
        set_epoch_for_batch_sampler(inner, epoch)

    inner = getattr(batch_sampler, "sampler", None)
    if inner is not None and inner is not batch_sampler:
        set_epoch_for_batch_sampler(inner, epoch)
def train(args: argparse.Namespace) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerator.even_batches = False
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
    if accelerator.distributed_type == DistributedType.DEEPSPEED:

        accelerator.even_batches = False

        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1

        logger.info(

            "Set DeepSpeed train_micro_batch_size_per_gpu=1 for packing dataloaders "

        )
    wandb_enabled = accelerator.is_main_process and args.wandb_mode != "disabled"
    if wandb_enabled:
        wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            config=vars(args),
            dir=args.log_dir,
            mode=args.wandb_mode
        )
        wandb.define_metric("train/update_step")
        wandb.define_metric("train/*", step_metric="train/update_step")

    logger.info(f"accelerator.distributed_type = {accelerator.distributed_type}")
    logger.info(f"accelerator.state = {accelerator.state}")

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        logger.info("DeepSpeed is ENABLED")
        logger.info(f"DeepSpeed plugin: {accelerator.state.deepspeed_plugin}")
    else:
        logger.info("DeepSpeed is NOT enabled")

    enable_batch_token_stats_logging = bool(getattr(args, "enable_batch_token_stats_logging", False))
    
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == "latest":
        resume_ckpt = find_latest_checkpoint(args.output_dir)

    if resume_ckpt is None:
        logger.info(f"No checkpoint found in {args.output_dir}, fallback to base model: {args.model_path}")
    else:
        logger.info(f"Found checkpoint for resume: {resume_ckpt}")

    # processor/config source
    processor_source = resolve_hf_model_dir(resume_ckpt) if resume_ckpt is not None else resolve_hf_model_dir(args.model_path)

    processor = ViMoProcessor.from_pretrained(processor_source)
    if "to_und_token" not in processor.image_processor._valid_kwargs_names:
        processor.image_processor._valid_kwargs_names.append("to_und_token")
    config = ViMoConfig.from_pretrained(processor_source)
    extra_vision_cfg = TSIMTokExtraCfg.load(args.extra_vision_cfg) if args.extra_vision_cfg else TSIMTokExtraCfg()

    model = build_model(args, config, extra_vision_cfg, resume_ckpt=resume_ckpt)
    if args.gradient_checkpointing:
        gc_kwargs = {"use_reentrant": False}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        logger.info(f"Gradient checkpointing enabled with kwargs={gc_kwargs}")
    else:
        logger.info("Gradient checkpointing disabled")

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

    if args.target_global_batch_size is not None and not getattr(args, "enable_packing", False):
        logger.warning(
            "--target_global_batch_size is set but --enable_packing is False; "
            "the target global batch-size constraint will not take effect."
        )

    task_loaders = build_task_dataloaders(args, processor, extra_vision_cfg)

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

    # must measure each rank's actual dataloader length after prepare
    task_num_batches = {task_name: len(loader) for task_name, loader in task_loaders.items()}
    for task_name, loader in task_loaders.items():
        logger.info(f"[Task={task_name}] computing prepared loader length ...")
        task_num_batches[task_name] = len(loader)
        logger.info(f"[Task={task_name}] prepared loader length done: {task_num_batches[task_name]}")
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
    oom_skip_record = None

    state_resume_ok = False
    state_resume_error = None
    if resume_ckpt is not None:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")
        try:
            accelerator.load_state(resume_ckpt)
            state_resume_ok = True
            oom_skip_record = load_oom_skip_record(args.output_dir)
        except Exception as e:
            state_resume_error = e
            logger.exception(f"Failed to load accelerator state from {resume_ckpt}: {e}")
            logger.warning(
                "Fallback to weights-only resume (tfmr). "
                "Optimizer/scheduler/rng states are not restored."
            )

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
            if oom_skip_record is not None:
                logger.warning(f"Loaded OOM skip record: {oom_skip_record}")
        else:
            logger.warning("training_state.json not found, fallback to epoch=1 / step=0")

        if not state_resume_ok and state_resume_error is not None:
            warmup_steps = int(args.warmup_rates * num_training_steps)
            resumed_lr = get_learning_rate(
                step=update_step,
                initial_lr=args.learning_rate,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=args.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = resumed_lr
            logger.warning(
                "Since accelerator state resume failed, this run continues from checkpoint "
                "weights and training_state only; optimizer momentum/adam states are reset. "
                f"Current LR is reset to resumed step value: {resumed_lr:.8e}."
            )

    logger.info(f"Per-rank task_num_batches: {task_num_batches}")
    logger.info(f"Per-rank total_batches_per_epoch: {total_batches_per_epoch}")
    logger.info(f"num_training_steps: {num_training_steps}")
    logger.info(f"process_index: {accelerator.process_index}")
    logger.info(f"is_main_process: {accelerator.is_main_process}")
    random_task_plan_per_rank = bool(int(getattr(args, "random_task_plan_per_rank", 0)))
    logger.info(f"random_task_plan_per_rank: {random_task_plan_per_rank}")
    logger.info(f"model class after prepare: {type(model)}")
    if oom_skip_record is not None:
        logger.warning(
            "OOM auto-skip is enabled for "
            f"epoch={oom_skip_record.get('epoch')} step_in_epoch={oom_skip_record.get('step_in_epoch')} "
            f"task={oom_skip_record.get('task_name')} radius={oom_skip_record.get('skip_radius', 0)}"
        )
    for epoch in range(start_epoch, args.n_epochs + 1):
        for task_name, loader in task_loaders.items():
            try:
                ds = loader.dataset
                set_epoch_for_dataset(ds, epoch)

                if hasattr(loader, "batch_sampler"):
                    set_epoch_for_batch_sampler(loader.batch_sampler, epoch)

            except Exception as e:
                logger.warning(f"set_epoch failed for {task_name}: {e}")
        current_task_num_batches = {
            task_name: len(loader) for task_name, loader in task_loaders.items()
        }
        task_iters = {}
        current_task_loaders = task_loaders
        step_offset = 0

        if resume_ckpt is not None and epoch == start_epoch and resume_step_in_epoch > 0:
            batch_plan, current_task_loaders, resume_skip_counts = build_resumed_epoch_plan_and_loaders(
                accelerator=accelerator,
                task_loaders=task_loaders,
                task_num_batches=current_task_num_batches,
                seed=args.seed,
                epoch=epoch,
                resume_step_in_epoch=resume_step_in_epoch,
                random_task_plan_per_rank=random_task_plan_per_rank,
                rank=int(accelerator.process_index),
            )
            step_offset = resume_step_in_epoch
            if accelerator.is_main_process:
                logger.info(
                    f"Resume epoch {epoch}: skip first {resume_step_in_epoch} mixed steps, "
                    f"per-task skips={resume_skip_counts}, remaining_steps={len(batch_plan)}"
                )
        else:
            batch_plan = build_batch_plan(
                current_task_num_batches,
                seed=args.seed,
                epoch=epoch,
                random_task_plan_per_rank=random_task_plan_per_rank,
                rank=int(accelerator.process_index),
            )

        train_iter = tqdm(batch_plan, total=len(batch_plan)) if accelerator.is_main_process else batch_plan

        pending_lr_bs_scale_sum = 0.0
        pending_lr_bs_scale_cnt = 0

        for local_step_idx, task_name in enumerate(train_iter):
            step_in_epoch = step_offset + local_step_idx

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if task_name not in task_iters:
                task_iters[task_name] = iter(current_task_loaders[task_name])

            batch = next(task_iters[task_name])


            if should_skip_due_to_oom_record(oom_skip_record, epoch, step_in_epoch, task_name):
                if accelerator.is_main_process:
                    logger.warning(
                        f"[Epoch {epoch}] skip step due to previous OOM record: "
                        f"step_in_epoch={step_in_epoch}, task={task_name}, global_step={global_step}"
                    )
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
            batch_token_stats = None
            cur_loss_bs_scale = 1.0
            cur_global_sample_num = None
            loss_before_bs_scale = None
            image_loss_before_bs_scale = None
            text_loss_before_bs_scale = None
            try:
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
                with autocast_ctx:
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
                    cur_image_loss_weight = batch["image_loss_weight"]
                    if isinstance(cur_image_loss_weight, torch.Tensor):
                        cur_image_loss_weight = cur_image_loss_weight.to(hidden_states.device).item()

                    batch["output_image_tokens"] = output_image_tokens.detach()
                    if enable_batch_token_stats_logging:
                        batch_token_stats = get_batch_token_stats(batch)

                    global_sample_count_for_scale = None
                    if args.enable_global_sample_mean_loss:
                        loss, image_loss, text_loss, global_sample_count_for_scale = compute_global_sample_mean_losses(
                            model=model,
                            text_logits=text_logits,
                            text_label=text_label,
                            output_image_logits=output_image_logits,
                            output_image_tokens=output_image_tokens,
                            output_token_counts=output_token_counts,
                            batch=batch,
                            image_loss_weight=cur_image_loss_weight,
                            loss_norm_mode=args.loss_norm_mode,
                        )
                    else:
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

                        loss = text_loss + cur_image_loss_weight * image_loss

                    if args.theoretical_global_sample_num is not None:
                        loss_before_bs_scale = loss
                        image_loss_before_bs_scale = image_loss
                        text_loss_before_bs_scale = text_loss
                        if global_sample_count_for_scale is None:
                            global_sample_count_for_scale = get_global_sample_count_tensor(
                                batch=batch,
                                device=loss.device,
                            )
                        bs_scale = (
                            global_sample_count_for_scale.to(dtype=loss.dtype)
                            / loss.new_tensor(float(args.theoretical_global_sample_num))
                        )
                        cur_loss_bs_scale = float(bs_scale.detach().item())
                        cur_global_sample_num = float(global_sample_count_for_scale.detach().item())
                        pending_lr_bs_scale_sum += cur_loss_bs_scale
                        pending_lr_bs_scale_cnt += 1
                    else:
                        loss_before_bs_scale = loss
                        image_loss_before_bs_scale = image_loss
                        text_loss_before_bs_scale = text_loss

                    if args.enable_global_sample_mean_loss:
                        # Keep compute_image_loss graph anchor (especially no-image dummy path)
                        # to prevent rank-wise grad bucket shape divergence under ZeRO.
                        loss = loss + image_loss_sum * 0.0
                    # Keep image-conditional token path params in graph on every rank.
                    # This avoids rank-wise parameter usage divergence on mixed image/no-image steps.
                    raw_model = unwrap_if_accelerated(model)
                    # Always keep LM path in graph on every rank to avoid collective desync
                    # when a corner batch contributes near-empty supervised tokens.
                    lm_dummy = text_logits.sum()
                    _dummy_ids = torch.zeros(1, dtype=torch.long, device=loss.device)
                    tok_proj_dummy = raw_model.gen_projector(raw_model.tok_embeddings(_dummy_ids)).sum()
                    merger_dummy = torch.zeros((), device=loss.device, dtype=loss.dtype)
                    if args.stage == 3:
                        for name, p in raw_model.named_parameters():
                            if p.requires_grad and "merger" in name:
                                merger_dummy = merger_dummy + p.sum()
                    loss = loss + (tok_proj_dummy + merger_dummy + lm_dummy) * 0.0

                metric(
                    output_image_logits,
                    output_image_tokens,
                    text_logits,
                    text_label,
                    loss,
                    image_loss,
                    text_loss,
                    get_batch_sample_count(batch),
                    loss_before_bs_scale,
                    image_loss_before_bs_scale,
                    text_loss_before_bs_scale,
                )

                accelerator.backward(loss)
            except RuntimeError as e:
                if not is_cuda_oom_error(e):
                    raise

                oom_record = {
                    "epoch": int(epoch),
                    "step_in_epoch": int(step_in_epoch),
                    "global_step": int(global_step),
                    "update_step": int(update_step),
                    "task_name": task_name,
                    "skip_radius": int(args.oom_skip_radius),
                    "message": str(e),
                    "recorded_at": datetime.utcnow().isoformat() + "Z",
                }

                try:
                    oom_record["rank"] = int(accelerator.process_index)
                    save_oom_skip_record(
                        args.output_dir,
                        oom_record,
                        rank=int(accelerator.process_index),
                    )
                except Exception as save_e:
                    logger.error(f"Failed to save OOM record on rank {accelerator.process_index}: {save_e}")

                logger.exception(
                    f"Caught CUDA OOM at epoch={epoch}, step_in_epoch={step_in_epoch}, "
                    f"global_step={global_step}, update_step={update_step}, task={task_name}, "
                    f"rank={accelerator.process_index}"
                )

                try:
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise

            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            did_update = False
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                applied_lr_bs_scale = 1.0
                if args.theoretical_global_sample_num is not None and pending_lr_bs_scale_cnt > 0:
                    applied_lr_bs_scale = pending_lr_bs_scale_sum / float(pending_lr_bs_scale_cnt)

                base_lrs = [pg["lr"] for pg in optimizer.param_groups]
                if args.theoretical_global_sample_num is not None:
                    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                        pg["lr"] = base_lr * applied_lr_bs_scale

                effective_lr_for_step = base_lrs[0] * applied_lr_bs_scale if len(base_lrs) > 0 else 0.0
                optimizer.step()
                if args.theoretical_global_sample_num is not None:
                    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                        pg["lr"] = base_lr
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.theoretical_global_sample_num is not None:
                    pending_lr_bs_scale_sum = 0.0
                    pending_lr_bs_scale_cnt = 0
                cur_loss_bs_scale = applied_lr_bs_scale

                update_step += 1
                did_update = True

                do_metric_sync = (
                    int(getattr(args, "metric_sync_interval", 1)) <= 1
                    or (update_step % int(getattr(args, "metric_sync_interval", 1)) == 0)
                )


                (
                    acc,
                    train_loss,
                    image_acc,
                    image_loss_v,
                    text_acc,
                    text_loss_v,
                    total_samples_v,
                    train_loss_unscaled,
                    image_loss_unscaled_v,
                    text_loss_unscaled_v,
                ) = metric.get_metric(sync=do_metric_sync, reset=do_metric_sync)

                if accelerator.is_main_process:
                    if hasattr(train_iter, "set_postfix"):
                        if do_metric_sync:
                            train_iter.set_postfix(
                                epoch=epoch,
                                step=update_step,
                                task=task_name,
                                loss=f"{train_loss:.3f} (img:{image_loss_v:.3f}, txt:{text_loss_v:.3f})",
                                loss0=f"{train_loss_unscaled:.3f} (img:{image_loss_unscaled_v:.3f}, txt:{text_loss_unscaled_v:.3f})",
                                acc=f"{acc:.3f} (img:{image_acc:.3f}, txt:{text_acc:.3f})",
                                samples=f"{total_samples_v}",
                                img_w=f"{cur_image_loss_weight:.2f}",
                                bs_s=f"{cur_loss_bs_scale:.3f}",
                                lr=f"{effective_lr_for_step:.2e}"
                            )
                        else:
                            train_iter.set_postfix(
                                epoch=epoch,
                                step=update_step,
                                task=task_name,
                                metric_sync=f"{update_step % int(getattr(args, 'metric_sync_interval', 1))}/{int(getattr(args, 'metric_sync_interval', 1))}",
                                img_w=f"{cur_image_loss_weight:.2f}",
                                bs_s=f"{cur_loss_bs_scale:.3f}",
                                lr=f"{effective_lr_for_step:.2e}"
                            )

                    if wandb_enabled and do_metric_sync:
                        log_payload = {
                            'train/update_step': update_step,
                            'train/loss_total': train_loss,
                            'train/loss_total_unscaled': train_loss_unscaled,
                            'train/loss_text': text_loss_v,
                            'train/loss_text_unscaled': text_loss_unscaled_v,
                            'train/loss_image': image_loss_v,
                            'train/loss_image_unscaled': image_loss_unscaled_v,
                            'train/acc': acc,
                            'train/image_acc': image_acc,
                            'train/text_acc': text_acc,
                            'train/total_samples': total_samples_v,
                            'train/lr': effective_lr_for_step,
                            'train/lr_base': lr_scheduler.get_last_lr()[0],
                            'train/task': task_name,
                            'train/image_loss_weight': cur_image_loss_weight,
                            'train/lr_bs_scale': cur_loss_bs_scale,
                            'train/metric_sync_interval': int(getattr(args, "metric_sync_interval", 1)),
                        }
                        if cur_global_sample_num is not None:
                            log_payload['train/global_sample_num'] = cur_global_sample_num
                        wandb.log(log_payload, step=update_step)

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


            if enable_batch_token_stats_logging and batch_token_stats is not None:
                mem_alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                mem_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
                mem_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                logger.info(
                    f"[TrainStep] epoch={epoch} step_in_epoch={step_in_epoch} global_step={global_step} task={task_name} "
                    f"samples={batch_token_stats['samples']} seq_len={batch_token_stats['seq_len']} "
                    f"input_tokens={batch_token_stats['input_tokens']} supervised_tokens={batch_token_stats['supervised_tokens']} "
                    f"text_tokens={batch_token_stats['text_tokens']} image_tokens={batch_token_stats['image_tokens']} "
                    f"sampler_lengths={batch_token_stats['sampler_lengths']} "
                    f"sampler_sum={batch_token_stats['sampler_length_sum']} sampler_max={batch_token_stats['sampler_length_max']} "
                    f"raw_lengths={batch_token_stats['raw_sample_lengths']} "
                    f"mem_alloc_mb={mem_alloc_mb:.2f} mem_reserved_mb={mem_reserved_mb:.2f} mem_peak_mb={mem_peak_mb:.2f}"
                )
                    # --- LOGGING BLOCK PATCH END ---

            global_step += 1

        accelerator.wait_for_everyone()
        if epoch == start_epoch and resume_step_in_epoch > 0:
            resume_step_in_epoch = 0

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
    Initialize the leading rows of model.tok_embeddings from visual.slot_quantize.embedding.weight.
    
    - if tok_embeddings has more rows than the vq embedding, copy only the first min_rows rows
    - if tok_embeddings has fewer rows than the vq embedding, copy only the first min_rows rows
    - the number of columns (embedding dim) must match
    - when zero_init_rest=True, the extra rows are zeroed; otherwise their original init is kept
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

    # with no resume checkpoint, initialize normally according to the stage
    if args.stage == 1:
        model = ViMoModel(config=config, extra_cfg=extra_vision_cfg)

        state_dict = load_hf_state_dict_any_layout(args.model_path)
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
        logger.info(f"[Stage {args.stage}] load base HF model from {resolved_model_path}")
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
        elif stage == 2:
            if 'visual' in name or 'merger' in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
        elif stage == 3:
            # Stage 3 = Stage 2 + unfreeze merger MLPs used by understanding encoder.
            if 'visual' in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
        else:
            raise ValueError(f"Unsupported stage: {stage}")

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
        # dummy path: ensure the vision-related params are "used" on every rank
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
    parser.add_argument(
        '--wandb_mode',
        type=str,
        default='offline',
        choices=['offline', 'online', 'disabled'],
        help='Weights & Biases mode: offline/online/disabled'
    )
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
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], required=True, help='Training stage: 1, 2 or 3')
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
        '--enable_global_sample_mean_loss',
        action='store_true',
        help=(
            "If set, compute loss by sample: normalize inside each sample first, "
            "then average by global sample count across all ranks (for packed training)."
        )
    )
    parser.add_argument(
        '--theoretical_global_sample_num',
        type=int,
        default=None,
        help=(
            "If set, scale learning rate each optimizer step by "
            "(real_global_sample_num / theoretical_global_sample_num)."
        )
    )
    parser.add_argument(
        '--use_json_num_tokens',
        action='store_true',
        help='If set, use num_tokens from json instead of random sampling'
    )
    parser.add_argument(
        '--drop_output_images',
        action='store_true',
        help='If set, remove all output_image placeholders/images and trim trailing num_tokens accordingly'
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
    parser.add_argument(
        '--oom_skip_radius',
        type=int,
        default=1,
        help='When resuming, automatically skip steps within this radius around the last recorded OOM step_in_epoch'
    )
    parser.add_argument(
        "--enable_batch_token_stats_logging",
        action="store_true",
        help="Enable per-step get_batch_token_stats computation and detailed TrainStep logging (adds overhead and lots of logs).",
    )
    parser.add_argument(
        "--metric_sync_interval",
        type=int,
        default=16,
        help="Sync/aggregate training metrics across ranks every N optimizer steps (1 = every step).",
    )
    parser.add_argument("--enable_packing", action="store_true")
    parser.add_argument("--pack_total_length", type=int, default=None)
    parser.add_argument("--pack_max_batch_size", type=int, default=None)
    parser.add_argument("--pack_total_length_threshold", type=int, default=None)
    parser.add_argument("--pack_total_length_threshold_ratio", type=float, default=0.9)
    parser.add_argument("--packing_shuffle", type=int, default=1)
    parser.add_argument("--max_rounds", type=int, default=32, help="Maximum number of packing rounds per epoch")
    parser.add_argument(
        "--random_task_plan_per_rank",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "If set to 1, each rank uses a different task mixing plan per epoch. "
            "If 0, all ranks share the same shuffled task plan."
        ),
    )
    parser.add_argument(
        "--target_global_batch_size",
        type=int,
        default=None,
        help=(
            "If set, constrain packing to keep per-step global sample count around this value "
            "(converted to per-rank target by world_size)."
        ),
    )
    parser.add_argument(
        "--target_global_batch_size_tolerance",
        type=int,
        default=0,
        help="Allowed deviation (±) around target_global_batch_size.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (activation recomputation) for supported model layers.",
    )
    args = parser.parse_args()

    if args.stage == 1 and not args.gen_weights_path:
        raise ValueError("stage 1 requires --gen_weights_path")
    if args.theoretical_global_sample_num is not None and args.theoretical_global_sample_num <= 0:
        raise ValueError("--theoretical_global_sample_num must be > 0 when set")
    if args.target_global_batch_size is not None and args.target_global_batch_size <= 0:
        raise ValueError("--target_global_batch_size must be > 0 when set")
    if args.target_global_batch_size_tolerance < 0:
        raise ValueError("--target_global_batch_size_tolerance must be >= 0")
    if args.metric_sync_interval <= 0:
        raise ValueError("--metric_sync_interval must be > 0")
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
