import torch
import numpy as np
import os.path as osp
from PIL import Image,UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import Dataset
from .augmentation import random_crop_arr, center_crop_arr, random_crop_to_target_aspect_ratio, random_pad_to_target_aspect_ratio, center_pad_to_target_aspect_ratio,center_crop_to_target_aspect_ratio
import torch.nn.functional as F_pad
# from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.image_utils import ChannelDimension, SizeDict
from torchvision.transforms import InterpolationMode
from transformers import Qwen2VLImageProcessorFast
import random

import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class ImageListDataset(Dataset, Qwen2VLImageProcessorFast):
    def __init__(
        self,
        json_path="data/your_dataset_list",
        is_train=True,
        image_size=(384, 384),
        max_samples=None,
        is_list=False,
        val_padding=False,
        use_pad=0,
        train_center_padding=False,
        train_center_crop=False,
        return_paths=False,
        return_num_tokens=False,
    ):
        """
        Args:
            list_dir: directory holding json list files; each json is a list of image paths
            is_train: whether this is the training split (decides random crop / center crop)
            image_size: (H, W)
            max_samples: optional, only take the first max_samples samples
        """
        self.is_train = is_train
        self.val_padding = val_padding
        self.image_size = image_size
        self.is_list=is_list
        self.return_paths=return_paths
        self.return_num_tokens=return_num_tokens
        self.train_center_padding=train_center_padding
        with open(json_path, "r", encoding="utf-8") as f:
            image_paths = json.load(f)
        self.image_paths = image_paths
        self.length = len(self.image_paths)
        self.use_pad=use_pad
        self.train_center_crop=train_center_crop
        # Optionally cap the number of samples
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
            self.length = len(self.image_paths)

        # Shuffle indices (rather than the paths directly, so the order can be changed later)
        self.indices = np.arange(self.length)
        np.random.seed(43)
        np.random.shuffle(self.indices)

        # Multiprocessing sharing strategy
        torch.multiprocessing.set_sharing_strategy("file_system")

    def __len__(self):
        return self.length
    def _process_one_image(self, img_path,  crop_u=None, crop_v=None, crop_seed=None, use_pad: bool = False):
        img = Image.open(img_path).convert("RGB")
        img.load()

        resized_height, resized_width = self.image_size

        image_size = resized_height  # e.g. 384

        # --- ADM crop: PIL in, PIL out ---
        if self.is_train:
            siglip_img = F.pil_to_tensor(img).to(dtype=torch.uint8)  # [C,H,W], uint8

            resized_height, resized_width = self.image_size

            if use_pad:
                # padding path
                if self.train_center_padding:
                    siglip_img = F.pil_to_tensor(img).to(dtype=torch.uint8)
                    siglip_img = center_pad_to_target_aspect_ratio(siglip_img, resized_height, resized_width)
                else:
                    siglip_img = random_pad_to_target_aspect_ratio(
                        siglip_img,
                        target_height=resized_height,
                        target_width=resized_width,
                        pad_u=crop_u,
                        pad_v=crop_v,
                        random_seed=crop_seed,  # reuse the same seed for the same sample
                )
            else:
                # crop path
                if self.train_center_crop:
                    img = center_crop_arr(img, image_size)
                    siglip_img = F.pil_to_tensor(img).to(dtype=torch.uint8)
                else:
                    siglip_img = random_crop_to_target_aspect_ratio(
                    siglip_img,
                    crop_u=crop_u,
                    crop_v=crop_v,
                    target_height=resized_height,
                    target_width=resized_width,
                    random_seed=crop_seed,  # reuse the same seed for the same sample
                )
        else:
            if self.val_padding:
                siglip_img = F.pil_to_tensor(img).to(dtype=torch.uint8)
                siglip_img = center_pad_to_target_aspect_ratio(siglip_img, resized_height, resized_width)
            else:
                siglip_img = F.pil_to_tensor(img).to(dtype=torch.uint8)  # [C,H,W], uint8
                siglip_img = center_crop_to_target_aspect_ratio(
                    siglip_img,
                    target_height=resized_height,
                    target_width=resized_width,
                )

        croped_siglip_img = siglip_img.contiguous()  # already the cropped result
        resized_siglip_img = self.resize(
            image=croped_siglip_img,
            size=SizeDict(height=resized_height, width=resized_width),
            interpolation=InterpolationMode.BICUBIC,
        )

        patches = self.rescale_and_normalize(
            resized_siglip_img,
            do_rescale=True,
            rescale_factor=1.0 / 255,
            do_normalize=True,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        ).unsqueeze(0)

        temporal_patch_size = 2
        patch_size = 16
        merge_size = 2
        target_img = patches[0]  # (C, H, W)

        if patches.ndim == 4:
            # add a temporal dimension
            patches = patches.unsqueeze(1)  # (B, T=1, C, H, W)

        # pad the temporal dimension if it is not divisible by temporal_patch_size
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(
                1, temporal_patch_size - 1, 1, 1, 1
            )  # pad with the last frame
            patches = torch.cat([patches, repeats], dim=1)

        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = patches.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )

        patches = patches.permute(
            0, 1, 4, 7, 5, 8, 3, 2, 6, 9
        )  # (B, grid_t, gh', gw', merge, merge, C, T, ph, pw)

        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )[0]  # take the first image in the batch

        image_grid_thw = torch.tensor(
            [grid_t, grid_h, grid_w], dtype=torch.int64
        )

        # ensure contiguity
        target_img = target_img.contiguous()
        flatten_patches = flatten_patches.contiguous()
        image_grid_thw = image_grid_thw.contiguous()

        return target_img, flatten_patches, image_grid_thw
    def get_seq_len(self, index: int) -> int:
        real_idx = self.indices[index]
        item = self.image_paths[real_idx]

        if self.is_list:
            if isinstance(item, dict):
                return len(item["img_paths"])
            return len(item)

        return 1
    def __getitem__(self, index):
        # use the shuffled indices to keep behavior consistent
        real_idx = self.indices[index]
        item = self.image_paths[real_idx]
        if self.return_num_tokens:
            img_paths = item['img_paths']
            num_tokens = item['num_tokens']
        else:
            img_paths=item
            num_tokens=None
        # Set the worker random seed (avoid random-number collisions across workers)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed =  random.randint(0, 10**9) + worker_info.id + index  # unique per worker+index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            seed =  random.randint(0, 10**9) + index  # unique per worker+index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        use_pad = (random.random() < self.use_pad)
        crop_u = random.random()
        crop_v = random.random()
        if self.is_list:

            images = []
            pixel_values_list = []
            image_grid_thw_list = []

            for i, img_path in enumerate(img_paths):
                img, pv, grid = self._process_one_image(img_path, crop_u=crop_u, crop_v=crop_v, crop_seed=seed,use_pad=use_pad)
                images.append(img)
                pixel_values_list.append(pv)
                image_grid_thw_list.append(grid)

            num_images = len(images)
            if self.return_paths:
                return images, pixel_values_list, image_grid_thw_list, num_images, img_paths
            elif self.return_num_tokens:
                return images, pixel_values_list, image_grid_thw_list, num_images, img_paths, num_tokens
            else:
                return images, pixel_values_list, image_grid_thw_list, num_images

        else:
            img, pv, grid = self._process_one_image(img_paths, crop_u=crop_u, crop_v=crop_v, crop_seed=seed, use_pad=use_pad)
            return img, pv, grid

class ImageNetListDataset(ImageListDataset):
    def __init__(
        self,
        json_path="data/your_dataset_list",
        image_size=(384, 384),
        is_train=False,
        max_samples=None,
        is_list=False,
        val_padding=False,
        use_pad=0,
        train_center_padding=False,
        train_center_crop=False,
        return_paths=False,
        return_num_tokens=False
    ):
        super().__init__(
            json_path=json_path,
            is_train=is_train,
            image_size=image_size,
            max_samples=max_samples,
            is_list=is_list,
            val_padding=val_padding,
            use_pad=use_pad,
            train_center_padding=train_center_padding,
            train_center_crop=train_center_crop,
            return_paths=return_paths,
            return_num_tokens=return_num_tokens
        )
        self.is_list = is_list
        self.return_paths = return_paths
        self.return_num_tokens = return_num_tokens

    def __getitem__(self, idx):
        max_retry = len(self)  # at most scan the whole dataset once, to avoid an infinite loop

        for offset in range(max_retry):
            cur_idx = (idx + offset) % len(self)
            try:
                return super().__getitem__(cur_idx)
            except (UnidentifiedImageError, OSError, ValueError, TypeError) as e:
                print(f"[WARN] skip bad image idx={cur_idx}, err={e}")
                continue

        raise RuntimeError(
            f"All samples failed after trying {max_retry} candidates starting from idx={idx}"
        )
        
from torch.utils.data import Sampler
from collections import defaultdict


class MultiSourceBatchSampler(Sampler):
    """
    Mix multiple child batch samplers after scaling each by its ratio.

    Semantics:
    - Each batch from the i-th sampler is repeated ratios[i] times
    - So the total number of batches = sum(len(sampler_i) * ratios[i])
    - Then a global shuffle is applied over all batches
    - Each child sampler already handles its own bucket / budget / DDP
    - This class only handles:
        1) reading batches
        2) adding the ConcatDataset offset
        3) repeating by ratio
        4) mixing the output
    """
    def __init__(
        self,
        samplers,
        offsets,
        ratios,
        shuffle=True,
        seed=0,
    ):
        assert len(samplers) == len(offsets) == len(ratios)
        assert all(r > 0 and isinstance(r, int) for r in ratios)

        self.samplers = samplers
        self.offsets = offsets
        self.ratios = ratios
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0

        self._batches = None
        self._built_epoch = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        for s in self.samplers:
            if hasattr(s, "set_epoch"):
                s.set_epoch(epoch)
        self._batches = None
        self._built_epoch = None

    def _build_batches(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        mixed_batches = []

        # 1) read each sampler's batches and add its offset
        # 2) append each batch to mixed_batches repeated by its ratio
        for sampler, offset, ratio in zip(self.samplers, self.offsets, self.ratios):
            batches = list(iter(sampler))
            batches = [[idx + offset for idx in batch] for batch in batches]

            for batch in batches:
                for _ in range(ratio):
                    mixed_batches.append(batch)

        # 3) global shuffle
        if self.shuffle and len(mixed_batches) > 1:
            perm = torch.randperm(len(mixed_batches), generator=g).tolist()
            mixed_batches = [mixed_batches[i] for i in perm]

        return mixed_batches

    def __iter__(self):
        if self._batches is None or self._built_epoch != self.epoch:
            self._batches = self._build_batches()
            self._built_epoch = self.epoch

        for batch in self._batches:
            yield batch

    def __len__(self):
        return sum(len(s) * r for s, r in zip(self.samplers, self.ratios))


from typing import Dict, Callable, Optional, Union

BucketMaxCounts = Union[int, Dict[int, int], Callable[[int], int], None]

class BucketedTokenBudgetBatchSampler_len(Sampler):
    """
    Scheme B (recommended for DDP):
    - Globally shuffle samples
    - Globally bucket + globally greedy packing to form all_batches
    - Then split at the batch level across ranks (rank::world_size)
    - Optional: trim the global batch count to a multiple of world_size so each rank has the same step count

    Also supports per-bucket caps on the number of samples in a batch (len(batch) <= max_count(bucket)).
    """
    def __init__(
        self,
        dataset,
        target_budget: int,
        bucket_width: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,     # drop_last_step (used to align with world_size)
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        bucket_max_counts: BucketMaxCounts = None,  # max samples per bucket
        default_max_count: int = 10_000,            # fallback cap for unspecified buckets
        sort_within_bucket: bool = False,   # whether to sort within a bucket by length (long to short)
    ):
        self.dataset = dataset
        self.target_budget = int(target_budget)
        self.bucket_width = int(bucket_width)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(rank)
        self.world_size = int(world_size)

        self.bucket_max_counts = bucket_max_counts
        self.default_max_count = int(default_max_count)

        self._batches = None
        self._built_epoch = None

        self.lengths = [int(dataset.get_seq_len(i)) for i in range(len(dataset))]
        self.sort_within_bucket = bool(sort_within_bucket)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._batches = self._build_batches()
        self._built_epoch = self.epoch

    def _bucket_id(self, L: int) -> int:
        return (max(1, L) - 1) // self.bucket_width

    def _max_count_for_bucket(self, bucket_id: int) -> int:
        """
        Return the cap on the number of samples in a batch for this bucket (len(batch) <= cap)
        """
        cfg = self.bucket_max_counts
        if cfg is None:
            return self.default_max_count
        if isinstance(cfg, int):
            return int(cfg)
        if isinstance(cfg, dict):
            return int(cfg.get(bucket_id, self.default_max_count))
        if callable(cfg):
            v = int(cfg(bucket_id))
            return max(1, v)
        # fallback
        return self.default_max_count

    def _build_global_batches(self):
        n = len(self.dataset)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1) globally shuffle indices (sample-level fairness)
        if self.shuffle:
            all_indices = torch.randperm(n, generator=g).tolist()
        else:
            all_indices = list(range(n))

        # 2) global bucketing (note: no DDP sharding here)
        buckets = defaultdict(list)
        overs = []
        for idx in all_indices:
            L = self.lengths[idx]
            if L >= self.target_budget:
                overs.append(idx)  # single sample over budget: its own batch
            else:
                buckets[self._bucket_id(L)].append(idx)

        all_batches = []

        # 3) turn overs into batches first (naturally len(batch)=1)
        for idx in overs:
            all_batches.append([idx])

        # 4) greedy packing within each bucket (satisfying both token budget and per-bucket max_count)
        bucket_keys = list(buckets.keys())
        if self.shuffle and len(bucket_keys) > 1:
            perm = torch.randperm(len(bucket_keys), generator=g).tolist()
            bucket_keys = [bucket_keys[i] for i in perm]

        for k in bucket_keys:
            b = buckets[k]
            if not b:
                continue

            max_count = self._max_count_for_bucket(k)
            if self.sort_within_bucket:
                # place longer samples first to better approach the budget
                b.sort(key=lambda i: self.lengths[i], reverse=True)

            batch = []
            budget = 0
            for idx in b:
                L = self.lengths[idx]

                # Two conditions that trigger flushing the current batch:
                # 1) adding this sample would exceed the token budget
                # 2) the current batch has reached this bucket's sample cap
                hit_budget = (batch and (budget + L > self.target_budget))
                hit_count = (len(batch) >= max_count)

                if hit_budget or hit_count:
                    all_batches.append(batch)
                    batch, budget = [], 0

                batch.append(idx)
                budget += L

                # if max_count is reached exactly, flush immediately (optional but cleaner)
                if len(batch) >= max_count:
                    all_batches.append(batch)
                    batch, budget = [], 0

            # collect the leftover batch at the tail of the bucket
            if batch:
                all_batches.append(batch)

        # 5) globally reshuffle batch order across buckets/overs once more
        if self.shuffle and len(all_batches) > 1:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        return all_batches

    def _build_batches(self):
        # A) build the global batch list first (identical across all ranks)
        all_batches = self._build_global_batches()

        # B) align step counts for DDP: trim the global batch count to a multiple of world_size
        if self.world_size > 1 and self.drop_last:
            m = (len(all_batches) // self.world_size) * self.world_size
            all_batches = all_batches[:m]

        # C) batch-level sharding: each rank takes its own share
        local_batches = all_batches[self.rank::self.world_size]

        # D) fallback: avoid an empty result (in extreme cases)
        if len(local_batches) == 0:
            return [[]]

        return local_batches

    def __iter__(self):
        if self._batches is None or self._built_epoch != self.epoch:
            self._batches = self._build_batches()
            self._built_epoch = self.epoch
        for batch in self._batches:
            if batch:
                yield batch

    def __len__(self):
        if self._batches is None or self._built_epoch != self.epoch:
            return 1
        return max(1, len(self._batches))