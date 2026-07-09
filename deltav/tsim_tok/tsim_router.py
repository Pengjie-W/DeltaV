"""TSIM Router — temporal-similarity driven token-budget allocation (paper Sec. 3.2).

Given an ordered sequence of visual states, the router measures temporal similarity
(TSIM) between each state and its history with a frozen visual extractor (DINOv2),
then maps each TSIM value to an incremental token budget via the offline-calibrated
TSIM-to-budget rule stored in ``tsim_intervals.json``.

This is the repository-native copy used by native inference / visualization when a
sample does not carry precomputed ``num_tokens`` and has more than one image. The
VLMEvalKit integration keeps its own independent copy.
"""
import json
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image
from torchvision import transforms


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dinov2_extractor(visual_extractor_repo_path: str, visual_extractor_ckpt_path: str, device: str):
    """Load the frozen DINOv2 ViT-B/14 used to compute temporal similarity (TSIM).

    Empty ``visual_extractor_repo_path`` -> torch.hub auto-downloads the code and
    pretrained weights from GitHub (cached under ``~/.cache/torch/hub/``).
    A non-empty path loads a local clone (offline/intranet) plus local weights.
    """
    if visual_extractor_repo_path:
        # Offline/intranet: local clone plus local weights
        model = torch.hub.load(
            visual_extractor_repo_path,
            "dinov2_vitb14",
            trust_repo=True,
            source="local",
            pretrained=False,
        )
        state_dict = torch.load(visual_extractor_ckpt_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
    else:
        # Default: torch.hub auto-downloads the code and pretrained weights
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            trust_repo=True,
        )
    model.eval().to(device).half()
    return model


class TSIMRouter:
    """Maps temporal visual variation to an adaptive incremental token budget."""

    def __init__(
        self,
        visual_extractor_ckpt_path,
        tsim_intervals_path,
        visual_extractor_repo_path,
        budget_key="budget",
        alpha=0.8,
        n_base=144,
        exclude_base_in_output=True,
        device="cuda",
    ):
        self.visual_extractor_ckpt_path = visual_extractor_ckpt_path
        self.visual_extractor_repo_path = visual_extractor_repo_path
        self.tsim_intervals_path = tsim_intervals_path
        self.budget_key = budget_key
        self.alpha = float(alpha)
        self.n_base = int(n_base)
        self.exclude_base_in_output = bool(exclude_base_in_output)
        self.device = device

        self.image_cache = {}
        self.visual_extractor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.visual_extractor = load_dinov2_extractor(
            visual_extractor_repo_path=self.visual_extractor_repo_path,
            visual_extractor_ckpt_path=self.visual_extractor_ckpt_path,
            device=self.device,
        )

        tsim_intervals = load_json(self.tsim_intervals_path)
        self.tsim_to_budget = self._build_tsim_to_budget_mapping(tsim_intervals, self.budget_key)
        self.alpha = float(tsim_intervals.get("config", {}).get("alpha", self.alpha))

    def center_crop_pil(self, pil_image, image_size: int):
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size),
                resample=PIL.Image.BOX,
            )
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size),
            resample=PIL.Image.BICUBIC,
        )
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return PIL.Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

    def load_image(self, path):
        if path not in self.image_cache:
            self.image_cache[path] = PIL.Image.open(path).convert("RGB")
        return self.image_cache[path]

    def _is_monotonic_non_increasing(self, seq: List[float]) -> bool:
        return all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))

    def _build_tsim_to_budget_mapping(self, tsim_intervals: Dict[str, Any], budget_key: str):
        intervals = tsim_intervals["tsim_intervals"]
        intervals_sorted = sorted(intervals, key=lambda x: float(x["tsim_left"]))
        tsim_lefts = [float(b["tsim_left"]) for b in intervals_sorted]
        tsim_rights = [float(b["tsim_right"]) for b in intervals_sorted]

        budgets: List[float] = []
        for b in intervals_sorted:
            v = b.get(budget_key, None)
            budgets.append(9.0 if v is None else float(v))

        if not self._is_monotonic_non_increasing(budgets):
            raise ValueError(f"{budget_key} in tsim_intervals is not monotonic non-increasing.")

        budgets = [int(round(t)) for t in budgets]

        def tsim_to_budget(tsim: float) -> int:
            for i in range(len(intervals_sorted)):
                left, right = tsim_lefts[i], tsim_rights[i]
                if i < len(intervals_sorted) - 1:
                    if left <= tsim < right:
                        return budgets[i]
                else:
                    if left <= tsim <= right:
                        return budgets[i]
            return budgets[0] if tsim < tsim_lefts[0] else budgets[-1]

        return tsim_to_budget

    def _compute_tsim(self, similarity_list: List[float], budget_list: List[float]) -> float:
        """History-weighted aggregation of pairwise similarities into a single TSIM value."""
        n = min(len(similarity_list), len(budget_list))
        if n <= 0:
            raise ValueError("Empty similarity_list or budget_list for TSIM computation.")
        num = 0.0
        den = 0.0
        for j in range(n):
            w = (self.alpha ** (n - 1 - j)) * float(budget_list[j])
            num += float(similarity_list[j]) * w
            den += w
        return num / den if den != 0.0 else 0.0

    @torch.no_grad()
    def extract_visual_features(self, image_paths):
        images = [self.center_crop_pil(self.load_image(p), 224) for p in image_paths]
        pixel_values = torch.stack([self.visual_extractor_transform(img) for img in images], dim=0)
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)
        with torch.inference_mode():
            feats = self.visual_extractor.forward_features(pixel_values)
            feats = feats["x_norm_patchtokens"].mean(dim=1)
        return F.normalize(feats.float(), p=2, dim=-1)

    @torch.no_grad()
    def build_pairwise_similarity_triangle(self, image_paths):
        feats = self.extract_visual_features(image_paths)
        sim_matrix = feats @ feats.transpose(0, 1)
        tsim_triangle = []
        for i in range(len(image_paths)):
            if i == 0:
                tsim_triangle.append([])
            else:
                tsim_triangle.append(sim_matrix[i, :i].detach().float().cpu().tolist())
        return tsim_triangle

    def allocate_token_budgets(self, image_paths):
        """Return the incremental token budget for each visual state in the sequence."""
        if len(image_paths) <= 0:
            return []
        if len(image_paths) == 1:
            return [] if self.exclude_base_in_output else [self.n_base]

        tsim_triangle = self.build_pairwise_similarity_triangle(image_paths)
        budgets = [self.n_base]
        for i in range(1, len(tsim_triangle)):
            row = tsim_triangle[i]
            n = min(i, len(row), len(budgets))
            similarity_list = [float(x) for x in row[:n]]
            budget_list = [float(x) for x in budgets[:n]]
            tsim_value = self._compute_tsim(similarity_list, budget_list)
            budgets.append(int(self.tsim_to_budget(tsim_value)))

        return budgets[1:] if self.exclude_base_in_output else budgets
