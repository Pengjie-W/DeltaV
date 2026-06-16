import os
import sys
import json
import random
from datetime import datetime
from typing import Any, Dict, List

import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

_vimo_repo = os.environ.get('VIMO_REPO')
if _vimo_repo and _vimo_repo not in sys.path:
    sys.path.insert(0, _vimo_repo)

from vimo.processing_vimo import ViMoProcessor
from vimo.modeling_vimo import ViMoModel, TSIMTokExtraCfg
from vimo.configuration_vimo import ViMoConfig

try:
    from vlmeval.vlm.base import BaseModel
except ImportError:
    class BaseModel:
        pass


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
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            trust_repo=True,
        )
    model.eval().to(device).half()
    return model


def build_incremental_visual_decode_inputs(
    mmgpt,
    generated_image_tokens,
    device,
    input_first_image_tokens=None,
    input_interleave_image_tokens=None,
    input_num_tokens=None,
):
    if input_num_tokens is None:
        input_num_tokens = []

    generated_token_lens = [int(t.shape[-1]) for t in generated_image_tokens if t.numel() > 0]

    if input_first_image_tokens is not None and input_first_image_tokens.numel() > 0:
        first_tokens = input_first_image_tokens.reshape(-1)

        interleave_parts = []
        all_num_tokens = []

        if input_interleave_image_tokens is not None and input_interleave_image_tokens.numel() > 0:
            interleave_parts.append(input_interleave_image_tokens.reshape(-1))
            all_num_tokens.extend(list(input_num_tokens))

        for t in generated_image_tokens:
            if t.numel() > 0:
                interleave_parts.append(t.squeeze(0))

        all_num_tokens.extend(generated_token_lens)

        if len(interleave_parts) > 0:
            interleave_tokens = torch.cat(interleave_parts, dim=0)
        else:
            interleave_tokens = torch.empty(0, dtype=torch.long, device=device)

        total_num_images = 1 + len(all_num_tokens)
    else:
        if len(generated_image_tokens) == 0:
            raise ValueError("No generated_image_tokens to decode.")

        first_tokens = generated_image_tokens[0].squeeze(0)
        if len(generated_image_tokens) > 1:
            interleave_tokens = torch.cat(
                [t.squeeze(0) for t in generated_image_tokens[1:] if t.numel() > 0],
                dim=0
            )
        else:
            interleave_tokens = torch.empty(0, dtype=torch.long, device=device)

        all_num_tokens = generated_token_lens[1:]
        total_num_images = len(generated_token_lens)

    grid_thw = torch.tensor([[1, 16, 16]] * total_num_images, device=device, dtype=torch.long)

    return first_tokens, interleave_tokens, grid_thw, total_num_images, all_num_tokens


class TSIMRouter:
    def __init__(
        self,
        visual_extractor_ckpt_path,
        tsim_intervals_path,
        visual_extractor_repo_path="",
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

        token_bins = load_json(self.tsim_intervals_path)
        self.tsim_to_budget = self._build_tsim_to_budget_mapping(token_bins, self.budget_key)
        self.alpha = float(token_bins.get("config", {}).get("alpha", self.alpha))


    def center_crop_pil(self, pil_image, image_size: int):
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size),
                resample=PIL.Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size),
            resample=PIL.Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return PIL.Image.fromarray(
            arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
        )
    def load_image(self, path):
        if path not in self.image_cache:
            self.image_cache[path] = PIL.Image.open(path).convert("RGB")
        return self.image_cache[path]

    def _is_monotonic_non_increasing(self, seq: List[float]) -> bool:
        for i in range(len(seq) - 1):
            if seq[i] < seq[i + 1]:
                return False
        return True

    def _build_tsim_to_budget_mapping(self, tsim_intervals: Dict[str, Any], budget_key: str):
        bins = tsim_intervals["tsim_intervals"]
        bins_sorted = sorted(bins, key=lambda x: float(x["tsim_left"]))
        sim_lefts = [float(b["tsim_left"]) for b in bins_sorted]
        sim_rights = [float(b["tsim_right"]) for b in bins_sorted]

        tokens: List[float] = []
        for b in bins_sorted:
            v = b.get(budget_key, None)
            tokens.append(9.0 if v is None else float(v))

        if not self._is_monotonic_non_increasing(tokens):
            raise ValueError(f"{budget_key} in tsim_intervals is not monotonic non-increasing.")

        tokens = [int(round(t)) for t in tokens]

        def tsim_to_budget(sim: float) -> int:
            for i in range(len(bins_sorted)):
                left, right = sim_lefts[i], sim_rights[i]
                if i < len(bins_sorted) - 1:
                    if left <= sim < right:
                        return tokens[i]
                else:
                    if left <= sim <= right:
                        return tokens[i]
            return tokens[0] if sim < sim_lefts[0] else tokens[-1]

        return tsim_to_budget

    def _compute_tsim(self, s_list: List[float], t_list: List[float]) -> float:
        n = min(len(s_list), len(t_list))
        if n <= 0:
            raise ValueError("Empty s_list or t_list for weighted mean.")

        num = 0.0
        den = 0.0
        for j in range(n):
            w = (self.alpha ** (n - 1 - j)) * float(t_list[j])
            num += float(s_list[j]) * w
            den += w
        return num / den if den != 0.0 else 0.0

    @torch.no_grad()
    def extract_visual_features(self, image_paths):
        images = [self.center_crop_pil(self.load_image(p), 224) for p in image_paths]
        pixel_values = torch.stack([self.visual_extractor_transform(img) for img in images], dim=0)
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)

        with torch.inference_mode():
            feats = self.visual_extractor.forward_features(pixel_values)
            patch_tokens = feats["x_norm_patchtokens"]
            feats = patch_tokens.mean(dim=1)

        feats = F.normalize(feats.float(), p=2, dim=-1)
        return feats

    @torch.no_grad()
    def build_pairwise_similarity_triangle(self, image_paths):
        feats = self.extract_visual_features(image_paths)
        sim_matrix = feats @ feats.transpose(0, 1)

        sim_triangle = []
        for i in range(len(image_paths)):
            if i == 0:
                sim_triangle.append([])
            else:
                sim_triangle.append(sim_matrix[i, :i].detach().float().cpu().tolist())
        return sim_triangle

    def allocate_token_budgets(self, image_paths):
        if len(image_paths) <= 0:
            return []
        if len(image_paths) == 1:
            return [] if self.exclude_base_in_output else [self.n_base]

        sim_triangle = self.build_pairwise_similarity_triangle(image_paths)
        tokens = [self.n_base]

        for i in range(1, len(sim_triangle)):
            row = sim_triangle[i]
            n = min(i, len(row), len(tokens))
            s_list = [float(x) for x in row[:n]]
            t_list = [float(x) for x in tokens[:n]]
            weighted_sim = self._compute_tsim(s_list, t_list)
            tokens.append(int(self.tsim_to_budget(weighted_sim)))

        return tokens[1:] if self.exclude_base_in_output else tokens

class ViMo(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path, extra_cfg_path, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.extra_cfg_path = extra_cfg_path
        self.kwargs = kwargs

        model_dtype_name = str(kwargs.get("model_dtype", os.environ.get("VIMO_MODEL_DTYPE", "fp16"))).lower()
        if model_dtype_name in ("bf16", "bfloat16"):
            self.model_torch_dtype = torch.bfloat16
        elif model_dtype_name in ("fp16", "float16", "half"):
            self.model_torch_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported model_dtype: {model_dtype_name}, expected bf16/fp16")

        decode_autocast_dtype_name = str(
            kwargs.get("decode_autocast_dtype", os.environ.get("VIMO_DECODE_AUTOCAST_DTYPE", "bf16"))
        ).lower()
        if decode_autocast_dtype_name in ("bf16", "bfloat16"):
            self.decode_autocast_dtype = torch.bfloat16
        elif decode_autocast_dtype_name in ("fp16", "float16", "half"):
            self.decode_autocast_dtype = torch.float16
        else:
            raise ValueError(
                f"Unsupported decode_autocast_dtype: {decode_autocast_dtype_name}, expected bf16/fp16"
            )
        
        self.vl_chat_processor = ViMoProcessor.from_pretrained(model_path)
        if "to_und_token" not in self.vl_chat_processor.image_processor._valid_kwargs_names:
            self.vl_chat_processor.image_processor._valid_kwargs_names.append("to_und_token")
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_chat_processor.pad_id = self.tokenizer.vocab.get("<|vision_pad|>")

        config = ViMoConfig.from_pretrained(model_path)
        self.extra_vision_cfg = TSIMTokExtraCfg.load(extra_cfg_path)

        self.model = ViMoModel.from_pretrained(
            model_path,
            config=config,
            extra_cfg=self.extra_vision_cfg,
            torch_dtype=self.model_torch_dtype
        ).cuda().eval()

        self.image_cache = {}
        self.min_pixels = 256 * 256
        self.max_pixels = 2048 * 2048
        self.rank = int(os.environ.get("RANK", 0))
        self.pid = os.getpid()
        self._sample_counter = 0
        self.decode_and_save_image = bool(kwargs.get("decode_and_save_image", True))
        self.vis_save_root = os.path.abspath(
            kwargs.get("vis_save_root", os.environ.get("VIMO_VIS_ROOT", "./seqimgvl_vis_outputs"))
        )
        os.makedirs(self.vis_save_root, exist_ok=True)

        self.token_sampler = None
        visual_extractor_ckpt_path = kwargs.get("visual_extractor_ckpt_path", None)
        tsim_intervals_path = kwargs.get("tsim_intervals_path", None)
        if visual_extractor_ckpt_path is not None and tsim_intervals_path is not None:
            self.token_sampler = TSIMRouter(
                visual_extractor_ckpt_path=visual_extractor_ckpt_path,
                tsim_intervals_path=tsim_intervals_path,
                visual_extractor_repo_path=kwargs.get("visual_extractor_repo_path", os.environ.get("VISUAL_EXTRACTOR_REPO", "")),
                budget_key=kwargs.get("budget_key", "budget"),
                alpha=kwargs.get("token_alpha", 0.8),
                n_base=kwargs.get("n_base", 144),
                exclude_base_in_output=kwargs.get("exclude_base_in_output", True),
                device=kwargs.get("dino_device", "cuda"),
            )

    @staticmethod
    def _safe_name(name: str) -> str:
        if name is None:
            return "unknown_benchmark"
        name = str(name).strip()
        if not name:
            return "unknown_benchmark"
        keep = []
        for ch in name:
            if ch.isalnum() or ch in ("-", "_", "."):
                keep.append(ch)
            else:
                keep.append("_")
        return "".join(keep)[:120]

    def _build_visual_output_path(self, benchmark: str) -> str:
        self._sample_counter += 1
        benchmark_dir = os.path.join(self.vis_save_root, self._safe_name(benchmark))
        os.makedirs(benchmark_dir, exist_ok=True)
        filename = f"sample_{self._sample_counter:08d}_rank{self.rank}_pid{self.pid}.png"
        return os.path.join(benchmark_dir, filename)

    def _record_path(self, benchmark: str) -> str:
        benchmark_dir = os.path.join(self.vis_save_root, self._safe_name(benchmark))
        os.makedirs(benchmark_dir, exist_ok=True)
        return os.path.join(benchmark_dir, f"records_rank{self.rank}_pid{self.pid}.jsonl")

    def _append_record(self, benchmark: str, payload: Dict[str, Any]) -> None:
        path = self._record_path(benchmark)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def _decode_and_save_visual_updates(
        self,
        generated_image_tokens: List[torch.Tensor],
        output_path: str,
        device: str,
        is_incremental_encoding: bool = False,
        image_token_num_per_image: int = 144,
        input_first_image_tokens: torch.Tensor = None,
        input_interleave_image_tokens: torch.Tensor = None,
        input_num_tokens: List[int] = None,
        save_all_images_in_chain: bool = True,
    ) -> (bool, str):
        mmgpt = self.model
        has_input_tokens = (
            input_first_image_tokens is not None
            and input_first_image_tokens.numel() > 0
        )

        if len(generated_image_tokens) == 0 and not has_input_tokens:
            return False, "no generated tokens and no input tokens"

        try:
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=self.decode_autocast_dtype):
                if is_incremental_encoding:
                    first_tokens, interleave_tokens, grid_thw, total_num_images, all_num_tokens = \
                        build_incremental_visual_decode_inputs(
                            mmgpt=mmgpt,
                            generated_image_tokens=generated_image_tokens,
                            device=device,
                            input_first_image_tokens=input_first_image_tokens,
                            input_interleave_image_tokens=input_interleave_image_tokens,
                            input_num_tokens=input_num_tokens,
                        )

                    dec, _ = mmgpt.visual.gen_decode_from_indices(
                        first_tokens,
                        interleave_tokens,
                        grid_thw,
                        num_images=[total_num_images],
                        num_tokens=all_num_tokens,
                    )

                    nrow = total_num_images if save_all_images_in_chain else max(len(generated_image_tokens), 1)
                else:
                    flat_token_parts = []
                    num_input_images = 0
                    if has_input_tokens:
                        flat_token_parts.append(input_first_image_tokens.reshape(-1))
                        num_input_images = 1

                        if input_interleave_image_tokens is not None and input_interleave_image_tokens.numel() > 0:
                            flat_token_parts.append(input_interleave_image_tokens.reshape(-1))
                            if input_num_tokens is not None:
                                num_input_images += len(input_num_tokens)
                            else:
                                n_base = int(mmgpt.vision_extra_cfg.gen_cfg.n_base)
                                num_input_images += int(input_interleave_image_tokens.numel() // n_base)

                    if len(generated_image_tokens) > 0:
                        flat_token_parts.append(torch.cat(generated_image_tokens, dim=0).flatten(0))

                    if len(flat_token_parts) == 0:
                        return False, "flat_token_parts is empty"

                    flat_tokens = torch.cat(flat_token_parts, dim=0)
                    num_total_images = num_input_images + len(generated_image_tokens)

                    dec, _ = mmgpt.visual.gen_decode_from_indices(
                        flat_tokens,
                        torch.empty(0, dtype=torch.long, device=device),
                        torch.tensor([[1, 16, 16]] * num_total_images, device=device, dtype=torch.long),
                        num_images=torch.ones(num_total_images, dtype=torch.long, device=device),
                        num_tokens=[],
                    )

                    nrow = num_total_images

            save_tensor = dec.detach().cpu().float()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_image(save_tensor, output_path, nrow=nrow, normalize=True)
            return True, ""
        except Exception as e:
            return False, str(e)

    def load_image(self, path):
        if path not in self.image_cache:
            self.image_cache[path] = PIL.Image.open(path).convert("RGB")
        return self.image_cache[path]

    def process_visual_input(self, image_paths, to_und_token):
        images = [self.load_image(p) for p in image_paths]
        images_outputs = self.vl_chat_processor.image_processor(
            images,
            to_und_token=to_und_token,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            return_tensors="pt"
        )
        return (
            images_outputs["pixel_values"].to(torch.bfloat16).cuda(),
            images_outputs["image_grid_thw"].cuda()
        )

    @torch.no_grad()
    def interleaved_generate(
        self,
        input_prompt,
        input_image_path=None,
        temperature=1.0,
        is_incremental_encoding=False,
        image_token_num_per_image=144,
        sample_num_tokens=None,
        max_runtime=16384,
    ):
        self.vl_chat_processor.image_start_tag = "<|vision_start|>"
        self.vl_chat_processor.image_tag = "<|image_pad|>"
        self.vl_chat_processor.image_end_tag = "<|vision_end|>"
        self.vl_chat_processor.pad_tag = "<|vision_pad|>"
        self.vl_chat_processor.first_gen_num_image_tokens = self.extra_vision_cfg.gen_cfg.n_base

        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        device = "cuda"
        input_images = input_image_path or []
        has_input_images = len(input_images) > 0

        torch.cuda.empty_cache()
        mmgpt = self.model

        first_image_tokens = None
        interleave_image_tokens = None
        image_mask = None
        deepstack_image_embeds = None
        num_tokens = []

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if has_input_images:
                if is_incremental_encoding:
                    img_len = [len(input_images)]
                else:
                    img_len = [1] * len(input_images)

                expected_num_tokens = max(len(input_images) - 1, 0)

                if sample_num_tokens is not None:
                    if not isinstance(sample_num_tokens, list):
                        raise ValueError(f"sample_num_tokens must be a list, got {type(sample_num_tokens)}")
                    num_tokens = [int(x) for x in sample_num_tokens[:expected_num_tokens]]
                elif self.token_sampler is not None and len(input_images) > 1:
                    num_tokens = [int(x) for x in self.token_sampler.allocate_token_budgets(input_images)[:expected_num_tokens]]
                else:
                    num_tokens = []

                if len(num_tokens) < expected_num_tokens:
                    num_tokens = num_tokens + [
                        random.choice(self.extra_vision_cfg.gen_cfg.n_delta)
                        for _ in range(expected_num_tokens - len(num_tokens))
                    ]

                und_image_input_pixel_values, und_input_pixel_values_grid_thw = self.process_visual_input(input_images, to_und_token=True)

                input_img_tokens = ""
                und_token_ptr = 0
                token_ptr = 0
                is_first_image = True

                for _ in range(len(input_images)):
                    t, h, w = und_input_pixel_values_grid_thw[und_token_ptr]
                    und_num_img_tokens = (h * w) // 4

                    if is_first_image:
                        temp_gen_num_image_tokens = self.vl_chat_processor.first_gen_num_image_tokens
                        is_first_image = False
                    else:
                        temp_gen_num_image_tokens = num_tokens[token_ptr]
                        token_ptr += 1

                    cur_img_tokens = (
                        self.vl_chat_processor.image_start_tag
                        + self.vl_chat_processor.image_tag * und_num_img_tokens
                        + self.vl_chat_processor.image_end_tag
                        + self.vl_chat_processor.image_start_tag
                        + self.vl_chat_processor.pad_tag * temp_gen_num_image_tokens
                        + self.vl_chat_processor.image_end_tag
                    )
                    input_img_tokens += cur_img_tokens
                    und_token_ptr += 1

                prompts = "<|im_start|>user\n" + input_img_tokens + input_prompt + "<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompts = "<|im_start|>user\n" + input_prompt + "<|im_end|>\n<|im_start|>assistant\n"

            input_ids = torch.LongTensor(self.vl_chat_processor.tokenizer.encode(prompts))
            tokens = input_ids.unsqueeze(0).cuda()
            inputs_embeds = mmgpt.get_input_embeddings()(tokens)

            if has_input_images:
                und_input_hidden_states, und_input_process_hidden_states = mmgpt.visual.backbone(
                    und_image_input_pixel_values, und_input_pixel_values_grid_thw
                )
                image_embeds, deepstack_image_embeds = mmgpt.und_encoder(
                    und_input_hidden_states, und_input_process_hidden_states
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                image_mask, _ = mmgpt.get_placeholder_mask(
                    input_ids.unsqueeze(0),
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                image_mask = image_mask[..., 0]

                input_image_pixel_values, input_pixel_values_grid_thw = self.process_visual_input(input_images, to_und_token=False)
                input_hidden_states, input_process_hidden_states = mmgpt.visual.backbone(
                    input_image_pixel_values, input_pixel_values_grid_thw
                )
                if is_incremental_encoding:
                    first_image_tokens, interleave_image_tokens = mmgpt.visual.encode_forward(
                        input_hidden_states,
                        input_process_hidden_states,
                        input_pixel_values_grid_thw,
                        img_len,
                        num_tokens
                    )
                else:
                    single_image_num = torch.ones(
                        len(input_images),
                        dtype=torch.int64,
                        device=input_hidden_states.device
                    )
                    first_image_tokens, interleave_image_tokens = mmgpt.visual.encode_forward(
                        input_hidden_states,
                        input_process_hidden_states,
                        input_pixel_values_grid_thw,
                        single_image_num,
                        []
                    )

                    num_merge_image = torch.tensor([len(input_images)], dtype=torch.int64, device=input_hidden_states.device)
                    first_image_tokens, interleave_image_tokens = mmgpt.split_first_and_interleave_vectorized(
                        first_image_tokens,
                        num_merge_image,
                        tokens_per_image=mmgpt.vision_extra_cfg.gen_cfg.n_base
                    )

                first_gen_image_embeddings = mmgpt.gen_projector(mmgpt.tok_embeddings(first_image_tokens))
                interleave_gen_image_embeddings = mmgpt.gen_projector(mmgpt.tok_embeddings(interleave_image_tokens))

                position_ids, _ = mmgpt.get_rope_index(input_ids.unsqueeze(0), und_input_pixel_values_grid_thw)
                position_ids = position_ids.cuda()

                image_gen_indices = (tokens == vision_end_id).nonzero()

                image_eos_embed = mmgpt.gen_projector(
                    mmgpt.tok_embeddings.weight[mmgpt.image_eos_id:mmgpt.image_eos_id + 1]
                ).to(inputs_embeds.device, inputs_embeds.dtype)

                first_ptr = 0
                inter_ptr = 0

                for img_idx, ind in enumerate(image_gen_indices):
                    if img_idx % 2 != 0:
                        continue

                    if img_idx == 0:
                        cur_token_len = mmgpt.vision_extra_cfg.gen_cfg.n_base
                        temp_image_embed = first_gen_image_embeddings[first_ptr:first_ptr + cur_token_len]
                        first_ptr += cur_token_len
                    else:
                        if is_incremental_encoding:
                            cur_token_len = num_tokens[(img_idx // 2) - 1]
                            temp_image_embed = interleave_gen_image_embeddings[inter_ptr:inter_ptr + cur_token_len]
                            inter_ptr += cur_token_len
                        else:
                            cur_token_len = mmgpt.vision_extra_cfg.gen_cfg.n_base
                            temp_image_embed = interleave_gen_image_embeddings[inter_ptr:inter_ptr + cur_token_len]
                            inter_ptr += cur_token_len

                    offset = ind[1] + 2
                    inputs_embeds[ind[0], offset: offset + cur_token_len, :] = temp_image_embed
                    inputs_embeds[ind[0], offset + cur_token_len, :] = image_eos_embed[0]
            else:
                position_ids, _ = mmgpt.get_rope_index(input_ids.unsqueeze(0))
                position_ids = position_ids.cuda()

            past_key_values = None
            generated_text_tokens = []
            generated_image_tokens = []
            mode = "text"
            finished = False
            is_first_token = True
            runtime = 0

            with torch.inference_mode():
                while not finished:
                    runtime += 1
                    if runtime > max_runtime:
                        break
                    if mode == "text":
                        if is_first_token:
                            cache_position = torch.arange(inputs_embeds.shape[1]).cuda()
                            outputs = mmgpt.language_model(
                                position_ids=position_ids,
                                inputs_embeds=inputs_embeds,
                                cache_position=cache_position,
                                past_key_values=past_key_values,
                                visual_pos_masks=image_mask,
                                deepstack_visual_embeds=deepstack_image_embeds,
                                use_cache=True,
                            )
                            is_first_token = False
                        else:
                            position_ids = position_ids[:, :, -1] + 1
                            position_ids = position_ids.unsqueeze(-1)
                            cache_position = cache_position[-1] + 1
                            cache_position = cache_position.unsqueeze(0)

                            outputs = mmgpt.language_model(
                                position_ids=position_ids,
                                inputs_embeds=inputs_embeds,
                                cache_position=cache_position,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )

                        hidden_states = outputs.last_hidden_state
                        past_key_values = outputs.past_key_values
                        logits = mmgpt.lm_head(hidden_states[:, -1])
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.argmax(probs, dim=-1, keepdim=True)
                        token_id = next_token[0, 0].item()
                        generated_text_tokens.append(token_id)

                        token_embed = mmgpt.get_input_embeddings()(next_token.view(-1))
                        inputs_embeds = token_embed.unsqueeze(1)

                        if token_id == vision_start_id:
                            mode = "image"
                        elif token_id == im_end_id:
                            finished = True
                            break

                    else:
                        image_tokens_list = []
                        for i in range(image_token_num_per_image):
                            runtime += 1
                            position_ids = position_ids[:, :, -1] + 1
                            position_ids = position_ids.unsqueeze(-1)
                            cache_position = cache_position[-1] + 1
                            cache_position = cache_position.unsqueeze(0)

                            outputs = mmgpt.language_model(
                                position_ids=position_ids,
                                inputs_embeds=inputs_embeds,
                                cache_position=cache_position,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                            hidden_states = outputs.last_hidden_state
                            past_key_values = outputs.past_key_values
                            output_image_hidden_states = mmgpt.norm(hidden_states[:, -1, :])
                            logits = mmgpt.vision_head(output_image_hidden_states)

                            probs = torch.softmax(logits / temperature, dim=-1)
                            next_token = torch.argmax(probs, dim=-1, keepdim=True)
                            image_eos_token_id = logits.shape[-1] - 1

                            token_id = next_token[0, 0].item()
                            if token_id == image_eos_token_id:
                                img_embed = mmgpt.gen_projector(mmgpt.tok_embeddings(next_token))
                                inputs_embeds = img_embed
                                break

                            image_tokens_list.append(token_id)
                            img_embed = mmgpt.gen_projector(mmgpt.tok_embeddings(next_token))
                            inputs_embeds = img_embed

                        if len(image_tokens_list) > 0:
                            image_tokens = torch.tensor(
                                image_tokens_list,
                                dtype=torch.long,
                                device=device
                            ).unsqueeze(0)
                            generated_image_tokens.append(image_tokens)
                        mode = "text"

        decoded_text = self.vl_chat_processor.tokenizer.decode(generated_text_tokens, skip_special_tokens=True)
        generated_image_token_counts = [int(t.shape[-1]) for t in generated_image_tokens]
        if not has_input_images:
            decode_input_num_tokens = None
        elif is_incremental_encoding:
            decode_input_num_tokens = num_tokens
        else:
            decode_input_num_tokens = [mmgpt.vision_extra_cfg.gen_cfg.n_base] * max(len(input_images) - 1, 0)
        return (
            decoded_text,
            generated_image_tokens,
            generated_image_token_counts,
            first_image_tokens,
            interleave_image_tokens,
            decode_input_num_tokens,
            is_incremental_encoding,
        )

    def generate_inner(self, message, dataset=None):
        input_images = []
        prompt = ""
        for msg in message:
            if msg['type'] == 'image':
                input_images.append(msg['value'])
            elif msg['type'] == 'text':
                prompt += msg['value']
        benchmark = dataset if dataset is not None else "unknown_benchmark"
        try:
            (
                decoded_text,
                generated_image_tokens,
                generated_image_token_num_list,
                input_first_image_tokens,
                input_interleave_image_tokens,
                decode_input_num_tokens,
                decode_is_incremental_encoding,
            ) = self.interleaved_generate(
                input_prompt=prompt,
                input_image_path=input_images,
                temperature=self.kwargs.get('temperature', 1.0),
                is_incremental_encoding=self.kwargs.get('is_incremental_encoding', False),
                image_token_num_per_image=self.kwargs.get('image_token_num_per_image', 145),
                sample_num_tokens=self.kwargs.get('sample_num_tokens', None),
                max_runtime=self.kwargs.get('max_new_tokens', 16384)
            )

            output_path = None
            decode_error = None
            has_generated_image = len(generated_image_tokens) > 0

            if self.decode_and_save_image:
                output_path = self._build_visual_output_path(benchmark)
                try:
                    saved, decode_error = self._decode_and_save_visual_updates(
                        generated_image_tokens=generated_image_tokens,
                        output_path=output_path,
                        device="cuda",
                        is_incremental_encoding=decode_is_incremental_encoding,
                        image_token_num_per_image=self.kwargs.get('image_token_num_per_image', 145),
                        input_first_image_tokens=input_first_image_tokens,
                        input_interleave_image_tokens=input_interleave_image_tokens,
                        input_num_tokens=decode_input_num_tokens,
                        save_all_images_in_chain=True,
                    )
                    if not saved:
                        output_path = None
                except Exception as decode_err:
                    decode_error = str(decode_err)
                    print(f"decode_and_save_generated_images failed: {decode_error}")
                    output_path = None

            vis_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "benchmark": benchmark,
                "has_generated_image": bool(has_generated_image),
                "model_output_text": decoded_text,
                "model_output_image": output_path,
                "model_output_image_token_num_list": generated_image_token_num_list,
                "input_image_count": len(input_images),
                "input_image_paths": input_images,
                "gt_available": False,
                "gt_image_path": None,
                "rank": self.rank,
                "pid": self.pid,
            }
            if decode_error is not None:
                vis_record["decode_error"] = decode_error
            self._append_record(benchmark, vis_record)
            return decoded_text
        except Exception as e:
            print(f"Error during generation: {e}")
            error_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "benchmark": benchmark,
                "has_generated_image": False,
                "model_output_text": "",
                "model_output_image": None,
                "model_output_image_token_num_list": [],
                "input_image_count": len(input_images),
                "input_image_paths": input_images,
                "gt_available": False,
                "gt_image_path": None,
                "rank": self.rank,
                "pid": self.pid,
                "error": str(e),
            }
            try:
                self._append_record(benchmark, error_record)
            except Exception as log_err:
                print(f"write vis record failed: {log_err}")
            return ""
