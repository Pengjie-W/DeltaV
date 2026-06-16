import os
import sys
import json
import random
import argparse

import PIL.Image
import torch
import torch.distributed as dist
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vimo.processing_vimo import ViMoProcessor
from vimo.modeling_vimo import ViMoModel, TSIMTokExtraCfg
from vimo.configuration_vimo import ViMoConfig
from torchvision.utils import save_image
from vimo.rope2d import get_rope_index_3
from tqdm import tqdm


# =========================
# Path configuration
# =========================
MODEL_PATH = "weights/vimo_2b"
EXTRA_CFG = "configs/vimo_cfg.json"
JSON_PATH = "data/zebra_test_sample.json"
SAVE_DIR = "./zebra_test/baseline_zebra_image_output"
RESULT_DIR = "./zebra_test/json_parts"
FINAL_JSON = "./zebra_test/baseline_zebra_image_output.json"

image_cache = {}

vl_chat_processor = None
vl_gpt = None
tokenizer = None
extra_vision_cfg = None
MIN_PIXELS = 65536
MAX_PIXELS = 1048576


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--extra_cfg", type=str, default=EXTRA_CFG)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--final_json", type=str, default=FINAL_JSON)
    parser.add_argument(
        "--image_token_num_per_image",
        type=int,
        default=144,
        help="number of image tokens generated per image"
    )

    parser.add_argument(
        "--is_incremental_encoding",
        action="store_true",
        help="enable incremental vision encoding"
    )
    parser.add_argument(
        "--strict_num_tokens",
        action="store_true",
        help="If set, raise error when json num_tokens is missing or length mismatch"
    )

    # Whether to read num_tokens directly from the json
    parser.add_argument(
        "--use_json_num_tokens",
        action="store_true",
        help="If set, use num_tokens from json instead of random sampling"
    )
    # TSIM Router fallback: when a sample has no num_tokens and >1 input image,
    # allocate incremental budgets by temporal similarity instead of random sampling.
    parser.add_argument("--use_tsim_router", action="store_true",
                        help="Use TSIM Router to allocate token budgets when num_tokens absent")
    parser.add_argument("--tsim_intervals_path", type=str, default="tools/data_processing/tsim_intervals.json")
    parser.add_argument("--visual_extractor_ckpt", type=str, default="weights/dinov2/dinov2_vitb14.pth")
    parser.add_argument("--visual_extractor_repo", type=str, default="",
                        help="Local path to facebookresearch/dinov2 repo (torch.hub source=local)")
    parser.add_argument(
        "--max_runtime",
        type=int,
        default=16384,
        help="maximum generation steps to avoid infinite loop"
    )
    # Whether to decode generated image tokens and save images. When enabled, output
    # switches to streaming jsonl and special tokens are kept (skip_special_tokens=False);
    # when disabled, behavior matches pure inference evaluation.
    parser.add_argument(
        "--decode_and_save_image",
        action="store_true",
        help="If set, decode generated image tokens and save image to output_path"
    )
    parser.add_argument(
        "--concat_gt_images",
        action="store_true",
        help="If set, save GT input/output concat image to a separate subfolder under save_dir"
    )
    parser.add_argument(
        "--gt_image_size",
        type=int,
        default=256,
        help="center-crop GT images to square then resize to this size before concat"
    )
    parser.add_argument(
        "--shuffle_dataset",
        action="store_true",
        help="If set, randomly shuffle dataset order before distributed sharding"
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=-1,
        help="Seed for dataset shuffle; use negative value to sample a random seed per run"
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=65536,
        help="Minimum image pixels for smart_resize (must match training)"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1048576,
        help="Maximum image pixels for smart_resize (must match training)"
    )
    return parser.parse_args()


def setup_distributed():
    """
    torchrun automatically injects:
      RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    """
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun rather than python xxx.py")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    from datetime import timedelta

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=600),
    )

    return rank, world_size, local_rank


def build_incremental_decode_inputs(
    mmgpt,
    generated_image_tokens,
    device,
    input_first_image_tokens=None,
    input_interleave_image_tokens=None,
    input_num_tokens=None,
):
    """
    Build the correct inputs for incremental gen_decode_from_indices.

    Rules:
    - The first image of the whole sequence uses first_tokens (fixed n_base)
    - All subsequent images use interleave_tokens
    - num_tokens must correspond to the real token count of every image except the first
    - If there are input images, they must also be merged into the whole chain
    """

    if input_num_tokens is None:
        input_num_tokens = []

    generated_token_lens = [int(t.shape[-1]) for t in generated_image_tokens if t.numel() > 0]

    # Case A: input images are present
    if input_first_image_tokens is not None and input_first_image_tokens.numel() > 0:
        first_tokens = input_first_image_tokens.reshape(-1)

        interleave_parts = []
        all_num_tokens = []

        # First append the input images except the first one
        if input_interleave_image_tokens is not None and input_interleave_image_tokens.numel() > 0:
            interleave_parts.append(input_interleave_image_tokens.reshape(-1))
            all_num_tokens.extend(list(input_num_tokens))

        # Then append generated images (all of them count as subsequent images)
        for t in generated_image_tokens:
            if t.numel() > 0:
                interleave_parts.append(t.squeeze(0))

        all_num_tokens.extend(generated_token_lens)

        if len(interleave_parts) > 0:
            interleave_tokens = torch.cat(interleave_parts, dim=0)
        else:
            interleave_tokens = torch.empty(0, dtype=torch.long, device=device)

        total_num_images = 1 + len(all_num_tokens)

    # Case B: no input images, the whole chain consists of generated images
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

        all_num_tokens = generated_token_lens[1:]   # the first image is fixed and excluded from num_tokens
        total_num_images = len(generated_token_lens)

    grid_thw = torch.tensor([[1, 16, 16]] * total_num_images, device=device, dtype=torch.long)

    return first_tokens, interleave_tokens, grid_thw, total_num_images, all_num_tokens


def cleanup_distributed():
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"cleanup_distributed failed: {e}", flush=True)


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def load_image(path):
    if path not in image_cache:
        image_cache[path] = PIL.Image.open(path).convert("RGB")
    return image_cache[path]


def process_image(image_paths, to_und_token):
    images = [load_image(p) for p in image_paths]
    images_outputs = vl_chat_processor.image_processor(
        images,
        to_und_token=to_und_token,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        return_tensors="pt"
    )
    return (
        images_outputs["pixel_values"].to(torch.bfloat16).cuda(),
        images_outputs["image_grid_thw"].cuda()
    )


def center_crop_and_resize_pil(image, image_size=256):
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if hasattr(PIL.Image, "Resampling"):
        resample = PIL.Image.Resampling.BICUBIC
    else:
        resample = PIL.Image.BICUBIC
    return image.resize((image_size, image_size), resample=resample)


def load_gt_images_as_tensor(input_image_paths=None, output_image_paths=None, image_size=256):
    paths = []
    if input_image_paths:
        paths.extend(input_image_paths)
    if output_image_paths:
        paths.extend(output_image_paths)

    image_tensors = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[GT] image not found, skip: {path}", flush=True)
            continue

        try:
            image = load_image(path).copy()
            image = center_crop_and_resize_pil(image, image_size=image_size)
            image_np = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
            # Align to the [-1, 1] range of the VQ decode output
            image_tensor = image_tensor * 2.0 - 1.0
            image_tensors.append(image_tensor)
        except Exception as e:
            print(f"[GT] failed to load image {path}: {e}", flush=True)

    if len(image_tensors) == 0:
        return None
    return torch.stack(image_tensors, dim=0)


@torch.no_grad()
def decode_and_save_generated_images(
    mmgpt,
    generated_image_tokens,
    output_path,
    device,
    is_incremental_encoding=False,
    image_token_num_per_image=144,
    input_first_image_tokens=None,
    input_interleave_image_tokens=None,
    input_num_tokens=None,
    save_all_images_in_chain=True,
    concat_gt_images=False,
    gt_input_image_paths=None,
    gt_output_image_paths=None,
    gt_image_size=256,
):
    has_input_tokens = (
        input_first_image_tokens is not None
        and input_first_image_tokens.numel() > 0
    )

    if len(generated_image_tokens) == 0 and not has_input_tokens:
        return

    try:
        if is_incremental_encoding:
            first_tokens, interleave_tokens, grid_thw, total_num_images, all_num_tokens = \
                build_incremental_decode_inputs(
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

            # With input images, the whole chain is usually saved
            nrow = total_num_images if save_all_images_in_chain else max(len(generated_image_tokens), 1)

        else:
            # Non-incremental: decode and save input images and generated images together
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
                return

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
        save_nrow = nrow

        save_image(
            save_tensor,
            output_path,
            nrow=save_nrow,
            normalize=True,
        )
        print('save', output_path)

        if concat_gt_images:
            gt_tensor = load_gt_images_as_tensor(
                input_image_paths=gt_input_image_paths,
                output_image_paths=gt_output_image_paths,
                image_size=gt_image_size,
            )
            if gt_tensor is not None and gt_tensor.shape[0] > 0:
                gt_dir = os.path.join(os.path.dirname(output_path), "gt_concat")
                os.makedirs(gt_dir, exist_ok=True)
                gt_output_path = os.path.join(gt_dir, os.path.basename(output_path))
                save_image(
                    gt_tensor.float(),
                    gt_output_path,
                    nrow=gt_tensor.shape[0],
                    normalize=True,
                )
                print('save gt', gt_output_path)
    except Exception as e:
        print(f"decode_and_save_generated_images failed: {e}", flush=True)


@torch.no_grad()
def generate(
    input_prompt,
    input_image_path,
    output_path,
    vl_chat_processor,
    vl_gpt,
    temperature=1.0,
    parallel_size=1,
    cfg_weight=5,
    is_incremental_encoding=False,
    image_token_num_per_image=144,
    sample_num_tokens=None,
    max_runtime=16384,
    decode_and_save_image=False,
    concat_gt_images=False,
    gt_input_image_paths=None,
    gt_output_image_paths=None,
    gt_image_size=256,
):
    """
    Unified entry point for interleaved text/image generation.

    - When input_image_path is non-empty, the image-text input path is used (including
      understanding/generation image encoding; the text forward pass uses
      is_first_token + advance-before, consistent with the text_image_to_text_image
      research implementation).
    - When input_image_path is empty, the text-only input path is used (the text forward
      pass uses advance-after, consistent with the original infer / seqimg_vl_token
      implementation).
    The rest (text processing after taking hidden_states, and the image token generation
    block) is shared by both paths.
    """
    vl_chat_processor.image_start_tag = "<|vision_start|>"
    vl_chat_processor.image_tag = "<|image_pad|>"
    vl_chat_processor.image_end_tag = "<|vision_end|>"
    vl_chat_processor.pad_tag = "<|vision_pad|>"
    vl_chat_processor.first_gen_num_image_tokens = extra_vision_cfg.gen_cfg.n_base

    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    device = "cuda"

    input_images = input_image_path or []
    has_input_images = len(input_images) > 0

    torch.cuda.empty_cache()
    mmgpt = vl_gpt

    # Only the image-text path uses these; the text-only path keeps them None
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

            # Number of tokens required for non-first input images
            expected_num_tokens = max(len(input_images) - 1, 0)

            if sample_num_tokens is not None:
                if not isinstance(sample_num_tokens, list):
                    raise ValueError(f"sample_num_tokens must be a list, got {type(sample_num_tokens)}")

                if len(sample_num_tokens) != expected_num_tokens:
                    sample_num_tokens = sample_num_tokens[:expected_num_tokens]

                num_tokens = sample_num_tokens
            else:
                num_tokens = [
                    random.choice(extra_vision_cfg.gen_cfg.n_delta)
                    for _ in range(expected_num_tokens)
                ]

            und_image_input_pixel_values, und_input_pixel_values_grid_thw = process_image(input_images, to_und_token=True)

            input_img_tokens = ""
            und_token_ptr = 0
            token_ptr = 0
            is_first_image = True

            for _ in range(len(input_images)):
                t, h, w = und_input_pixel_values_grid_thw[und_token_ptr]
                und_num_img_tokens = (h * w) // 4

                if is_first_image:
                    temp_gen_num_image_tokens = vl_chat_processor.first_gen_num_image_tokens
                    is_first_image = False
                else:
                    temp_gen_num_image_tokens = num_tokens[token_ptr]
                    token_ptr += 1

                cur_img_tokens = (
                    vl_chat_processor.image_start_tag
                    + vl_chat_processor.image_tag * und_num_img_tokens
                    + vl_chat_processor.image_end_tag
                    + vl_chat_processor.image_start_tag
                    + vl_chat_processor.pad_tag * temp_gen_num_image_tokens
                    + vl_chat_processor.image_end_tag
                )
                input_img_tokens += cur_img_tokens
                und_token_ptr += 1

            prompts = "<|im_start|>user\n" + input_img_tokens + input_prompt + "<|im_end|>\n<|im_start|>assistant\n"
            input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(prompts))
            tokens = input_ids.unsqueeze(0).cuda()
            inputs_embeds = mmgpt.get_input_embeddings()(tokens)

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

            input_image_pixel_values, input_pixel_values_grid_thw = process_image(input_images, to_und_token=False)
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
                # First encode treating every image as a first image
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

                # Then, consistent with training, split back into first / interleave
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
            ).to(inputs_embeds.device, inputs_embeds.dtype)   # [1, hidden_dim]

            first_ptr = 0
            inter_ptr = 0

            for img_idx, ind in enumerate(image_gen_indices):
                # The prompt structure for each input image is:
                # <vision_start><image_pad...><vision_end><vision_start><vision_pad...><vision_end>
                # so only the first vision_end of each pair is processed
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

                # In training, writing starts at +2 after the first vision_end,
                # i.e. skipping the current <vision_end> and the next <vision_start>
                offset = ind[1] + 2

                # Write the image token embedding
                inputs_embeds[ind[0], offset: offset + cur_token_len, :] = temp_image_embed

                # In training, the following <vision_end> is also replaced with the image eos embedding
                inputs_embeds[ind[0], offset + cur_token_len, :] = image_eos_embed[0]
        else:
            prompts = "<|im_start|>user\n" + input_prompt + "<|im_end|>\n<|im_start|>assistant\n"
            input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(prompts))
            tokens = input_ids.unsqueeze(0).cuda()
            inputs_embeds = mmgpt.get_input_embeddings()(tokens)

            position_ids, _ = mmgpt.get_rope_index(input_ids.unsqueeze(0))
            position_ids = position_ids.cuda()

        past_key_values = None
        generated_text_tokens = []
        generated_image_tokens = []
        mode = "text"
        finished = False
        is_first_token = True
        runtime = 0
        # text-only path: set cache_position once outside the loop (advance-after semantics).
        if not has_input_images:
            cache_position = torch.arange(inputs_embeds.shape[1]).cuda()

        with torch.inference_mode():
            num_images = 0
            while not finished:
                runtime += 1
                if runtime > max_runtime:
                    break
                if mode == "text":
                    if has_input_images:
                        # interleaved path: is_first_token + advance-before
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
                    else:
                        # text-only path: advance-after
                        outputs = mmgpt.language_model(
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            cache_position=cache_position,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )

                        position_ids = position_ids[:, :, -1] + 1
                        position_ids = position_ids.unsqueeze(-1)
                        cache_position = cache_position[-1] + 1
                        cache_position = cache_position.unsqueeze(0)

                    # ---- shared by both paths below ----
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
                    num_images += 1
                    image_tokens_list = []
                    max_image_tokens = image_token_num_per_image  # treated as an upper bound now, not a fixed length

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
                        # the last token acts as the image-end token
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
                        ).unsqueeze(0)   # shape: [1, seq_len]
                        generated_image_tokens.append(image_tokens)
                    mode = "text"

    # decode and save the generated images only when requested
    if decode_and_save_image:
        # with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        if has_input_images:
            if is_incremental_encoding:
                decode_input_num_tokens = num_tokens
            else:
                decode_input_num_tokens = [mmgpt.vision_extra_cfg.gen_cfg.n_base] * max(len(input_images) - 1, 0)

            decode_and_save_generated_images(
                mmgpt=mmgpt,
                generated_image_tokens=generated_image_tokens,
                output_path=output_path,
                device=device,
                is_incremental_encoding=is_incremental_encoding,
                image_token_num_per_image=image_token_num_per_image,
                input_first_image_tokens=first_image_tokens,
                input_interleave_image_tokens=interleave_image_tokens,
                input_num_tokens=decode_input_num_tokens,
                save_all_images_in_chain=True,
                concat_gt_images=concat_gt_images,
                gt_input_image_paths=gt_input_image_paths,
                gt_output_image_paths=gt_output_image_paths,
                gt_image_size=gt_image_size,
            )
        elif len(generated_image_tokens) > 0:
            decode_and_save_generated_images(
                mmgpt=mmgpt,
                generated_image_tokens=generated_image_tokens,
                output_path=output_path,
                device=device,
                is_incremental_encoding=is_incremental_encoding,
                image_token_num_per_image=image_token_num_per_image,
                # the text-only path has no input-image encoding result; avoid referencing undefined variables
                input_first_image_tokens=None,
                input_interleave_image_tokens=None,
                input_num_tokens=None,
                save_all_images_in_chain=True,
                concat_gt_images=concat_gt_images,
                gt_input_image_paths=gt_input_image_paths,
                gt_output_image_paths=gt_output_image_paths,
                gt_image_size=gt_image_size,
            )

    # When not saving images (pure inference/eval), skip_special_tokens=True to match the original
    # infer behavior; when saving images (visualization), keep the special tokens.
    skip_special_tokens = not decode_and_save_image
    decoded_text = vl_chat_processor.tokenizer.decode(generated_text_tokens, skip_special_tokens=skip_special_tokens)
    generated_image_token_num_list = [int(t.shape[-1]) for t in generated_image_tokens]
    return decoded_text, generated_image_token_num_list


def load_model(model_path, extra_cfg_path):
    global vl_chat_processor
    global vl_gpt
    global tokenizer
    global extra_vision_cfg

    vl_chat_processor = ViMoProcessor.from_pretrained(model_path)
    if "to_und_token" not in vl_chat_processor.image_processor._valid_kwargs_names:
        vl_chat_processor.image_processor._valid_kwargs_names.append("to_und_token")
    tokenizer = vl_chat_processor.tokenizer
    vl_chat_processor.pad_id = tokenizer.vocab.get("<|vision_pad|>")

    config = ViMoConfig.from_pretrained(model_path)
    extra_vision_cfg = TSIMTokExtraCfg.load(extra_cfg_path)

    vl_gpt = ViMoModel.from_pretrained(
        model_path,
        config=config,
        extra_cfg=extra_vision_cfg,
        torch_dtype=torch.bfloat16
    ).cuda().eval()


def run_inference(rank, world_size, local_rank, args, dataset):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    print(f"[Rank {rank}] loading model on local gpu {local_rank} ...", flush=True)
    load_model(args.model_path, args.extra_cfg)

    tsim_router = None
    if getattr(args, 'use_tsim_router', False):
        from vimo.tsim_tok.tsim_router import TSIMRouter
        tsim_router = TSIMRouter(
            visual_extractor_ckpt_path=args.visual_extractor_ckpt,
            tsim_intervals_path=args.tsim_intervals_path,
            visual_extractor_repo_path=args.visual_extractor_repo,
            n_base=extra_vision_cfg.gen_cfg.n_base,
            device='cuda',
        )

    # shard by global rank, suitable for multi-node multi-GPU
    shard = dataset[rank::world_size]
    print(f"[Rank {rank}] processing {len(shard)} samples", flush=True)

    # When saving images (visualization), stream to jsonl, one line per sample; otherwise keep the
    # original infer behavior: accumulate into a list and dump once to .json at the end (compatible
    # with the evaluation pipeline in merge_results.py).
    if args.decode_and_save_image:
        part_path = os.path.join(args.result_dir, f"result_part_rank{rank}.jsonl")
        # streaming output: clear the old file first
        with open(part_path, "w", encoding="utf-8"):
            pass
    else:
        part_path = os.path.join(args.result_dir, f"result_part_rank{rank}.json")
        results = []

    for local_idx, sample in enumerate(tqdm(shard, disable=(rank != 0))):
        prompt = sample["input_prompt"]
        input_image_path = sample.get("input_image", [])
        gt_output_image_path = sample.get("output_image", [])

        sample_num_tokens = None
        if args.use_json_num_tokens:
            raw_num_tokens = sample.get("num_tokens", None)

            # match the training script: by default drop the first one and use the rest,
            # because the first image always uses n_base
            if raw_num_tokens is None:
                msg = f"[Rank {rank}] sample {local_idx} has no 'num_tokens' field"
                if args.strict_num_tokens:
                    raise ValueError(msg)
                else:
                    print(msg + ", fallback to random sampling", flush=True)
            else:
                if not isinstance(raw_num_tokens, list):
                    raise ValueError(
                        f"[Rank {rank}] sample {local_idx} num_tokens must be list, got {type(raw_num_tokens)}"
                    )

                # match training: drop the first element
                sample_num_tokens = raw_num_tokens[1:] if len(raw_num_tokens) > 0 else []

        # TSIM Router fallback when no precomputed num_tokens and multiple input images
        if sample_num_tokens is None and tsim_router is not None and len(input_image_path) > 1:
            sample_num_tokens = tsim_router.allocate_token_budgets(input_image_path)
        # record the original global index for later sorting
        global_idx = rank + local_idx * world_size

        output_path = os.path.join(
            args.save_dir,
            f"sample_{global_idx:06d}.png"
        )

        try:
            with torch.no_grad():
                decoded_text, generated_image_token_num_list = generate(
                    prompt,
                    input_image_path,
                    output_path,
                    vl_chat_processor,
                    vl_gpt,
                    parallel_size=1,
                    is_incremental_encoding=args.is_incremental_encoding,
                    image_token_num_per_image=args.image_token_num_per_image,
                    sample_num_tokens=sample_num_tokens,
                    max_runtime=args.max_runtime,
                    decode_and_save_image=args.decode_and_save_image,
                    concat_gt_images=args.concat_gt_images,
                    gt_input_image_paths=input_image_path,
                    gt_output_image_paths=gt_output_image_path,
                    gt_image_size=args.gt_image_size,
                )
        except Exception as e:
            print(f"[Rank {rank}] error on sample {global_idx}: {e}", flush=True)
            decoded_text = ""
            generated_image_token_num_list = []

        item = dict(sample)
        item["_global_idx"] = global_idx
        item["model_output_text"] = decoded_text
        item["model_output_image"] = output_path

        if args.decode_and_save_image:
            item["model_output_image_token_num_list"] = generated_image_token_num_list
            # stream writes to avoid rewriting the whole result file each time (which gets slower over time)
            with open(part_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            results.append(item)

    if not args.decode_and_save_image:
        with open(part_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Rank {rank}] finished, saved to {part_path}", flush=True)


def merge_json(world_size, result_dir, final_json):
    print("[Rank 0] Merging results...", flush=True)

    final = []
    for rank in range(world_size):
        part_path_jsonl = os.path.join(result_dir, f"result_part_rank{rank}.jsonl")
        part_path_json = os.path.join(result_dir, f"result_part_rank{rank}.json")

        if os.path.exists(part_path_jsonl):
            with open(part_path_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    final.append(json.loads(line))
        elif os.path.exists(part_path_json):
            with open(part_path_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            final.extend(data)
        else:
            print(f"[Rank 0] warning: missing {part_path_jsonl} and {part_path_json}", flush=True)

    # restore the original order
    final.sort(key=lambda x: x["_global_idx"])
    for x in final:
        x.pop("_global_idx", None)

    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"[Rank 0] Final JSON saved to: {final_json}", flush=True)


def main():
    args = parse_args()

    global MIN_PIXELS, MAX_PIXELS
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels

    rank, world_size, local_rank = setup_distributed()

    print(f"[Rank {rank}] loading model on local gpu {local_rank} ...", flush=True)
    if is_main_process():
        print(f"Total world size: {world_size}", flush=True)

    with open(args.json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.shuffle_dataset:
        if args.shuffle_seed >= 0:
            shuffle_seed = int(args.shuffle_seed)
        else:
            if rank == 0:
                shuffle_seed = int.from_bytes(os.urandom(8), byteorder="big") % (2**31)
            else:
                shuffle_seed = 0

            seed_tensor = torch.tensor([shuffle_seed], dtype=torch.long, device=f"cuda:{local_rank}")
            dist.broadcast(seed_tensor, src=0)
            shuffle_seed = int(seed_tensor.item())

        rng = random.Random(shuffle_seed)
        rng.shuffle(dataset)
        print(f"[Rank {rank}] dataset shuffled with seed={shuffle_seed}", flush=True)

    if is_main_process():
        print(f"Total samples: {len(dataset)}", flush=True)

    try:
        run_inference(rank, world_size, local_rank, args, dataset)

        # synchronize only after all ranks finish normally
        dist.barrier()


    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
