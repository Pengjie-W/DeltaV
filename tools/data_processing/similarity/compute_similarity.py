import os
import json
import ast
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms


# ----------------------------
# DINOv2 loader
# ----------------------------
def load_dinov2_extractor(visual_extractor_repo_path: str, visual_extractor_ckpt_path: str, device: str):
    """Load the frozen DINOv2 ViT-B/14 used to compute temporal similarity (TSIM).

    Empty ``visual_extractor_repo_path`` -> torch.hub auto-downloads the code and
    pretrained weights from GitHub (cached under ``~/.cache/torch/hub/``).
    A non-empty path loads a local clone (offline/intranet) plus local weights.
    """
    if visual_extractor_repo_path:
        # offline/intranet: local clone + local weights
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
        # default: torch.hub auto-downloads the code + pretrained weights
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            trust_repo=True,
        )
    model.eval().to(device).half()
    return model


# ----------------------------
# preprocess
# ----------------------------
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2

    return Image.fromarray(
        arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    )


dinov2_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def load_image_threadsafe(path):
    img = Image.open(path).convert("RGB")
    img = center_crop_arr(img, 224)
    return dinov2_transform(img)


# ----------------------------
# distributed utils
# ----------------------------
def init_distributed():
    if "RANK" not in os.environ:
        raise RuntimeError("This script must be launched with torchrun.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank


def split_by_rank(data, rank, world_size):
    return data[rank::world_size]


def load_input_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"input json must be a list, but got {type(data)}")

    return data


# ----------------------------
# main worker logic
# ----------------------------
@torch.no_grad()
def process_rank(keys, ckpt_path, device, rank, num_workers=8, repo_path=""):
    model = load_dinov2_extractor(
        visual_extractor_repo_path=repo_path,
        visual_extractor_ckpt_path=ckpt_path,
        device=device,
    )
    result = {}

    pbar = tqdm(keys, disable=(rank != 0), desc=f"Rank {rank}")
    for image_paths in pbar:
        try:
            if isinstance(image_paths, str):
                image_paths = ast.literal_eval(image_paths)

            if not isinstance(image_paths, list):
                continue

            if len(image_paths) == 1:
                result[str(image_paths)] = [[1.0]]
                continue
        except Exception:
            continue

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                images = list(executor.map(load_image_threadsafe, image_paths))

            if len(images) == 0:
                continue

            if len(images) == 1:
                result[str(image_paths)] = [[1.0]]
                continue

            images_tensor = torch.stack(images).to(device, non_blocking=True).half()

            feats = model.forward_features(images_tensor)
            patch_tokens = feats["x_norm_patchtokens"]
            F_all = patch_tokens.mean(dim=1)

            F_all = F.normalize(F_all, dim=-1)
            sim_matrix = torch.matmul(F_all, F_all.T)

            sims = []
            T = F_all.size(0)
            for t in range(T):
                if t == 0:
                    sims.append([1.0])
                else:
                    sims.append(sim_matrix[t, :t].float().cpu().tolist())

            result[str(image_paths)] = sims

        except Exception as e:
            print(f"[Rank {rank}] failed on sample: {e}", flush=True)
            continue

    return result


# ----------------------------
# Main entry (multi-node, multi-GPU)
# ----------------------------
def process_multi_node(json_path, output_path, ckpt_path, num_workers=8, repo_path=""):
    rank, world_size, local_rank = init_distributed()
    device = f"cuda:{local_rank}"

    data = load_input_data(json_path)
    local_keys = split_by_rank(data, rank, world_size)

    print(
        f"[Rank {rank}] world_size={world_size}, local_rank={local_rank}, "
        f"num_samples={len(local_keys)}",
        flush=True
    )

    local_result = process_rank(local_keys, ckpt_path, device, rank, num_workers=num_workers, repo_path=repo_path)

    output_path = Path(output_path)
    tmp_dir = output_path.parent / (output_path.stem + "_parts")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part_file = tmp_dir / f"part_rank{rank:05d}.json"
    with open(part_file, "w") as f:
        json.dump(local_result, f, indent=2, ensure_ascii=False)

    print(f"[Rank {rank}] wrote {part_file}", flush=True)

    dist.barrier()

    if rank == 0:
        final_result = {}
        for r in range(world_size):
            pf = tmp_dir / f"part_rank{r:05d}.json"
            with open(pf, "r") as f:
                part = json.load(f)
            final_result.update(part)

        with open(output_path, "w") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

        print(f"[Rank 0] final merged json saved to: {output_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument(
        "--visual_extractor_ckpt",
        type=str,
        default="weights/dinov2/dinov2_vitb14.pth",
        help="Local DINOv2 weights; only used by the offline branch (when --visual_extractor_repo is non-empty)",
    )
    parser.add_argument(
        "--visual_extractor_repo",
        type=str,
        default="",
        help="Path to a local DINOv2 clone (offline/intranet); leave empty to let torch.hub download it",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    process_multi_node(
        json_path=args.input_json,
        output_path=args.output_json,
        ckpt_path=args.visual_extractor_ckpt,
        num_workers=args.num_workers,
        repo_path=args.visual_extractor_repo,
    )


if __name__ == "__main__":
    main()