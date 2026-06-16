import numpy as np
import argparse
import os.path as osp
from tqdm import tqdm
import re
from omegaconf import OmegaConf
import torch, os, time, warnings
from torch.utils.tensorboard import SummaryWriter
from datetime import timedelta
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vimo.tsim_tok.training.engine.logger import create_logger
from vimo.tsim_tok.training.tokenizer.vq_loss import VQLoss
from vimo.tsim_tok.training.engine.ema import update_ema, requires_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from vimo.tsim_tok.training.engine.distributed import init_distributed_mode
from vimo.tsim_tok.training.engine.lr_scheduler import build_scheduler
from vimo.tsim_tok.training.data.dataset import ImageNetListDataset as ImageNetDataset
from vimo.tsim_tok.training.engine.misc import is_main_process, all_reduce_mean, concat_all_gather, concat_all_gather_variable_1d, get_world_size, get_rank
import torch.nn as nn
from transformers import PretrainedConfig, AutoConfig
from safetensors.torch import load_file
import random
import json
import torch.nn.functional as F
import math
def import_tsim_tok(args):
    # Open-source release ships only the TSIM-Tok ("ours") path.
    from vimo.tsim_tok.modeling_tsim_tok import TSIMTokExtraCfg as ExtraVisionCfg, TSIMTok
    from inference.eval_recon import val
    return ExtraVisionCfg, TSIMTok, val

def load_vision_config(base_model_path):
    """Read vision_config from the Qwen3-VL base model dir.

    Returns a PretrainedConfig identical (field-for-field) to the previously
    hard-coded dict. Prefers AutoConfig; falls back to parsing config.json.
    """
    try:
        base_cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        vision_config = base_cfg.vision_config
        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig(**vision_config)
        return vision_config
    except Exception as e:
        print(f"[load_vision_config] AutoConfig failed ({e}); falling back to config.json")
        with open(osp.join(base_model_path, "config.json"), "r") as f:
            cfg = json.load(f)
        return PretrainedConfig(**cfg["vision_config"])

def extract_epoch(path: str):
    """
    Extract the training progress from a checkpoint filename; supports both epoch / step naming:
      vit_vqgan_epoch_4.pt              -> ('epoch', 4)
      vit_vqgan_final_epoch_199.pt      -> ('epoch', 199)
      vit_vqgan_final_step_1.pt         -> ('step', 1)
      vit_vqgan_epoch_3_step_5000.pt    -> ('step', 5000)   # the trailing step wins
    Returns (tag, number), where tag is 'epoch' or 'step'.
    """
    filename = os.path.basename(path)  # keep only the filename
    # match step first: for names ending in epoch_E_step_N.pt the trailing step takes precedence
    match = re.search(r'_step_(\d+)\.pt$', filename)
    if match:
        return 'step', int(match.group(1))
    match = re.search(r'_epoch_(\d+)\.pt$', filename)
    if match:
        return 'epoch', int(match.group(1))
    # filename has no epoch/step marker (e.g. weights/tsim_tok/tsim_tok.pt);
    # this value is only used for log naming, so fall back instead of raising.
    print(f"Cannot extract epoch/step from: {path}; falling back to ('step', 0)")
    return 'step', 0

def interleave_flatten_collate(batch):
    """
    batch:
        List[
            (
                target_imgs: List[Tensor],
                flatten_patches: List[Tensor],
                image_grid_thw: List[Tensor],
                num_images: int,
                img_paths: Optional[List[str]],
                num_tokens: List[int]  # the added 6th element
            )
        ]
    """

    all_imgs = []
    all_patches = []
    all_grids = []
    all_num_images = []
    all_img_paths = []
    all_num_tokens = []

    # inspect a single sample length to determine the input structure
    sample_len = len(batch[0])
    has_paths = sample_len >= 5
    has_tokens = sample_len == 6

    for sample in batch:
        # unpack dynamically based on the tuple length
        if has_tokens:
            imgs_list, patches_list, grids_list, n, paths_list, tokens_list = sample
            all_num_tokens.extend(tokens_list[1:])
            all_img_paths.extend(paths_list)
        elif sample_len == 5:
            imgs_list, patches_list, grids_list, n, paths_list = sample
            all_img_paths.extend(paths_list)
        else:
            imgs_list, patches_list, grids_list, n = sample

        all_num_images.append(int(n))
        all_imgs.extend(imgs_list)
        all_patches.extend(patches_list)
        all_grids.extend(grids_list)

    # stack tensors
    imgs = torch.stack(all_imgs, dim=0)               # [sum(num_images), C, H, W]
    flatten_patches = torch.stack(all_patches, dim=0) # [sum(num_images), N, D]
    image_grid_thw = torch.stack(all_grids, dim=0)    # [sum(num_images), 3]
    num_images = torch.tensor(all_num_images, dtype=torch.long)  # [B]

    # build the return tuple
    return_values = (imgs, flatten_patches, image_grid_thw, num_images)

    if has_paths:
        return_values += (all_img_paths,)
    
    if has_tokens:
        return_values += (all_num_tokens,)

    return return_values

def get_args_parser():

    parser = argparse.ArgumentParser('VFMTok evaluation', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    #* Dataset parameters
    parser.add_argument('--output_dir', default='./recons',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output/logs/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_false',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    #* Feature genration
    parser.add_argument('--evaluate', action='store_true', help="perform only evaluation")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-slots-embed-dim", type=int, default=8, help="codebook dimension for queries quantization")
    parser.add_argument("--image-size", type=int, choices=[128, 192, 256, 336, 384, 512], default=256)
    parser.add_argument("--transformer-config-file", type=str, default='configs/vit_transformer.yaml',)
    parser.add_argument("--z-channels", type=int, default=512,)
    parser.add_argument("--val-data", type=str, required=True)
    
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument('--save_dir', default=None,)
    parser.add_argument('--is_save', action='store_true')
    parser.add_argument("--extra_cfg", type=str, default="configs/vision_extra.json",)
    parser.add_argument("--base_model_path", type=str, default=os.environ.get("BASE_MODEL_PATH", None),
                        help="Qwen3-VL-2B base dir: provides vision_config (config.json) and visual init weights (model.safetensors)")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--val_padding", action='store_true', default=False)
    parser.add_argument("--val_save", action='store_true', default=False)
    parser.add_argument("--train_baseline", action='store_true', default=False)
    parser.add_argument("--val_single", action='store_true', default=False)
    parser.add_argument("--val_n_delta", type=int, nargs="+", default=[144],help="List of integer deltas for validation")
    parser.add_argument("--baseline", action='store_true', default=False)
    parser.add_argument("--model_baseline", action='store_true', default=False)
    parser.add_argument("--val_bs", type=int, default=4)
    parser.add_argument("--ddp_timeout_min", type=int, default=240)
    parser.add_argument("--use_token", action='store_true', default=False)
    parser.add_argument(
        "--metric_image_select",
        type=str,
        default="all",
        choices=["all", "first", "exclude_first"],
        help="which generated images to use for metric computation"
    )
    return parser

def val_model(ema, val_loader, device, args, write_val, checkpoint_dir, logger, writer, global_step, global_epoch, val, batch_size=4, OBJ_GROUP=None, val_n_delta=None, save_image=False):
    if args.val_single:
        val_result=val(ema, val_loader, device, args, write_val, checkpoint_dir, batch_size=batch_size,save_image=save_image)
    else:
        val_result=val(ema, val_loader, device, args, write_val, checkpoint_dir, batch_size=batch_size, OBJ_GROUP=OBJ_GROUP, val_n_delta=val_n_delta,save_image=save_image)
    if is_main_process():
        FID, IS, usage_img, usage_slot, psnr_val_rgb, ssim_val_rgb = val_result
        msg = (
            f"{write_val}"
            f"FID={FID:.4f} IS={IS:.4f} "
            f"img_usage={usage_img:.4f} slot_usage={usage_slot:.4f} "
            f"PSNR={psnr_val_rgb:.4f} SSIM={ssim_val_rgb:.4f}"
        )
        # tqdm-safe output to the screen
        tqdm.write(msg)
        # write to the log file (for traceability)
        logger.info(msg)
        if writer is not None and val_n_delta is None:

            writer.add_scalar("val_epoch/FID", FID, global_epoch)
            writer.add_scalar("val_epoch/IS", IS, global_epoch)
            writer.add_scalar("val_epoch/img_usage", usage_img, global_epoch)
            writer.add_scalar("val_epoch/slot_usage", usage_slot, global_epoch)
            writer.add_scalar("val_epoch/PSNR", psnr_val_rgb, global_epoch)
            writer.add_scalar("val_epoch/SSIM", ssim_val_rgb, global_epoch)
        elif writer is not None:

            writer.add_scalar(f"val_epoch_{val_n_delta[0]}/FID", FID, global_epoch)
            writer.add_scalar(f"val_epoch_{val_n_delta[0]}/IS", IS, global_epoch)
            writer.add_scalar(f"val_epoch_{val_n_delta[0]}/img_usage", usage_img, global_epoch)
            writer.add_scalar(f"val_epoch_{val_n_delta[0]}/slot_usage", usage_slot, global_epoch)
            writer.add_scalar(f"val_epoch_{val_n_delta[0]}/PSNR", psnr_val_rgb, global_epoch)
            writer.add_scalar(f"val_epoch_{val_n_delta[0]}/SSIM", ssim_val_rgb, global_epoch)


def main(args):
    assert args.base_model_path, "--base_model_path (or env BASE_MODEL_PATH) is required"
    ExtraVisionCfg, TSIMTok, val = import_tsim_tok(args)
    # Setup DDP:
    init_distributed_mode(args)
    print('job dir: {}'.format(osp.dirname(osp.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    num_tasks = get_world_size()
    global_rank = get_rank()
    print(f'num_tasks: {num_tasks}')
    #* Setup an experiment folder:
    if is_main_process():
        #* Make results folder (holds all experiment subfolders)
        os.makedirs(args.results_dir, exist_ok=True)  
        checkpoint_dir = osp.join(args.results_dir, 'model_dump')
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger_dir = osp.join(args.results_dir, 'logs')
        os.makedirs(logger_dir, exist_ok=True)
        logger = create_logger(logger_dir)
        logger.info(f"Experiment directory created at {args.results_dir}")
    else:
        logger = create_logger(None)
    writer = None
    if is_main_process():
        tb_dir = os.getenv("TENSORBOARD_LOG_PATH")
        if tb_dir is None or len(tb_dir) == 0:
            logger.warning("TENSORBOARD_LOG_PATH is empty. TensorBoard will be disabled.")
        else:
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"TensorBoard enabled. log_dir={tb_dir}")
    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={global_rank}, seed={seed}, world_size={dist.get_world_size()}.")
    # vision_config is read from the base model dir (config.json); it is field-for-field
    # identical to the Qwen3-VL-2B vision_config.
    vision_config = load_vision_config(args.base_model_path)
    extra_cfg = ExtraVisionCfg.load(args.extra_cfg) if args.extra_cfg else ExtraVisionCfg()
    extra_cfg.gen_cfg.image_size=args.image_size
    extra_cfg.gen_cfg.codebook_size=args.codebook_size
    vq_model = TSIMTok(vision_config, extra_cfg=extra_cfg)
    state_dict = load_file(osp.join(args.base_model_path, "model.safetensors"))
    visual_keys = [key for key in state_dict.keys() if key.startswith("model.visual")]
    frozen_keys = [
        key for key in visual_keys
        if key.startswith("model.visual.patch_embed")
        or key.startswith("model.visual.pos_embed")
        or key.startswith("model.visual.rotary_pos_emb")
        or key.startswith("model.visual.blocks")
    ]
    # extract the frozen key/value pairs
    frozen_state_dict = {key: state_dict[key] for key in frozen_keys}
    frozen_visual_keys = [key for key in frozen_state_dict.keys() if key.startswith("model.visual")]
    new_state_dict = {}
    for key in frozen_visual_keys:
        # strip the "model.visual." prefix
        new_key = key.replace("model.visual.", "")
        new_state_dict[new_key] = frozen_state_dict[key]

    missing_keys, unexpected_keys = vq_model.backbone.load_state_dict(new_state_dict, strict=False)
    print(f"missing_keys:{missing_keys}")
    print(f"unexpected_keys:{unexpected_keys}")
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    vq_model = vq_model.to(device)

    val_dataset = ImageNetDataset(
        json_path=args.val_data,
        image_size=(args.image_size, args.image_size),
        is_train=False,
        is_list = not args.baseline,
        max_samples=args.max_samples,
        val_padding=args.val_padding,
        return_num_tokens=args.use_token
    )
    val_sampler = DistributedSampler(val_dataset, rank=global_rank, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=None if args.baseline else interleave_flatten_collate
    )


    checkpoint = torch.load(args.vq_ckpt, weights_only=False)
    # extract the weights
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # DDP or plain model
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    if isinstance(model_weight, nn.Module):
        model_weight = model_weight.state_dict()
    # handle the DDP "module." prefix (optional but recommended)
    if isinstance(model_weight, dict) and any(k.startswith("module.") for k in model_weight.keys()):
        model_weight = {k[len("module."):]: v for k, v in model_weight.items()}
    missing, unexpected = vq_model.load_state_dict(model_weight, strict=False)
    print("missing",missing)
    print("unexpected",unexpected)
    # filter out missing keys that contain "backbone"
    missing = [k for k in missing if "backbone" not in k]

    print("Filtered missing keys:", missing)

    # wrap the models (vq_model and vq_loss) in DistributedDataParallel
    vq_model.eval()
    OBJ_GROUP = None
    if dist.is_available() and dist.is_initialized():
        # dedicated group for communicating python objects (CPU)
        OBJ_GROUP = dist.new_group(
            backend="gloo",
            timeout=timedelta(minutes=args.ddp_timeout_min)
        )
    ckpt_tag, ckpt_num = extract_epoch(args.vq_ckpt)
    print(args.val_save)
    val_model(vq_model, val_loader, device, args, f"{ckpt_tag}_{ckpt_num}:val_loader", args.results_dir, logger, writer=writer, global_step=None, global_epoch=ckpt_num, val=val,  batch_size=args.val_bs, OBJ_GROUP=OBJ_GROUP,save_image=args.val_save)
   

if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()

    main(args)