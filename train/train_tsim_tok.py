import numpy as np
import argparse, pdb
from glob import glob
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from omegaconf import OmegaConf
import torch, os, time, warnings
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
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
from vimo.tsim_tok.training.data.dataset import ImageNetListDataset as ImageNetDataset, MultiSourceBatchSampler, BucketedTokenBudgetBatchSampler_len
from vimo.tsim_tok.training.engine.misc import is_main_process, all_reduce_mean, concat_all_gather, concat_all_gather_variable_1d, get_world_size
import torch.nn as nn
from transformers import PretrainedConfig, AutoConfig
from safetensors.torch import load_file
import random
import json
from datetime import timedelta
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


def build_checkpoint(
    vq_model,
    vq_loss,
    optimizer,
    optimizer_disc,
    train_steps,
    args,
    optimizer_backbone=None,
    ema=None,
    epoch=None,
    step_in_epoch=None,
):
    """
    Build the checkpoint dict (does not write to disk).
    Note: vq_model/vq_loss may be DDP-wrapped; they are unwrapped internally.
    """
    model_obj = vq_model.module if isinstance(vq_model, DDP) else vq_model
    vqloss_obj = vq_loss.module if isinstance(vq_loss, DDP) else vq_loss

    if args.train_stage == 1:
        model_weight = filter_frozen_visual_keys(model_obj.state_dict())[0]
    else:
        model_weight = model_obj.state_dict()

    ckpt = {
        "model": model_weight,
        "optimizer": optimizer.state_dict(),
        "optimizer_backbone": optimizer_backbone.state_dict() if (optimizer_backbone is not None and args.train_stage == 2) else None,
        "discriminator": vqloss_obj.discriminator.state_dict(),
        "optimizer_disc": optimizer_disc.state_dict(),
        "steps": train_steps,
        "args": args,
    }

    if epoch is not None:
        ckpt["epoch"] = epoch
    if step_in_epoch is not None:
        ckpt["step_in_epoch"] = step_in_epoch

    if args.ema and (ema is not None):
        if args.train_stage == 1:
            ema_weight = filter_frozen_visual_keys(ema.state_dict())[0]
        else:
            ema_weight = ema.state_dict()
        ckpt["ema"] = ema_weight

    # Save RNG state so resumed runs stay as consistent as possible.
    ckpt["rng_state"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }

    return ckpt

def save_checkpoint(
    ckpt_path: str,
    vq_model,
    vq_loss,
    optimizer,
    optimizer_disc,
    train_steps,
    args,
    logger=None,
    optimizer_backbone=None,
    ema=None,
    epoch=None,
    step_in_epoch=None,
):
    checkpoint = build_checkpoint(
        vq_model=vq_model,
        vq_loss=vq_loss,
        optimizer=optimizer,
        optimizer_disc=optimizer_disc,
        optimizer_backbone=optimizer_backbone,
        ema=ema,
        train_steps=train_steps,
        args=args,
        epoch=epoch,
        step_in_epoch=step_in_epoch,
    )
    torch.save(checkpoint, ckpt_path)
    if logger is not None:
        logger.info(f"Saved checkpoint to {ckpt_path}")
    else:
        print(f"Saved checkpoint to {ckpt_path}")

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
                num_tokens: List[int]  # optional 6th element
            )
        ]
    """

    all_imgs = []
    all_patches = []
    all_grids = []
    all_num_images = []
    all_img_paths = []
    all_num_tokens = []

    # Inspect a single sample's length to determine the input structure.
    sample_len = len(batch[0])
    has_paths = sample_len >= 5
    has_tokens = sample_len == 6

    for sample in batch:
        # Unpack dynamically based on length.
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

    # Stack tensors.
    imgs = torch.stack(all_imgs, dim=0)               # [sum(num_images), C, H, W]
    flatten_patches = torch.stack(all_patches, dim=0) # [sum(num_images), N, D]
    image_grid_thw = torch.stack(all_grids, dim=0)    # [sum(num_images), 3]
    num_images = torch.tensor(all_num_images, dtype=torch.long)  # [B]

    # Build the return tuple.
    return_values = (imgs, flatten_patches, image_grid_thw, num_images)

    if has_paths:
        return_values += (all_img_paths,)

    if has_tokens:
        return_values += (all_num_tokens,)

    return return_values


def unwrap_model(m):
    return m.module if isinstance(m, DDP) else m

def count_params_frozen_trainable(model):
    """
    Return total / trainable / frozen parameter counts (by numel).
    Note: counts are based only on requires_grad; whether a param is in the
    optimizer is not considered.
    """
    model = unwrap_model(model)
    total = 0
    trainable = 0
    frozen = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        else:
            frozen += n

    return total, trainable, frozen

def log_param_status(model, logger, prefix="vq_model"):
    total, trainable, frozen = count_params_frozen_trainable(model)
    pct_train = 100.0 * trainable / max(total, 1)
    pct_frozen = 100.0 * frozen / max(total, 1)

    msg = (f"[{prefix}] total={total:,} | trainable={trainable:,} ({pct_train:.2f}%) "
           f"| frozen={frozen:,} ({pct_frozen:.2f}%)")
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

    # Optional: break down counts by module name (e.g. backbone vs others).
    model = unwrap_model(model)
    bb_total = bb_train = bb_frozen = 0
    other_total = other_train = other_frozen = 0
    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith("backbone."):
            bb_total += n
            if p.requires_grad: bb_train += n
            else: bb_frozen += n
        else:
            other_total += n
            if p.requires_grad: other_train += n
            else: other_frozen += n

    msg2 = (f"[{prefix}] backbone: total={bb_total:,} trainable={bb_train:,} frozen={bb_frozen:,} | "
            f"other: total={other_total:,} trainable={other_train:,} frozen={other_frozen:,}")
    if logger is not None:
        logger.info(msg2)
    else:
        print(msg2)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def val_model_multi(
    ema,
    val_loaders,
    val_names,
    device,
    args,
    write_val,
    checkpoint_dir,
    logger,
    writer,
    global_step,
    global_epoch,
    val,
    batch_size=4,
    OBJ_GROUP=None,
    val_n_delta=None,
):
    for i, (val_loader, val_name) in enumerate(zip(val_loaders, val_names)):
        # Give each validation set its own tag to make logs easier to distinguish.
        tag_name = osp.splitext(osp.basename(val_name))[0]
        write_val_i = f"{write_val} [{tag_name}]"

        val_result = val(
            ema,
            val_loader,
            device,
            args,
            write_val_i,
            checkpoint_dir,
            batch_size=batch_size,
            OBJ_GROUP=OBJ_GROUP,
            val_n_delta=val_n_delta
        )

        if is_main_process():
            FID, IS, usage_img, usage_slot, psnr_val_rgb, ssim_val_rgb = val_result

            msg = (
                f"{write_val_i} "
                f"FID={FID:.4f} IS={IS:.4f} "
                f"img_usage={usage_img:.4f} slot_usage={usage_slot:.4f} "
                f"PSNR={psnr_val_rgb:.4f} SSIM={ssim_val_rgb:.4f}"
            )
            tqdm.write(msg)
            logger.info(msg)

            if writer is not None:
                prefix = f"val/{tag_name}"
                if val_n_delta is not None:
                    prefix = f"{prefix}_delta_{val_n_delta[0]}"

                writer.add_scalar(f"{prefix}/step/FID", FID, global_step)
                writer.add_scalar(f"{prefix}/step/IS", IS, global_step)
                writer.add_scalar(f"{prefix}/step/img_usage", usage_img, global_step)
                writer.add_scalar(f"{prefix}/step/slot_usage", usage_slot, global_step)
                writer.add_scalar(f"{prefix}/step/PSNR", psnr_val_rgb, global_step)
                writer.add_scalar(f"{prefix}/step/SSIM", ssim_val_rgb, global_step)

                writer.add_scalar(f"{prefix}/epoch/FID", FID, global_epoch)
                writer.add_scalar(f"{prefix}/epoch/IS", IS, global_epoch)
                writer.add_scalar(f"{prefix}/epoch/img_usage", usage_img, global_epoch)
                writer.add_scalar(f"{prefix}/epoch/slot_usage", usage_slot, global_epoch)
                writer.add_scalar(f"{prefix}/epoch/PSNR", psnr_val_rgb, global_epoch)
                writer.add_scalar(f"{prefix}/epoch/SSIM", ssim_val_rgb, global_epoch)

def remap_state_dict_keys(state_dict, key_map, prefix_map=None):
    """
    key_map: exact key replacement {"old_key": "new_key"}
    prefix_map: prefix replacement {"old_prefix.": "new_prefix."}
    """
    if state_dict is None:
        return None

    new_sd = {}
    for k, v in state_dict.items():
        new_k = k

        # 1) Exact key-name replacement.
        if key_map and new_k in key_map:
            new_k = key_map[new_k]

        # 2) Prefix replacement (e.g. module path).
        if prefix_map:
            for old_p, new_p in prefix_map.items():
                if new_k.startswith(old_p):
                    new_k = new_p + new_k[len(old_p):]
                    break

        new_sd[new_k] = v

    return new_sd

def filter_frozen_visual_keys(state_dict: dict):
    """
    Remove frozen visual encoder params from a state_dict before saving.
    Covers common prefixes: module., _orig_mod., visual., etc.
    """
    drop_keywords = [
        "backbone.",
    ]

    # Possible hierarchy prefixes that may appear in keys (e.g. visual.* or
    # encoder.visual.* depending on the project layout).
    possible_prefixes = [
        "",  # keys are directly backbone.*
    ]

    drop_prefixes = []
    for p in possible_prefixes:
        for kw in drop_keywords:
            drop_prefixes.append(p + kw)

    new_sd = {}
    removed = []
    for k, v in state_dict.items():
        if any(k.startswith(dp) for dp in drop_prefixes):
            removed.append(k)
            continue
        new_sd[k] = v
    return new_sd, removed

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.base_model_path, "--base_model_path (or env BASE_MODEL_PATH) is required"
    ExtraVisionCfg, TSIMTok, val = import_tsim_tok(args)
    # Setup DDP:
    init_distributed_mode(args)
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    for gb in args.global_batch_size:
        assert gb % dist.get_world_size() == 0, \
            f"Each global batch size must be divisible by world size, but got {gb} and world_size={dist.get_world_size()}"
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    np.random.seed(os.getpid())
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
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    #* Create and load model
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
    # Extract the frozen key/value pairs.
    frozen_state_dict = {key: state_dict[key] for key in frozen_keys}
    frozen_visual_keys = [key for key in frozen_state_dict.keys() if key.startswith("model.visual")]
    new_state_dict = {}
    for key in frozen_visual_keys:
        # Strip the "model.visual." prefix.
        new_key = key.replace("model.visual.", "")
        new_state_dict[new_key] = frozen_state_dict[key]

    missing_keys, unexpected_keys = vq_model.backbone.load_state_dict(new_state_dict, strict=False)
    print(f"missing_keys:{missing_keys}")
    print(f"unexpected_keys:{unexpected_keys}")
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)  # EMA weights are fully frozen.
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)

    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
        config=vision_config,
        stage=args.train_stage,
        use_warmup=args.disc_use_warmup,

    ).to(device)
    if args.train_stage==2:
        missing_keys, unexpected_keys = vq_loss.DistillLoss.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"missing_keys:{missing_keys}")
        print(f"unexpected_keys:{unexpected_keys}")

    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    #* Initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    
    #* Setup optimizer
    base_model = unwrap_model(vq_model)  # safe even if vq_model is not yet DDP-wrapped

    # Split parameters by name (backbone / others).
    backbone_params = []
    other_params = []
    for n, p in base_model.named_parameters():
        if n.startswith("backbone."):
            backbone_params.append(p)
        else:
            other_params.append(p)

    # Stage control: freeze / unfreeze + lr setup.
    if args.train_stage == 1:
        base_model.backbone.freeze()
        for p in backbone_params:
            p.requires_grad_(False)
        backbone_lr = 0.0
    else:
        # Do not freeze the backbone (so it can be trained).
        # Use backbone.unfreeze() if available; otherwise set requires_grad_(True).
        if hasattr(base_model.backbone, "unfreeze"):
            base_model.backbone.unfreeze()
        for p in backbone_params:
            p.requires_grad_(True)
        backbone_lr = args.lr * args.backbone_lr_mult

    optimizer = torch.optim.Adam(other_params, lr=args.lr, betas=(args.beta1, args.beta2))
    if args.train_stage == 2:
        optimizer_backbone = torch.optim.Adam(backbone_params, lr=args.lr * args.backbone_lr_mult, betas=(args.beta1, args.beta2))
    else:
        optimizer_backbone = None

    min_lr = args.min_lr if args.min_lr is not None else args.lr

    disc_lr = args.disc_lr if args.disc_lr is not None else args.lr
    disc_min_lr = args.disc_min_lr if args.disc_min_lr is not None else disc_lr
    optimizer_disc = torch.optim.Adam(
        vq_loss.discriminator.parameters(),
        lr=disc_lr,
        betas=(args.beta1, args.beta2)
    )
    logger.info(f"Generator lr={args.lr}, Backbone lr={args.lr * args.backbone_lr_mult if args.train_stage == 2 else 0}, Discriminator lr={disc_lr}")

    num_datasets = len(args.data_paths)

    assert len(args.data_ratios) == num_datasets, \
        f"data_ratios len={len(args.data_ratios)} != num_datasets={num_datasets}"

    assert len(args.global_batch_size) == num_datasets, \
        f"global_batch_size len={len(args.global_batch_size)} != num_datasets={num_datasets}"

    assert len(args.bucket_max_counts) == num_datasets, \
        f"bucket_max_counts len={len(args.bucket_max_counts)} != num_datasets={num_datasets}"

    if args.bucket_widths is not None:
        assert len(args.bucket_widths) == num_datasets, \
            f"bucket_widths len={len(args.bucket_widths)} != num_datasets={num_datasets}"

    if args.dataset_seeds is not None:
        assert len(args.dataset_seeds) == num_datasets, \
            f"dataset_seeds len={len(args.dataset_seeds)} != num_datasets={num_datasets}"
    from torch.utils.data import ConcatDataset

    datasets = []
    samplers = []

    num_datasets = len(args.data_paths)

    for i in range(num_datasets):
        ds = ImageNetDataset(
            json_path=args.data_paths[i],
            image_size=(args.image_size, args.image_size),
            is_train=True,
            is_list=not args.baseline,
            max_samples=args.max_samples,
            use_pad=args.use_pad,
            train_center_padding=args.train_center_padding,
            train_center_crop=args.train_center_crop,
            return_num_tokens=bool(args.return_num_tokens),
        )
        datasets.append(ds)

    train_dataset = ConcatDataset(datasets)

    offsets = []
    cur = 0
    for ds in datasets:
        offsets.append(cur)
        cur += len(ds)

    for i in range(num_datasets):
        sampler = BucketedTokenBudgetBatchSampler_len(
            dataset=datasets[i],
            target_budget=int(args.global_batch_size[i] // dist.get_world_size()),
            bucket_width=args.bucket_widths[i] if args.bucket_widths is not None else args.bucket_width,
            shuffle=True,
            drop_last=True,
            seed=args.dataset_seeds[i] if args.dataset_seeds is not None else (args.global_seed + i * 1000),
            bucket_max_counts=int(args.bucket_max_counts[i] // dist.get_world_size()),
            sort_within_bucket=args.sort_within_bucket,
            rank=rank,
            world_size=dist.get_world_size(),
        )
        samplers.append(sampler)

    batch_sampler = MultiSourceBatchSampler(
        samplers=samplers,
        offsets=offsets,
        ratios=args.data_ratios,
        seed=args.global_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=interleave_flatten_collate,
        worker_init_fn=worker_init_fn,
    )
    batch_sampler.set_epoch(0)
    logger.info(f"Dataset contains {len(train_dataset):,} samples from {len(args.data_paths)} datasets: {args.data_paths}")

    # ===== build multiple val loaders =====
    val_datasets = []
    val_samplers = []
    val_loaders = []

    for val_json in args.val_data_paths:
        val_dataset = ImageNetDataset(
            json_path=val_json,
            image_size=(args.image_size, args.image_size),
            is_train=False,
            is_list=not args.baseline,
            max_samples=args.max_samples,
            val_padding=args.val_padding,
            return_num_tokens=bool(args.return_num_tokens)
        )
        val_sampler = DistributedSampler(
            val_dataset,
            rank=rank,
            num_replicas=dist.get_world_size(),
            shuffle=False
        )
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

        val_datasets.append(val_dataset)
        val_samplers.append(val_sampler)
        val_loaders.append(val_loader)

    logger.info(f"Built {len(val_loaders)} validation loaders: {args.val_data_paths}")

   
    # warmup_lr_init, warmup_steps = args.warming_up_init_lr, int(args.warming_up_epochs * len(train_loader))
    warmup_lr_init, warmup_steps = args.warming_up_init_lr, args.warming_up_steps
    # 3) two schedulers (key line: different warmup_lr_init)
    g_sched = build_scheduler(optimizer, args.epochs, len(train_loader),
                                lr_min=min_lr, warmup_steps=warmup_steps,
                                warmup_lr_init=args.warming_up_init_lr)
    disc_sched = build_scheduler(
        optimizer_disc,
        args.epochs,
        len(train_loader),
        lr_min=disc_min_lr,
        warmup_steps=warmup_steps,
        warmup_lr_init=args.warming_up_init_lr
    )
    if args.train_stage == 2:
        g_sched_backbone    = build_scheduler(optimizer_backbone, args.epochs, len(train_loader),
                                    lr_min=min_lr * args.backbone_lr_mult,
                                    warmup_steps=warmup_steps,
                                    warmup_lr_init=args.warming_up_init_lr * args.backbone_lr_mult)
    #* Prepare models for training:
    resume_step_in_epoch = 0
    print('len(train_loader)', len(train_loader))
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)

        # --- model ---
        model_weight = checkpoint.get("model", checkpoint.get("state_dict", None))
        if model_weight is None:
            raise KeyError(f"Checkpoint missing model weights. keys={list(checkpoint.keys())}")
        if isinstance(model_weight, nn.Module):
            model_weight = model_weight.state_dict()
        if isinstance(model_weight, dict) and any(k.startswith("module.") for k in model_weight.keys()):
            model_weight = {k[len("module."):]: v for k, v in model_weight.items()}

        key_map = {
            "carrier_tokens": "base_state_queries",
            "slots_pos": "base_query_pos",
        }
        prefix_map = {
            "decode_transformer.slot_position_embedding": 
                "decode_transformer.base_position_embedding"
        }

        model_weight = remap_state_dict_keys(
            model_weight,
            key_map=key_map,
            prefix_map=prefix_map,
        )

        missing, unexpected = vq_model.load_state_dict(model_weight, strict=False)
        print("missing", missing)
        print("unexpected", unexpected)
        missing = [k for k in missing if "backbone" not in k]
        print("Filtered missing keys:", missing)

        # --- ema ---
        if args.ema:
            ema_weight = checkpoint.get("ema", None)
            if ema_weight is None:
                raise KeyError(f"args.ema=True but checkpoint has no 'ema'. keys={list(checkpoint.keys())}")
            if isinstance(ema_weight, nn.Module):
                ema_weight = ema_weight.state_dict()
            if isinstance(ema_weight, dict) and any(k.startswith("module.") for k in ema_weight.keys()):
                ema_weight = {k[len("module."):]: v for k, v in ema_weight.items()}

            ema_weight = remap_state_dict_keys(
                ema_weight,
                key_map=key_map,
                prefix_map=prefix_map,
            )
            ema.load_state_dict(ema_weight, strict=False)

        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"], strict=False)

        if not args.finetune:
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
            if checkpoint.get("optimizer_backbone") is not None and args.train_stage == 2:
                optimizer_backbone.load_state_dict(checkpoint["optimizer_backbone"])

            train_steps = checkpoint.get("steps", 0)
            ckpt_epoch = checkpoint.get("epoch", 0)
            ckpt_step_in_epoch = checkpoint.get("step_in_epoch", 0)

            # if this ckpt was saved at the end of an epoch, resume from the next epoch
            if ckpt_step_in_epoch >= len(train_loader):
                start_epoch = ckpt_epoch + 1
                resume_step_in_epoch = 0
            else:
                start_epoch = ckpt_epoch
                resume_step_in_epoch = ckpt_step_in_epoch

            # restore RNG state (optional but recommended)
            rng_state = checkpoint.get("rng_state", None)
            if rng_state is not None:
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                torch.set_rng_state(rng_state["torch"])
                torch.cuda.set_rng_state_all(rng_state["cuda"])

        else:
            train_steps = 0
            start_epoch = 0
            resume_step_in_epoch = 0

        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, start_epoch={start_epoch}, resume_step_in_epoch={resume_step_in_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        resume_step_in_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)

    if args.compile:  # optimize the model compute graph to generate more efficient code and speed up execution
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0        
    
    # wrap the models (vq_model and vq_loss) in DistributedDataParallel
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters,)  # set to True because some parameters are unused
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    
    model_dirs = osp.realpath(__file__).split('/')
    this_model_dir = '/'.join(model_dirs[2:-1])
    if args.train_stage == 1:
        if isinstance(vq_model, DDP):
            vq_model.module.backbone.freeze()
        else:
            vq_model.backbone.freeze()
    else:
        if isinstance(vq_model, DDP):
            for p in vq_model.module.parameters():
                p.requires_grad = True
        else:
            for p in vq_model.parameters():
                p.requires_grad = True

    use_epoch_ckpt = args.ckpt_every_epoch is not None and args.ckpt_every_epoch > 0
    if use_epoch_ckpt and is_main_process():
        logger.info(f"Checkpointing by epoch every {args.ckpt_every_epoch} epochs (step-based disabled).")
    elif is_main_process():
        logger.info(f"Checkpointing by step every {args.ckpt_every} steps (epoch-based disabled).")
    use_epoch_eval = args.eval_every_epoch is not None and args.eval_every_epoch > 0
    log_param_status(vq_model, logger, prefix=f"vq_model(stage={args.train_stage})")
    if args.train_stage==2:
        loss_obj = unwrap_model(vq_loss)          # this yields the VQLoss
        log_param_status(loss_obj.DistillLoss, logger, prefix=f"vq_loss.DistillLoss(stage={args.train_stage})")
    if is_main_process():
        if use_epoch_eval:
            logger.info(f"Evaluation by epoch every {args.eval_every_epoch} epochs (step-based eval disabled).")
        else:
            logger.info(f"Evaluation by step every {args.eval_every} steps (epoch-based eval disabled).")
    OBJ_GROUP = None
    if dist.is_available() and dist.is_initialized():
        # dedicated group for communicating python objects (CPU)
        OBJ_GROUP = dist.new_group(
            backend="gloo",
            timeout=timedelta(minutes=args.ddp_timeout_min)
        )
    # ===== restore the scheduler to the current train_steps =====
    if train_steps > 0:
        g_sched.step_update(train_steps)
        disc_sched.step_update(train_steps)
        if args.train_stage == 2:
            g_sched_backbone.step_update(train_steps)
    logger.info(f"Training for {args.epochs} epochs, current project dir is {this_model_dir}.")
    for epoch in range(start_epoch, args.epochs):

        batch_sampler.set_epoch(epoch)

        logger.info(f"Beginning epoch {epoch}...")

        epoch_loss_total = 0.0
        epoch_loss_gen = 0.0
        epoch_loss_discr = 0.0
        epoch_loss_codebook = 0.0
        epoch_loss_distill = 0.0
        epoch_usage_slot = 0.0
        epoch_n_slots = 0.0
        epoch_n_slots_q1 = 0.0
        epoch_n_slots_q2 = 0.0
        epoch_log_cnt = 0

        with tqdm(train_loader, dynamic_ncols=True, disable=not is_main_process()) as train_dl:
            for data_iter_step, batch_data in enumerate(train_dl):

                # ===== on resume, skip the iters already completed in the current epoch =====
                if epoch == start_epoch and data_iter_step < resume_step_in_epoch:
                    continue
                num_tokens=None
                if args.baseline:
                    (imgs, flatten_patches, image_grid_thw)=batch_data
                else:
                    # ---- parse batch for both cases ----
                    if len(batch_data) == 3:
                        imgs, flatten_patches, image_grid_thw = batch_data
                        num_images = None
                        mode = "single"
                    elif len(batch_data) == 4:
                        imgs, flatten_patches, image_grid_thw, num_images = batch_data
                        mode = "multi"
                    elif len(batch_data) == 5:
                        imgs, flatten_patches, image_grid_thw, num_images, img_paths = batch_data
                        mode = "multi"
                    elif len(batch_data) == 6:
                        imgs, flatten_patches, image_grid_thw, num_images, img_paths, num_tokens = batch_data
                        mode = "multi"
                    else:
                        raise ValueError(f"Unsupported batch format with len={len(batch_data)}. Expect 3 (single) or 4 (multi).")
                    # (imgs, flatten_patches, image_grid_thw, num_images)=batch_data
                if not os.path.exists(args.results_dir):
                    sys.exit(1)
                imgs = imgs.to(device, non_blocking=True)
                flatten_patches = flatten_patches.view(-1, 1536)
                #* Generator training
                optimizer.zero_grad()
                if args.train_stage == 2:
                    optimizer_backbone.zero_grad()
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    if args.baseline:
                        recons_imgs, codebook_loss, q_indices = vq_model(flatten_patches, image_grid_thw)
                    else:
                        if args.train_baseline:
                            num_images = torch.ones(
                                num_images.sum(),
                                dtype=torch.long,
                                device=num_images.device
                            )
                        if num_tokens is None:
                            # compute the total length
                            total_len = sum(num_images) - len(num_images)

                            # randomly sample total_len times
                            num_tokens = [random.choice(extra_cfg.gen_cfg.n_delta) for _ in range(total_len)]
                        
                        recons_imgs, codebook_loss, q_indices = vq_model(flatten_patches, image_grid_thw, num_images, num_tokens)

                    loss_gen, distill_loss = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps+1, 
                                    last_layer=vq_model.module.decoder.last_layer, 
                                    log_every=args.log_every, hidden_states = flatten_patches, grid_thw=image_grid_thw)

                scaler.scale(loss_gen).backward()
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    if args.train_stage == 2:
                        scaler.unscale_(optimizer_backbone)
                    torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
                
                g_sched.step_update(train_steps + 1)
                disc_sched.step_update(train_steps + 1)
                if args.train_stage == 2:
                    g_sched_backbone.step_update(train_steps + 1)
                scaler.step(optimizer)
                if args.train_stage == 2:
                    scaler.step(optimizer_backbone)
                scaler.update()
                if args.ema:
                    update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

                #* Discriminator training            
                optimizer_disc.zero_grad()
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    loss_disc = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1,
                                        log_every=args.log_every)
                scaler_disc.scale(loss_disc).backward()
                if args.max_grad_norm != 0.0:
                    scaler_disc.unscale_(optimizer_disc)
                    torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()
                # if train_steps % 1000 == 0:
                #     torch.cuda.empty_cache()
                train_steps += 1
                if train_steps % args.log_every == 0:
                    cur_lr = optimizer.param_groups[0]['lr']
                    if args.train_stage == 2:
                        lr_backbone = optimizer_backbone.param_groups[0]["lr"]
                    else:
                        lr_backbone=0
                    # # Log loss values:
                    torch.cuda.synchronize()
                    avg_loss = all_reduce_mean(loss_gen + loss_disc)
                    loss_gen = all_reduce_mean(loss_gen)
                    loss_discr = all_reduce_mean(loss_disc)
                    loss_codebook = all_reduce_mean(sum(codebook_loss))
                    loss_distill = all_reduce_mean(distill_loss)
                    if args.baseline or args.model_baseline:
                        slot_indices = torch.unique(concat_all_gather_variable_1d(q_indices))
                    else:
                        (q1, q2) = q_indices
                        slot_indices_1 = torch.unique(concat_all_gather_variable_1d(q1))
                        slot_indices_2 = torch.unique(concat_all_gather_variable_1d(q2))
                        slot_indices = torch.unique(torch.cat([slot_indices_1, slot_indices_2], dim=0))
                    usage_slot = slot_indices.size(0) / args.codebook_size
                    # accumulate for epoch-level logging
                    epoch_loss_total += avg_loss.item()
                    epoch_loss_gen += loss_gen.item()
                    epoch_loss_discr += loss_discr.item()
                    epoch_loss_codebook += loss_codebook.item()
                    epoch_loss_distill += loss_distill.item()
                    epoch_usage_slot += float(usage_slot)
                    epoch_n_slots += float(slot_indices.size(0))

                    if not (args.baseline or args.model_baseline):
                        epoch_n_slots_q1 += float(slot_indices_1.size(0))
                        epoch_n_slots_q2 += float(slot_indices_2.size(0))

                    epoch_log_cnt += 1

                    world_size = get_world_size()
                    if is_main_process():
                        if args.baseline or args.model_baseline:
                            train_dl.set_postfix(
                                ordered_dict={
                                    "epoch"          : epoch,
                                    "iters"          : train_steps,
                                    "total_loss"     : avg_loss.item(),
                                    "loss_gen"       : loss_gen.item(),
                                    "loss_discr"     : loss_discr.item(),
                                    'loss_distill'   : loss_distill.item(),
                                    'loss_codebook'  : loss_codebook.item(),
                                    'usage_slot'     : usage_slot,
                                    'n_slots'        : slot_indices.size(0),
                                    'learning_rate'  : cur_lr,
                                    'b_learning_rate': lr_backbone,
                                    'this_model_dir' : this_model_dir,
                                    "world_size"     : world_size,
                                }
                            )
                        else:
                            train_dl.set_postfix(
                                ordered_dict={
                                    "epoch"          : epoch,
                                    "iters"          : train_steps,
                                    "total_loss"     : avg_loss.item(),
                                    "loss_gen"       : loss_gen.item(),
                                    "loss_discr"     : loss_discr.item(),
                                    'loss_distill'   : loss_distill.item(),
                                    'loss_codebook'  : loss_codebook.item(),
                                    'usage_slot'     : usage_slot,
                                    'n_slots'        : slot_indices.size(0),
                                    'n_slots_q1'     : slot_indices_1.size(0),
                                    'n_slots_q2'     : slot_indices_2.size(0),
                                    'learning_rate'  : cur_lr,
                                    'b_learning_rate': lr_backbone,
                                    'this_model_dir' : this_model_dir,
                                    "world_size"     : world_size,
                                }
                            )
                    if is_main_process() and writer is not None:
                        step = train_steps

                        writer.add_scalar("train_step/loss/total_loss", avg_loss.item(), step)
                        writer.add_scalar("train_step/loss/loss_gen", loss_gen.item(), step)
                        writer.add_scalar("train_step/loss/loss_discr", loss_discr.item(), step)
                        writer.add_scalar("train_step/loss/loss_codebook", loss_codebook.item(), step)
                        writer.add_scalar("train_step/loss/loss_distill", loss_distill.item(), step)

                        writer.add_scalar("train_step/usage/usage_slot", float(usage_slot), step)
                        writer.add_scalar("train_step/usage/n_slots", int(slot_indices.size(0)), step)

                        writer.add_scalar("train_step/lr/base", float(cur_lr), step)
                        writer.add_scalar("train_step/lr/backbone", float(lr_backbone), step)

                        if (not (args.baseline or args.model_baseline)):
                            writer.add_scalar("train_step/usage/n_slots_q1", int(slot_indices_1.size(0)), step)
                            writer.add_scalar("train_step/usage/n_slots_q2", int(slot_indices_2.size(0)), step)

                #* Save checkpoint:
                if (not use_epoch_ckpt) and (train_steps > 0) and (train_steps % args.ckpt_every == 0) and is_main_process():
                    ckpt_path = osp.join(checkpoint_dir, f'vit_vqgan_epoch_{epoch}_step_{train_steps}.pt')
                    save_checkpoint(
                        ckpt_path=ckpt_path,
                        vq_model=vq_model,
                        vq_loss=vq_loss,
                        optimizer=optimizer,
                        optimizer_backbone=optimizer_backbone,
                        optimizer_disc=optimizer_disc,
                        ema=ema if args.ema else None,
                        train_steps=train_steps,
                        epoch=epoch,
                        step_in_epoch=data_iter_step + 1,   # important
                        args=args,
                        logger=logger,
                    )

                dist.barrier()  # synchronize all processes so they share the same state after the checkpoint is saved

                if (not use_epoch_eval) and (train_steps > 0) and (train_steps % args.eval_every == 0):
                    val_model_multi(
                        ema=ema,
                        val_loaders=val_loaders,
                        val_names=args.val_data_paths,
                        device=device,
                        args=args,
                        write_val=f"{train_steps}:val_loader",
                        checkpoint_dir=args.results_dir,
                        logger=logger,
                        writer=writer,
                        global_step=train_steps,
                        global_epoch=epoch,
                        val=val,
                        batch_size=args.val_bs,
                        OBJ_GROUP=OBJ_GROUP,
                    )
                    

        # ---- end of epoch: save by epoch (if enabled) ----
        if use_epoch_ckpt and is_main_process():
            if ((epoch + 1) % args.ckpt_every_epoch) == 0:
                ckpt_path = osp.join(checkpoint_dir, f'vit_vqgan_epoch_{epoch}.pt')
                save_checkpoint(
                    ckpt_path=ckpt_path,
                    vq_model=vq_model,
                    vq_loss=vq_loss,
                    optimizer=optimizer,
                    optimizer_backbone=optimizer_backbone,
                    optimizer_disc=optimizer_disc,
                    ema=ema if args.ema else None,
                    train_steps=train_steps,
                    epoch=epoch,
                    step_in_epoch=len(train_loader),   # important
                    args=args,
                    logger=logger,
                )
            

        if use_epoch_eval:
            if ((epoch + 1) % args.eval_every_epoch) == 0:
                val_model_multi(
                    ema=ema,
                    val_loaders=val_loaders,
                    val_names=args.val_data_paths,
                    device=device,
                    args=args,
                    write_val=f"epoch{epoch}:val_dataset",
                    checkpoint_dir=args.results_dir,
                    logger=logger,
                    writer=writer,
                    global_step=train_steps,
                    global_epoch=epoch,
                    val=val,
                    batch_size=args.val_bs,
                    OBJ_GROUP=OBJ_GROUP,
                )
                if args.val_n_delta_one is not None:
                    for delta in args.val_n_delta_one:
                        val_model_multi(
                            ema=ema,
                            val_loaders=val_loaders,
                            val_names=args.val_data_paths,
                            device=device,
                            args=args,
                            write_val=f"{train_steps}:val_loader:delta_{delta}",
                            checkpoint_dir=args.results_dir,
                            logger=logger,
                            writer=writer,
                            global_step=train_steps,
                            global_epoch=epoch,
                            val=val,
                            batch_size=args.val_bs,
                            OBJ_GROUP=OBJ_GROUP,
                            val_n_delta=[delta],
                        )
        # after an epoch completes, do not skip iters in subsequent epochs
        resume_step_in_epoch = 0
        if is_main_process() and writer is not None and epoch_log_cnt > 0:
            e = epoch  # or epoch + 1, depending on whether epochs should start at 0 or 1

            writer.add_scalar("train_epoch/loss/total_loss", epoch_loss_total / epoch_log_cnt, e)
            writer.add_scalar("train_epoch/loss/loss_gen", epoch_loss_gen / epoch_log_cnt, e)
            writer.add_scalar("train_epoch/loss/loss_discr", epoch_loss_discr / epoch_log_cnt, e)
            writer.add_scalar("train_epoch/loss/loss_codebook", epoch_loss_codebook / epoch_log_cnt, e)
            writer.add_scalar("train_epoch/loss/loss_distill", epoch_loss_distill / epoch_log_cnt, e)

            writer.add_scalar("train_epoch/usage/usage_slot", epoch_usage_slot / epoch_log_cnt, e)
            writer.add_scalar("train_epoch/usage/n_slots", epoch_n_slots / epoch_log_cnt, e)

            # learning rate: at epoch end the current value is sufficient
            writer.add_scalar("train_epoch/lr/base", float(optimizer.param_groups[0]["lr"]), e)
            writer.add_scalar("train_epoch/lr/backbone", float(optimizer_backbone.param_groups[0]["lr"]) if optimizer_backbone is not None else 0.0, e)

            if not (args.baseline or args.model_baseline):
                writer.add_scalar("train_epoch/usage/n_slots_q1", epoch_n_slots_q1 / epoch_log_cnt, e)
                writer.add_scalar("train_epoch/usage/n_slots_q2", epoch_n_slots_q2 / epoch_log_cnt, e)

    if is_main_process():
        if use_epoch_ckpt:
            ckpt_path = osp.join(checkpoint_dir, f'vit_vqgan_final_epoch_{args.epochs-1}.pt')
        else:
            ckpt_path = osp.join(checkpoint_dir, f'vit_vqgan_final_step_{train_steps}.pt')

        save_checkpoint(
            ckpt_path=ckpt_path,
            vq_model=vq_model,
            vq_loss=vq_loss,
            optimizer=optimizer,
            optimizer_backbone=optimizer_backbone,
            optimizer_disc=optimizer_disc,
            ema=ema if args.ema else None,
            train_steps=train_steps,
            epoch=args.epochs - 1,
            step_in_epoch=len(train_loader),
            args=args,
            logger=logger,
        )

    val_model_multi(
        ema=ema,
        val_loaders=val_loaders,
        val_names=args.val_data_paths,
        device=device,
        args=args,
        write_val=f"vit_vqgan_final_epoch_{args.epochs-1}_step{train_steps}:val_loader",
        checkpoint_dir=args.results_dir,
        logger=logger,
        writer=writer,
        global_step=train_steps,
        global_epoch=epoch,
        val=val,
        batch_size=args.val_bs,
        OBJ_GROUP=OBJ_GROUP,
    )
    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()
    if is_main_process() and writer is not None:
        writer.flush()
        writer.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", nargs="+", type=str, required=True)
    parser.add_argument("--data_ratios", nargs="+", type=int, required=True)
    parser.add_argument("--global-batch-size", nargs="+", type=int, required=True)
    parser.add_argument("--bucket_max_counts", nargs="+", type=int, required=True)

    parser.add_argument("--bucket_widths", nargs="+", type=int, default=None)
    parser.add_argument("--dataset_seeds", nargs="+", type=int, default=None)
    
    parser.add_argument("--transformer-config-file", type=str, default='configs/vit_transformer.yaml',)
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    # parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--warming-up-epochs", type=float, default=4, help="warming up iterations")
    parser.add_argument("--warming-up-steps", type=float, default=20000, help="warming up steps")
    parser.add_argument("--warming-up-init-lr", type=float, default=1e-7, help="warming up initial learning rate")
    
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")

    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-use-warmup",  action='store_true')
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan', 'dinogan'], default='dinogan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating', 'softplus_g_loss'], default='softplus_g_loss', help="generator loss for gan training")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[128, 192, 256, 336, 384, 512], default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    # parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--val-data-paths", nargs="+", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use_pad", type=float, default=0)
    parser.add_argument("--train_center_padding", action='store_true', default=False)
    parser.add_argument("--train_center_crop", action='store_true', default=False)
    # 1: samples carry num_tokens from the dataset (stage 2, multi-image); 0: they do not, and token
    # budgets are randomly sampled from n_delta during training (reproduces stage 1)
    parser.add_argument("--return_num_tokens", type=int, default=1)
    parser.add_argument("--val_padding", action='store_true', default=False)
    parser.add_argument("--val_n_delta", type=int, nargs="+", default=[144],help="List of integer deltas for validation")
    parser.add_argument("--val_n_delta_one", type=int, nargs="+", default=None, help="List of integer deltas for validation")
    parser.add_argument("--ckpt-every-epoch", type=int, default=0, help="If >0, save checkpoint every N epochs and disable step-based saving.")
    parser.add_argument("--extra_cfg", type=str, default="configs/vision_extra.json",)
    parser.add_argument("--base_model_path", type=str, default=os.environ.get("BASE_MODEL_PATH", None),
                        help="Qwen3-VL-2B base dir: provides vision_config (config.json) and visual init weights (model.safetensors)")
    parser.add_argument("--eval-every-epoch", type=int, default=0, help="If >0, run validation every N epochs and disable step-based eval.")
    parser.add_argument("--train-stage", type=int, default=1)
    parser.add_argument("--sort_within_bucket", action='store_true', default=False)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1, help="backbone lr multiplier vs base lr (used when train_stage==2)")
    parser.add_argument("--baseline", action='store_true', default=False)
    parser.add_argument("--find_unused_parameters", action='store_true', default=False)
    parser.add_argument("--model_baseline", action='store_true', default=False)
    parser.add_argument("--train_baseline", action='store_true', default=False)
    parser.add_argument("--ratio", action='store_true', default=False)
    parser.add_argument("--enable_zoom", action='store_true', default=False)
    parser.add_argument("--val_bs", type=int, default=4)
    parser.add_argument("--ddp_timeout_min", type=int, default=240)
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=None,
        help="discriminator learning rate; if not set, fallback to --lr",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="final learning rate after cosine decay; if not set, fallback to --lr",
    )
    parser.add_argument(
        "--disc-min-lr",
        type=float,
        default=None,
        help="final discriminator learning rate after cosine decay; if not set, fallback to --disc-lr or --lr",
    )
    args = parser.parse_args()
    main(args)
