import os
import os.path as osp
import sys
import warnings
import gc
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vimo.tsim_tok.training.evaluations.evaluator import Evaluator
from skimage.metrics import structural_similarity as ssim_loss
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from vimo.tsim_tok.training.engine.misc import (
    is_main_process,
    concat_all_gather_varlen,
    get_rank
)

warnings.filterwarnings('ignore')

def gather_object_to_rank0(obj, group=None):
    """Gather python object from all ranks to rank0. Return list on rank0, else None."""
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]

    if group is None:
        group = dist.group.WORLD

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        gathered = [None for _ in range(world_size)]
        dist.gather_object(obj, object_gather_list=gathered, dst=0, group=group)
        return gathered
    else:
        dist.gather_object(obj, object_gather_list=None, dst=0, group=group)
        return None

def _local_unique_1d(x: torch.Tensor) -> torch.Tensor:
    """Make sure we always unique on a 1D tensor (works for any shape)."""
    if x is None:
        return None
    return torch.unique(x.detach().reshape(-1))


def val(model, val_loader, device, args, step, checkpoint_dir, batch_size=64, OBJ_GROUP=None, val_n_delta=None, save_image=False):
    """
    Unified val for both single/multi:
      - each rank computes its local samples/gt/psnr/ssim
      - gather python payloads to rank0
      - gather slot/img indices tensors to rank0 (via concat_all_gather_varlen)
      - only rank0 runs TF Evaluator, writes results.md
    """
    num_protos, num_slots, eps = args.codebook_size, args.codebook_size, 1e-6

    # local per-rank gen
    img_idx_local, slot_idx_local, (samples_local, gt_local, psnr_local, ssim_local) = gen_images(
        model, val_loader, device, args, val_n_delta, save_image
    )

    # gather python payloads to rank0
    payload = {"samples": samples_local, "gt": gt_local, "psnr": psnr_local, "ssim": ssim_local}
    gathered_payloads = gather_object_to_rank0(payload, group=OBJ_GROUP)

    # gather tensor indices across ranks
    slot_idx_all = concat_all_gather_varlen(slot_idx_local)
    slot_idx_all = torch.unique(slot_idx_all)

    img_idx_all = concat_all_gather_varlen(img_idx_local)
    img_idx_all = torch.unique(img_idx_all) if img_idx_all.numel() > 0 else img_idx_all

    out = None
    if is_main_process():
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

        # merge all ranks' python lists
        samples_all, gt_all, psnr_all, ssim_all = [], [], [], []
        for p in gathered_payloads:
            samples_all.extend(p["samples"])
            gt_all.extend(p["gt"])
            psnr_all.extend(p["psnr"])
            ssim_all.extend(p["ssim"])

        samples_all = np.stack(samples_all, axis=0)
        gt_all = np.stack(gt_all, axis=0)
        print(f'len(samples):{samples_all.shape[0]}, len(gt): {gt_all.shape[0]}')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        evaluator = Evaluator(sess, batch_size=batch_size)
        evaluator.warmup()

        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(gt_all)

        print("computing/reading reference batch statistics...")
        ref_stats, _ = evaluator.read_statistics(gt_all, ref_acts)

        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(samples_all)

        print("computing/reading sample batch statistics...")
        sample_stats, _ = evaluator.read_statistics(samples_all, sample_acts)

        FID = sample_stats.frechet_distance(ref_stats)
        IS = evaluator.compute_inception_score(sample_acts[0])
        print(f"rFID: {FID:04f}, rIS: {IS:04f}.")

        usage_img = (img_idx_all.size(0) / num_protos) if img_idx_all.numel() > 0 else 0.0
        usage_slot = slot_idx_all.size(0) / num_slots

        psnr_val_rgb = sum(psnr_all) / (len(psnr_all) + eps)
        ssim_val_rgb = sum(ssim_all) / (len(ssim_all) + eps)

        sess.close()
        del evaluator

        print('usage_img:{:.4f}, usage_slot: {:.4f},  psnr: {:.4f}, ssim: {:.4f}'.format(
            usage_img, usage_slot, psnr_val_rgb, ssim_val_rgb
        ))

        with open(os.path.join(checkpoint_dir, 'results.md'), 'a') as fid:
            fid.write(f'\n{step}:\n')
            fid.write(f'rFID: {FID:04f}, rIS: {IS:04f}.\n')
            fid.write('usage_img:{:.4f}, usage_slot: {:.5f}, PSNR: {:.4f}, SSIM: {:.4f}.\n'.format(
                usage_img, usage_slot, psnr_val_rgb, ssim_val_rgb
            ))

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        tf.reset_default_graph()
        del sess, ref_acts, sample_acts, ref_stats, sample_stats
        del samples_all, gt_all, psnr_all, ssim_all

        out = (FID, IS, usage_img, usage_slot, psnr_val_rgb, ssim_val_rgb)

    del gathered_payloads
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return out

@torch.no_grad()
def gen_images(model, dataloader, device, args, val_n_delta, save_image=False):
    """
    Unified generator for:
      - single-image loader yields: (images, flatten_patches, image_grid_thw)
      - multi-image loader yields: (images, flatten_patches, image_grid_thw, num_images)

    Output format is kept the same:
      return img_indices, slot_indices, (samples, gt, psnr_val_rgb, ssim_val_rgb)
    """
    model.eval()

    total = len(dataloader)
    model_dirs = osp.realpath(__file__).split('/')
    idx = np.argmax([len(p) for p in model_dirs])
    this_model_dir = model_dirs[idx]

    # Keep placeholders (same as your current logic: img_indices not filled)
    img_indices = torch.empty(0, device=device)
    slot_indices = torch.empty(0, device=device)

    samples, gt = [], []
    psnr_val_rgb, ssim_val_rgb = [], []

    # (optional) keep your save_dir creation behavior
    # NOTE: you had different dirs in single/multi; here choose one or keep as-is by args if you want
    save_dir = getattr(args, "save_dir", None) or "./vis_results"
    os.makedirs(save_dir, exist_ok=True)
    gen_dir = os.path.join(save_dir, "gen")
    gt_dir = os.path.join(save_dir, "gt")

    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    rank=get_rank()
    for i, batch in tqdm(enumerate(dataloader), total=total, mininterval=10, miniters=200):
        num_images = None
        num_tokens = None
        # ---- parse batch for both cases ----
        if len(batch) == 3:
            images, flatten_patches, image_grid_thw = batch
            mode = "single"
        elif len(batch) == 4:
            images, flatten_patches, image_grid_thw, num_images = batch
            mode = "multi"
        elif len(batch) == 5:
            images, flatten_patches, image_grid_thw, num_images, img_paths = batch
            mode = "multi"
        elif len(batch) == 6:
            images, flatten_patches, image_grid_thw, num_images, img_paths, num_tokens = batch
            mode = "multi"
        else:
            raise ValueError(f"Unsupported batch format with len={len(batch)}. Expect 3 (single) or 4 (multi).")

        images = images.to(device)
        flatten_patches = flatten_patches.to(device)
        image_grid_thw = image_grid_thw.to(device)

        flatten_patches = flatten_patches.view(-1, 1536)

        # ---- forward for both cases, keeping original calling conventions ----
        if mode == "single":
            # original single: (gen_imgs, _, _), _, q_indices = model(flatten_patches, image_grid_thw)
            (gen_imgs, _, _), _, q_indices = model(flatten_patches, image_grid_thw)
        else:
            # original multi: recons_imgs, codebook_loss, q_indices = model(flatten_patches, image_grid_thw, num_images)
            if args.train_baseline:
                num_images = torch.ones(
                    num_images.sum(),
                    dtype=torch.long,
                    device=num_images.device
                )
            if num_tokens is None:
                total_len = sum(num_images) - len(num_images)

                # randomly sample total_len times
                if val_n_delta is not None:
                    num_tokens = [random.choice(val_n_delta) for _ in range(total_len)]
                else:
                    num_tokens = [random.choice(args.val_n_delta) for _ in range(total_len)]
            recons_imgs, _, q_indices = model(flatten_patches, image_grid_thw, num_images, num_tokens)
            gen_imgs, _, _ = recons_imgs

        # ---- slot usage (local) ----
        # single: q_indices is a tensor
        # multi:  q_indices is (q1, q2)
        if isinstance(q_indices, (tuple, list)) and len(q_indices) == 2:
            q1, q2 = q_indices
            q1_u = _local_unique_1d(q1)
            q2_u = _local_unique_1d(q2)
            slot_indices = torch.unique(torch.cat([slot_indices, q1_u, q2_u], dim=0))
        else:
            q_u = _local_unique_1d(q_indices)
            slot_indices = torch.unique(torch.cat([slot_indices, q_u], dim=0))

        # ---- compute metrics locally ----
        gen_images_np = (127.5 * gen_imgs.permute(0, 2, 3, 1).float() + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        images_np     = (127.5 * images.permute(0, 2, 3, 1).float() + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        if is_main_process() and i % 100 == 0:
            print('{}, iter-{}/{}, gen_imgs.shape:{}'.format(this_model_dir, i, total, gen_images_np.shape))
        
        metric_image_select = getattr(args, "metric_image_select", "all")

        if metric_image_select == "all":
            selected_indices = range(len(gen_images_np))
        elif metric_image_select == "first":
            selected_indices = [0] if len(gen_images_np) > 0 else []
        elif metric_image_select == "exclude_first":
            selected_indices = range(1, len(gen_images_np))
        else:
            raise ValueError(
                f"Unsupported args.metric_image_select={metric_image_select}, "
                f"expected one of ['all', 'first', 'exclude_first']"
            )
        for k in selected_indices:
            recon_arr = gen_images_np[k]
            rec = Image.fromarray(np.uint8(recon_arr))
            img = Image.fromarray(np.uint8(images_np[k]))

            rec = rec.resize((args.image_size, args.image_size))
            img = img.resize((args.image_size, args.image_size))

            rgb_restored = np.array(rec).astype(np.float32) / 255.
            rgb_gt = np.array(img).astype(np.float32) / 255.

            psnr = psnr_loss(rgb_restored, rgb_gt)
            if not np.isfinite(psnr):
                psnr = 100.0

            # keep your original ssim parameters unchanged
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)

            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)

            samples.append(np.array(rec))
            gt.append(np.array(img))

            # keep saving disabled as in your snippets
            if save_image:
                rec.save(os.path.join(gen_dir, f"{rank}_{i}_{k}.png"))
                img.save(os.path.join(gt_dir,  f"{rank}_{i}_{k}.png"))

    return img_indices, slot_indices, (samples, gt, psnr_val_rgb, ssim_val_rgb)