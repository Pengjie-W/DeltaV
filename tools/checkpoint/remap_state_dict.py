"""Checkpoint compatibility checker / remapper for the renamed ViMo model.

Renaming Python identifiers (class names, methods, local variables) does NOT change
``state_dict`` keys — keys derive from ``nn.Module`` *attribute* names, which were kept
unchanged on purpose. So existing checkpoints load into the renamed ``ViMoModel`` as-is.

This script (a) verifies a checkpoint loads with ``strict=True`` into ``ViMoModel``, and
(b) optionally writes a remapped copy if a future rename ever touches submodule attribute
names. The original checkpoint is never modified — output goes to a new path.

Usage:
    # validate that the released checkpoint loads into the renamed model
    python tools/checkpoint/remap_state_dict.py --check \
        --ckpt weights/vimo_2b --extra_cfg configs/vimo_cfg.json

    # write a remapped copy (identity mapping by default)
    python tools/checkpoint/remap_state_dict.py \
        --ckpt /path/to/old.pt --out weights/vimo_2b_remapped.pt --extra_cfg configs/vimo_cfg.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

# Add submodule-attribute renames here if a future rename changes nn.Module attribute
# names (left side = old key prefix/name, right side = new). Empty = identity.
KEY_RENAMES: dict = {}


def remap_keys(state_dict):
    if not KEY_RENAMES:
        return state_dict, 0
    out, n = {}, 0
    for k, v in state_dict.items():
        nk = k
        for old, new in KEY_RENAMES.items():
            if nk.startswith(old):
                nk = new + nk[len(old):]
                n += 1
        out[nk] = v
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (HF) or .pt/.safetensors file")
    ap.add_argument("--extra_cfg", default="configs/vimo_cfg.json")
    ap.add_argument("--out", default=None, help="output path for a remapped copy")
    ap.add_argument("--check", action="store_true", help="instantiate ViMoModel and strict-load")
    args = ap.parse_args()

    import torch
    from vimo.modeling_vimo import ViMoModel, TSIMTokExtraCfg
    from vimo.configuration_vimo import ViMoConfig

    if os.path.isdir(args.ckpt):
        # HF-style checkpoint dir
        config = ViMoConfig.from_pretrained(args.ckpt)
        extra_cfg = TSIMTokExtraCfg.load(args.extra_cfg)
        model = ViMoModel.from_pretrained(args.ckpt, config=config, extra_cfg=extra_cfg)
        print(f"[OK] ViMoModel.from_pretrained loaded '{args.ckpt}' (strict).")
        return

    state = torch.load(args.ckpt, map_location="cpu")
    state = state.get("model", state) if isinstance(state, dict) else state
    new_state, n = remap_keys(state)
    print(f"[remap] {n} keys rewritten ({'identity' if not KEY_RENAMES else 'custom mapping'}).")

    if args.check:
        config = ViMoConfig.from_pretrained(os.path.dirname(args.ckpt) or ".")
        extra_cfg = TSIMTokExtraCfg.load(args.extra_cfg)
        model = ViMoModel(config=config, extra_cfg=extra_cfg)
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[check] missing={len(missing)} unexpected={len(unexpected)}")
        if missing or unexpected:
            print("  missing[:5]:", list(missing)[:5])
            print("  unexpected[:5]:", list(unexpected)[:5])

    if args.out:
        assert os.path.abspath(args.out) != os.path.abspath(args.ckpt), "refuse to overwrite original"
        torch.save({"model": new_state}, args.out)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
