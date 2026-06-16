#!/usr/bin/env python3
"""Strip training-only / private metadata from a TSIM-Tok checkpoint.

The released TSIM-Tok checkpoint is a full training checkpoint that embeds an
``args`` namespace (private absolute paths + training hyper-parameters), an
``rng_state`` blob and step counters. None of these are read by the inference or
training loaders (which only touch ``ema`` / ``model`` / ``state_dict``), so they
are dropped here. Tensor storages are copied through byte-for-byte (torch.save of
the same float32 storages is lossless) — no re-quantisation, no precision change.

A numpy RandomState lives inside ``rng_state``; on environments whose numpy is
older than the one that wrote the checkpoint, unpickling it raises
``ModuleNotFoundError: numpy._core``. Since ``rng_state`` is being deleted anyway,
a custom pickle module stubs out every ``numpy.*`` global during load so the
weight tensors decode without needing a numpy version bump.

The output is written with the *file name* ``tsim_tok.pt`` because torch derives
the zip archive's top-level directory name from the output path stem; saving as
``tsim_tok.pt`` rewrites the stale ``vit_vqgan_epoch_0_step_100000`` directory
name to ``tsim_tok``.

Usage:
    python tools/checkpoint/clean_tsim_tok.py \
        --src weights/tsim_tok/tsim_tok.pt \
        --out-dir /tmp/tsim_tok_clean
"""
import argparse
import os
import pickle
import subprocess
import sys
import types
import zipfile

import torch

DROP_KEYS = ("args", "rng_state", "steps", "epoch", "step_in_epoch")
KEEP_KEYS = ("model", "optimizer", "optimizer_backbone", "optimizer_disc",
             "discriminator", "ema")


class _NumpyStub:
    """Absorbs any numpy reconstruct call; only used by the rng_state we drop."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        pass

    def __reduce__(self):
        return (_NumpyStub, ())


def _stub(*a, **k):
    return _NumpyStub()


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith("numpy") or (mod == "_codecs" and name == "encode"):
            return _stub
        return super().find_class(mod, name)


def _numpy_safe_pickle_module():
    shim = types.ModuleType("shim_pickle")
    shim.Unpickler = SafeUnpickler
    shim.load = pickle.load
    shim.UnpicklingError = pickle.UnpicklingError
    return shim


def load_numpy_safe(path):
    return torch.load(path, map_location="cpu", weights_only=False,
                      pickle_module=_numpy_safe_pickle_module())


def clean(ckpt):
    dropped = [k for k in DROP_KEYS if k in ckpt]
    for k in DROP_KEYS:
        ckpt.pop(k, None)
    return ckpt, dropped


def assert_tensors_equal(old, new, label):
    ok = old.keys() == new.keys()
    if not ok:
        raise AssertionError(f"[{label}] key set differs: "
                             f"only_old={set(old)-set(new)} only_new={set(new)-set(old)}")
    n_tensor = 0
    for k, ov in old.items():
        nv = new[k]
        if torch.is_tensor(ov):
            n_tensor += 1
            if not torch.equal(ov, nv):
                raise AssertionError(f"[{label}] tensor '{k}' changed after re-save")
    return n_tensor


def verify(src, out):
    print("\n=== verify ===")
    # 1) strings: no private paths / stale archive name
    res = subprocess.run(
        f"strings '{out}' | grep -E '/mnt|/home|vit_vqgan' || true",
        shell=True, capture_output=True, text=True)
    hits = [l for l in res.stdout.splitlines() if l.strip()]
    print(f"[strings] private-path hits: {len(hits)}")
    if hits:
        for h in hits[:10]:
            print("   !!", h)
        raise AssertionError("private strings still present in cleaned checkpoint")

    # 2) internal archive dir name
    tops = sorted({n.split("/")[0] for n in zipfile.ZipFile(out).namelist()})
    print(f"[zip] internal top-level dirs: {tops}")
    assert tops == ["tsim_tok"], f"unexpected archive dir name: {tops}"

    # 3) top-level keys
    new_ckpt = torch.load(out, map_location="cpu", weights_only=False)  # no numpy now
    keys = list(new_ckpt.keys())
    print(f"[keys] {keys}")
    for k in DROP_KEYS:
        assert k not in new_ckpt, f"dropped key '{k}' still present"

    # 4) bit-exact weights for model & ema
    old_ckpt = load_numpy_safe(src)
    for label in ("model", "ema"):
        n = assert_tensors_equal(old_ckpt[label], new_ckpt[label], label)
        print(f"[equal] {label}: {n} tensors all torch.equal ✓")

    print("=== all checks passed ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="weights/tsim_tok/tsim_tok.pt")
    ap.add_argument("--out-dir", default="/tmp/tsim_tok_clean")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "tsim_tok.pt")  # name drives zip archive dir

    print(f"src = {src}")
    print(f"out = {out}")

    print("\n=== load (numpy-safe) ===")
    ckpt = load_numpy_safe(src)
    print("top-level keys:", list(ckpt.keys()))

    ckpt, dropped = clean(ckpt)
    print("dropped:", dropped)
    print("remaining:", list(ckpt.keys()))

    print("\n=== save ===")
    torch.save(ckpt, out)
    sz = os.path.getsize(out) / 1e6
    print(f"wrote {out}  ({sz:.1f} MB)")

    verify(src, out)
    print(f"\nDONE. Cleaned checkpoint at: {out}")
    print("Review, then overwrite the original when satisfied.")


if __name__ == "__main__":
    sys.exit(main())
