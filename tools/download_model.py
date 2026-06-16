#!/usr/bin/env python3
"""Download ViMo checkpoints from Hugging Face or ModelScope.

Examples
--------
# From Hugging Face (default)
python tools/download_model.py -n ViMo-2B
python tools/download_model.py -n TSIM-Tok

# From ModelScope
python tools/download_model.py -t modelscope -n ViMo-2B
python tools/download_model.py -t modelscope -n TSIM-Tok

By default all checkpoints are placed under ``weights/``:

    weights/
      vimo_2b/
      tsim_tok/
        tsim_tok.pt
"""

import argparse
import os
import shutil

# Model name -> {source: repo_id}, and where it lands under the weights dir.
MODELS = {
    "ViMo-2B": {
        "huggingface": "dle666/ViMo-2B",
        "modelscope": "wpj2003/ViMo-2B",
        "target_subdir": "vimo_2b",
        "kind": "repo",          # download the whole repo
    },
    "TSIM-Tok": {
        "huggingface": "wpj20000/TSIM-Tok",
        "modelscope": "wpj2003/TSIM-Tok",
        "target_subdir": "tsim_tok",
        "kind": "file",          # only need the .pt checkpoint
        "filename": "tsim_tok.pt",
    },
}


def download_huggingface(repo_id, local_dir, kind, filename=None):
    if kind == "file":
        from huggingface_hub import hf_hub_download

        os.makedirs(local_dir, exist_ok=True)
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        dst = os.path.join(local_dir, filename)
        shutil.copy(path, dst)
        return dst
    else:
        from huggingface_hub import snapshot_download

        os.makedirs(local_dir, exist_ok=True)
        return snapshot_download(repo_id=repo_id, local_dir=local_dir)


def download_modelscope(repo_id, local_dir, kind, filename=None):
    if kind == "file":
        from modelscope.hub.file_download import model_file_download

        os.makedirs(local_dir, exist_ok=True)
        path = model_file_download(model_id=repo_id, file_path=filename)
        dst = os.path.join(local_dir, filename)
        shutil.copy(path, dst)
        return dst
    else:
        from modelscope import snapshot_download

        os.makedirs(local_dir, exist_ok=True)
        return snapshot_download(repo_id, local_dir=local_dir)


def main():
    parser = argparse.ArgumentParser(description="Download ViMo checkpoints.")
    parser.add_argument(
        "-n", "--name", required=True, choices=list(MODELS.keys()),
        help="Which checkpoint to download: ViMo-2B or TSIM-Tok.",
    )
    parser.add_argument(
        "-t", "--type", default="huggingface",
        choices=["huggingface", "modelscope"],
        help="Download source (default: huggingface).",
    )
    parser.add_argument(
        "-d", "--weights-dir", default="weights",
        help="Root directory to store checkpoints (default: weights).",
    )
    args = parser.parse_args()

    info = MODELS[args.name]
    repo_id = info[args.type]
    local_dir = os.path.join(args.weights_dir, info["target_subdir"])

    print(f"Downloading {args.name} from {args.type} ({repo_id}) -> {local_dir}")

    if args.type == "huggingface":
        out = download_huggingface(
            repo_id, local_dir, info["kind"], info.get("filename")
        )
    else:
        out = download_modelscope(
            repo_id, local_dir, info["kind"], info.get("filename")
        )

    print(f"Done. Saved to: {out}")


if __name__ == "__main__":
    main()
