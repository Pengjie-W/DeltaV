"""
Local copy of wpj's convert_zebra_to_extract_format_wpj.py with one fix:
pass through the input `config` field instead of writing "" — required so
extract_zebra_pred_with_gt_cache_v2.py's check_alignment() does not raise
"Config mismatch for id=...".
"""

import os
import json
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input zebra JSON/JSONL file")
    parser.add_argument("--output", required=True, help="Output normalized JSONL file")
    parser.add_argument(
        "--id_mode",
        default="sample_png",
        choices=["sample_png", "index", "keep"],
        help=(
            "sample_png: parse id from model_output_image like sample_000000.png\n"
            "index: use item index as id, zero-padded to 6 digits\n"
            "keep: use original item['id'] if exists"
        ),
    )
    return parser.parse_args()


def read_json_or_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            try:
                obj = json.load(f)
                if isinstance(obj, list):
                    data = obj
                else:
                    data = [obj]
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
    return data


def write_jsonl(data, path):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_id_from_path(path_text):
    if not path_text:
        return None

    text = str(path_text)

    m = re.search(r"sample_(\d+)\.(png|jpg|jpeg|webp)$", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).zfill(6)

    m = re.search(r"sample_(\d+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).zfill(6)

    return None


def build_id(item, idx, id_mode):
    if id_mode == "keep":
        if item.get("id") is None:
            raise ValueError(f"item {idx} has no 'id' but id_mode=keep")
        return str(item["id"]).zfill(6)

    if id_mode == "index":
        return str(idx).zfill(6)

    model_output_image = item.get("model_output_image")
    parsed = parse_id_from_path(model_output_image)
    if parsed is not None:
        return parsed

    if item.get("id") is not None:
        return str(item["id"]).zfill(6)

    return str(idx).zfill(6)


def convert_one(item, idx, id_mode):
    new_id = build_id(item, idx, id_mode)

    prompt = item.get("input_prompt", "")
    pred_text = item.get("model_output_text", "")
    config = item.get("config", "") or ""

    output = {
        "id": new_id,
        "config": config,
        "prompt": prompt,
        "reference_image": item.get("input_image", []),
        "pred_text": pred_text,
        "pred_image": item.get("model_output_image", []),
    }

    if isinstance(output["pred_image"], str):
        output["pred_image"] = [output["pred_image"]]
    elif output["pred_image"] is None:
        output["pred_image"] = []

    return output


def main():
    args = parse_args()

    data = read_json_or_jsonl(args.input)
    print(f"Loaded {len(data)} items from: {args.input}")

    converted = []
    for idx, item in enumerate(data):
        converted.append(convert_one(item, idx, args.id_mode))

    write_jsonl(converted, args.output)
    print(f"Saved converted file to: {args.output}")
    print(f"id_mode = {args.id_mode}")


if __name__ == "__main__":
    main()
