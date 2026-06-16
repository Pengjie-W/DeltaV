#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import os
from multiprocessing import Pool
from tqdm import tqdm

# ======================
# Configuration
# ======================

NUM_WORKERS = 8  # file-level parallelism
IMG_PATTERN = re.compile(r"<(input_image_\d+|output_image_\d+)>")

# unified output directory
OUTPUT_DIR = "data/tokens_merged"


# ======================
# Step 1: build the token index
# ======================

def build_token_index(token_path):
    index = {}

    with open(token_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"[Token] {os.path.basename(token_path)}", leave=False):
        key = tuple(item["img_paths"])
        index[key] = item["num_tokens"]

    return index


# ======================
# Step 2: parse the image order within messages
# ======================

def extract_image_order(messages):
    order = []
    for msg in messages:
        matches = IMG_PATTERN.findall(msg["content"])
        order.extend(matches)
    return order


# ======================
# Step 3: build img_paths
# ======================

def build_img_paths(sample, order):
    id2path = {}

    for img in sample.get("input_images", []):
        id2path[img["id"]] = img["path"]

    for img in sample.get("output_images", []):
        id2path[img["id"]] = img["path"]

    paths = []
    for oid in order:
        if oid in id2path:
            paths.append(id2path[oid])
        else:
            raise ValueError(f"Missing image id: {oid}")

    return paths


# ======================
# Step 4: per-record processing (called serially)
# ======================

def process_one(sample, token_index):
    try:
        order = extract_image_order(sample["messages"])
        img_paths = build_img_paths(sample, order)
        key = tuple(img_paths)

        if key in token_index:
            sample["num_tokens"] = token_index[key]
        else:
            sample["num_tokens"] = None
            sample["_warn"] = "token_not_found"

    except Exception as e:
        sample["num_tokens"] = None
        sample["_error"] = str(e)

    return sample


# ======================
# Step 5: streaming read
# ======================

def stream_json(input_path):
    if input_path.endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                yield item


# ======================
# Step 6: single-file processing (no in-file parallelism)
# ======================

def build_output_path(input_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name = os.path.basename(input_path)

    if base_name.endswith(".jsonl"):
        output_name = base_name.replace(".jsonl", "_with_tokens.jsonl")
    elif base_name.endswith(".json"):
        output_name = base_name.replace(".json", "_with_tokens.json")
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    return os.path.join(OUTPUT_DIR, output_name)


def process_file(task):
    group_name = task["group_name"]
    dataset_idx = task["dataset_idx"]
    input_path = task["input_path"]
    token_path = task["token_path"]
    output_path = build_output_path(input_path)

    print("\n========== Processing ==========")
    print(f"Group : {group_name}")
    print(f"Index : {dataset_idx}")
    print(f"Input : {input_path}")
    print(f"Token : {token_path}")
    print(f"Output: {output_path}")

    token_index = build_token_index(token_path)

    count = 0
    matched = 0
    token_not_found = 0
    errors = 0
    bad_cases = []

    with open(output_path, "w", encoding="utf-8") as out:
        if input_path.endswith(".jsonl"):
            for sample in tqdm(
                stream_json(input_path),
                desc=f"Processing {os.path.basename(input_path)}",
                leave=False
            ):
                result = process_one(sample, token_index)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1

                if result.get("num_tokens") is not None:
                    matched += 1
                elif result.get("_warn") == "token_not_found":
                    token_not_found += 1
                    if len(bad_cases) < 10:
                        bad_cases.append({
                            "type": "token_not_found",
                            "id": result.get("id"),
                        })
                elif "_error" in result:
                    errors += 1
                    if len(bad_cases) < 10:
                        bad_cases.append({
                            "type": "error",
                            "id": result.get("id"),
                            "error": result["_error"],
                        })
        else:
            out.write("[\n")
            first = True
            for sample in tqdm(
                stream_json(input_path),
                desc=f"Processing {os.path.basename(input_path)}",
                leave=False
            ):
                result = process_one(sample, token_index)
                if not first:
                    out.write(",\n")
                first = False
                out.write(json.dumps(result, ensure_ascii=False))
                count += 1

                if result.get("num_tokens") is not None:
                    matched += 1
                elif result.get("_warn") == "token_not_found":
                    token_not_found += 1
                    if len(bad_cases) < 10:
                        bad_cases.append({
                            "type": "token_not_found",
                            "id": result.get("id"),
                        })
                elif "_error" in result:
                    errors += 1
                    if len(bad_cases) < 10:
                        bad_cases.append({
                            "type": "error",
                            "id": result.get("id"),
                            "error": result["_error"],
                        })
            out.write("\n]")

    print(f"✅ Done! total={count}")
    print(f"   matched={matched}, token_not_found={token_not_found}, errors={errors}")

    if bad_cases:
        print("   example mismatched samples:")
        for x in bad_cases:
            print("   ", x)

    return {
        "group_name": group_name,
        "dataset_idx": dataset_idx,
        "input_path": input_path,
        "token_path": token_path,
        "output_path": output_path,
        "num_samples": count,
        "matched": matched,
        "token_not_found": token_not_found,
        "errors": errors,
        "match_rate": matched / count if count > 0 else 0.0,
    }


# ======================
# Step 7: batch processing (multiple files in parallel)
# ======================

def flatten_config_to_tasks(config):
    tasks = []
    for group_name, datasets in config.items():
        for idx, ds in enumerate(datasets):
            tasks.append({
                "group_name": group_name,
                "dataset_idx": idx,
                "input_path": ds["path"],
                "token_path": ds["token"],
            })
    return tasks


def process_all(config, new_config_path):
    tasks = flatten_config_to_tasks(config)

    print(f"Total tasks: {len(tasks)}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")

    # create the output directory first
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with Pool(NUM_WORKERS) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_file, tasks),
                total=len(tasks),
                desc="All Files"
            )
        )

    # write output_path back into the original config
    new_config = json.loads(json.dumps(config, ensure_ascii=False))

    for res in results:
        group_name = res["group_name"]
        dataset_idx = res["dataset_idx"]

        new_config[group_name][dataset_idx]["output_path"] = res["output_path"]
        new_config[group_name][dataset_idx]["num_samples"] = res["num_samples"]

        # add statistics
        new_config[group_name][dataset_idx]["matched"] = res["matched"]
        new_config[group_name][dataset_idx]["token_not_found"] = res["token_not_found"]
        new_config[group_name][dataset_idx]["errors"] = res["errors"]
        new_config[group_name][dataset_idx]["match_rate"] = res["match_rate"]

    # save the new config file
    with open(new_config_path, "w", encoding="utf-8") as f:
        json.dump(new_config, f, ensure_ascii=False, indent=2)

    print(f"\n✅ New config saved to: {new_config_path}")


# ======================
# Main entry
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge num_tokens from tokens.json back into the original dataset, keyed by tuple(img_paths)"
    )
    parser.add_argument(
        "--config", type=str,
        default="data/tokens_driver_config.json",
        help="driver config: {group: [{path: dataset, token: tokens.json}, ...]}",
    )
    parser.add_argument(
        "--new_config", type=str,
        default="data/tokens_driver_config.with_output.json",
        help="output path for the new config after writing back output_path / statistics",
    )
    parser.add_argument(
        "--out_dir", type=str, default=OUTPUT_DIR,
        help="unified output directory for *_with_tokens.{json,jsonl}",
    )
    parser.add_argument(
        "--num_workers", type=int, default=NUM_WORKERS, help="file-level parallelism",
    )
    args = parser.parse_args()

    # Override the module-level constants with the CLI values so build_output_path / process_all
    # do not need signature changes. The defaults match the original hardcoded values verbatim, so
    # behavior is identical when no args are passed.
    OUTPUT_DIR = args.out_dir
    NUM_WORKERS = args.num_workers

    with open(args.config, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)

    process_all(CONFIG, args.new_config)