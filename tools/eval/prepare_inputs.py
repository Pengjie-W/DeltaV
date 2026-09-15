#!/usr/bin/env python3
"""Prepare downloaded public test data for inference, preserving public IDs."""

import argparse
import json
import sys
from pathlib import Path

from evaluate import BENCHMARKS, atomic_json, image_names, load_cache


def prepare_inputs(benchmark, dataset_root, output):
    dataset_root = dataset_root.resolve()
    source = dataset_root / "test" / BENCHMARKS[benchmark]["test_file"]
    if source.resolve() == output.resolve():
        raise ValueError("Do not overwrite the published test file; choose another output path.")
    records = json.loads(source.read_text(encoding="utf-8"))
    cache, _ = load_cache(benchmark)
    if not isinstance(records, list) or len(records) != len(cache):
        raise ValueError("Downloaded test split does not match the bundled evaluation release.")
    prepared = []
    for sample_id, (record, reference) in enumerate(zip(records, cache)):
        if record.get("input_prompt") != reference["prompt"] or image_names(record.get("input_image", [])) != image_names(reference["reference_image"]):
            raise ValueError(f"Test release/order mismatch at public ID {sample_id}.")
        result = {**record, "id": sample_id}
        for field in ("input_image", "output_image"):
            paths = []
            for relative in record.get(field, []):
                image = (dataset_root / relative).resolve()
                if not image.is_relative_to(dataset_root):
                    raise ValueError(f"Image path escapes the dataset root at ID {sample_id}.")
                if not image.is_file():
                    raise ValueError(f"Missing image: {image}. Extract dist_test/StructCoT_test_images.tar in the dataset root first.")
                paths.append(str(image))
            result[field] = paths
        prepared.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, prepared)
    print(f"Prepared {len(prepared)} samples with public IDs and absolute image paths: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare_inputs(args.benchmark, args.dataset_root, args.output)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
