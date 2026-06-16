#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import sys
import json
import argparse
from typing import Dict, List, Iterator, Any, Optional

import ijson
import orjson


IMAGE_TOKEN_RE = re.compile(r"<([a-zA-Z_]+\d+)>")


def build_image_map(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Build an {id: path} mapping from input_images / output_images.
    """
    image_map = {}

    for img in sample.get("input_images", []) or []:
        img_id = img.get("id")
        img_path = img.get("path")
        if img_id is not None and img_path is not None:
            image_map[img_id] = img_path

    for img in sample.get("output_images", []) or []:
        img_id = img.get("id")
        img_path = img.get("path")
        if img_id is not None and img_path is not None:
            image_map[img_id] = img_path

    return image_map


def extract_images_in_message_order(sample: Dict[str, Any], keep_missing: bool = False) -> List[Optional[str]]:
    """
    Extract image paths in the order their image tokens appear in messages.

    For example:
    if messages contain <input_image_2> then <input_image_1>,
    the output is [path_of_input_image_2, path_of_input_image_1]

    keep_missing:
      - False: skip when no mapping is found
      - True : keep None when no mapping is found
    """
    image_map = build_image_map(sample)
    result: List[Optional[str]] = []

    for msg in sample.get("messages", []) or []:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        for token in IMAGE_TOKEN_RE.findall(content):
            if token in image_map:
                result.append(image_map[token])
            elif keep_missing:
                result.append(None)

    return result


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """
    Read JSONL line by line.
    """
    with open(path, "rb") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except Exception as e:
                raise ValueError(f"failed to parse JSONL at line {line_no}: {e}") from e


def iter_json_array(path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream-read a very large JSON array file:
    [
      {...},
      {...},
      ...
    ]
    """
    with open(path, "rb") as f:
        # the top level is a list, so iterate items one by one
        for obj in ijson.items(f, "item"):
            yield obj


def detect_format(path: str, fmt: str) -> str:
    """
    Auto-detect whether the file is jsonl or a json array.
    The user can override this explicitly via --format.
    """
    if fmt != "auto":
        return fmt

    with open(path, "rb") as f:
        while True:
            ch = f.read(1)
            if not ch:
                raise ValueError("empty file; cannot determine the format")
            if ch in b" \t\r\n":
                continue
            if ch == b"[":
                return "json"
            return "jsonl"


def stream_extract(
    input_path: str,
    output_path: str,
    input_format: str = "auto",
    keep_missing: bool = False,
    flush_every: int = 10000,
    progress_every: int = 100000,
) -> None:
    """
    Read and write incrementally, producing a nested-list JSON:
    [
      ["img1", "img2"],
      ["img3"],
      ...
    ]
    """
    real_format = detect_format(input_path, input_format)

    if real_format == "jsonl":
        iterator = iter_jsonl(input_path)
    elif real_format == "json":
        iterator = iter_json_array(input_path)
    else:
        raise ValueError(f"unsupported format: {real_format}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    count = 0
    with open(output_path, "wb") as out:
        out.write(b"[")

        first = True
        for sample in iterator:
            image_list = extract_images_in_message_order(sample, keep_missing=keep_missing)
            data = orjson.dumps(image_list)

            if first:
                out.write(data)
                first = False
            else:
                out.write(b",")
                out.write(data)

            count += 1

            if flush_every > 0 and count % flush_every == 0:
                out.flush()
                os.fsync(out.fileno())

            if progress_every > 0 and count % progress_every == 0:
                print(f"[INFO] processed {count} records", file=sys.stderr)

        out.write(b"]")
        out.flush()
        os.fsync(out.fileno())

    print(f"[DONE] processed {count} records, output saved to: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Extract image paths in the order image tokens appear in messages, and output a nested-list JSON")
    parser.add_argument("--input", required=True, help="input file path, either a JSON array or JSONL")
    parser.add_argument("--output", required=True, help="output JSON path")
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "json", "jsonl"],
        help="input format: auto/json/jsonl",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="if a message references a non-existent image id, keep null in the result; skipped by default",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10000,
        help="flush the output file every N processed records (default 10000)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="print progress every N processed records (default 100000)",
    )

    args = parser.parse_args()

    stream_extract(
        input_path=args.input,
        output_path=args.output,
        input_format=args.format,
        keep_missing=args.keep_missing,
        flush_every=args.flush_every,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
