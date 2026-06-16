#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For all jsonl datasets registered in converted.json, stream-append length
information and write the results back to the directory of each original jsonl.
The output filename appends a suffix to the original name, e.g.:
    train.jsonl -> train.with_total_length.jsonl

Design: parallelism within a single jsonl + workers do not touch sqlite.
- No longer parallelize at the "file level".
- Instead: the main process reads a jsonl line by line -> splits into chunks ->
  multiple worker processes handle chunks in parallel.
- The main process writes results back in the original chunk order, keeping the
  output order stable.
- The main process owns sqlite exclusively: workers only compute and neither read
  nor write sqlite, fully avoiding multi-process sqlite lock protocol issues.
- Multiple files are processed serially by default, to avoid excessive I/O jitter
  from combining "in-file parallelism" with "cross-file parallelism".

Features
- Parallel: chunk-level multi-process processing within a single jsonl.
- Streaming: the main process reads line by line and writes chunk by chunk in
  order, without loading the whole jsonl into memory.
- The main process writes the sqlite cache: image sizes / text token lengths /
  run_file_stats.
- Workers have no sqlite dependency, sidestepping the locking protocol.
- Image token estimation: based on the Qwen smart_resize logic, estimated only
  from the original image size without actually resizing.
- Output files are saved by default in the same directory as the original jsonl,
  with the suffix appended automatically.

Notes
- This version prioritizes stability: workers do not read the cache from sqlite,
  only deduplicating locally in memory within the current chunk.
- The main process still writes computed results into sqlite, making it easy to
  later extend into a "main process pre-checks cache, workers only compute misses"
  version.

To fully align with the length definition in a given training script, prefer
adjusting these parameters first:
- --image-wrapper-tokens-per-image
- --tokenizer-path
- --tokenizer-use-fast
- build_text_for_tokenize()
"""

import os
import re
import gc
import json
import math
import time
import sqlite3
import hashlib
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from PIL import Image

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# ----------------------------
# Global variables (reused across worker processes)
# ----------------------------
G_TOKENIZER = None
G_ARGS = None


# ----------------------------
# Basic utilities
# ----------------------------
def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_output_path(src_path: str, suffix_tag: str) -> Path:
    """
    Write to the directory of the original jsonl, appending a suffix to the filename.
    For example:
        /a/b/train.jsonl
    ->  /a/b/train.with_total_length.jsonl
    """
    p = Path(src_path)
    ext = p.suffix or ".jsonl"
    return p.with_name(f"{p.stem}.{suffix_tag}{ext}")


def append_jsonl(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


# ----------------------------
# smart_resize approximation / reproduction
# ----------------------------
def smart_resize_hw(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 256 * 256,
    max_pixels: int = 1024 * 1024,
) -> Tuple[int, int]:
    """
    Approximate Qwen2/3-VL smart_resize, computed only from the size without actually reading/resizing the image.
    factor is usually = patch_size * merge_size
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {(height, width)}")

    if max(height, width) / min(height, width) > 200:
        # keep the guard logic; in practice this could also just be allowed through
        pass

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    pixels_bar = h_bar * w_bar
    raw_pixels = height * width

    if pixels_bar > max_pixels:
        beta = math.sqrt(raw_pixels / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif pixels_bar < min_pixels:
        beta = math.sqrt(min_pixels / raw_pixels)
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)

    return int(h_bar), int(w_bar)


def estimate_image_tokens_from_size(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    patch_size: int = 14,
    merge_size: int = 2,
) -> Tuple[int, int, int]:
    """
    Returns:
    - resized_height
    - resized_width
    - num_image_tokens
    """
    factor = patch_size * merge_size
    rh, rw = smart_resize_hw(
        height=height,
        width=width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    grid_h = rh // patch_size
    grid_w = rw // patch_size
    num_patches = grid_h * grid_w
    num_tokens = num_patches // (merge_size ** 2)
    return rh, rw, int(num_tokens)


# ----------------------------
# sqlite cache (accessed by the main process only)
# ----------------------------
def open_db(db_path: Path) -> sqlite3.Connection:
    timeout_sec = max(1, int(getattr(G_ARGS, "db_busy_timeout_ms", 120000)) // 1000) if G_ARGS else 120
    conn = sqlite3.connect(str(db_path), timeout=timeout_sec)

    if G_ARGS is not None:
        conn.execute(f"PRAGMA busy_timeout={int(G_ARGS.db_busy_timeout_ms)};")
    else:
        conn.execute("PRAGMA busy_timeout=120000;")

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA mmap_size=268435456;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS image_meta_cache (
            image_path TEXT PRIMARY KEY,
            image_mtime_ns INTEGER,
            image_size_bytes INTEGER,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS text_token_cache (
            text_sha1 TEXT NOT NULL,
            tokenizer_key TEXT NOT NULL,
            text_token_len INTEGER NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (text_sha1, tokenizer_key)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_file_stats (
            src_path TEXT NOT NULL,
            output_path TEXT NOT NULL,
            processed_lines INTEGER NOT NULL,
            error_lines INTEGER NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (src_path, output_path)
        )
        """
    )
    conn.commit()


def db_execute_with_retry(conn: sqlite3.Connection, sql: str, params=()):
    retry_times = int(getattr(G_ARGS, "db_retry_times", 8)) if G_ARGS else 8
    retry_sleep = float(getattr(G_ARGS, "db_retry_sleep", 0.2)) if G_ARGS else 0.2

    last_err = None
    for i in range(retry_times + 1):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            last_err = e
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                if i < retry_times:
                    time.sleep(retry_sleep * (i + 1))
                    continue
            raise
    raise last_err


def db_executemany_with_retry(conn: sqlite3.Connection, sql: str, seq_of_params):
    retry_times = int(getattr(G_ARGS, "db_retry_times", 8)) if G_ARGS else 8
    retry_sleep = float(getattr(G_ARGS, "db_retry_sleep", 0.2)) if G_ARGS else 0.2

    last_err = None
    for i in range(retry_times + 1):
        try:
            conn.executemany(sql, seq_of_params)
            return
        except sqlite3.OperationalError as e:
            last_err = e
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                if i < retry_times:
                    time.sleep(retry_sleep * (i + 1))
                    continue
            raise
    raise last_err


def db_commit_with_retry(conn: sqlite3.Connection):
    retry_times = int(getattr(G_ARGS, "db_retry_times", 8)) if G_ARGS else 8
    retry_sleep = float(getattr(G_ARGS, "db_retry_sleep", 0.2)) if G_ARGS else 0.2

    last_err = None
    for i in range(retry_times + 1):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            last_err = e
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                if i < retry_times:
                    time.sleep(retry_sleep * (i + 1))
                    continue
            raise
    raise last_err


def bulk_upsert_text_token_cache(conn: sqlite3.Connection, rows: List[Tuple[str, str, int]]) -> None:
    if not rows:
        return
    now = time.time()
    db_executemany_with_retry(
        conn,
        """
        INSERT INTO text_token_cache (text_sha1, tokenizer_key, text_token_len, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(text_sha1, tokenizer_key) DO UPDATE SET
            text_token_len=excluded.text_token_len,
            updated_at=excluded.updated_at
        """,
        [(text_sha1, tokenizer_key, int(text_token_len), now) for text_sha1, tokenizer_key, text_token_len in rows],
    )


def bulk_upsert_image_meta_cache(
    conn: sqlite3.Connection,
    rows: List[Tuple[str, int, int, int, int]],
) -> None:
    if not rows:
        return
    now = time.time()
    db_executemany_with_retry(
        conn,
        """
        INSERT INTO image_meta_cache (
            image_path, image_mtime_ns, image_size_bytes, width, height, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(image_path) DO UPDATE SET
            image_mtime_ns=excluded.image_mtime_ns,
            image_size_bytes=excluded.image_size_bytes,
            width=excluded.width,
            height=excluded.height,
            updated_at=excluded.updated_at
        """,
        [
            (image_path, int(image_mtime_ns), int(image_size_bytes), int(width), int(height), now)
            for image_path, image_mtime_ns, image_size_bytes, width, height in rows
        ],
    )


# ----------------------------
# tokenizer logic
# ----------------------------
def init_worker(args_dict: dict) -> None:
    global G_TOKENIZER, G_ARGS
    G_ARGS = argparse.Namespace(**args_dict)

    if G_ARGS.tokenizer_path:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not installed, but --tokenizer-path was provided")
        G_TOKENIZER = AutoTokenizer.from_pretrained(
            G_ARGS.tokenizer_path,
            use_fast=bool(G_ARGS.tokenizer_use_fast),
            trust_remote_code=True,
        )
    else:
        G_TOKENIZER = None


_IMAGE_PLACEHOLDER_RE = re.compile(r"<(?:input|output)_image_\d+>")


def build_text_for_tokenize(sample: dict, strip_image_placeholders: bool = True) -> str:
    """
    Build the same text structure used during training:
    <|im_start|>{role}\n
    content
    <|im_end|>\n
    """
    messages = sample.get("messages", [])
    chunks: List[str] = []

    for m in messages:
        role = str(m.get("role", ""))
        content = m.get("content", "")

        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        if strip_image_placeholders:
            content = _IMAGE_PLACEHOLDER_RE.sub("", content)

        prefix = f"<|im_start|>{role}\n"
        suffix = "<|im_end|>\n"
        chunks.append(prefix + content + suffix)

    return "".join(chunks)


def get_text_token_len_nocache(text: str) -> Tuple[int, str]:
    global G_TOKENIZER
    if G_TOKENIZER is None:
        raise RuntimeError("no tokenizer provided; cannot compute text token length")
    text_id = sha1_text(text)
    token_len = len(G_TOKENIZER.encode(text, add_special_tokens=False))
    return int(token_len), text_id


# ----------------------------
# sample processing
# ----------------------------
def file_stat_tuple(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    return int(st.st_mtime_ns), int(st.st_size)


def get_image_size_nocache(image_path: str) -> Tuple[int, int, int, int]:
    """
    Returns:
    - width
    - height
    - mtime_ns
    - size_bytes
    """
    mtime_ns, size_bytes = file_stat_tuple(image_path)
    with Image.open(image_path) as img:
        width, height = img.size
    return int(width), int(height), int(mtime_ns), int(size_bytes)


def collect_input_images(sample: dict) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for img in sample.get("input_images", []) or []:
        img_id = str(img.get("id", ""))
        img_path = str(img.get("path", ""))
        if img_path:
            items.append((img_id, img_path))
    return items


def collect_output_images(sample: dict) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for img in sample.get("output_images", []) or []:
        img_id = str(img.get("id", ""))
        img_path = str(img.get("path", ""))
        if img_path:
            items.append((img_id, img_path))
    return items


def split_num_tokens_by_images(sample: dict) -> Tuple[List[int], List[int]]:
    """
    Split sample['num_tokens'] by convention into:
    - the token list for input_images
    - the token list for output_images

    The original num_tokens order is assumed to be:
        input_images + output_images
    """
    raw_num_tokens = sample.get("num_tokens", None)
    if raw_num_tokens is None:
        raise ValueError("sample has no 'num_tokens' field")

    if not isinstance(raw_num_tokens, list):
        raise ValueError(f"sample['num_tokens'] is not a list: {type(raw_num_tokens)}")

    n_input = len(sample.get("input_images", []) or [])
    n_output = len(sample.get("output_images", []) or [])

    expected_len = n_input + n_output
    if len(raw_num_tokens) < expected_len:
        raise ValueError(
            f"sample['num_tokens'] length ({len(raw_num_tokens)}) < "
            f"len(input_images)+len(output_images) ({expected_len})"
        )

    input_tokens = raw_num_tokens[:n_input]
    output_tokens = raw_num_tokens[n_input:n_input + n_output]

    def norm_list(xs, name):
        out = []
        for i, x in enumerate(xs):
            if isinstance(x, bool):
                raise ValueError(f"invalid bool token in {name}[{i}]: {x}")
            if not isinstance(x, (int, float)):
                raise ValueError(f"invalid token type in {name}[{i}]: {type(x)}")
            out.append(int(x))
        return out

    return norm_list(input_tokens, "input_tokens"), norm_list(output_tokens, "output_tokens")


def process_one_sample(
    sample: dict,
    tokenizer_key: str,
    min_pixels: int,
    max_pixels: int,
    patch_size: int,
    merge_size: int,
    image_wrapper_tokens_per_image: int,
    strip_image_placeholders: bool,
    keep_original_num_tokens: bool,
    local_text_cache: Optional[Dict[str, Tuple[int, str]]] = None,
    local_image_cache: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
) -> Tuple[dict, Tuple[str, str, int], List[Tuple[str, int, int, int, int]]]:
    """
    New accounting:
    - text token: as before
    - input_images:
        1) vision tokens estimated from the image size
        2) the input_images tokens in sample['num_tokens']
        both parts add +2 per image
    - output_images:
        use only the output_images tokens in sample['num_tokens']
        adding +2 per image

    Finally:
        image_num_tokens =
            input_vision_total
            + input_num_tokens_total
            + output_num_tokens_total
    """
    text = build_text_for_tokenize(sample, strip_image_placeholders=strip_image_placeholders)

    if local_text_cache is not None and text in local_text_cache:
        text_token_len, text_id = local_text_cache[text]
    else:
        text_token_len, text_id = get_text_token_len_nocache(text)
        if local_text_cache is not None:
            local_text_cache[text] = (text_token_len, text_id)

    input_image_items = collect_input_images(sample)
    output_image_items = collect_output_images(sample)

    # Text-only samples (no input/output images) should not hard-depend on the num_tokens field.
    # Some data explicitly sets num_tokens=null; treat that as "no image tokens" as well.
    if len(input_image_items) == 0 and len(output_image_items) == 0:
        input_num_tokens_list, output_num_tokens_list = [], []
    else:
        input_num_tokens_list, output_num_tokens_list = split_num_tokens_by_images(sample)

    # 1) input_images "size-estimated tokens"
    input_vision_token_list: List[int] = []
    input_vision_detail: List[dict] = []
    image_cache_rows: List[Tuple[str, int, int, int, int]] = []

    for image_id, image_path in input_image_items:
        if local_image_cache is not None and image_path in local_image_cache:
            width, height, mtime_ns, size_bytes = local_image_cache[image_path]
        else:
            width, height, mtime_ns, size_bytes = get_image_size_nocache(image_path)
            if local_image_cache is not None:
                local_image_cache[image_path] = (width, height, mtime_ns, size_bytes)

        resized_h, resized_w, image_tokens = estimate_image_tokens_from_size(
            height=height,
            width=width,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            patch_size=patch_size,
            merge_size=merge_size,
        )

        input_vision_token_list.append(int(image_tokens))
        input_vision_detail.append(
            {
                "id": image_id,
                "path": image_path,
                "width": int(width),
                "height": int(height),
                "resized_width": int(resized_w),
                "resized_height": int(resized_h),
                "vision_tokens": int(image_tokens),
            }
        )
        image_cache_rows.append(
            (image_path, int(mtime_ns), int(size_bytes), int(width), int(height))
        )

    input_vision_raw_sum = int(sum(input_vision_token_list))
    input_vision_wrapper_sum = int(len(input_image_items) * image_wrapper_tokens_per_image)
    input_vision_total = int(input_vision_raw_sum + input_vision_wrapper_sum)

    # 2) input_images' original num_tokens
    input_num_tokens_raw_sum = int(sum(input_num_tokens_list))
    input_num_tokens_wrapper_sum = int(len(input_num_tokens_list) * image_wrapper_tokens_per_image)
    input_num_tokens_total = int(input_num_tokens_raw_sum + input_num_tokens_wrapper_sum)

    input_num_tokens_detail: List[dict] = []
    for idx, (image_id, image_path) in enumerate(input_image_items):
        tok = input_num_tokens_list[idx] if idx < len(input_num_tokens_list) else 0
        input_num_tokens_detail.append(
            {
                "id": image_id,
                "path": image_path,
                "tokens_from_num_tokens": int(tok),
            }
        )

    # 3) output_images' original num_tokens
    output_num_tokens_raw_sum = int(sum(output_num_tokens_list))
    output_num_tokens_wrapper_sum = int(len(output_num_tokens_list) * image_wrapper_tokens_per_image)
    output_num_tokens_total = int(output_num_tokens_raw_sum + output_num_tokens_wrapper_sum)

    output_num_tokens_detail: List[dict] = []
    for idx, (image_id, image_path) in enumerate(output_image_items):
        tok = output_num_tokens_list[idx] if idx < len(output_num_tokens_list) else 0
        output_num_tokens_detail.append(
            {
                "id": image_id,
                "path": image_path,
                "tokens_from_num_tokens": int(tok),
            }
        )

    # 4) total image tokens
    image_total_token_len = int(
        input_vision_total
        + input_num_tokens_total
        + output_num_tokens_total
    )
    total_token_len = int(text_token_len + image_total_token_len)

    # 5) write fields back
    sample["text_num_tokens"] = int(text_token_len)

    sample["input_image_vision_num_tokens"] = int(input_vision_total)
    sample["input_image_vision_num_tokens_list"] = input_vision_token_list
    sample["input_image_size_info"] = input_vision_detail

    sample["input_image_num_tokens_from_sample"] = int(input_num_tokens_total)
    sample["input_image_num_tokens_from_sample_list"] = input_num_tokens_list

    sample["output_image_num_tokens"] = int(output_num_tokens_total)
    sample["output_image_num_tokens_list"] = output_num_tokens_list

    sample["image_num_tokens"] = int(image_total_token_len)
    sample["token_length"] = int(total_token_len)

    text_cache_row = (text_id, tokenizer_key, int(text_token_len))
    return sample, text_cache_row, image_cache_rows


# ----------------------------
# single-chunk processing (worker)
# ----------------------------
def process_one_chunk(task: dict) -> dict:
    """
    A worker processes one chunk:
    - does not access sqlite
    - performs pure computation only
    - returns processed jsonl lines + cache records for the main process to write
    """
    global G_ARGS

    args = G_ARGS
    src_path = task["src_path"]
    dataset_name = task["dataset_name"]
    chunk_idx = task["chunk_idx"]
    lines = task["lines"]

    tokenizer_key = (
        f"{args.tokenizer_path}|fast={int(args.tokenizer_use_fast)}|"
        f"strip={int(args.strip_image_placeholders)}"
    )

    out_lines: List[str] = []
    error_rows: List[dict] = []
    processed = 0
    errors = 0

    local_text_cache: Dict[str, Tuple[int, str]] = {}
    local_image_cache: Dict[str, Tuple[int, int, int, int]] = {}

    text_cache_map: Dict[Tuple[str, str], Tuple[str, str, int]] = {}
    image_cache_map: Dict[str, Tuple[str, int, int, int, int]] = {}

    for item in lines:
        line_idx = item["line_idx"]
        raw_line = item["line"]

        line = raw_line.strip()
        if not line:
            continue

        try:
            sample = json.loads(line)
            sample, text_cache_row, image_cache_rows = process_one_sample(
                sample=sample,
                tokenizer_key=tokenizer_key,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                patch_size=args.patch_size,
                merge_size=args.merge_size,
                image_wrapper_tokens_per_image=args.image_wrapper_tokens_per_image,
                strip_image_placeholders=bool(args.strip_image_placeholders),
                keep_original_num_tokens=bool(args.keep_original_num_tokens),
                local_text_cache=local_text_cache,
                local_image_cache=local_image_cache,
            )

            out_lines.append(json.dumps(sample, ensure_ascii=False) + "\n")
            processed += 1

            text_cache_map[(text_cache_row[0], text_cache_row[1])] = text_cache_row
            for row in image_cache_rows:
                image_cache_map[row[0]] = row

        except Exception as e:
            errors += 1
            if args.skip_bad_lines:
                error_rows.append(
                    {
                        "src_path": src_path,
                        "dataset_name": dataset_name,
                        "chunk_idx": chunk_idx,
                        "line_idx": line_idx,
                        "error": repr(e),
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
                continue
            raise

    gc.collect()

    return {
        "src_path": src_path,
        "dataset_name": dataset_name,
        "chunk_idx": chunk_idx,
        "out_lines": out_lines,
        "processed_lines": processed,
        "error_lines": errors,
        "error_rows": error_rows,
        "text_cache_rows": list(text_cache_map.values()),
        "image_cache_rows": list(image_cache_map.values()),
    }


# ----------------------------
# in-file parallel processing
# ----------------------------
def process_one_jsonl_file_parallel(
    task: dict,
    ex: ProcessPoolExecutor,
    args,
    cache_conn: sqlite3.Connection,
) -> dict:
    """
    In-file parallelism for a single jsonl file:
    - main process: stream-read -> submit chunks -> write out in chunk_idx order
    - workers: pure computation, no sqlite access
    - main process: exclusively writes the sqlite cache
    """
    src_path = task["src_path"]
    dst_path = task["dst_path"]
    dataset_name = task["dataset_name"]

    ensure_dir(Path(dst_path).parent)

    chunk_size_lines = max(1, int(args.chunk_size_lines))
    max_inflight_chunks = max(1, int(args.max_inflight_chunks))

    processed_total = 0
    errors_total = 0
    next_chunk_to_write = 0
    chunk_idx = 0

    pending_futs = set()
    fut_to_chunk_idx = {}
    ready_results = {}

    def submit_chunk(chunk_idx_: int, chunk_lines_: List[dict]):
        fut = ex.submit(
            process_one_chunk,
            {
                "src_path": src_path,
                "dst_path": dst_path,
                "dataset_name": dataset_name,
                "chunk_idx": chunk_idx_,
                "lines": chunk_lines_,
            },
        )
        pending_futs.add(fut)
        fut_to_chunk_idx[fut] = chunk_idx_

    def flush_completed(done_futs, fout):
        nonlocal processed_total, errors_total, next_chunk_to_write

        for fut in done_futs:
            pending_futs.discard(fut)
            fut_to_chunk_idx.pop(fut, None)
            res = fut.result()
            ready_results[res["chunk_idx"]] = res

        while next_chunk_to_write in ready_results:
            res = ready_results.pop(next_chunk_to_write)

            if res.get("text_cache_rows"):
                bulk_upsert_text_token_cache(cache_conn, res["text_cache_rows"])

            if res.get("image_cache_rows"):
                bulk_upsert_image_meta_cache(cache_conn, res["image_cache_rows"])

            db_commit_with_retry(cache_conn)

            if res["out_lines"]:
                fout.writelines(res["out_lines"])

            processed_total += int(res["processed_lines"])
            errors_total += int(res["error_lines"])

            if res["error_rows"]:
                append_jsonl(args.error_log, res["error_rows"])

            if args.log_every > 0 and processed_total > 0 and processed_total % args.log_every == 0:
                log(f"[{dataset_name}] {src_path} -> {processed_total} lines done")

            next_chunk_to_write += 1

    with open(src_path, "r", encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        chunk_lines: List[dict] = []
        line_idx = 0

        for raw_line in fin:
            line_idx += 1
            chunk_lines.append(
                {
                    "line_idx": line_idx,
                    "line": raw_line,
                }
            )

            if len(chunk_lines) >= chunk_size_lines:
                submit_chunk(chunk_idx, chunk_lines)
                chunk_idx += 1
                chunk_lines = []

                if len(pending_futs) >= max_inflight_chunks:
                    done, _ = wait(pending_futs, return_when=FIRST_COMPLETED)
                    flush_completed(done, fout)

        if chunk_lines:
            submit_chunk(chunk_idx, chunk_lines)
            chunk_idx += 1

        while pending_futs:
            done, _ = wait(pending_futs, return_when=FIRST_COMPLETED)
            flush_completed(done, fout)

    db_execute_with_retry(
        cache_conn,
        """
        INSERT INTO run_file_stats (src_path, output_path, processed_lines, error_lines, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(src_path, output_path) DO UPDATE SET
            processed_lines=excluded.processed_lines,
            error_lines=excluded.error_lines,
            updated_at=excluded.updated_at
        """,
        (src_path, dst_path, int(processed_total), int(errors_total), time.time()),
    )
    db_commit_with_retry(cache_conn)

    return {
        "src_path": src_path,
        "dst_path": dst_path,
        "dataset_name": dataset_name,
        "processed_lines": int(processed_total),
        "error_lines": int(errors_total),
        "num_chunks": int(chunk_idx),
    }


# ----------------------------
# config loading
# ----------------------------
def load_tasks_from_converted_json(config_path: Path, suffix_tag: str) -> List[dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    tasks: List[dict] = []
    for category, category_info in obj.items():
        datasets = (category_info or {}).get("datasets", []) or []
        for ds in datasets:
            src_path = ds.get("output_path")
            dataset_name = ds.get("name", "")
            if not src_path:
                continue

            dst_path = build_output_path(src_path, suffix_tag)
            tasks.append(
                {
                    "category": category,
                    "dataset_name": dataset_name,
                    "src_path": str(src_path),
                    "dst_path": str(dst_path),
                }
            )
    return tasks


def write_rewritten_config(
    src_config_path: Path,
    dst_config_path: Path,
    suffix_tag: str,
) -> None:
    with open(src_config_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    for _, category_info in obj.items():
        datasets = (category_info or {}).get("datasets", []) or []
        for ds in datasets:
            src_path = ds.get("output_path")
            if not src_path:
                continue
            ds["output_path"] = str(build_output_path(src_path, suffix_tag))

    ensure_dir(dst_config_path.parent)
    with open(dst_config_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ----------------------------
# main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream-parallel appending of jsonl sample lengths, caching image sizes / text token lengths (workers do not access sqlite; output written back to the original directory)"
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default="data/your_dataset.json",
        help="path to your converted.json",
    )
    parser.add_argument(
        "--cache-db",
        type=str,
        default="./cache/add_lengths_cache.sqlite",
        help="sqlite cache path; written by the main process only",
    )
    parser.add_argument(
        "--rewritten-config-path",
        type=str,
        default="data/your_dataset.with_length.json",
        help="output path of the new config after rewriting output_path",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="with_total_length",
        help="output filename suffix; results are saved in the same directory as the original jsonl",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="e.g. Qwen/Qwen2.5-VL-7B-Instruct or a local tokenizer path",
    )
    parser.add_argument("--tokenizer-use-fast", type=int, default=1)

    parser.add_argument("--min-pixels", type=int, default=256 * 256)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--merge-size", type=int, default=2)

    parser.add_argument(
        "--image-wrapper-tokens-per-image",
        type=int,
        default=2,
        help="extra wrapper tokens per image (before + after); default 2",
    )
    parser.add_argument(
        "--strip-image-placeholders",
        type=int,
        default=1,
        help="whether to strip placeholders like <input_image_1> when counting text tokens; default 1",
    )
    parser.add_argument(
        "--keep-original-num-tokens",
        type=int,
        default=1,
        help="keep the sample's original num_tokens field by default; set to 0 to overwrite it with image_num_tokens_list",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() // 2),
        help="number of parallel workers within a single jsonl",
    )
    parser.add_argument(
        "--chunk-size-lines",
        type=int,
        default=256,
        help="lines per chunk; larger means lower scheduling overhead, smaller means more streaming",
    )
    parser.add_argument(
        "--max-inflight-chunks",
        type=int,
        default=0,
        help="max in-flight chunks; 0 means auto-set to num_workers * 4",
    )

    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--skip-bad-lines", type=int, default=1)
    parser.add_argument(
        "--error-log",
        type=str,
        default="./with_length_errors.jsonl",
    )

    parser.add_argument("--db-busy-timeout-ms", type=int, default=120000)
    parser.add_argument("--db-retry-times", type=int, default=8)
    parser.add_argument("--db-retry-sleep", type=float, default=0.2)

    return parser.parse_args()


def main():
    global G_ARGS

    args = parse_args()
    G_ARGS = args

    if args.max_inflight_chunks <= 0:
        args.max_inflight_chunks = max(1, args.num_workers * 4)

    config_path = Path(args.config_path)
    cache_db = Path(args.cache_db)
    rewritten_config_path = Path(args.rewritten_config_path)

    ensure_dir(cache_db.parent)
    ensure_dir(Path(args.error_log).resolve().parent)
    ensure_dir(rewritten_config_path.parent)

    cache_conn = open_db(cache_db)
    init_db(cache_conn)

    suffix_tag = args.output_suffix
    tasks = load_tasks_from_converted_json(config_path, suffix_tag=suffix_tag)

    write_rewritten_config(
        src_config_path=config_path,
        dst_config_path=rewritten_config_path,
        suffix_tag=suffix_tag,
    )

    log(f"loaded {len(tasks)} files from: {config_path}")
    log(f"rewritten config saved to: {rewritten_config_path}")
    log("output mode: save beside original jsonl")
    log(f"output suffix: {suffix_tag}")
    log(f"cache db: {cache_db}")
    log(f"num_workers (per file): {args.num_workers}")
    log(f"chunk_size_lines: {args.chunk_size_lines}")
    log(f"max_inflight_chunks: {args.max_inflight_chunks}")

    results = []
    t0 = time.time()

    worker_args = vars(args).copy()

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(worker_args,),
    ) as ex:
        for i, task in enumerate(tasks, start=1):
            log(f"start [{i}/{len(tasks)}]: {task['dataset_name']} | {task['src_path']}")
            res = process_one_jsonl_file_parallel(
                task=task,
                ex=ex,
                args=args,
                cache_conn=cache_conn,
            )
            results.append(res)
            log(
                f"done: {res['dataset_name']} | "
                f"lines={res['processed_lines']} | errors={res['error_lines']} | "
                f"chunks={res['num_chunks']} | {res['dst_path']}"
            )

    total_lines = sum(x["processed_lines"] for x in results)
    total_errors = sum(x["error_lines"] for x in results)
    dt = time.time() - t0

    summary = {
        "config_path": str(config_path),
        "rewritten_config_path": str(rewritten_config_path),
        "cache_db": str(cache_db),
        "num_files": len(results),
        "total_lines": int(total_lines),
        "total_errors": int(total_errors),
        "seconds": round(dt, 2),
        "parallel_mode": "per_jsonl_chunk_parallel_worker_no_sqlite",
        "output_mode": "same_dir_as_source",
        "output_suffix": suffix_tag,
        "num_workers": int(args.num_workers),
        "chunk_size_lines": int(args.chunk_size_lines),
        "max_inflight_chunks": int(args.max_inflight_chunks),
    }

    summary_path = rewritten_config_path.with_suffix(".run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cache_conn.close()

    log("all done")
    log(f"summary saved to: {summary_path}")
    log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
