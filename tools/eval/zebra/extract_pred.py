#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gt_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--model", default="qwen2.5-72b-instruct")
    parser.add_argument(
        "--retry_log_every",
        type=int,
        default=10,
        help="Print a warning every N retries for the same item.",
    )
    parser.add_argument(
        "--retry_backoff_cap",
        type=float,
        default=60.0,
        help="Maximum sleep seconds between retries.",
    )
    parser.add_argument(
        "--progress_path",
        default="",
        help="Optional explicit progress (resume) file. Default: <output>.progress.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reset_progress",
        action="store_true",
        help="Ignore and delete any existing progress file before starting.",
    )
    return parser.parse_args()


args = parse_args()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

PROMPT_TEMPLATE = """You are an expert answer extractor. Your strictly single task is to extract the final conclusion from the provided response to a given question. You must NOT judge whether the response is correct or wrong; only extract what the response claims is the answer.

Extraction Rules:
1. [Multiple-Choice Tasks]:
   - If the response selects a specific option letter, output ONLY the single uppercase letter.
   - If the response gives both an option letter and its content, return only the letter.

2. [Open-Ended / Value Tasks]:
   - Extract the final requested answer only.
   - Do not add explanation.
   - Keep the answer concise and faithful.

3. [No Valid Answer]:
   - If no usable answer exists, output "Z".

Output Format:
Directly output the extracted answer ONLY.

Question: {question}
Answer: {response}"""


_progress_lock = threading.Lock()


def read_json(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Input must be list/jsonl records: {path}")

    return data


def build_map(data, name):
    mp = {}
    for item in data:
        item_id = str(item.get("id", ""))
        if not item_id:
            raise ValueError(f"{name} has sample without id")
        if item_id in mp:
            raise ValueError(f"Duplicate id in {name}: {item_id}")
        mp[item_id] = item
    return mp


def check_alignment(gt_item, pred_item):
    item_id = str(gt_item.get("id", ""))

    gt_prompt = str(gt_item.get("prompt", "")).strip()
    pred_prompt = str(pred_item.get("prompt", "")).strip()
    if gt_prompt != pred_prompt:
        raise ValueError(f"Prompt mismatch for id={item_id}")

    gt_config = str(gt_item.get("config", "")).strip()
    pred_config = str(pred_item.get("config", "")).strip()
    if gt_config != pred_config:
        raise ValueError(f"Config mismatch for id={item_id}")

    # Compare by basename only: GT cache and predictions may carry different path
    # prefixes (e.g. dataset/.../images/foo.jpg vs data/image/.../foo.jpg) while
    # referring to the same file. Path prefixes differ but the file is the same.
    def _basenames(imgs):
        if isinstance(imgs, str):
            imgs = [imgs]
        elif imgs is None:
            imgs = []
        return [os.path.basename(str(p)) for p in imgs]

    if _basenames(gt_item.get("reference_image", [])) != _basenames(
        pred_item.get("reference_image", [])
    ):
        raise ValueError(f"Reference image mismatch for id={item_id}")


def extract_answer_with_api(question, response_text, item_id_for_log=""):
    """Call the API and retry forever until success."""
    response_text = response_text or ""
    prompt = PROMPT_TEMPLATE.format(question=question, response=response_text)

    attempt = 0
    backoff = 1.0
    while True:
        attempt += 1
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                extra_body={"enable_thinking": False},
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 1 or attempt % max(1, args.retry_log_every) == 0:
                print(
                    f"[retry] extract id={item_id_for_log} attempt={attempt} "
                    f"err={type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(min(backoff, args.retry_backoff_cap))
            backoff = min(backoff * 2.0, args.retry_backoff_cap)


def get_major_category(config):
    config = str(config).strip()
    if " - " in config:
        return config.split(" - ")[0].strip()
    return config


def append_progress(progress_path, record):
    line = json.dumps(record, ensure_ascii=False)
    with _progress_lock:
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


def load_progress(progress_path):
    """Returns (records_by_id, ordered_ids_loaded)."""
    records = {}
    if not os.path.exists(progress_path):
        return records

    with open(progress_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(
                    f"[progress] skipping malformed line {ln}: {e}",
                    file=sys.stderr,
                )
                continue
            rid = str(rec.get("id", ""))
            if rid:
                records[rid] = rec

    return records


def process_one(gt_item, pred_map, progress_path):
    item_id = str(gt_item["id"])
    idx = int(item_id) if item_id.isdigit() else item_id

    pred_item = pred_map.get(item_id)

    question = gt_item.get("prompt", "")
    config = gt_item.get("config", "")
    gt = gt_item.get("gt", None)

    if pred_item is None:
        record = {
            "_global_idx": idx,
            "id": item_id,
            "config": config,
            "major_category": get_major_category(config),
            "question": question,
            "gt": gt,
            "pred": None,
            "pred_missing": True,
            "pred_extract_method": "missing_pred",
            "gt_extract_method": gt_item.get("gt_extract_method", ""),
        }
        append_progress(progress_path, record)
        return record

    check_alignment(gt_item, pred_item)

    pred_source = pred_item.get("pred_text", "")
    pred = extract_answer_with_api(question, pred_source, item_id_for_log=item_id)

    record = {
        "_global_idx": idx,
        "id": item_id,
        "config": config,
        "major_category": get_major_category(config),
        "question": question,
        "gt": gt,
        "pred": pred,
        "pred_missing": False,
        "pred_extract_method": "api",
        "gt_extract_method": gt_item.get("gt_extract_method", ""),
    }
    append_progress(progress_path, record)
    return record


def main():
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}\n"
            f"Use --overwrite if you really want to overwrite."
        )

    progress_path = args.progress_path or (args.output + ".progress.jsonl")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.reset_progress and os.path.exists(progress_path):
        os.remove(progress_path)
        print(f"[progress] removed existing progress file: {progress_path}")

    pred_data = read_json(args.input)
    gt_data = read_json(args.gt_cache)

    pred_map = build_map(pred_data, "prediction file")

    done_records = load_progress(progress_path)
    if done_records:
        print(
            f"[resume] loaded {len(done_records)} existing records from {progress_path}"
        )

    pending = [
        gt_item for gt_item in gt_data if str(gt_item["id"]) not in done_records
    ]
    print(
        f"[plan] total={len(gt_data)} done={len(done_records)} pending={len(pending)}"
    )

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_one, gt_item, pred_map, progress_path)
                for gt_item in pending
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting Zebra Pred",
            ):
                future.result()

    final_records = load_progress(progress_path)

    results = list(final_records.values())
    results.sort(
        key=lambda x: x["_global_idx"]
        if isinstance(x["_global_idx"], int)
        else str(x["_global_idx"])
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"GT total: {len(gt_data)}")
    print(f"Prediction total: {len(pred_data)}")
    print(f"Output total: {len(results)}")
    print(f"Missing pred: {sum(bool(x.get('pred_missing')) for x in results)}")
    print(f"Saved to: {args.output}")
    print(f"Progress kept at: {progress_path}")


if __name__ == "__main__":
    main()
