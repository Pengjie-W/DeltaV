#!/usr/bin/env python3
import os
import re
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
    parser.add_argument("--input", required=True, help="Model prediction json/jsonl file, may contain 7000 samples")
    parser.add_argument("--gt_cache", required=True, help="New GT json file with gt + major_category, 5600 samples")
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
1. If the response contains an explicit final answer, extract the final answer content exactly.
2. If the final answer is multi-line code, HTML, a list, a table, or structured text, keep the full content.
3. If the response selects a specific multiple-choice option letter, output only the single uppercase letter.
4. If no usable answer exists, output "Z".

Output Format:
Directly output the extracted answer ONLY.

Question: {question}
Answer: {response}
"""


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


def clean_text(x):
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None


def normalize_final_answer(ans):
    ans = clean_text(ans)
    if ans is None:
        return None

    ans = ans.strip()
    ans = re.sub(r"[ \t\r\n]+$", "", ans)

    if re.fullmatch(r"[A-Z]", ans):
        return ans

    m = re.fullmatch(r"([A-Z])[\.\)、:：]\s*", ans)
    if m:
        return m.group(1)

    return ans


def extract_by_rules(text):
    text = clean_text(text)
    if text is None:
        return None

    m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text, flags=re.IGNORECASE)
    if m:
        return normalize_final_answer(m.group(1))

    m = re.search(r"<\\answer>\s*([\s\S]*?)\s*</\\answer>", text, flags=re.IGNORECASE)
    if m:
        return normalize_final_answer(m.group(1))

    m = re.search(r"Final\s+Answer\s*[:：]\s*([\s\S]*)", text, flags=re.IGNORECASE)
    if m:
        return normalize_final_answer(m.group(1))

    matches = list(re.finditer(
        r"(?:\*\*)?\bAnswer\s*[:：](?:\*\*)?\s*([\s\S]*)",
        text,
        flags=re.IGNORECASE
    ))
    if matches:
        return normalize_final_answer(matches[-1].group(1))

    matches = list(re.finditer(
        r"The\s+answer\s+is\s*[:：]?\s*([\s\S]*)",
        text,
        flags=re.IGNORECASE
    ))
    if matches:
        return normalize_final_answer(matches[-1].group(1))

    return None


def extract_with_api(question, response_text, item_id_for_log=""):
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
            return normalize_final_answer(completion.choices[0].message.content)
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


def build_gt_map(gt_data):
    gt_map = {}
    for item in gt_data:
        item_id = str(item["id"])
        if item_id in gt_map:
            raise ValueError(f"Duplicate id in gt_cache: {item_id}")
        gt_map[item_id] = item
    return gt_map


def build_pred_map(pred_data):
    pred_map = {}
    duplicate_count = 0

    for item in pred_data:
        item_id = str(item.get("id", ""))
        if item_id in pred_map:
            duplicate_count += 1
            continue
        pred_map[item_id] = item

    return pred_map, duplicate_count


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
    """Returns dict keyed by id (str)."""
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
    pred_item = pred_map.get(item_id)

    if pred_item is None:
        record = {
            "_global_idx": int(item_id) if item_id.isdigit() else item_id,
            "id": item_id,
            "question": gt_item.get("prompt", ""),
            "gt": gt_item.get("gt", None),
            "pred": None,
            "pred_extract_method": "missing_pred",
            "pred_missing": True,
            "major_category": gt_item.get("major_category", ""),
            "gt_extract_method": gt_item.get("gt_extract_method", ""),
        }
        append_progress(progress_path, record)
        return record

    gt_prompt = clean_text(gt_item.get("prompt", ""))
    pred_prompt = clean_text(pred_item.get("prompt", ""))

    if gt_prompt != pred_prompt:
        raise ValueError(
            f"Prompt mismatch for id={item_id}\n\n"
            f"[GT prompt]\n{gt_prompt[:1000]}\n\n"
            f"[PRED prompt]\n{pred_prompt[:1000]}"
        )

    question = gt_item.get("prompt", "")
    pred_source = pred_item.get("pred_text", "")

    pred = extract_by_rules(pred_source)
    method = "rule"

    if pred is None:
        pred = extract_with_api(question, pred_source, item_id_for_log=item_id)
        method = "api_fallback"

    record = {
        "_global_idx": int(item_id) if item_id.isdigit() else item_id,
        "id": item_id,
        "question": question,
        "gt": gt_item.get("gt", None),
        "pred": pred,
        "pred_extract_method": method,
        "pred_missing": False,
        "major_category": gt_item.get("major_category", ""),
        "gt_extract_method": gt_item.get("gt_extract_method", ""),
    }
    append_progress(progress_path, record)
    return record


def main():
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}\nUse --overwrite to overwrite.")

    progress_path = args.progress_path or (args.output + ".progress.jsonl")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.reset_progress and os.path.exists(progress_path):
        os.remove(progress_path)
        print(f"[progress] removed existing progress file: {progress_path}")

    pred_data = read_json(args.input)
    gt_data = read_json(args.gt_cache)

    gt_map = build_gt_map(gt_data)
    pred_map, duplicate_pred_count = build_pred_map(pred_data)

    done_records = load_progress(progress_path)
    if done_records:
        print(
            f"[resume] loaded {len(done_records)} existing records from {progress_path}"
        )

    sorted_gt_items = [
        gt_item
        for _, gt_item in sorted(
            gt_map.items(),
            key=lambda x: int(x[0]) if x[0].isdigit() else x[0],
        )
    ]
    pending = [
        gt_item for gt_item in sorted_gt_items if str(gt_item["id"]) not in done_records
    ]
    print(
        f"[plan] total={len(sorted_gt_items)} done={len(done_records)} pending={len(pending)}"
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
                desc="Extracting subset pred",
            ):
                future.result()

    final_records = load_progress(progress_path)
    results = list(final_records.values())
    results.sort(
        key=lambda x: x["_global_idx"]
        if isinstance(x["_global_idx"], int)
        else str(x["_global_idx"])
    )

    missing_pred_count = sum(bool(x.get("pred_missing")) for x in results)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Done!")
    print(f"GT subset total: {len(gt_data)}")
    print(f"Prediction input total: {len(pred_data)}")
    print(f"Output total: {len(results)}")
    print(f"Missing pred: {missing_pred_count}")
    print(f"Duplicate pred ids ignored: {duplicate_pred_count}")
    print(f"Saved to: {args.output}")
    print(f"Progress kept at: {progress_path}")


if __name__ == "__main__":
    main()
