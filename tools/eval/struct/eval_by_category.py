#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import threading
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", required=True)
    parser.add_argument("--output_json_path", default=None)
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


SYSTEM_PROMPT = """You are an AI assistant who will help judge whether a model answer is consistent with the standard answer for a given question.

Your task:
Given a question, a standard answer, and a model answer, determine whether the model answer is consistent with the standard answer.

Evaluation Criteria:

1. Multiple-choice questions
If the question contains explicit options such as A/B/C/D:
- The predicted answer must match the correct option.
- Acceptable forms include the option letter, the option letter with description, or the option content itself if it clearly matches the standard answer.

2. Numerical or counting questions
- The predicted numeric value must match the ground truth numeric value.
- Equivalent formats are acceptable, such as 36% and 36 percent.
- If the numeric value differs, mark Incorrect.

3. Open-ended answers
- Mark Correct if the prediction conveys the same answer as the standard answer.
- Minor wording differences are acceptable.

4. Procedural or step-based tasks
- Mark Correct if the predicted sequence or final result is consistent with the standard answer.
- Mark Incorrect if key actions/results contradict the standard answer.

5. Strict incorrect cases
Mark Incorrect if:
- the key entity/object differs;
- the selected option differs;
- the numeric value differs;
- the predicted answer is empty, missing, or only says it cannot answer;
- the predicted answer contradicts the standard answer.

Output format:
Output ONLY one label:
Correct
Incorrect
"""


_progress_lock = threading.Lock()


def build_user_prompt(question, prediction, ground_truth):
    return f"""Question:
{question}

[Model Answer]:
{prediction}

[Standard Answer]:
{ground_truth}

Your output:
"""


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_output_path(input_path):
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    return os.path.join(dir_name, f"{name}_evaluated_by_category{ext}")


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
    """Returns dict keyed by progress_key (str)."""
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
            key = str(rec.get("_progress_key", ""))
            if key:
                records[key] = rec

    return records


def make_progress_key(item):
    """Stable identifier for an input item across runs."""
    gid = item.get("_global_idx")
    iid = item.get("id", "")
    if gid is not None:
        return f"idx:{gid}"
    return f"id:{iid}"


def call_judge_api(user_prompt, item_id_for_log=""):
    """Call the judge API and retry forever until success."""
    attempt = 0
    backoff = 1.0
    while True:
        attempt += 1
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                extra_body={"enable_thinking": False},
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 1 or attempt % max(1, args.retry_log_every) == 0:
                print(
                    f"[retry] judge id={item_id_for_log} attempt={attempt} "
                    f"err={type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(min(backoff, args.retry_backoff_cap))
            backoff = min(backoff * 2.0, args.retry_backoff_cap)


def judge_one_example(item, progress_path):
    item_id = item.get("id", "")
    idx = item.get("_global_idx", item_id)
    question = item.get("question", "")
    ground_truth = item.get("gt", "")
    prediction = item.get("pred", "")
    major_category = item.get("major_category", "")
    progress_key = make_progress_key(item)

    if (
        item.get("pred_missing") is True
        or prediction is None
        or str(prediction).strip() == ""
    ):
        record = {
            "_progress_key": progress_key,
            "_global_idx": idx,
            "id": item_id,
            "question": question,
            "gt": ground_truth,
            "pred": prediction,
            "major_category": major_category,
            "verdict": "Incorrect",
            "raw_judge_output": "Missing prediction",
        }
        append_progress(progress_path, record)
        return record

    user_prompt = build_user_prompt(question, prediction, ground_truth)
    verdict_raw = call_judge_api(user_prompt, item_id_for_log=item_id)

    if "Incorrect" in verdict_raw:
        verdict = "Incorrect"
    elif "Correct" in verdict_raw:
        verdict = "Correct"
    else:
        verdict = "Invalid"

    record = {
        "_progress_key": progress_key,
        "_global_idx": idx,
        "id": item_id,
        "question": question,
        "gt": ground_truth,
        "pred": prediction,
        "major_category": major_category,
        "verdict": verdict,
        "raw_judge_output": verdict_raw,
    }
    append_progress(progress_path, record)
    return record


def main():
    input_path = args.input_json_path
    output_path = args.output_json_path or build_output_path(input_path)

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}\nUse --overwrite to overwrite.")

    progress_path = args.progress_path or (output_path + ".progress.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.reset_progress and os.path.exists(progress_path):
        os.remove(progress_path)
        print(f"[progress] removed existing progress file: {progress_path}")

    data = read_json(input_path)

    done_records = load_progress(progress_path)
    if done_records:
        print(
            f"[resume] loaded {len(done_records)} existing records from {progress_path}"
        )

    pending = [
        item for item in data if make_progress_key(item) not in done_records
    ]
    print(
        f"[plan] total={len(data)} done={len(done_records)} pending={len(pending)}"
    )

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(judge_one_example, item, progress_path)
                for item in pending
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Judging"
            ):
                future.result()

    final_records = load_progress(progress_path)
    results = list(final_records.values())
    for rec in results:
        rec.pop("_progress_key", None)

    results.sort(key=lambda x: x.get("_global_idx", -1) if isinstance(x.get("_global_idx"), int) else -1)

    correct_count = sum(x["verdict"] == "Correct" for x in results)
    incorrect_count = sum(x["verdict"] == "Incorrect" for x in results)
    invalid_count = sum(x["verdict"] == "Invalid" for x in results)
    error_count = sum(str(x["verdict"]).startswith("Error") for x in results)

    total = len(results)
    accuracy = correct_count / total if total else 0

    category_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "invalid": 0,
        "error": 0,
    })

    for result in results:
        category = result.get("major_category", "")
        if not category:
            category = "__MISSING_CATEGORY__"

        category_stats[category]["total"] += 1

        if result["verdict"] == "Correct":
            category_stats[category]["correct"] += 1
        elif result["verdict"] == "Incorrect":
            category_stats[category]["incorrect"] += 1
        elif result["verdict"] == "Invalid":
            category_stats[category]["invalid"] += 1
        elif str(result["verdict"]).startswith("Error"):
            category_stats[category]["error"] += 1

    category_summary = {}

    for category, stats in sorted(category_stats.items()):
        total_cat = stats["total"]
        category_summary[category] = {
            "total": total_cat,
            "correct": stats["correct"],
            "incorrect": stats["incorrect"],
            "invalid": stats["invalid"],
            "error": stats["error"],
            "accuracy": round(stats["correct"] / total_cat, 6) if total_cat else 0,
        }

    output_data = {
        "summary": {
            "total": total,
            "correct": correct_count,
            "incorrect": incorrect_count,
            "invalid": invalid_count,
            "error": error_count,
            "accuracy": round(accuracy, 6),
        },
        "category_summary": category_summary,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")
    print(f"Progress kept at: {progress_path}")
    print(f"Total: {total}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {incorrect_count}")
    print(f"Invalid: {invalid_count}")
    print(f"Error: {error_count}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nCategory summary:")
    for category, stats in category_summary.items():
        print(
            f"{category}: "
            f"{stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
