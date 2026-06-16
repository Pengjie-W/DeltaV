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
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", required=True)
    parser.add_argument("--output_json_path", required=True)
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

Evaluation Criteria for Answer Matching

1. Multiple-choice questions
If the question contains explicit options (e.g., A/B/C/D):
- The predicted answer must match the correct option.
- Acceptable forms include:
  - The option letter only (e.g., A)
  - The option letter with description (e.g., A. Qc2)
  - A sentence that clearly identifies the correct option (e.g., The answer is A.)
  - The option content itself, if it clearly matches the correct option content

2. True/False or option-equivalent questions
If the answer is expected to match one of several discrete options:
- The predicted answer must correspond to the same option or the same option content as the ground truth.
- Minor wording differences are acceptable only if the selected option/content is the same.

3. Numerical or counting questions
For problems where the numeric value is a key part of the answer (e.g., counting, calculation, measurements):
- The predicted numeric value must match the ground truth value.
- Equivalent formats are acceptable:
  - 30 days vs 30 Days
  - 36% vs 36 percent
- If the numeric value differs, mark it Incorrect even if the reasoning is similar.

4. Procedural or step-based tasks
For tasks involving sequences of actions or steps (e.g., robot planning, navigation, manipulation):
- The predicted sequence should achieve the same goal and include the required actions in the ground truth.
- Additional intermediate steps are allowed if they do NOT introduce incorrect operations and do NOT change the intended outcome.
- Mark Incorrect if:
  - required actions are missing
  - key actions are replaced with different actions
  - extra actions cause additional state changes beyond the ground truth requirements

5. Open-ended descriptive answers
For other question types:
- Mark Correct if the prediction conveys the same meaning and answers the question correctly, even if the wording differs.
- Minor paraphrasing is acceptable.

6. Strict incorrect conditions
Mark Incorrect if any of the following occurs:
- The key entity/object differs from the ground truth
- The numeric value differs
- The selected option differs
- The described actions/results contradict the ground truth

Output format:
- Output ONLY one label:
  - Correct
  - Incorrect
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
    idx = item.get("_global_idx")
    question = item.get("question", "")
    ground_truth = item.get("gt", "")
    prediction = item.get("pred", "")
    config = item.get("config", "")
    major_category = item.get("major_category", "")
    progress_key = make_progress_key(item)

    if (
        item.get("pred_missing")
        or prediction is None
        or str(prediction).strip() == ""
    ):
        record = {
            "_progress_key": progress_key,
            "_global_idx": idx,
            "id": item.get("id", ""),
            "config": config,
            "major_category": major_category,
            "question": question,
            "gt": ground_truth,
            "pred": prediction,
            "verdict": "Incorrect",
            "raw_judge_output": "Missing prediction",
        }
        append_progress(progress_path, record)
        return record

    user_prompt = build_user_prompt(question, prediction, ground_truth)
    verdict_raw = call_judge_api(user_prompt, item_id_for_log=item.get("id", ""))

    if "Incorrect" in verdict_raw:
        verdict = "Incorrect"
    elif "Correct" in verdict_raw:
        verdict = "Correct"
    else:
        verdict = "Invalid"

    record = {
        "_progress_key": progress_key,
        "_global_idx": idx,
        "id": item.get("id", ""),
        "config": config,
        "major_category": major_category,
        "question": question,
        "gt": ground_truth,
        "pred": prediction,
        "verdict": verdict,
        "raw_judge_output": verdict_raw,
    }
    append_progress(progress_path, record)
    return record


def update_stats(stats, verdict):
    stats["total"] += 1

    if verdict == "Correct":
        stats["correct"] += 1
    elif verdict == "Incorrect":
        stats["incorrect"] += 1
    elif verdict == "Invalid":
        stats["invalid"] += 1
    elif str(verdict).startswith("Error"):
        stats["error"] += 1


def finalize_stats(stats_dict):
    output = {}

    for key, stats in sorted(stats_dict.items()):
        total = stats["total"]
        output[key] = {
            "total": total,
            "correct": stats["correct"],
            "incorrect": stats["incorrect"],
            "invalid": stats["invalid"],
            "error": stats["error"],
            "accuracy": round(stats["correct"] / total, 6) if total > 0 else 0,
        }

    return output


def main():
    if os.path.exists(args.output_json_path) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_json_path}\n"
            f"Use --overwrite if you really want to overwrite."
        )

    progress_path = args.progress_path or (args.output_json_path + ".progress.jsonl")
    os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)

    if args.reset_progress and os.path.exists(progress_path):
        os.remove(progress_path)
        print(f"[progress] removed existing progress file: {progress_path}")

    data = read_json(args.input_json_path)

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
                as_completed(futures), total=len(futures), desc="Judging Zebra"
            ):
                future.result()

    final_records = load_progress(progress_path)
    results = list(final_records.values())
    for rec in results:
        rec.pop("_progress_key", None)

    results.sort(key=lambda x: x.get("_global_idx", -1) if isinstance(x.get("_global_idx"), int) else -1)

    summary_stats = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "invalid": 0,
        "error": 0,
    }

    category_stats = defaultdict(
        lambda: {"total": 0, "correct": 0, "incorrect": 0, "invalid": 0, "error": 0}
    )

    config_stats = defaultdict(
        lambda: {"total": 0, "correct": 0, "incorrect": 0, "invalid": 0, "error": 0}
    )

    for result in results:
        verdict = result["verdict"]
        update_stats(summary_stats, verdict)

        major_category = (
            result.get("major_category", "") or "__MISSING_MAJOR_CATEGORY__"
        )
        config = result.get("config", "") or "__MISSING_CONFIG__"

        update_stats(category_stats[major_category], verdict)
        update_stats(config_stats[config], verdict)

    total = summary_stats["total"]

    output_data = {
        "summary": {
            "total": total,
            "correct": summary_stats["correct"],
            "incorrect": summary_stats["incorrect"],
            "invalid": summary_stats["invalid"],
            "error": summary_stats["error"],
            "accuracy": round(summary_stats["correct"] / total, 6) if total > 0 else 0,
        },
        "category_summary": finalize_stats(category_stats),
        "config_summary": finalize_stats(config_stats),
        "results": results,
    }

    with open(args.output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {args.output_json_path}")
    print(f"Progress kept at: {progress_path}")
    print(f"Total: {output_data['summary']['total']}")
    print(f"Correct: {output_data['summary']['correct']}")
    print(f"Incorrect: {output_data['summary']['incorrect']}")
    print(f"Invalid: {output_data['summary']['invalid']}")
    print(f"Error: {output_data['summary']['error']}")
    print(f"Accuracy: {output_data['summary']['accuracy']:.6f}")


if __name__ == "__main__":
    main()
