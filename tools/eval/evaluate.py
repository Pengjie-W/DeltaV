#!/usr/bin/env python3
"""Score public StructCoT/Zebra-CoT predictions without model dependencies."""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from http.client import HTTPException
from pathlib import Path

from protocol import (
    BENCHMARKS,
    EXTRACTION_PROMPTS,
    JUDGE_PROMPTS,
    extract_by_rules,
    normalize_final_answer,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen2.5-72b-instruct"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        if Path(path).suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("Predictions must be a nonempty JSON array or JSONL file.")
    return records


def normalize_prompt(text):
    return re.sub(r"<image>\s*", "", text).strip()


def image_names(images):
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list) or not all(isinstance(image, str) for image in images):
        raise ValueError("Image metadata must be a string or a list of paths.")
    return [Path(image).name for image in images]


def load_cache(benchmark):
    manifest = json.loads((ROOT / "gt_cache/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["benchmarks"][benchmark]
    cache_bytes = (ROOT / "gt_cache" / entry["cache_file"]).read_bytes()
    if hashlib.sha256(cache_bytes).hexdigest() != entry["cache_sha256"]:
        raise ValueError("Bundled GT cache checksum mismatch.")
    records = json.loads(cache_bytes)
    if len(records) != BENCHMARKS[benchmark]["size"]:
        raise ValueError("Unexpected benchmark size.")
    if [record["id"] for record in records] != list(range(len(records))):
        raise ValueError("GT cache must use contiguous public test IDs.")
    return records, manifest


def parse_id(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Invalid sample ID: {value!r}")
    if not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"Invalid sample ID: {value!r}")
    return int(value)


def normalize_predictions(records, cache, allow_partial=False):
    predictions = {}
    for position, record in enumerate(records):
        explicit_id = None
        if isinstance(record, str):
            prediction = record
            metadata = {}
        elif isinstance(record, dict):
            metadata = record
            fields = [field for field in ("prediction", "model_output_text") if field in record]
            if len(fields) != 1:
                raise ValueError(f"Record {position}: provide exactly one of prediction/model_output_text.")
            prediction = record[fields[0]]
            if prediction is None:
                prediction = ""
            if "id" in record:
                explicit_id = parse_id(record["id"])
            elif "_global_idx" in record:
                explicit_id = parse_id(record["_global_idx"])
            elif "model_output_text" in record:
                output_image = record.get("model_output_image")
                if isinstance(output_image, str):
                    match = re.fullmatch(r"sample_(\d+)\.(?:png|jpg|jpeg|webp)", Path(output_image).name, re.IGNORECASE)
                    if match:
                        explicit_id = int(match.group(1))
        else:
            raise ValueError(f"Record {position} must be a prediction string or object.")
        if not isinstance(prediction, str):
            raise ValueError(f"Record {position}: prediction must be text or null.")
        if explicit_id is None and len(records) != len(cache):
            raise ValueError("Predictions without IDs must cover the full test set in its original order.")
        sample_id = position if explicit_id is None else explicit_id
        if sample_id >= len(cache):
            raise ValueError(f"Unknown public test ID: {sample_id}")
        if sample_id in predictions:
            raise ValueError(f"Duplicate public test ID: {sample_id}")
        reference = cache[sample_id]
        if "input_prompt" in metadata:
            prompt = metadata["input_prompt"]
            if not isinstance(prompt, str) or normalize_prompt(prompt) != normalize_prompt(reference["prompt"]):
                raise ValueError(f"Prompt mismatch for ID {sample_id}; check the test release and original IDs.")
        if "input_image" in metadata and image_names(metadata["input_image"]) != image_names(reference["reference_image"]):
            raise ValueError(f"Input image mismatch for ID {sample_id}; do not renumber a subset.")
        predictions[sample_id] = prediction
    if len(predictions) != len(cache) and not allow_partial:
        raise ValueError(f"Expected {len(cache)} predictions, got {len(predictions)}. Use --allow_partial for a non-leaderboard run.")
    return predictions


class JudgeClient:
    def __init__(self, api_key, base_url, model, timeout, max_attempts, stop_event):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use a plain HTTP(S) API base URL without embedded credentials, query, or fragment.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.stop_event = stop_event
        self.disable_thinking = parsed.hostname.endswith(".aliyuncs.com") and "dashscope" in parsed.hostname

    def complete(self, messages, sample_id, stage):
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if self.disable_thinking:
            payload["enable_thinking"] = False
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(1, self.max_attempts + 1):
            if self.stop_event.is_set():
                raise RuntimeError("Evaluation interrupted; completed progress is saved.")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                text = result["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Empty API response")
                return text.strip()
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                if status not in (408, 429) and status < 500:
                    raise RuntimeError(f"API HTTP {status} at {stage}, ID {sample_id}. Check key, region, model access and request settings.") from None
                reason = f"HTTP {status}"
            except (urllib.error.URLError, TimeoutError, OSError, HTTPException):
                reason = "network/timeout error"
            except (KeyError, IndexError, TypeError, ValueError):
                reason = "invalid API response"
            if attempt == self.max_attempts:
                raise RuntimeError(f"API {reason} at {stage}, ID {sample_id}; exhausted {attempt} attempts. Resume with the same command.")
            print(f"[retry] {stage} ID={sample_id}, {reason}, attempt={attempt}", file=sys.stderr)
            self.stop_event.wait(min(2 ** (attempt - 1), 30))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Progress:
    def __init__(self, path, allowed_ids):
        self.path = path
        self.lock = threading.Lock()
        self.extracted = {}
        self.completed = {}
        if not path.exists():
            return
        content = path.read_bytes()
        offset = 0
        for index, line in enumerate(content.splitlines(keepends=True)):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if offset + len(line) != len(content) or line.endswith(b"\n"):
                    raise ValueError(f"Malformed progress at line {index + 1}.") from None
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                print("[resume] removed interrupted final progress line", file=sys.stderr)
                break
            if not isinstance(record, dict) or record.get("id") not in allowed_ids or record.get("stage") not in ("extract", "judge"):
                raise ValueError("Progress contains an unknown sample or stage.")
            target = self.extracted if record["stage"] == "extract" else self.completed
            target[record["id"]] = record
            offset += len(line)
        if offset == len(content) and content and not content.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")

    def append(self, record):
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            target = self.extracted if record["stage"] == "extract" else self.completed
            target[record["id"]] = record


def score_one(benchmark, reference, prediction, client, progress):
    sample_id = reference["id"]
    if sample_id in progress.extracted:
        extracted = progress.extracted[sample_id]
    else:
        answer = extract_by_rules(benchmark, prediction)
        method = "rule"
        if not prediction.strip():
            answer, method = "", "empty"
        elif answer is None:
            prompt = EXTRACTION_PROMPTS[benchmark].format(question=reference["prompt"], response=prediction)
            answer = client.complete([{"role": "user", "content": prompt}], sample_id, "extract")
            if benchmark == "structcot":
                answer = normalize_final_answer(answer)
            method = "api"
        extracted = {"stage": "extract", "id": sample_id, "pred": answer, "pred_extract_method": method}
        progress.append(extracted)
    answer = extracted["pred"]
    if not answer:
        verdict, raw = "Incorrect", "Empty prediction"
    else:
        user_prompt = f"Question:\n{reference['prompt']}\n\n[Model Answer]:\n{answer}\n\n[Standard Answer]:\n{reference['gt']}\n\nYour output:\n"
        raw = client.complete([
            {"role": "system", "content": JUDGE_PROMPTS[benchmark]},
            {"role": "user", "content": user_prompt},
        ], sample_id, "judge")
        verdict = "Incorrect" if "Incorrect" in raw else "Correct" if "Correct" in raw else "Invalid"
    record = {
        **extracted, "stage": "judge", "question": reference["prompt"], "gt": reference["gt"],
        "major_category": reference["major_category"], "verdict": verdict, "raw_judge_output": raw,
    }
    progress.append(record)
    return record


def summarize(benchmark, results, model_name, metadata):
    def stats(records):
        total = len(records)
        correct = sum(record["verdict"] == "Correct" for record in records)
        return {
            "total": total, "correct": correct,
            "incorrect": sum(record["verdict"] == "Incorrect" for record in records),
            "invalid": sum(record["verdict"] == "Invalid" for record in records),
            "accuracy": correct / total if total else None,
        }

    categories = BENCHMARKS[benchmark]["categories"]
    category_summary = {name: stats([record for record in results if record["major_category"] == name]) for name, _ in categories}
    summary = stats(results)
    summary["expected_total"] = BENCHMARKS[benchmark]["size"]
    summary["coverage"] = len(results) / summary["expected_total"]
    summary["leaderboard_eligible"] = len(results) == summary["expected_total"]
    report = {"metadata": metadata, "summary": summary, "category_summary": category_summary, "results": results}
    headers = ["Model", "#Param"] + [label for _, label in categories] + ["Overall"]
    cells = []
    for entry in list(category_summary.values()) + [summary]:
        cells.append("-" if entry["accuracy"] is None else f"{100 * entry['accuracy']:.1f}")
    safe_name = re.sub(r"[\r\n|]", " ", model_name)
    row = "| " + " | ".join([safe_name, "-"] + cells) + " |"
    note = "Full test set; accuracy (%)." if summary["leaderboard_eligible"] else f"PARTIAL RUN — not a leaderboard result. Coverage: {len(results)}/{summary['expected_total']}; accuracy uses submitted samples only."
    markdown = note + "\n\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + row + "\n"
    return report, markdown


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="My model", help="Name of the evaluated model, not the judge")
    parser.add_argument("--judge_model", default=os.getenv("JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base_url", default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--reset_progress", action="store_true", help="Explicitly discard prior outputs/progress in this output directory")
    parser.add_argument("--validate_only", action="store_true", help="Validate predictions and cache without API calls or output writes")
    args = parser.parse_args()
    if args.max_workers < 1 or args.timeout <= 0 or args.max_attempts < 1:
        parser.error("Workers, timeout and attempts must be positive.")
    return args


def run(args):
    cache, manifest = load_cache(args.benchmark)
    predictions = normalize_predictions(read_records(args.predictions), cache, args.allow_partial)
    print(f"[validated] {args.benchmark}: {len(predictions)}/{len(cache)} samples", flush=True)
    if args.validate_only:
        return
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY to your judge API key; this is NOT a ModelScope upload token.")
    stop_event = threading.Event()
    client = JudgeClient(api_key, args.base_url, args.judge_model, args.timeout, args.max_attempts, stop_event)
    metadata = {
        "benchmark": args.benchmark, "benchmark_version": manifest["version"],
        "cache_sha256": manifest["benchmarks"][args.benchmark]["cache_sha256"],
        "predictions_sha256": digest(sorted(predictions.items())),
        "protocol_sha256": hashlib.sha256((ROOT / "protocol.py").read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_name": args.model_name, "judge_model": args.judge_model,
        "base_url": client.base_url, "temperature": 0, "disable_thinking": client.disable_thinking,
        "allow_partial": args.allow_partial, "max_workers": args.max_workers,
        "timeout": args.timeout, "max_attempts": args.max_attempts,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / ".eval.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError(f"Output directory is locked: {lock_path}. Stop the other run before removing a stale lock.") from None
    os.close(lock_descriptor)
    try:
        run_locked(args, cache, predictions, metadata, client, stop_event)
    finally:
        lock_path.unlink()


def run_locked(args, cache, predictions, metadata, client, stop_event):
    metadata_path = args.output_dir / "run.json"
    progress_path = args.output_dir / "progress.jsonl"
    if args.reset_progress:
        for name in ("run.json", "progress.jsonl", "results.json", "leaderboard.md"):
            (args.output_dir / name).unlink(missing_ok=True)
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Input, judge or evaluation settings changed. Use a new output directory or --reset_progress.")
    elif any((args.output_dir / name).exists() for name in ("progress.jsonl", "results.json", "leaderboard.md")):
        raise ValueError("Existing outputs have no run metadata. Use a new directory or --reset_progress.")
    else:
        atomic_json(metadata_path, metadata)
    progress = Progress(progress_path, set(predictions))
    pending_ids = iter(sample_id for sample_id in sorted(predictions) if sample_id not in progress.completed)
    print(f"[resume] {len(progress.completed)} completed; {len(predictions) - len(progress.completed)} pending", flush=True)
    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    futures = set()

    def submit_next():
        sample_id = next(pending_ids, None)
        if sample_id is not None:
            futures.add(executor.submit(score_one, args.benchmark, cache[sample_id], predictions[sample_id], client, progress))

    try:
        for _ in range(args.max_workers):
            submit_next()
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
            futures.difference_update(finished)
            for _ in finished:
                submit_next()
            print(f"[score] {len(progress.completed)}/{len(predictions)}", flush=True)
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    results = [progress.completed[sample_id] for sample_id in sorted(predictions)]
    report, markdown = summarize(args.benchmark, results, args.model_name, metadata)
    atomic_json(args.output_dir / "results.json", report)
    (args.output_dir / "leaderboard.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved results to {args.output_dir / 'results.json'}")


def main():
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted; completed progress is saved.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
