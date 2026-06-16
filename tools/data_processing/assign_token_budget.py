import os
import json
import ast
import argparse
from typing import Any, Dict, List, Tuple, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


ENFORCE_MONOTONIC_DECREASING = True


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_monotonic_non_increasing(seq: List[float]) -> bool:
    for i in range(len(seq) - 1):
        if seq[i] < seq[i + 1]:
            return False
    return True


def build_sim_to_token_fn(token_bins: Dict[str, Any], budget_key: str):
    bins = token_bins["tsim_intervals"]

    bins_sorted = sorted(bins, key=lambda x: float(x["tsim_left"]))
    sim_lefts = [float(b["tsim_left"]) for b in bins_sorted]
    sim_rights = [float(b["tsim_right"]) for b in bins_sorted]

    tokens: List[float] = []
    for b in bins_sorted:
        v = b.get(budget_key, None)
        tokens.append(9.0 if v is None else float(v))

    if ENFORCE_MONOTONIC_DECREASING and (not is_monotonic_non_increasing(tokens)):
        return None

    tokens = [int(round(t)) for t in tokens]

    def sim_to_token(sim: float) -> int:
        for i in range(len(bins_sorted)):
            left, right = sim_lefts[i], sim_rights[i]
            if i < len(bins_sorted) - 1:
                if left <= sim < right:
                    return tokens[i]
            else:
                if left <= sim <= right:
                    return tokens[i]
        return tokens[0] if sim < sim_lefts[0] else tokens[-1]

    return sim_to_token


def ew_token_weighted_mean(s_list: List[float], t_list: List[float], alpha: float) -> float:
    n = min(len(s_list), len(t_list))
    if n <= 0:
        raise ValueError("Empty s_list or t_list for weighted mean.")
    num = 0.0
    den = 0.0
    for j in range(n):
        w = (alpha ** (n - 1 - j)) * float(t_list[j])
        num += float(s_list[j]) * w
        den += w
    return num / den if den != 0.0 else 0.0


def compute_tokens_for_sequence(
    sim_triangle: List[List[float]],
    alpha: float,
    sim_to_token,
    first_token: int = 144,
    include_first_token: bool = True,
) -> List[int]:
    m = len(sim_triangle)
    if m <= 0:
        return []

    tokens: List[int] = [first_token]
    for i in range(1, m):
        row = sim_triangle[i]
        n = min(i, len(row), len(tokens))
        s_list = [float(x) for x in row[:n]]
        t_list = [float(x) for x in tokens[:n]]
        wsim = ew_token_weighted_mean(s_list, t_list, alpha)
        tokens.append(int(sim_to_token(wsim)))

    return tokens if include_first_token else tokens[1:]


def calc_avg_token(out_list: List[Dict[str, Any]]) -> float:
    total_token_sum = 0.0
    total_images = 0

    for item in out_list:
        img_n = len(item.get("img_paths", []))
        toks = item.get("num_tokens", [])
        if img_n <= 0:
            continue
        total_token_sum += sum(float(t) for t in toks)
        total_images += img_n

    return (total_token_sum / total_images) if total_images > 0 else 0.0


def chunked(seq, chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


def _process_chunk(
    items_chunk: List[Tuple[str, Any]],
    alpha: float,
    token_bins_path: str,
    budget_key: str,
    first_token: int,
    include_first_token: bool,
) -> List[Dict[str, Any]]:
    token_bins = load_json(token_bins_path)
    sim_to_token = build_sim_to_token_fn(token_bins, budget_key)
    if sim_to_token is None:
        return []

    out_chunk: List[Dict[str, Any]] = []
    for k, tri in items_chunk:
        try:
            img_paths = ast.literal_eval(k)
            if not isinstance(img_paths, list) or not isinstance(tri, list):
                continue
            toks = compute_tokens_for_sequence(
                tri,
                alpha,
                sim_to_token,
                first_token=first_token,
                include_first_token=include_first_token,
            )
            out_chunk.append({"img_paths": img_paths, "num_tokens": toks})
        except Exception:
            # skip a single bad record so it does not bring down the whole job
            continue
    return out_chunk


def process_one_scheme_parallel(
    sim_data: Dict[str, Any],
    token_bins_path: str,
    budget_key: str,
    out_path: str,
    alpha: float,
    include_first_token: bool = True,
    first_token: int = 144,
    num_workers: int = 8,
    chunk_size: int = 500,
    show_progress: bool = True,
) -> Optional[Dict[str, Any]]:
    token_bins = load_json(token_bins_path)
    sim_to_token_check = build_sim_to_token_fn(token_bins, budget_key)
    if sim_to_token_check is None:
        print(f"[skip] {budget_key} is not monotonic non-increasing.")
        return None

    items = list(sim_data.items())
    if not items:
        write_json(out_path, [])
        return {
            "budget_key": budget_key,
            "token_bins_json_path": token_bins_path,
            "out_path": out_path,
            "avg_token": 0.0,
            "alpha": alpha,
        }

    chunks = list(chunked(items, chunk_size))

    worker_fn = partial(
        _process_chunk,
        alpha=alpha,
        token_bins_path=token_bins_path,
        budget_key=budget_key,
        first_token=first_token,
        include_first_token=include_first_token,
    )

    out_all: List[Dict[str, Any]] = []

    if num_workers <= 1:
        iterator = chunks
        if show_progress:
            iterator = tqdm(iterator, total=len(chunks), desc=f"{budget_key}")
        for ch in iterator:
            out_all.extend(worker_fn(ch))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(worker_fn, ch) for ch in chunks]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=f"{budget_key}")
            for fut in iterator:
                out_all.extend(fut.result())

    write_json(out_path, out_all)

    return {
        "budget_key": budget_key,
        "token_bins_json_path": token_bins_path,
        "out_path": out_path,
        "avg_token": calc_avg_token(out_all),
        "alpha": alpha,
    }


def process_one_token_bins_to_outdir(
    sim_data: Dict[str, Any],
    token_bins_path: str,
    out_dir: str,
    budget_key: str = "budget",
    include_first_token: bool = True,
    first_token: int = 144,
    num_workers: int = 8,
    chunk_size: int = 500,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    token_bins = load_json(token_bins_path)
    alpha = float(token_bins["config"]["alpha"])

    ensure_dir(out_dir)

    summary: List[Dict[str, Any]] = []

    out_path = os.path.join(out_dir, "tokens.json")
    result = process_one_scheme_parallel(
        sim_data=sim_data,
        token_bins_path=token_bins_path,
        budget_key=budget_key,
        out_path=out_path,
        alpha=alpha,
        include_first_token=include_first_token,
        first_token=first_token,
        num_workers=num_workers,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
    if result is not None:
        summary.append(result)

    return summary


def run_for_one_split(
    split_name: str,
    sim_json_path: str,
    token_bins_path: str,
    out_dir: str,
    budget_key: str,
    include_first_token: bool,
    first_token: int,
    num_workers: int,
    chunk_size: int,
    show_progress: bool,
) -> Dict[str, Any]:
    print(f"[load] {split_name}: {sim_json_path}")
    sim_data = load_json(sim_json_path)

    summary = process_one_token_bins_to_outdir(
        sim_data=sim_data,
        token_bins_path=token_bins_path,
        out_dir=out_dir,
        budget_key=budget_key,
        include_first_token=include_first_token,
        first_token=first_token,
        num_workers=num_workers,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )

    summary_path = os.path.join(out_dir, "tokens_summary.json")
    write_json(summary_path, summary)

    return {
        "split": split_name,
        "sim_json": sim_json_path,
        "token_bins_json": token_bins_path,
        "out_dir": out_dir,
        "summary_path": summary_path,
        "summary_records": len(summary),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_sim_json",
        type=str,
        required=True,
        help="Path to train similarity json."
    )
    parser.add_argument(
        "--test_sim_json",
        type=str,
        default=None,
        help="Path to test similarity json. If not provided, test split will be skipped."
    )

    parser.add_argument(
        "--token_bins_jsons",
        type=str,
        nargs="+",
        required=True,
        help="A list of token_bins.json paths."
    )

    parser.add_argument(
        "--train_out_dirs",
        type=str,
        nargs="+",
        required=True,
        help="A list of output directories for train split, aligned with --token_bins_jsons."
    )
    parser.add_argument(
        "--test_out_dirs",
        type=str,
        nargs="+",
        default=None,
        help="A list of output directories for test split, aligned with --token_bins_jsons."
    )

    parser.add_argument(
        "--budget_key",
        type=str,
        default="budget",
        help="Key in each tsim_intervals entry holding the token budget."
    )

    parser.add_argument(
        "--include_first_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include first_token in output."
    )
    parser.add_argument("--first_token", type=int, default=144)

    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar."
    )

    args = parser.parse_args()

    n_bins = len(args.token_bins_jsons)
    n_train = len(args.train_out_dirs)
    has_test = args.test_sim_json is not None

    if has_test:
        if args.test_out_dirs is None:
            raise ValueError("When --test_sim_json is provided, --test_out_dirs must also be provided.")
        n_test = len(args.test_out_dirs)
        if not (n_bins == n_train == n_test):
            raise ValueError(
                f"Length mismatch: len(token_bins_jsons)={n_bins}, "
                f"len(train_out_dirs)={n_train}, len(test_out_dirs)={n_test}."
            )
    else:
        if args.test_out_dirs is not None:
            raise ValueError("When --test_out_dirs is provided, --test_sim_json must also be provided.")
        if n_bins != n_train:
            raise ValueError(
                f"Length mismatch: len(token_bins_jsons)={n_bins}, len(train_out_dirs)={n_train}."
            )

    all_results: List[Dict[str, Any]] = []

    for i, token_bins_path in enumerate(args.token_bins_jsons):
        train_out_dir = args.train_out_dirs[i]

        print("=" * 100)
        print(f"[{i + 1}/{n_bins}] token_bins_json = {token_bins_path}")

        print(f"--> TRAIN: sim_json={args.train_sim_json}")
        print(f"--> TRAIN: out_dir={train_out_dir}")
        train_result = run_for_one_split(
            split_name="train",
            sim_json_path=args.train_sim_json,
            token_bins_path=token_bins_path,
            out_dir=train_out_dir,
            budget_key=args.budget_key,
            include_first_token=args.include_first_token,
            first_token=args.first_token,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            show_progress=not args.no_progress,
        )
        all_results.append(train_result)

        if has_test:
            test_out_dir = args.test_out_dirs[i]
            print(f"--> TEST : sim_json={args.test_sim_json}")
            print(f"--> TEST : out_dir={test_out_dir}")
            test_result = run_for_one_split(
                split_name="test",
                sim_json_path=args.test_sim_json,
                token_bins_path=token_bins_path,
                out_dir=test_out_dir,
                budget_key=args.budget_key,
                include_first_token=args.include_first_token,
                first_token=args.first_token,
                num_workers=args.num_workers,
                chunk_size=args.chunk_size,
                show_progress=not args.no_progress,
            )
            all_results.append(test_result)

    if all_results:
        all_out_dirs = list(args.train_out_dirs)
        if has_test:
            all_out_dirs += list(args.test_out_dirs)

        common_parent = os.path.commonpath([os.path.abspath(p) for p in all_out_dirs])
        global_summary_path = os.path.join(common_parent, "global_run_summary.json")
        write_json(global_summary_path, all_results)
        print("=" * 100)
        print(f"global_summary_path={global_summary_path}")

    print("=" * 100)
    print(f"finished token_bins_count={n_bins}")
    print(f"budget_key={args.budget_key}")
    print(f"num_workers={args.num_workers}")
    print(f"chunk_size={args.chunk_size}")
    print(f"enforce_monotonic={ENFORCE_MONOTONIC_DECREASING}")


if __name__ == "__main__":
    main()

# Example:
#   python tools/data_processing/assign_token_budget.py \
#     --train_sim_json /path/to/<dataset>_sim.json \
#     --token_bins_jsons tools/data_processing/tsim_intervals.json \
#     --train_out_dirs /path/to/out_dir \
#     --include_first_token \
#     --first_token 144 \
#     --num_workers 16 \
#     --chunk_size 1000
# Output: <out_dir>/tokens.json  (per-sequence num_tokens, keyed by image-path list)

