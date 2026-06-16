import os
import json
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Merge distributed inference json parts.")
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory containing result_part_rank*.json"
    )
    parser.add_argument(
        "--final_json",
        type=str,
        required=True,
        help="Path to save merged final json"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Expected number of ranks. If set, will check missing rank files."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if any expected rank file is missing"
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_json(result_dir, final_json, world_size=None, strict=False):
    os.makedirs(os.path.dirname(final_json) or ".", exist_ok=True)

    if world_size is not None:
        part_files = [
            os.path.join(result_dir, f"result_part_rank{rank}.json")
            for rank in range(world_size)
        ]
    else:
        part_files = sorted(
            glob.glob(os.path.join(result_dir, "result_part_rank*.json"))
        )

    if not part_files:
        raise FileNotFoundError(f"No part files found in: {result_dir}")

    final = []
    missing_files = []

    for part_path in part_files:
        if not os.path.exists(part_path):
            missing_files.append(part_path)
            continue

        data = load_json(part_path)
        if not isinstance(data, list):
            raise ValueError(f"{part_path} does not contain a list")
        final.extend(data)

    if missing_files:
        msg = "Missing part files:\n" + "\n".join(missing_files)
        if strict:
            raise FileNotFoundError(msg)
        else:
            print("[Warning]")
            print(msg)

    # Restore original order: sort by _global_idx when available, otherwise keep order
    has_global_idx = all(isinstance(x, dict) and "_global_idx" in x for x in final)
    if has_global_idx:
        final.sort(key=lambda x: x["_global_idx"])
        for x in final:
            x.pop("_global_idx", None)
    else:
        print("[Warning] Some items do not have _global_idx, skip sorting.")

    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(final)} samples into: {final_json}")


def main():
    args = parse_args()
    merge_json(
        result_dir=args.result_dir,
        final_json=args.final_json,
        world_size=args.world_size,
        strict=args.strict
    )


if __name__ == "__main__":
    main()