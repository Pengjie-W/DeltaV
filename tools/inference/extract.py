import json
import re
import argparse
from pathlib import Path


def extract_final_answer(text):
    """
    Extract only the content that strictly follows '\n\nFinal Answer:'.
    """
    if not text:
        return None

    match = re.search(r"\n\nFinal Answer:([\s\S]*)", text)

    if match:
        return match.group(1).strip()

    return None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final_json",
        type=str,
        required=True,
        help="input json file"
    )

    args = parser.parse_args()

    input_path = Path(args.final_json)

    output_path = input_path.with_name(
        input_path.stem + "_extracted.json"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for idx, item in enumerate(data):

        gt = extract_final_answer(item.get("output_text", ""))
        pred = extract_final_answer(item.get("model_output_text", ""))

        results.append({
            "_global_idx": item.get("_global_idx", idx),
            "config": item.get("config"),
            "question": item.get("input_prompt"),
            "gt": gt,
            "pred": pred
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")
    print(f"Processed {len(results)} items")


if __name__ == "__main__":
    main()