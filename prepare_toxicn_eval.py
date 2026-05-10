import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def build_frame(records, tokenizer):
    posts = [record["content"] for record in records]
    labels = [int(record["toxic"]) for record in records]
    tokenized_post = tokenizer.batch_encode_plus(posts).input_ids
    return pd.DataFrame.from_dict(
        {
            "tokenized_post": tokenized_post,
            "label": labels,
            "cluster_label": [0] * len(records),
            "post": posts,
            "topic": [record.get("topic") for record in records],
            "platform": [record.get("platform") for record in records],
            "toxic_type": [record.get("toxic_type") for record in records],
            "expression": [record.get("expression") for record in records],
        }
    )


def resolve_local_model_path(model_name):
    if os.path.isdir(model_name):
        return model_name

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        snapshot_dir = repo_dir / "snapshots" / revision
        if snapshot_dir.is_dir():
            return str(snapshot_dir)

    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.is_dir():
        snapshot_candidates = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        if snapshot_candidates:
            return str(snapshot_candidates[-1])

    raise FileNotFoundError(
        f"Could not resolve local Hugging Face cache for {model_name}. "
        "Download it once online or pass --tokenizer with an explicit local path."
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare ToxiCN test.json for eval.py")
    parser.add_argument(
        "--input",
        default="raw_dataset/ToxiCN/data/test.json",
        help="Path to ToxiCN test.json",
    )
    parser.add_argument(
        "--output",
        default="preprocessed_data/preprocessed_toxicn.pkl",
        help="Output pickle path",
    )
    parser.add_argument(
        "--tokenizer",
        default="xlm-roberta-base",
        help="Tokenizer/model name",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use local Hugging Face cache",
    )
    args = parser.parse_args()

    local_files_only = args.local_files_only or os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"

    with open(args.input, "r") as f:
        records = json.load(f)

    tokenizer_source = args.tokenizer
    if local_files_only:
        tokenizer_source = resolve_local_model_path(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_files_only)
    test_frame = build_frame(records, tokenizer)
    payload = {
        "train": test_frame.copy(),
        "valid": test_frame.copy(),
        "test": test_frame.copy(),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)

    print(f"Prepared {len(records)} ToxiCN test records")
    print(f"Saved preprocessed dataset to {args.output}")


if __name__ == "__main__":
    main()
