import argparse
import json
import os
import pickle

import pandas as pd
from transformers import BertTokenizer

from util import ensure_output_dir_is_new, ensure_output_path_is_new

def build_split(frame, tokenizer):
    posts = frame["TEXT"].fillna("").astype(str).tolist()
    labels = frame["label"].astype(int).tolist()
    tokenized_post = tokenizer.batch_encode_plus(posts).input_ids

    processed_data = {
        "tokenized_post": tokenized_post,
        "label": labels,
        # Kept for shape compatibility with the existing dataset classes.
        "cluster_label": [0] * len(labels),
        "post": posts,
    }
    return pd.DataFrame.from_dict(processed_data)


def main():
    parser = argparse.ArgumentParser(description="Prepare COLDataset for eval.py without retraining.")
    parser.add_argument(
        "--input_dir",
        default="raw_dataset/COLDataset",
        type=str,
        help="Directory containing train.csv, dev.csv, and test.csv.",
    )
    parser.add_argument(
        "--tokenizer",
        default="bert-base-multilingual-cased",
        type=str,
        help="Tokenizer used by the evaluation model.",
    )
    parser.add_argument(
        "--output",
        default="preprocessed_data/preprocessed_cold.pkl",
        type=str,
        help="Output pickle path consumed by eval.py.",
    )
    parser.add_argument(
        "--exclude_samples",
        default=None,
        type=str,
        help="Optional CSV with a row_id column. Matching train rows are removed before preprocessing.",
    )
    parser.add_argument(
        "--filtered_output_dir",
        default=None,
        type=str,
        help="Optional directory to write the filtered raw COLD train/dev/test CSV files.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load tokenizer from local Hugging Face cache only.",
    )
    args = parser.parse_args()

    ensure_output_path_is_new(args.output, label="preprocessed COLD output file")
    if args.filtered_output_dir:
        ensure_output_dir_is_new(args.filtered_output_dir, label="filtered COLD output directory")
        os.makedirs(args.filtered_output_dir, exist_ok=False)

    split_files = {
        "train": "train.csv",
        "valid": "dev.csv",
        "test": "test.csv",
    }

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
    )
    data_dict = {}
    metadata = {
        "input_dir": args.input_dir,
        "exclude_samples": args.exclude_samples,
    }

    excluded_row_ids = set()
    if args.exclude_samples:
        excluded_df = pd.read_csv(args.exclude_samples)
        if "row_id" not in excluded_df.columns:
            raise ValueError(f"Expected row_id column in {args.exclude_samples}")
        excluded_row_ids = set(excluded_df["row_id"].astype(int).tolist())
        metadata["excluded_train_rows"] = len(excluded_row_ids)

    for split, filename in split_files.items():
        path = os.path.join(args.input_dir, filename)
        frame = pd.read_csv(path)
        if split == "train" and excluded_row_ids:
            frame = frame.loc[~frame.index.isin(excluded_row_ids)].reset_index(drop=True)

        if args.filtered_output_dir:
            filtered_path = os.path.join(args.filtered_output_dir, filename)
            frame.to_csv(filtered_path, index=False)

        data_dict[split] = build_split(frame, tokenizer)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data_dict, f)

    if args.filtered_output_dir:
        meta_path = os.path.join(args.filtered_output_dir, "filter_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

    print(f"Saved {args.output}")
    for split in ["train", "valid", "test"]:
        print(f"{split}: {len(data_dict[split])}")
    if excluded_row_ids:
        print(f"Excluded train rows: {len(excluded_row_ids)}")
    if args.filtered_output_dir:
        print(f"Saved filtered raw dataset to {args.filtered_output_dir}")


if __name__ == "__main__":
    main()
