import argparse
import os
import pickle

import pandas as pd
from transformers import BertTokenizer


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
        default="COLDataset/COLDataset",
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
        "--local_files_only",
        action="store_true",
        help="Load tokenizer from local Hugging Face cache only.",
    )
    args = parser.parse_args()

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

    for split, filename in split_files.items():
        path = os.path.join(args.input_dir, filename)
        frame = pd.read_csv(path)
        data_dict[split] = build_split(frame, tokenizer)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"Saved {args.output}")
    for split in ["train", "valid", "test"]:
        print(f"{split}: {len(data_dict[split])}")


if __name__ == "__main__":
    main()
