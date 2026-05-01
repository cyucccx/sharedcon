import argparse
import json
import os
import pickle

import pandas as pd
from transformers import AutoTokenizer, BertTokenizer

from util import ensure_output_dir_is_new, ensure_output_path_is_new


def load_tokenizer(tokenizer_name, local_files_only=False):
    if tokenizer_name.startswith("bert-"):
        return BertTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
    return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)


def build_split_frame(records, tokenizer, label_field):
    posts = [record["content"] for record in records]
    labels = [int(record[label_field]) for record in records]
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


def save_preprocessed_dataset(train_records, eval_records, output_path, tokenizer_name, label_field, local_files_only):
    tokenizer = load_tokenizer(tokenizer_name, local_files_only=local_files_only)
    train_frame = build_split_frame(train_records, tokenizer, label_field)
    eval_frame = build_split_frame(eval_records, tokenizer, label_field)

    payload = {
        "train": train_frame,
        "valid": eval_frame.copy(),
        "test": eval_frame.copy(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)


def prepare_test_only_eval(input_path, output_path, tokenizer_name, label_field, local_files_only):
    ensure_output_path_is_new(output_path, label="ToxiCN preprocessed output file")
    with open(input_path, "r") as f:
        eval_records = json.load(f)

    save_preprocessed_dataset(
        train_records=eval_records,
        eval_records=eval_records,
        output_path=output_path,
        tokenizer_name=tokenizer_name,
        label_field=label_field,
        local_files_only=local_files_only,
    )

    print(f"Prepared {len(eval_records)} ToxiCN records from {input_path}")
    print(f"Label field: {label_field}")
    print(f"Saved preprocessed dataset to {output_path}")


def load_excluded_row_ids(exclude_samples_path):
    excluded_df = pd.read_csv(exclude_samples_path)
    if "row_id" not in excluded_df.columns:
        raise ValueError(f"Missing row_id column in exclude_samples file: {exclude_samples_path}")
    return excluded_df


def prepare_filtered_train_pool(
    input_dir,
    exclude_samples_path,
    filtered_output_dir,
    output_path,
    tokenizer_name,
    label_field,
    local_files_only,
):
    ensure_output_path_is_new(output_path, label="filtered ToxiCN preprocessed output file")

    train_json_path = os.path.join(input_dir, "train.json")
    test_json_path = os.path.join(input_dir, "test.json")
    with open(train_json_path, "r") as f:
        train_records = json.load(f)
    with open(test_json_path, "r") as f:
        test_records = json.load(f)

    filtered_train_records = train_records
    excluded_row_ids = set()
    if exclude_samples_path:
        excluded_df = load_excluded_row_ids(exclude_samples_path)
        excluded_row_ids = set(excluded_df["row_id"].astype(int).tolist())

        if excluded_row_ids:
            max_row_id = max(excluded_row_ids)
            if max_row_id >= len(train_records):
                raise ValueError(
                    f"Excluded row_id {max_row_id} is out of range for ToxiCN train size {len(train_records)}"
                )

        mismatched_posts = []
        for _, row in excluded_df.iterrows():
            row_id = int(row["row_id"])
            source_post = str(train_records[row_id]["content"])
            selected_post = str(row["post"])
            if source_post != selected_post:
                mismatched_posts.append({"row_id": row_id, "source_post": source_post, "selected_post": selected_post})
            if len(mismatched_posts) >= 5:
                break
        if mismatched_posts:
            raise ValueError(
                "Selected pseudo samples do not match ToxiCN train rows by row_id. "
                f"Examples: {mismatched_posts}"
            )

        filtered_train_records = [
            record for idx, record in enumerate(train_records)
            if idx not in excluded_row_ids
        ]

    if filtered_output_dir:
        ensure_output_dir_is_new(filtered_output_dir, label="filtered ToxiCN raw dataset output directory")
        os.makedirs(filtered_output_dir, exist_ok=False)
        filtered_train_json = os.path.join(filtered_output_dir, "train.json")
        filtered_test_json = os.path.join(filtered_output_dir, "test.json")
        filtered_meta_json = os.path.join(filtered_output_dir, "filter_metadata.json")

        with open(filtered_train_json, "w") as f:
            json.dump(filtered_train_records, f, ensure_ascii=False, indent=2)
        with open(filtered_test_json, "w") as f:
            json.dump(test_records, f, ensure_ascii=False, indent=2)

        metadata = {
            "input_dir": input_dir,
            "exclude_samples_path": exclude_samples_path,
            "label_field": label_field,
            "original_train_size": len(train_records),
            "excluded_train_rows": len(excluded_row_ids),
            "filtered_train_size": len(filtered_train_records),
            "test_size": len(test_records),
        }
        with open(filtered_meta_json, "w") as f:
            json.dump(metadata, f, indent=2)

    save_preprocessed_dataset(
        train_records=filtered_train_records,
        eval_records=test_records,
        output_path=output_path,
        tokenizer_name=tokenizer_name,
        label_field=label_field,
        local_files_only=local_files_only,
    )

    if filtered_output_dir:
        print(f"Saved filtered raw dataset to {filtered_output_dir}")
    print(f"Saved filtered preprocessed dataset to {output_path}")
    print(f"Filtered train size: {len(filtered_train_records)}")
    print(f"Removed rows: {len(excluded_row_ids)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ToxiCN datasets for eval.py")
    parser.add_argument(
        "--input",
        default="raw_dataset/ToxiCN/data/test.json",
        help="Path to ToxiCN JSON file for direct test-set preprocessing.",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Directory containing ToxiCN train.json and test.json for filtered-train-pool preparation.",
    )
    parser.add_argument(
        "--exclude_samples",
        default=None,
        help="CSV of selected pseudo samples with row_id/post columns. Optional with --input_dir.",
    )
    parser.add_argument(
        "--filtered_output_dir",
        default=None,
        help="Where to write filtered ToxiCN raw JSON files. Optional with --input_dir.",
    )
    parser.add_argument(
        "--output",
        default="preprocessed_data/preprocessed_toxicn.pkl",
        help="Output pickle path. Existing files are not overwritten.",
    )
    parser.add_argument(
        "--tokenizer",
        default="bert-base-multilingual-cased",
        help="Tokenizer name used by the model.",
    )
    parser.add_argument(
        "--label-field",
        default="toxic",
        choices=["toxic", "toxic_type"],
        help="Which ToxiCN label field to evaluate.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load tokenizer files from local Hugging Face cache.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env_local_only = os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"
    local_files_only = args.local_files_only or env_local_only

    if args.input_dir:
        prepare_filtered_train_pool(
            input_dir=args.input_dir,
            exclude_samples_path=args.exclude_samples,
            filtered_output_dir=args.filtered_output_dir,
            output_path=args.output,
            tokenizer_name=args.tokenizer,
            label_field=args.label_field,
            local_files_only=local_files_only,
        )
    else:
        prepare_test_only_eval(
            input_path=args.input,
            output_path=args.output,
            tokenizer_name=args.tokenizer,
            label_field=args.label_field,
            local_files_only=local_files_only,
        )
