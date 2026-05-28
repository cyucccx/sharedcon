import argparse
import json
import os

import pandas as pd

from util import ensure_output_dir_is_new


TOXIC_TYPE_NAMES = {
    0: "non_toxic",
    1: "general_offensive_language",
    2: "hate_speech",
}

EXPRESSION_NAMES = {
    0: "non_hate",
    1: "explicit_hate_speech",
    2: "implicit_hate_speech",
    3: "reporting",
}

TARGET_NAMES = ["LGBTQ", "Region", "Sexism", "Racism", "Others", "non_hate"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter ToxiCN by gold labels so the remaining data matches an IHC-style "
            "binary label space: not_hate vs hate. By default, explicit and implicit "
            "hate are both kept as positive because explicit ToxiCN samples can still "
            "be useful when they are learned well."
        )
    )
    parser.add_argument(
        "--input_dir",
        default="raw_dataset/ToxiCN/data",
        help="Directory containing ToxiCN train.json and test.json.",
    )
    parser.add_argument(
        "--output_dir",
        default="raw_dataset/ToxiCN_filtered",
        help="New output directory for filtered train/test JSON and reports.",
    )
    parser.add_argument(
        "--filter_test",
        action="store_true",
        default=True,
        help="Apply the same label filter to test.json.",
    )
    parser.add_argument(
        "--keep_original_test",
        action="store_false",
        dest="filter_test",
        help="Copy original test.json without filtering.",
    )
    parser.add_argument(
        "--require_single_positive_target",
        action="store_true",
        default=True,
        help="Require positive hate samples to have exactly one active hate target.",
    )
    parser.add_argument(
        "--allow_multi_target_positive",
        action="store_false",
        dest="require_single_positive_target",
        help="Allow positive hate samples with multiple active targets.",
    )
    parser.add_argument(
        "--positive_expressions",
        default="1,2",
        help=(
            "Comma-separated expression labels to keep as positive. "
            "Default keeps explicit hate speech and implicit hate speech: 1,2."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=0,
        help="Optional max content length. Use 0 to disable.",
    )
    return parser.parse_args()


def parse_positive_expressions(value):
    expressions = {int(item.strip()) for item in value.split(",") if item.strip()}
    invalid = expressions.difference(EXPRESSION_NAMES)
    if invalid:
        raise ValueError(f"Unsupported expression labels: {sorted(invalid)}")
    if not expressions:
        raise ValueError("--positive_expressions must contain at least one expression label.")
    return expressions


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_target(values):
    if not isinstance(values, list):
        return []
    return [int(value) for value in values]


def active_hate_target_count(values):
    target = normalize_target(values)
    if len(target) >= 6:
        return sum(target[:5])
    return sum(target)


def is_non_hate_target(values):
    target = normalize_target(values)
    if len(target) >= 6:
        return bool(target[5]) and sum(target[:5]) == 0
    return sum(target) == 0


def attach_features(records):
    df = pd.DataFrame(records).copy()
    df["source_row_id"] = range(len(df))
    df["toxic"] = df["toxic"].astype(int)
    df["toxic_type"] = df["toxic_type"].astype(int)
    df["expression"] = df["expression"].astype(int)
    df["active_hate_target_count"] = df["target"].apply(active_hate_target_count)
    df["is_non_hate_target"] = df["target"].apply(is_non_hate_target)
    df["content_length"] = df["content"].astype(str).str.len()
    return df


def filter_split(records, args, positive_expressions):
    df = attach_features(records)

    positive_mask = (
        df["toxic"].eq(1)
        & df["toxic_type"].eq(2)
        & df["expression"].isin(positive_expressions)
    )
    if args.require_single_positive_target:
        positive_mask &= df["active_hate_target_count"].eq(1)
    else:
        positive_mask &= df["active_hate_target_count"].ge(1)

    negative_mask = (
        df["toxic"].eq(0)
        & df["toxic_type"].eq(0)
        & df["expression"].eq(0)
        & df["is_non_hate_target"]
    )

    keep_mask = positive_mask | negative_mask
    if args.max_length > 0:
        keep_mask &= df["content_length"].le(args.max_length)

    df["ihc_label"] = None
    df.loc[negative_mask, "ihc_label"] = 0
    df.loc[positive_mask, "ihc_label"] = 1
    df["keep_ihc_label_aligned"] = keep_mask
    df["drop_reason"] = "kept"
    df.loc[~(positive_mask | negative_mask), "drop_reason"] = "label_not_in_ihc_pure"
    df.loc[df["toxic_type"].eq(1), "drop_reason"] = "general_offensive_language"
    df.loc[df["expression"].eq(3), "drop_reason"] = "reporting"
    if args.max_length > 0:
        df.loc[df["content_length"].gt(args.max_length), "drop_reason"] = "too_long"
    df.loc[keep_mask, "drop_reason"] = "kept"

    return df


def clean_records(df):
    records = []
    helper_columns = {
        "active_hate_target_count",
        "is_non_hate_target",
        "content_length",
        "keep_ihc_label_aligned",
        "drop_reason",
    }
    for record in df.to_dict(orient="records"):
        cleaned = {}
        for key, value in record.items():
            if key in helper_columns:
                continue
            if hasattr(value, "item"):
                value = value.item()
            cleaned[key] = value
        if cleaned["ihc_label"] is not None and not pd.isna(cleaned["ihc_label"]):
            cleaned["ihc_label"] = int(cleaned["ihc_label"])
        records.append(cleaned)
    return records


def distribution(series, names=None):
    counts = series.value_counts().sort_index()
    result = {}
    for value, count in counts.items():
        key = str(int(value)) if isinstance(value, (int, float)) else str(value)
        item = {"count": int(count)}
        if names is not None:
            item["name"] = names.get(int(value), str(value))
        result[key] = item
    return result


def summarize(df):
    kept = df.loc[df["keep_ihc_label_aligned"]]
    dropped = df.loc[~df["keep_ihc_label_aligned"]]
    return {
        "original_size": int(len(df)),
        "kept": int(len(kept)),
        "dropped": int(len(dropped)),
        "kept_ihc_label_distribution": distribution(kept["ihc_label"].dropna().astype(int)),
        "drop_reason_distribution": distribution(dropped["drop_reason"]),
        "kept_toxic_type_distribution": distribution(kept["toxic_type"], TOXIC_TYPE_NAMES),
        "kept_expression_distribution": distribution(kept["expression"], EXPRESSION_NAMES),
        "dropped_toxic_type_distribution": distribution(dropped["toxic_type"], TOXIC_TYPE_NAMES),
        "dropped_expression_distribution": distribution(dropped["expression"], EXPRESSION_NAMES),
        "kept_topic_distribution": distribution(kept["topic"]),
    }


def write_report(df, path):
    columns = [
        "source_row_id",
        "content",
        "toxic",
        "toxic_type",
        "expression",
        "target",
        "topic",
        "ihc_label",
        "keep_ihc_label_aligned",
        "drop_reason",
    ]
    df[columns].to_csv(path, index=False)


def main():
    args = parse_args()
    positive_expressions = parse_positive_expressions(args.positive_expressions)
    ensure_output_dir_is_new(args.output_dir, label="IHC-label-aligned ToxiCN output directory")

    train_records = load_json(os.path.join(args.input_dir, "train.json"))
    test_records = load_json(os.path.join(args.input_dir, "test.json"))

    train_df = filter_split(train_records, args, positive_expressions)
    if args.filter_test:
        test_df = filter_split(test_records, args, positive_expressions)
    else:
        test_df = attach_features(test_records)
        test_df["ihc_label"] = None
        test_df["keep_ihc_label_aligned"] = True
        test_df["drop_reason"] = "kept_original_test"

    kept_train = train_df.loc[train_df["keep_ihc_label_aligned"]].copy()
    kept_test = test_df.loc[test_df["keep_ihc_label_aligned"]].copy()

    os.makedirs(args.output_dir, exist_ok=False)
    with open(os.path.join(args.output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(clean_records(kept_train), f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(clean_records(kept_test), f, ensure_ascii=False, indent=2)

    write_report(train_df, os.path.join(args.output_dir, "train_filter_report.csv"))
    write_report(test_df, os.path.join(args.output_dir, "test_filter_report.csv"))

    metadata = {
        "input_dir": args.input_dir,
        "filter_policy": {
            "negative_kept": "toxic=0, toxic_type=0, expression=0, no active hate target",
            "positive_kept": "toxic=1, toxic_type=2, expression in configured positive_expressions",
            "positive_expressions": {
                str(expression): EXPRESSION_NAMES[expression]
                for expression in sorted(positive_expressions)
            },
            "filtered_out": [
                "toxic_type=1 general offensive language",
                "expression=3 reporting",
                "other label combinations outside the selected binary label space",
            ],
            "require_single_positive_target": args.require_single_positive_target,
            "max_length": args.max_length,
            "filter_test": args.filter_test,
        },
        "train": summarize(train_df),
        "test": summarize(test_df),
    }
    with open(os.path.join(args.output_dir, "filter_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("===== ToxiCN IHC-label alignment summary =====")
    print(f"Train kept: {metadata['train']['kept']} / {metadata['train']['original_size']}")
    print(f"Test kept:  {metadata['test']['kept']} / {metadata['test']['original_size']}")
    print("Train drop reasons:", metadata["train"]["drop_reason_distribution"])
    print(f"Saved filtered dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
