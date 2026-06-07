import argparse
import json
import os
import random

import pandas as pd

from util import ensure_output_dir_is_new


PSEUDO_LABEL_TO_CLASS = {
    0: "not_hate",
    1: "implicit_hate",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select top cross-score ToxiCN pseudo-labeled samples and mix them into IHC pure. "
            "Unlike labeled balancing, this selects globally by the winning class cross score "
            "and does not force per-class pseudo quotas."
        )
    )
    parser.add_argument("--ihc_dir", default="raw_dataset/ihc_pure")
    parser.add_argument(
        "--combined_predictions",
        default="openai_batch/combined_llm_ihc_cross_predictions.csv",
        help="CSV from combine_prediction_scores.py.",
    )
    parser.add_argument(
        "--output_dir",
        default="raw_dataset/ihc_pure_toxicn_filtered_cross_pseudo_top5500",
    )
    parser.add_argument("--target_total", type=int, default=5500)
    parser.add_argument(
        "--remove_policy",
        choices=["random_total", "class_matched"],
        default="random_total",
        help=(
            "random_total removes target_total random IHC train rows, allowing final label distribution to shift. "
            "class_matched removes IHC rows matching the selected pseudo class counts when possible."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_ihc_split(ihc_dir, split):
    return pd.read_csv(os.path.join(ihc_dir, f"{split}.tsv"), sep="\t")


def add_winning_cross_score(predictions):
    df = predictions.copy()
    df["combined_pred_label"] = df["combined_pred_label"].astype(int)
    df["winning_cross_score"] = df.apply(
        lambda row: row["cross_score_1"] if int(row["combined_pred_label"]) == 1 else row["cross_score_0"],
        axis=1,
    )
    df["pseudo_class"] = df["combined_pred_label"].map(PSEUDO_LABEL_TO_CLASS)
    if df["pseudo_class"].isna().any():
        bad = df.loc[df["pseudo_class"].isna(), "combined_pred_label"].unique().tolist()
        raise ValueError(f"Unsupported combined_pred_label values: {bad}")
    return df


def select_top_pseudo(predictions, target_total):
    if target_total <= 0:
        raise ValueError("--target_total must be positive.")
    if target_total > len(predictions):
        raise ValueError(f"--target_total {target_total} exceeds available predictions {len(predictions)}.")

    return (
        predictions.sort_values(
            ["winning_cross_score", "combined_confidence", "row_id"],
            ascending=[False, False, True],
        )
        .head(target_total)
        .copy()
        .reset_index(drop=True)
    )


def remove_ihc_rows_random_total(ihc_train, total_n, seed):
    shuffled = ihc_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    removed = shuffled.iloc[:total_n].copy()
    kept = shuffled.iloc[total_n:].copy()
    return kept, removed


def remove_ihc_rows_class_matched(ihc_train, selected_pseudo, seed):
    selected_counts = selected_pseudo["pseudo_class"].value_counts().to_dict()
    removed_parts = []
    kept_parts = []

    for class_name, class_df in ihc_train.groupby("class"):
        class_df = class_df.sample(frac=1, random_state=seed + abs(hash(class_name)) % 10000).reset_index(drop=True)
        remove_n = min(int(selected_counts.get(class_name, 0)), len(class_df))
        removed_parts.append(class_df.iloc[:remove_n].copy())
        kept_parts.append(class_df.iloc[remove_n:].copy())

    removed = pd.concat(removed_parts, ignore_index=True)
    kept = pd.concat(kept_parts, ignore_index=True)
    return kept, removed


def build_pseudo_train_rows(selected_pseudo):
    records = []
    for row in selected_pseudo.to_dict(orient="records"):
        text = row["post"]
        records.append(
            {
                "ID": f"toxicn_cross_pseudo_{int(row['row_id'])}",
                "class": row["pseudo_class"],
                "implied_statement": "",
                "post": text,
                "aug_sent1_of_post": text,
                "aug_sent2_of_post": text,
                "is_pseudo": 1,
                "pseudo_source_row_id": int(row["row_id"]),
                "pseudo_label": int(row["combined_pred_label"]),
                "pseudo_score": float(row["winning_cross_score"]),
            }
        )
    return pd.DataFrame(records)


def add_source_flags(ihc_df):
    df = ihc_df.copy()
    df["is_pseudo"] = 0
    df["pseudo_source_row_id"] = pd.NA
    df["pseudo_label"] = pd.NA
    df["pseudo_score"] = pd.NA
    return df


def summarize_dataset(mixed_train, kept_ihc, removed_ihc, selected_pseudo, args):
    pseudo_counts = selected_pseudo["pseudo_class"].value_counts().sort_index().to_dict()
    return {
        "ihc_dir": args.ihc_dir,
        "combined_predictions": args.combined_predictions,
        "target_total": args.target_total,
        "remove_policy": args.remove_policy,
        "seed": args.seed,
        "mixed_train_size": int(len(mixed_train)),
        "kept_ihc_train_size": int(len(kept_ihc)),
        "removed_ihc_train_size": int(len(removed_ihc)),
        "selected_pseudo_size": int(len(selected_pseudo)),
        "selected_pseudo_class_distribution": {
            str(k): int(v) for k, v in pseudo_counts.items()
        },
        "selected_pseudo_true_label_distribution": {
            str(int(k)): int(v)
            for k, v in selected_pseudo["true_label"].value_counts().sort_index().to_dict().items()
        },
        "selected_pseudo_combined_pred_distribution": {
            str(int(k)): int(v)
            for k, v in selected_pseudo["combined_pred_label"].value_counts().sort_index().to_dict().items()
        },
        "mixed_train_class_distribution": {
            str(k): int(v)
            for k, v in mixed_train["class"].value_counts().sort_index().to_dict().items()
        },
        "winning_cross_score": {
            key: float(value)
            for key, value in selected_pseudo["winning_cross_score"].describe().to_dict().items()
        },
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    ensure_output_dir_is_new(args.output_dir, label="cross-score pseudo mixed dataset output directory")

    ihc_train = load_ihc_split(args.ihc_dir, "train")
    ihc_valid = load_ihc_split(args.ihc_dir, "valid")
    ihc_test = load_ihc_split(args.ihc_dir, "test")

    predictions = pd.read_csv(args.combined_predictions)
    predictions = add_winning_cross_score(predictions)
    selected_pseudo = select_top_pseudo(predictions, args.target_total)

    if args.remove_policy == "random_total":
        kept_ihc, removed_ihc = remove_ihc_rows_random_total(ihc_train, len(selected_pseudo), args.seed)
    elif args.remove_policy == "class_matched":
        kept_ihc, removed_ihc = remove_ihc_rows_class_matched(ihc_train, selected_pseudo, args.seed)
    else:
        raise NotImplementedError(args.remove_policy)

    pseudo_train = build_pseudo_train_rows(selected_pseudo)
    mixed_train = pd.concat([add_source_flags(kept_ihc), pseudo_train], ignore_index=True)
    mixed_train = mixed_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=False)
    mixed_train.to_csv(os.path.join(args.output_dir, "train.tsv"), sep="\t", index=False)
    ihc_valid.to_csv(os.path.join(args.output_dir, "valid.tsv"), sep="\t", index=False)
    ihc_test.to_csv(os.path.join(args.output_dir, "test.tsv"), sep="\t", index=False)

    selected_pseudo.to_csv(os.path.join(args.output_dir, "selected_pseudo_samples.csv"), index=False)
    removed_ihc.to_csv(os.path.join(args.output_dir, "removed_ihc_train.csv"), sep="\t", index=False)
    kept_ihc.to_csv(os.path.join(args.output_dir, "kept_ihc_train.csv"), sep="\t", index=False)

    metadata = summarize_dataset(mixed_train, kept_ihc, removed_ihc, selected_pseudo, args)
    with open(os.path.join(args.output_dir, "mix_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved cross-score pseudo mixed dataset to {args.output_dir}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
