import argparse
import json
import os
import random
import shutil
import warnings

import pandas as pd


LABEL_MAP = {
    0: "not_hate",
    1: "implicit_hate",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inject high-confidence pseudo-labeled samples into IHC train data "
        "while removing the same number of original IHC samples."
    )
    parser.add_argument(
        "--ihc_train",
        default="raw_dataset/ihc_pure/train.tsv",
        type=str,
        help="Path to the original IHC train.tsv.",
    )
    parser.add_argument(
        "--ihc_valid",
        default="raw_dataset/ihc_pure/valid.tsv",
        type=str,
        help="Path to the original IHC valid.tsv.",
    )
    parser.add_argument(
        "--ihc_test",
        default="raw_dataset/ihc_pure/test.tsv",
        type=str,
        help="Path to the original IHC test.tsv.",
    )
    parser.add_argument(
        "--pseudo_predictions",
        default="pseudo_clusters/sbert-multi_cold_conf099_k50/selected_cluster_samples.csv",
        type=str,
        help="Path to eval.py output CSV containing pseudo predictions.",
    )
    parser.add_argument(
        "--output_dir",
        default="raw_dataset/ihc_pure_cold_pseudo",
        type=str,
        help="Directory to write the new train/valid/test files.",
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.95,
        type=float,
        help="Keep pseudo samples with confidence >= this threshold.",
    )
    parser.add_argument(
        "--max_pseudo_samples",
        default=None,
        type=int,
        help="Optional hard cap after confidence filtering. Highest-confidence samples are kept first.",
    )
    parser.add_argument(
        "--remove_strategy",
        choices=["random", "label_matched_random"],
        default="label_matched_random",
        help="How to remove original IHC samples to keep train size unchanged.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for reproducible sampling/shuffling.",
    )
    return parser.parse_args()


def select_pseudo_samples(predictions, threshold, max_pseudo_samples):
    pseudo = predictions.loc[predictions["confidence"] >= threshold].copy()
    pseudo = pseudo.sort_values("confidence", ascending=False).reset_index(drop=True)

    if max_pseudo_samples is not None:
        pseudo = pseudo.iloc[:max_pseudo_samples].copy()

    return pseudo


def fit_pseudo_samples_to_train_capacity(pseudo_df, train_df, remove_strategy):
    if remove_strategy != "label_matched_random" or len(pseudo_df) == 0:
        return pseudo_df

    pseudo = pseudo_df.copy()
    pseudo["target_class"] = pseudo["pred_label"].astype(int).map(LABEL_MAP)

    capped_parts = []
    train_capacity = train_df["class"].value_counts().to_dict()
    for class_name, capacity in train_capacity.items():
        class_pseudo = pseudo.loc[pseudo["target_class"] == class_name].copy()
        class_pseudo = class_pseudo.sort_values("confidence", ascending=False).reset_index(drop=True)
        capped_parts.append(class_pseudo.iloc[:capacity].copy())

    if not capped_parts:
        return pseudo_df.iloc[0:0].copy()

    pseudo = pd.concat(capped_parts, ignore_index=True)
    pseudo = pseudo.drop(columns=["target_class"])
    pseudo = pseudo.sort_values("confidence", ascending=False).reset_index(drop=True)
    return pseudo


def build_pseudo_rows(pseudo_df, train_columns):
    rows = []
    for pseudo in pseudo_df.to_dict(orient="records"):
        pred_label = int(pseudo["pred_label"])
        row = {column: "" for column in train_columns}

        if "ID" in row:
            row["ID"] = f"pseudo_cold_{int(pseudo['row_id'])}"
        if "class" in row:
            row["class"] = LABEL_MAP[pred_label]
        if "implied_statement" in row:
            row["implied_statement"] = ""
        if "post" in row:
            row["post"] = pseudo["post"]
        if "aug_sent1_of_post" in row:
            row["aug_sent1_of_post"] = pseudo["post"]
        if "aug_sent2_of_post" in row:
            row["aug_sent2_of_post"] = pseudo["post"]

        rows.append(row)

    pseudo_rows = pd.DataFrame(rows, columns=train_columns)
    pseudo_rows["is_pseudo"] = 1
    pseudo_rows["pseudo_confidence"] = pseudo_df["confidence"].tolist()
    pseudo_rows["pseudo_pred_label"] = pseudo_df["pred_label"].astype(int).tolist()
    if "true_label" in pseudo_df.columns:
        pseudo_rows["cold_true_label"] = pseudo_df["true_label"].tolist()

    return pseudo_rows


def sample_rows_to_remove(train_df, pseudo_rows, strategy, seed):
    if len(pseudo_rows) == 0:
        return train_df.iloc[0:0].copy()

    if strategy == "random":
        return train_df.sample(n=len(pseudo_rows), random_state=seed).copy()

    to_remove_parts = []
    pseudo_counts = pseudo_rows["class"].value_counts().to_dict()
    for class_name, count in pseudo_counts.items():
        class_rows = train_df.loc[train_df["class"] == class_name]
        if len(class_rows) < count:
            raise ValueError(
                f"Not enough IHC samples in class={class_name} to remove {count} rows. "
                f"Only found {len(class_rows)}."
            )
        sampled = class_rows.sample(n=count, random_state=seed).copy()
        to_remove_parts.append(sampled)

    return pd.concat(to_remove_parts, ignore_index=False)


def main():
    args = parse_args()
    random.seed(args.seed)

    train_df = pd.read_csv(args.ihc_train, sep="\t")
    predictions = pd.read_csv(args.pseudo_predictions)

    raw_pseudo_candidates = select_pseudo_samples(
        predictions,
        threshold=args.confidence_threshold,
        max_pseudo_samples=args.max_pseudo_samples,
    )
    pseudo_candidates = fit_pseudo_samples_to_train_capacity(
        raw_pseudo_candidates,
        train_df,
        remove_strategy=args.remove_strategy,
    )
    pseudo_rows = build_pseudo_rows(pseudo_candidates, train_df.columns.tolist())

    if len(pseudo_rows) > len(train_df):
        raise ValueError(
            f"Pseudo sample count ({len(pseudo_rows)}) is larger than train set size ({len(train_df)}). "
            "Raise the threshold or lower --max_pseudo_samples."
        )

    removed_rows = sample_rows_to_remove(
        train_df,
        pseudo_rows,
        strategy=args.remove_strategy,
        seed=args.seed,
    )
    kept_train = train_df.drop(index=removed_rows.index).copy()

    kept_train["is_pseudo"] = 0
    kept_train["pseudo_confidence"] = pd.NA
    kept_train["pseudo_pred_label"] = pd.NA
    kept_train["cold_true_label"] = pd.NA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        new_train = pd.concat([kept_train, pseudo_rows], ignore_index=True)
    new_train = new_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    assert len(new_train) == len(train_df)

    os.makedirs(args.output_dir, exist_ok=True)
    train_out = os.path.join(args.output_dir, "train.tsv")
    valid_out = os.path.join(args.output_dir, "valid.tsv")
    test_out = os.path.join(args.output_dir, "test.tsv")
    meta_out = os.path.join(args.output_dir, "pseudo_injection_metadata.json")
    pseudo_out = os.path.join(args.output_dir, "selected_pseudo_samples.csv")

    new_train.to_csv(train_out, sep="\t", index=False)
    shutil.copyfile(args.ihc_valid, valid_out)
    shutil.copyfile(args.ihc_test, test_out)
    pseudo_candidates.to_csv(pseudo_out, index=False)

    metadata = {
        "ihc_train_path": args.ihc_train,
        "pseudo_predictions_path": args.pseudo_predictions,
        "confidence_threshold": args.confidence_threshold,
        "max_pseudo_samples": args.max_pseudo_samples,
        "remove_strategy": args.remove_strategy,
        "seed": args.seed,
        "original_train_size": len(train_df),
        "raw_selected_pseudo_samples_before_capacity_fit": len(raw_pseudo_candidates),
        "selected_pseudo_samples": len(pseudo_rows),
        "removed_ihc_samples": len(removed_rows),
        "new_train_size": len(new_train),
        "selected_pseudo_class_distribution": pseudo_rows["class"].value_counts().to_dict(),
        "removed_ihc_class_distribution": removed_rows["class"].value_counts().to_dict(),
    }

    with open(meta_out, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {train_out}")
    print(f"Selected pseudo samples: {len(pseudo_rows)}")
    print(f"Removed IHC samples: {len(removed_rows)}")
    print(f"Train size kept at: {len(new_train)}")


if __name__ == "__main__":
    main()
