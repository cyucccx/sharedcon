import argparse
import json
import os
import random
import shutil
import warnings

import pandas as pd

from util import ensure_output_dir_is_new

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
        default="pseudo_clusters/sbert-multi_cold_conf099_global_k50_aligned/selected_cluster_samples.csv",
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
        choices=["random", "label_matched_random", "cluster_matched_random"],
        default="label_matched_random",
        help="How to remove original IHC samples to keep train size unchanged.",
    )
    parser.add_argument(
        "--base_clustered_train",
        default="clustered_dataset/sbert-multi/ihc_pure_c10/train.tsv",
        type=str,
        help="Clustered IHC train.tsv used to remove original IHC rows from the pseudo-aligned cluster.",
    )
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="Keep existing pseudo rows in --ihc_train and only remove non-pseudo rows when adding new pseudo samples.",
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


def load_base_cluster_by_id(base_clustered_train):
    base_df = pd.read_csv(base_clustered_train, sep="\t", usecols=["ID", "cluster"])
    if base_df["ID"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate ID found in base clustered train: {base_clustered_train}")
    return dict(zip(base_df["ID"].astype(str), base_df["cluster"].astype(int)))


def attach_removal_clusters(train_df, base_cluster_by_id, removable_mask):
    removal_clusters = pd.Series(pd.NA, index=train_df.index, dtype="Int64")
    removable_ids = train_df.loc[removable_mask, "ID"].astype(str)
    mapped_clusters = removable_ids.map(base_cluster_by_id)
    if mapped_clusters.isna().any():
        missing_ids = removable_ids.loc[mapped_clusters.isna()].head(5).tolist()
        raise ValueError(
            "Missing base IHC cluster labels for removable rows. "
            f"Example IDs: {missing_ids}"
        )
    removal_clusters.loc[removable_mask] = mapped_clusters.astype(int).to_numpy()
    return removal_clusters


def fit_pseudo_samples_to_train_capacity(
    pseudo_df,
    train_df,
    remove_strategy,
    removable_train_df=None,
    removable_clusters=None,
):
    if remove_strategy not in {"label_matched_random", "cluster_matched_random"} or len(pseudo_df) == 0:
        return pseudo_df

    pseudo = pseudo_df.copy()
    capacity_df = removable_train_df if removable_train_df is not None else train_df

    if remove_strategy == "label_matched_random":
        pseudo["target_class"] = pseudo["pred_label"].astype(int).map(LABEL_MAP)
        train_capacity = capacity_df["class"].value_counts().to_dict()
        group_column = "target_class"
    else:
        if "aligned_target_cluster" not in pseudo.columns:
            raise ValueError("cluster_matched_random requires aligned_target_cluster in pseudo predictions.")
        if removable_clusters is None:
            raise ValueError("cluster_matched_random requires removable IHC cluster labels.")
        pseudo["target_cluster"] = pseudo["aligned_target_cluster"].astype(int)
        train_capacity = removable_clusters.dropna().astype(int).value_counts().to_dict()
        group_column = "target_cluster"

    capped_parts = []
    for group_value, capacity in train_capacity.items():
        group_pseudo = pseudo.loc[pseudo[group_column] == group_value].copy()
        group_pseudo = group_pseudo.sort_values("confidence", ascending=False).reset_index(drop=True)
        capped_parts.append(group_pseudo.iloc[:capacity].copy())

    if not capped_parts:
        return pseudo_df.iloc[0:0].copy()

    pseudo = pd.concat(capped_parts, ignore_index=True)
    pseudo = pseudo.drop(columns=[column for column in ["target_class", "target_cluster"] if column in pseudo.columns])
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
    if "cluster" in pseudo_df.columns:
        pseudo_rows["pseudo_source_cluster"] = pseudo_df["cluster"].astype(int).tolist()
    else:
        pseudo_rows["pseudo_source_cluster"] = pd.NA
    if "cluster_size" in pseudo_df.columns:
        pseudo_rows["pseudo_source_cluster_size"] = pseudo_df["cluster_size"].tolist()
    else:
        pseudo_rows["pseudo_source_cluster_size"] = pd.NA
    if "majority_ratio" in pseudo_df.columns:
        pseudo_rows["pseudo_source_cluster_majority_ratio"] = pseudo_df["majority_ratio"].tolist()
    else:
        pseudo_rows["pseudo_source_cluster_majority_ratio"] = pd.NA
    if "aligned_target_cluster" in pseudo_df.columns:
        pseudo_rows["aligned_target_cluster"] = pseudo_df["aligned_target_cluster"].tolist()
    else:
        pseudo_rows["aligned_target_cluster"] = pd.NA
    if "aligned_centroid_sample" in pseudo_df.columns:
        pseudo_rows["aligned_centroid_sample"] = pseudo_df["aligned_centroid_sample"].tolist()
    else:
        pseudo_rows["aligned_centroid_sample"] = pd.NA
    if "aligned_similarity" in pseudo_df.columns:
        pseudo_rows["aligned_similarity"] = pseudo_df["aligned_similarity"].tolist()
    else:
        pseudo_rows["aligned_similarity"] = pd.NA
    if "aligned_base_class_name" in pseudo_df.columns:
        pseudo_rows["aligned_base_class_name"] = pseudo_df["aligned_base_class_name"].tolist()
    else:
        pseudo_rows["aligned_base_class_name"] = pd.NA
    if "class_ratio_gap" in pseudo_df.columns:
        pseudo_rows["class_ratio_gap"] = pseudo_df["class_ratio_gap"].tolist()
    else:
        pseudo_rows["class_ratio_gap"] = pd.NA
    if "true_label" in pseudo_df.columns:
        pseudo_rows["cold_true_label"] = pseudo_df["true_label"].tolist()

    return pseudo_rows


def sample_rows_to_remove(train_df, pseudo_rows, strategy, seed, removable_mask=None, removal_clusters=None):
    if len(pseudo_rows) == 0:
        return train_df.iloc[0:0].copy()

    removable_df = train_df.loc[removable_mask].copy() if removable_mask is not None else train_df

    if strategy == "random":
        if len(removable_df) < len(pseudo_rows):
            raise ValueError(
                f"Not enough removable samples to remove {len(pseudo_rows)} rows. "
                f"Only found {len(removable_df)}."
        )
        return removable_df.sample(n=len(pseudo_rows), random_state=seed).copy()

    if strategy == "cluster_matched_random":
        if "aligned_target_cluster" not in pseudo_rows.columns:
            raise ValueError("cluster_matched_random requires aligned_target_cluster in pseudo rows.")
        if removal_clusters is None:
            raise ValueError("cluster_matched_random requires removal_clusters.")

        removable_df = removable_df.copy()
        removable_df["_removal_cluster"] = removal_clusters.loc[removable_df.index].astype(int)
        to_remove_parts = []
        pseudo_counts = pseudo_rows["aligned_target_cluster"].astype(int).value_counts().to_dict()
        for cluster_id, count in pseudo_counts.items():
            cluster_rows = removable_df.loc[removable_df["_removal_cluster"] == int(cluster_id)]
            if len(cluster_rows) < count:
                raise ValueError(
                    f"Not enough removable IHC samples in cluster={cluster_id} to remove {count} rows. "
                    f"Only found {len(cluster_rows)}."
                )
            sampled = cluster_rows.sample(n=count, random_state=seed).copy()
            to_remove_parts.append(sampled.drop(columns=["_removal_cluster"]))
        return pd.concat(to_remove_parts, ignore_index=False)

    to_remove_parts = []
    pseudo_counts = pseudo_rows["class"].value_counts().to_dict()
    for class_name, count in pseudo_counts.items():
        class_rows = removable_df.loc[removable_df["class"] == class_name]
        if len(class_rows) < count:
            raise ValueError(
                f"Not enough removable IHC samples in class={class_name} to remove {count} rows. "
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
    existing_pseudo_mask = (
        train_df["is_pseudo"].fillna(0).astype(int).eq(1)
        if args.cumulative and "is_pseudo" in train_df.columns
        else pd.Series(False, index=train_df.index)
    )
    removable_mask = ~existing_pseudo_mask
    removable_train_df = train_df.loc[removable_mask].copy()
    base_cluster_by_id = None
    removal_clusters = None
    if args.remove_strategy == "cluster_matched_random":
        base_cluster_by_id = load_base_cluster_by_id(args.base_clustered_train)
        removal_clusters = attach_removal_clusters(train_df, base_cluster_by_id, removable_mask)

    raw_pseudo_candidates = select_pseudo_samples(
        predictions,
        threshold=args.confidence_threshold,
        max_pseudo_samples=args.max_pseudo_samples,
    )
    pseudo_candidates = fit_pseudo_samples_to_train_capacity(
        raw_pseudo_candidates,
        train_df,
        remove_strategy=args.remove_strategy,
        removable_train_df=removable_train_df,
        removable_clusters=removal_clusters.loc[removable_mask] if removal_clusters is not None else None,
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
        removable_mask=removable_mask,
        removal_clusters=removal_clusters,
    )
    kept_train = train_df.drop(index=removed_rows.index).copy()

    new_metadata_columns = {
        "is_pseudo": 0,
        "pseudo_confidence": pd.NA,
        "pseudo_pred_label": pd.NA,
        "pseudo_source_cluster": pd.NA,
        "pseudo_source_cluster_size": pd.NA,
        "pseudo_source_cluster_majority_ratio": pd.NA,
        "aligned_target_cluster": pd.NA,
        "aligned_centroid_sample": pd.NA,
        "aligned_similarity": pd.NA,
        "aligned_base_class_name": pd.NA,
        "class_ratio_gap": pd.NA,
        "cold_true_label": pd.NA,
    }
    for column, default_value in new_metadata_columns.items():
        if column not in kept_train.columns:
            kept_train[column] = default_value
    non_pseudo_mask = kept_train["is_pseudo"].fillna(0).astype(int).ne(1)
    for column, default_value in new_metadata_columns.items():
        if column == "is_pseudo":
            kept_train.loc[non_pseudo_mask, column] = 0
        else:
            kept_train.loc[non_pseudo_mask, column] = default_value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        new_train = pd.concat([kept_train, pseudo_rows], ignore_index=True)
    new_train = new_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    assert len(new_train) == len(train_df)

    ensure_output_dir_is_new(args.output_dir, label="mixed raw dataset output directory")
    os.makedirs(args.output_dir, exist_ok=False)
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
        "base_clustered_train": args.base_clustered_train if args.remove_strategy == "cluster_matched_random" else None,
        "pseudo_predictions_path": args.pseudo_predictions,
        "confidence_threshold": args.confidence_threshold,
        "max_pseudo_samples": args.max_pseudo_samples,
        "remove_strategy": args.remove_strategy,
        "seed": args.seed,
        "cumulative": args.cumulative,
        "original_train_size": len(train_df),
        "existing_pseudo_samples_before_injection": int(existing_pseudo_mask.sum()),
        "removable_train_size": len(removable_train_df),
        "raw_selected_pseudo_samples_before_capacity_fit": len(raw_pseudo_candidates),
        "selected_pseudo_samples": len(pseudo_rows),
        "removed_ihc_samples": len(removed_rows),
        "new_train_size": len(new_train),
        "selected_pseudo_class_distribution": pseudo_rows["class"].value_counts().to_dict(),
        "removed_ihc_class_distribution": removed_rows["class"].value_counts().to_dict(),
        "selected_pseudo_aligned_cluster_distribution": (
            pseudo_rows["aligned_target_cluster"].astype(int).value_counts().sort_index().to_dict()
            if args.remove_strategy == "cluster_matched_random" and len(pseudo_rows) > 0
            else {}
        ),
        "removed_ihc_cluster_distribution": (
            removal_clusters.loc[removed_rows.index].astype(int).value_counts().sort_index().to_dict()
            if args.remove_strategy == "cluster_matched_random" and len(removed_rows) > 0
            else {}
        ),
    }

    with open(meta_out, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {train_out}")
    print(f"Selected pseudo samples: {len(pseudo_rows)}")
    print(f"Removed IHC samples: {len(removed_rows)}")
    print(f"Train size kept at: {len(new_train)}")


if __name__ == "__main__":
    main()
