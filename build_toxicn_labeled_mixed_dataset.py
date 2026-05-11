import argparse
import json
import os
import random

import pandas as pd

from util import ensure_output_dir_is_new

IHC_LABELS = {
    0: "not_hate",
    1: "implicit_hate",
}

IHC_TO_TOXICN = {
    "not_hate": 0,
    "implicit_hate": 1,
}

LABEL_SEED_OFFSET = {
    "not_hate": 101,
    "implicit_hate": 202,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replace part of IHC train/test with gold-labeled ToxiCN samples while "
            "matching labels and balancing ToxiCN topics."
        )
    )
    parser.add_argument("--ihc_dir", default="raw_dataset/ihc_pure", type=str)
    parser.add_argument("--toxicn_dir", default="raw_dataset/ToxiCN/data", type=str)
    parser.add_argument(
        "--output_dir",
        default="raw_dataset/ihc_pure_toxicn_labeled_balanced50_260511_v0",
        type=str,
    )
    parser.add_argument(
        "--replace_ratio",
        default=0.5,
        type=float,
        help="Target fraction of each IHC label to replace in train/test.",
    )
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def load_toxicn_split(path):
    with open(path, "r") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["mapped_class"] = df["toxic"].map(IHC_LABELS)
    return df


def round_robin_topic_sample(df, total_n, seed):
    if total_n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()

    topic_frames = {}
    selected_counts = {}
    topic_order = sorted(df["topic"].dropna().unique().tolist())

    for idx, topic in enumerate(topic_order):
        topic_df = df.loc[df["topic"] == topic].sample(frac=1, random_state=seed + idx).reset_index(drop=True)
        topic_frames[topic] = topic_df
        selected_counts[topic] = 0

    cursors = {topic: 0 for topic in topic_order}
    selected_parts = []

    while len(selected_parts) < total_n:
        available_topics = [
            topic for topic in topic_order
            if cursors[topic] < len(topic_frames[topic])
        ]
        if not available_topics:
            break

        available_topics.sort(key=lambda topic: (selected_counts[topic], topic))
        for topic in available_topics:
            if len(selected_parts) >= total_n:
                break
            selected_parts.append(topic_frames[topic].iloc[[cursors[topic]]])
            cursors[topic] += 1
            selected_counts[topic] += 1

    if not selected_parts:
        return df.iloc[0:0].copy()
    return pd.concat(selected_parts, ignore_index=True)


def select_toxicn_rows(source_df, replacement_targets, seed):
    selected_parts = []
    stats = {}

    for label_name, target_n in replacement_targets.items():
        toxicn_label = IHC_TO_TOXICN[label_name]
        label_source = source_df.loc[source_df["toxic"] == toxicn_label].copy().reset_index(drop=True)
        selected = round_robin_topic_sample(label_source, target_n, seed=seed + toxicn_label * 1000)

        selected_parts.append(selected)
        stats[label_name] = {
            "requested": int(target_n),
            "available": int(len(label_source)),
            "selected": int(len(selected)),
            "topic_distribution": {
                str(topic): int(count)
                for topic, count in selected["topic"].value_counts().sort_index().to_dict().items()
            },
        }

    selected_df = pd.concat(selected_parts, ignore_index=True)
    return selected_df, stats


def remove_ihc_rows(ihc_df, replacement_counts, seed):
    removed_parts = []
    kept_parts = []

    for label_name, target_n in replacement_counts.items():
        label_rows = ihc_df.loc[ihc_df["class"] == label_name].copy()
        label_rows = label_rows.sample(
            frac=1,
            random_state=seed + LABEL_SEED_OFFSET[label_name],
        ).reset_index(drop=True)

        remove_n = min(int(target_n), len(label_rows))
        removed_parts.append(label_rows.iloc[:remove_n].copy())
        kept_parts.append(label_rows.iloc[remove_n:].copy())

    removed_df = pd.concat(removed_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    kept_df = pd.concat(kept_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return kept_df, removed_df


def build_train_replacements(selected_toxicn):
    records = []
    for row in selected_toxicn.to_dict(orient="records"):
        text = row["content"]
        records.append(
            {
                "ID": f"toxicn_train_{int(row['source_row_id'])}",
                "class": row["mapped_class"],
                "implied_statement": "",
                "post": text,
                "aug_sent1_of_post": text,
                "aug_sent2_of_post": text,
            }
        )
    return pd.DataFrame(records)


def build_eval_replacements(selected_toxicn, split_name):
    records = []
    for row in selected_toxicn.to_dict(orient="records"):
        text = row["content"]
        records.append(
            {
                "ID": f"toxicn_{split_name}_{int(row['source_row_id'])}",
                "class": row["mapped_class"],
                "implied_statement": "",
                "post": text,
            }
        )
    return pd.DataFrame(records)


def add_source_row_id(df):
    df = df.copy().reset_index(drop=True)
    df["source_row_id"] = df.index
    return df


def build_mixed_split(ihc_df, toxicn_df, replace_ratio, seed, split_name):
    target_counts = {
        label_name: int(len(ihc_df.loc[ihc_df["class"] == label_name]) * replace_ratio)
        for label_name in ["not_hate", "implicit_hate"]
    }
    selected_toxicn, toxicn_stats = select_toxicn_rows(toxicn_df, target_counts, seed=seed)

    actual_counts = {
        label_name: int((selected_toxicn["mapped_class"] == label_name).sum())
        for label_name in ["not_hate", "implicit_hate"]
    }
    kept_ihc, removed_ihc = remove_ihc_rows(ihc_df, actual_counts, seed=seed)

    if split_name == "train":
        toxicn_rows = build_train_replacements(selected_toxicn)
    else:
        toxicn_rows = build_eval_replacements(selected_toxicn, split_name=split_name)

    mixed_df = pd.concat([kept_ihc, toxicn_rows], ignore_index=True)
    mixed_df = mixed_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    split_stats = {
        "ihc_original_size": int(len(ihc_df)),
        "mixed_size": int(len(mixed_df)),
        "requested_replacements": target_counts,
        "actual_replacements": actual_counts,
        "actual_replace_ratio": {
            label_name: (
                actual_counts[label_name] / int(len(ihc_df.loc[ihc_df["class"] == label_name]))
                if len(ihc_df.loc[ihc_df["class"] == label_name]) > 0
                else 0.0
            )
            for label_name in ["not_hate", "implicit_hate"]
        },
        "selected_toxicn": toxicn_stats,
    }

    return mixed_df, selected_toxicn, removed_ihc, split_stats


def main():
    args = parse_args()
    random.seed(args.seed)

    ensure_output_dir_is_new(args.output_dir, label="mixed labeled dataset output directory")

    ihc_train = pd.read_csv(os.path.join(args.ihc_dir, "train.tsv"), sep="\t")
    ihc_valid = pd.read_csv(os.path.join(args.ihc_dir, "valid.tsv"), sep="\t")
    ihc_test = pd.read_csv(os.path.join(args.ihc_dir, "test.tsv"), sep="\t")

    toxicn_train = add_source_row_id(load_toxicn_split(os.path.join(args.toxicn_dir, "train.json")))
    toxicn_test = add_source_row_id(load_toxicn_split(os.path.join(args.toxicn_dir, "test.json")))

    mixed_train, selected_train, removed_train, train_stats = build_mixed_split(
        ihc_df=ihc_train,
        toxicn_df=toxicn_train,
        replace_ratio=args.replace_ratio,
        seed=args.seed,
        split_name="train",
    )
    mixed_test, selected_test, removed_test, test_stats = build_mixed_split(
        ihc_df=ihc_test,
        toxicn_df=toxicn_test,
        replace_ratio=args.replace_ratio,
        seed=args.seed + 17,
        split_name="test",
    )

    os.makedirs(args.output_dir, exist_ok=False)
    mixed_train.to_csv(os.path.join(args.output_dir, "train.tsv"), sep="\t", index=False)
    ihc_valid.to_csv(os.path.join(args.output_dir, "valid.tsv"), sep="\t", index=False)
    mixed_test.to_csv(os.path.join(args.output_dir, "test.tsv"), sep="\t", index=False)

    selected_train.to_csv(os.path.join(args.output_dir, "selected_toxicn_train.csv"), index=False)
    selected_test.to_csv(os.path.join(args.output_dir, "selected_toxicn_test.csv"), index=False)
    removed_train.to_csv(os.path.join(args.output_dir, "removed_ihc_train.csv"), sep="\t", index=False)
    removed_test.to_csv(os.path.join(args.output_dir, "removed_ihc_test.csv"), sep="\t", index=False)

    metadata = {
        "ihc_dir": args.ihc_dir,
        "toxicn_dir": args.toxicn_dir,
        "replace_ratio": args.replace_ratio,
        "seed": args.seed,
        "valid_policy": "kept original IHC valid.tsv unchanged",
        "train": train_stats,
        "test": test_stats,
    }
    with open(os.path.join(args.output_dir, "mix_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Mixed dataset is saved at {args.output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
