import argparse
import json
import os

import pandas as pd

from util import ensure_output_dir_is_new


LABEL_TO_CLASS = {
    0: "not_hate",
    1: "implicit_hate",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a pseudo-labeled ToxiCN train dataset from a prediction CSV. "
            "By default the ToxiCN prediction rows are appended to IHC train; "
            "with --pseudo_only_train, train.tsv contains only the prediction rows. "
            "IHC valid/test are copied unchanged."
        )
    )
    parser.add_argument("--ihc_dir", default="raw_dataset/ihc_pure")
    parser.add_argument(
        "--predictions",
        default=(
            "save/sbert-multi/sbert-multi_ihc_pure_c10/0/"
            "toxicn_filtered_trainpool_train_predictions_260530_v0.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="raw_dataset/ihc_pure_toxicn_filtered_trainpool_260530_pseudo_all",
    )
    parser.add_argument(
        "--pseudo_only_train",
        action="store_true",
        help="Use only prediction pseudo rows in train.tsv instead of appending them to IHC train.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_ihc_split(ihc_dir, split):
    return pd.read_csv(os.path.join(ihc_dir, f"{split}.tsv"), sep="\t")


def load_predictions(path):
    df = pd.read_csv(path)
    required = {"row_id", "post", "true_label", "pred_label", "confidence", "prob_0", "prob_1"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

    df = df.copy()
    df["row_id"] = df["row_id"].astype(int)
    df["true_label"] = df["true_label"].astype(int)
    df["pred_label"] = df["pred_label"].astype(int)
    df["confidence"] = df["confidence"].astype(float)
    df["prob_0"] = df["prob_0"].astype(float)
    df["prob_1"] = df["prob_1"].astype(float)

    duplicate_mask = df["row_id"].duplicated(keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, "row_id"].head(20).tolist()
        raise ValueError(f"Duplicate row_id values in prediction CSV. Examples: {examples}")
    if not df["pred_label"].isin([0, 1]).all():
        raise ValueError("pred_label must contain only 0/1.")
    return df


def add_ihc_source_columns(df):
    output = df.copy()
    output["is_pseudo"] = 0
    output["pseudo_source_row_id"] = pd.NA
    output["pseudo_label"] = pd.NA
    output["pseudo_confidence"] = pd.NA
    output["pseudo_prob_0"] = pd.NA
    output["pseudo_prob_1"] = pd.NA
    output["pseudo_true_label"] = pd.NA
    return output


def build_pseudo_rows(predictions):
    rows = []
    for record in predictions.to_dict(orient="records"):
        text = record["post"]
        rows.append(
            {
                "ID": f"toxicn_filtered_trainpool_pseudo_{int(record['row_id'])}",
                "class": LABEL_TO_CLASS[int(record["pred_label"])],
                "implied_statement": "",
                "post": text,
                "aug_sent1_of_post": text,
                "aug_sent2_of_post": text,
                "is_pseudo": 1,
                "pseudo_source_row_id": int(record["row_id"]),
                "pseudo_label": int(record["pred_label"]),
                "pseudo_confidence": float(record["confidence"]),
                "pseudo_prob_0": float(record["prob_0"]),
                "pseudo_prob_1": float(record["prob_1"]),
                "pseudo_true_label": int(record["true_label"]),
            }
        )
    return pd.DataFrame(rows)


def summarize(ihc_train, pseudo_rows, mixed_train, args):
    return {
        "ihc_dir": args.ihc_dir,
        "predictions": args.predictions,
        "seed": args.seed,
            "policy": (
                "use only prediction pseudo rows in train; keep IHC valid/test unchanged"
                if args.pseudo_only_train
                else "append all prediction rows as pseudo labels; no ToxiCN split; keep IHC valid/test unchanged"
            ),
        "ihc_train_size": int(len(ihc_train)),
        "pseudo_train_size": int(len(pseudo_rows)),
        "mixed_train_size": int(len(mixed_train)),
        "ihc_train_class_distribution": {
            str(k): int(v)
            for k, v in ihc_train["class"].value_counts().sort_index().to_dict().items()
        },
        "pseudo_class_distribution": {
            str(k): int(v)
            for k, v in pseudo_rows["class"].value_counts().sort_index().to_dict().items()
        },
        "pseudo_pred_label_distribution": {
            str(int(k)): int(v)
            for k, v in pseudo_rows["pseudo_label"].value_counts().sort_index().to_dict().items()
        },
        "pseudo_original_true_label_distribution": {
            str(int(k)): int(v)
            for k, v in pseudo_rows["pseudo_true_label"].value_counts().sort_index().to_dict().items()
        },
        "mixed_train_class_distribution": {
            str(k): int(v)
            for k, v in mixed_train["class"].value_counts().sort_index().to_dict().items()
        },
        "pseudo_confidence": {
            str(k): float(v)
            for k, v in pseudo_rows["pseudo_confidence"].describe().to_dict().items()
        },
    }


def main():
    args = parse_args()
    ensure_output_dir_is_new(args.output_dir, label="prediction pseudo mixed dataset output directory")

    ihc_train = load_ihc_split(args.ihc_dir, "train")
    ihc_valid = load_ihc_split(args.ihc_dir, "valid")
    ihc_test = load_ihc_split(args.ihc_dir, "test")
    predictions = load_predictions(args.predictions)

    pseudo_rows = build_pseudo_rows(predictions)
    if args.pseudo_only_train:
        mixed_train = pseudo_rows.copy()
    else:
        mixed_train = pd.concat([add_ihc_source_columns(ihc_train), pseudo_rows], ignore_index=True)
    mixed_train = mixed_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=False)
    mixed_train.to_csv(os.path.join(args.output_dir, "train.tsv"), sep="\t", index=False)
    ihc_valid.to_csv(os.path.join(args.output_dir, "valid.tsv"), sep="\t", index=False)
    ihc_test.to_csv(os.path.join(args.output_dir, "test.tsv"), sep="\t", index=False)
    pseudo_rows.to_csv(os.path.join(args.output_dir, "selected_pseudo_samples.csv"), index=False)

    metadata = summarize(ihc_train, pseudo_rows, mixed_train, args)
    with open(os.path.join(args.output_dir, "mix_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved prediction pseudo mixed dataset to {args.output_dir}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
