import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

from util import ensure_output_dir_is_new


LABEL_TO_CLASS = {
    0: "not_hate",
    1: "implicit_hate",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build one iteration of a pseudo-only Chinese fine-tune dataset from "
            "LLM x model combined predictions. Selection is global top-K by the "
            "winning class raw cross score, with optional exclusion of previous rounds."
        )
    )
    parser.add_argument(
        "--combined_predictions",
        default="openai_batch/combined_llm_ihc_cross_predictions.csv",
        help="CSV from combine_prediction_scores.py for the current baseline/current model.",
    )
    parser.add_argument("--ihc_dir", default="raw_dataset/ihc_pure")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Output raw dataset directory. Defaults to "
            "raw_dataset/ihc_pure_toxicn_filtered_iter<iteration>_cross_top<top_k>."
        ),
    )
    parser.add_argument(
        "--previous_selected_csv",
        action="append",
        default=[],
        help="CSV from a previous iteration selected_pseudo_samples.csv. Can be repeated.",
    )
    parser.add_argument(
        "--previous_dataset_dir",
        action="append",
        default=[],
        help="Previous raw dataset dir containing selected_pseudo_samples.csv. Can be repeated.",
    )
    parser.add_argument(
        "--history_glob",
        default=None,
        help="Optional glob for previous selected_pseudo_samples.csv files to exclude.",
    )
    parser.add_argument(
        "--require_agreement",
        action="store_true",
        help="Only select rows where llm_pred_label == model_pred_label == combined_pred_label.",
    )
    parser.add_argument(
        "--min_cross_score",
        type=float,
        default=None,
        help="Optional minimum winning_cross_score before top-K selection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sent_emb_model", default="sbert-multi")
    parser.add_argument("--cluster_num", type=int, default=10)
    parser.add_argument("--tokenizer", default="bert-base-multilingual-cased")
    parser.add_argument(
        "--init_checkpoint_dir",
        default="save/sbert-multi/sbert-multi_ihc_pure_c10/0",
        help="Checkpoint dir to initialize fine-tuning from in printed commands.",
    )
    parser.add_argument(
        "--init_checkpoint_filename",
        default="model_260519_v0.pt",
        help="Checkpoint filename to initialize fine-tuning from in printed commands.",
    )
    return parser.parse_args()


def read_ihc_split(ihc_dir, split):
    return pd.read_csv(os.path.join(ihc_dir, f"{split}.tsv"), sep="\t")


def collect_previous_row_ids(args):
    paths = []
    paths.extend(args.previous_selected_csv)
    for dataset_dir in args.previous_dataset_dir:
        paths.append(os.path.join(dataset_dir, "selected_pseudo_samples.csv"))
    if args.history_glob:
        paths.extend(glob.glob(args.history_glob))

    previous_ids = set()
    used_paths = []
    for path in sorted(set(paths)):
        if not path or not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        if "row_id" not in df.columns:
            raise ValueError(f"Previous selected file has no row_id column: {path}")
        previous_ids.update(df["row_id"].astype(int).tolist())
        used_paths.append(path)
    return previous_ids, used_paths


def load_combined_predictions(path):
    df = pd.read_csv(path)
    required = {
        "row_id",
        "post",
        "true_label",
        "llm_pred_label",
        "model_pred_label",
        "cross_score_0",
        "cross_score_1",
        "combined_pred_label",
        "combined_confidence",
        "combined_prob_0",
        "combined_prob_1",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Combined prediction CSV is missing columns: {sorted(missing)}")

    df = df.copy()
    df["row_id"] = df["row_id"].astype(int)
    df["true_label"] = df["true_label"].astype(int)
    df["llm_pred_label"] = df["llm_pred_label"].astype(int)
    df["model_pred_label"] = df["model_pred_label"].astype(int)
    df["combined_pred_label"] = df["combined_pred_label"].astype(int)
    df["cross_score_0"] = df["cross_score_0"].astype(float)
    df["cross_score_1"] = df["cross_score_1"].astype(float)
    df["combined_confidence"] = df["combined_confidence"].astype(float)
    df["winning_cross_score"] = df.apply(
        lambda row: row["cross_score_1"]
        if int(row["combined_pred_label"]) == 1
        else row["cross_score_0"],
        axis=1,
    )
    df["pseudo_class"] = df["combined_pred_label"].map(LABEL_TO_CLASS)
    if df["pseudo_class"].isna().any():
        bad = df.loc[df["pseudo_class"].isna(), "combined_pred_label"].unique().tolist()
        raise ValueError(f"Unsupported combined_pred_label values: {bad}")
    return df


def select_pseudo_rows(predictions, previous_ids, args):
    candidates = predictions.loc[~predictions["row_id"].isin(previous_ids)].copy()

    if args.require_agreement:
        candidates = candidates.loc[
            candidates["llm_pred_label"].eq(candidates["combined_pred_label"])
            & candidates["model_pred_label"].eq(candidates["combined_pred_label"])
        ].copy()

    if args.min_cross_score is not None:
        candidates = candidates.loc[candidates["winning_cross_score"].ge(args.min_cross_score)].copy()

    if len(candidates) < args.top_k:
        raise ValueError(
            f"Only {len(candidates)} candidates remain, but --top_k={args.top_k}. "
            "Lower top_k or relax filters."
        )

    selected = (
        candidates.sort_values(
            ["winning_cross_score", "combined_confidence", "row_id"],
            ascending=[False, False, True],
        )
        .head(args.top_k)
        .copy()
        .reset_index(drop=True)
    )
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["iteration"] = args.iteration
    return selected, candidates


def build_train_rows(selected):
    rows = []
    for record in selected.to_dict(orient="records"):
        text = record["post"]
        rows.append(
            {
                "ID": f"iter{int(record['iteration']):02d}_toxicn_pseudo_{int(record['row_id'])}",
                "class": record["pseudo_class"],
                "implied_statement": "",
                "post": text,
                "aug_sent1_of_post": text,
                "aug_sent2_of_post": text,
                "is_pseudo": 1,
                "pseudo_source_row_id": int(record["row_id"]),
                "pseudo_label": int(record["combined_pred_label"]),
                "pseudo_score": float(record["winning_cross_score"]),
                "pseudo_confidence": float(record["combined_confidence"]),
                "pseudo_true_label": int(record["true_label"]),
                "iteration": int(record["iteration"]),
                "selection_rank": int(record["selection_rank"]),
            }
        )
    return pd.DataFrame(rows)


def build_commands(dataset_name, args):
    clustered_dataset = f"{dataset_name}_c{args.cluster_num}"
    return {
        "cluster": (
            "python shared_semantics.py "
            f"--cluster_num {args.cluster_num} "
            f"--load_dataset {dataset_name} "
            f"--load_sent_emb_model {args.sent_emb_model}"
        ),
        "preprocess": (
            "python preprocess_dataset.py "
            f"-m {args.sent_emb_model} "
            f"-d {clustered_dataset} "
            f"-t {args.tokenizer}"
        ),
        "train": (
            f"TRAIN_DATASET={clustered_dataset} "
            f"INIT_CHECKPOINT_DIR={args.init_checkpoint_dir} "
            f"INIT_CHECKPOINT_FILENAME={args.init_checkpoint_filename} "
            "python train.py"
        ),
        "eval": (
            "EVAL_DATASETS=ihc_pure_c10,toxicn_filtered "
            f"EVAL_LOAD_DIR=save/{args.sent_emb_model}/{clustered_dataset}/0 "
            "EVAL_MODEL_FILENAME=<new_model_filename.pt> "
            "python eval.py"
        ),
        "predict_next_trainpool": (
            "EVAL_DATASETS=toxicn_filtered_trainpool "
            f"EVAL_LOAD_DIR=save/{args.sent_emb_model}/{clustered_dataset}/0 "
            "EVAL_MODEL_FILENAME=<new_model_filename.pt> "
            "EVAL_TRAIN_ONLY=1 "
            "python eval.py"
        ),
        "combine_next_predictions": (
            "python combine_prediction_scores.py "
            "--llm_predictions openai_batch/llm_predictions_final_all.csv "
            "--model_predictions <next_toxicn_filtered_trainpool_predictions.csv> "
            f"--output openai_batch/combined_iter{args.iteration + 1:02d}_predictions.csv"
        ),
    }


def summarize(selected, candidates, previous_ids, previous_paths, train_rows, args, dataset_name):
    return {
        "dataset_name": dataset_name,
        "combined_predictions": args.combined_predictions,
        "iteration": args.iteration,
        "top_k": args.top_k,
        "seed": args.seed,
        "previous_selected_paths": previous_paths,
        "previous_selected_count": int(len(previous_ids)),
        "candidate_count_after_exclusion_and_filters": int(len(candidates)),
        "require_agreement": bool(args.require_agreement),
        "min_cross_score": args.min_cross_score,
        "train_size": int(len(train_rows)),
        "selected_class_distribution": {
            str(k): int(v)
            for k, v in selected["pseudo_class"].value_counts().sort_index().to_dict().items()
        },
        "selected_pred_label_distribution": {
            str(int(k)): int(v)
            for k, v in selected["combined_pred_label"].value_counts().sort_index().to_dict().items()
        },
        "selected_true_label_distribution": {
            str(int(k)): int(v)
            for k, v in selected["true_label"].value_counts().sort_index().to_dict().items()
        },
        "winning_cross_score": {
            str(k): float(v)
            for k, v in selected["winning_cross_score"].describe().to_dict().items()
        },
    }


def main():
    args = parse_args()
    output_dir = args.output_dir or (
        f"raw_dataset/ihc_pure_toxicn_filtered_iter{args.iteration:02d}_cross_top{args.top_k}"
    )
    dataset_name = os.path.basename(output_dir.rstrip(os.sep))
    ensure_output_dir_is_new(output_dir, label="iterative cross pseudo dataset output directory")

    ihc_valid = read_ihc_split(args.ihc_dir, "valid")
    ihc_test = read_ihc_split(args.ihc_dir, "test")
    previous_ids, previous_paths = collect_previous_row_ids(args)
    predictions = load_combined_predictions(args.combined_predictions)
    selected, candidates = select_pseudo_rows(predictions, previous_ids, args)
    train_rows = build_train_rows(selected)

    os.makedirs(output_dir, exist_ok=False)
    train_rows.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False)
    ihc_valid.to_csv(os.path.join(output_dir, "valid.tsv"), sep="\t", index=False)
    ihc_test.to_csv(os.path.join(output_dir, "test.tsv"), sep="\t", index=False)
    selected.to_csv(os.path.join(output_dir, "selected_pseudo_samples.csv"), index=False)

    commands = build_commands(dataset_name, args)
    metadata = summarize(
        selected=selected,
        candidates=candidates,
        previous_ids=previous_ids,
        previous_paths=previous_paths,
        train_rows=train_rows,
        args=args,
        dataset_name=dataset_name,
    )
    metadata["policy"] = (
        "Pseudo-only train set; selected by global top-K winning raw cross score. "
        "IHC valid/test are copied only for pipeline compatibility."
    )
    metadata["commands"] = commands

    with open(os.path.join(output_dir, "mix_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "commands.sh"), "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for name, command in commands.items():
            f.write(f"# {name}\n{command}\n\n")

    print(f"Saved iterative pseudo dataset to {output_dir}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
