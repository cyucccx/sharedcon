import argparse
import json
import os

import pandas as pd

from build_iterative_cross_pseudo_dataset import (
    build_commands,
    build_train_rows,
    collect_previous_row_ids,
    load_combined_predictions,
    read_ihc_split,
    summarize,
)
from util import ensure_output_dir_is_new


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced pseudo-only Chinese fine-tune dataset from LLM x model "
            "combined predictions. Selection is top-K by winning raw cross score within "
            "each predicted label, so label 0 and label 1 are kept close to 1:1."
        )
    )
    parser.add_argument(
        "--combined_predictions",
        default="openai_batch/combined_llm_ihc_cross_predictions.csv",
        help="CSV from combine_prediction_scores.py.",
    )
    parser.add_argument("--ihc_dir", default="raw_dataset/ihc_pure")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=2000)
    parser.add_argument(
        "--label0_k",
        type=int,
        default=None,
        help="Number of predicted label 0 rows to select. Defaults to top_k // 2.",
    )
    parser.add_argument(
        "--label1_k",
        type=int,
        default=None,
        help="Number of predicted label 1 rows to select. Defaults to top_k - label0_k.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Output raw dataset directory. Defaults to "
            "raw_dataset/ihc_pure_toxicn_filtered_iter<iteration>_cross_balanced_top<top_k>."
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
        help="Optional minimum winning_cross_score before balanced top-K selection.",
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


def resolve_label_counts(args):
    if args.label0_k is None and args.label1_k is None:
        label0_k = args.top_k // 2
        label1_k = args.top_k - label0_k
    elif args.label0_k is None:
        label1_k = args.label1_k
        label0_k = args.top_k - label1_k
    elif args.label1_k is None:
        label0_k = args.label0_k
        label1_k = args.top_k - label0_k
    else:
        label0_k = args.label0_k
        label1_k = args.label1_k

    if label0_k < 0 or label1_k < 0:
        raise ValueError("--label0_k and --label1_k must be non-negative.")
    if label0_k + label1_k != args.top_k:
        raise ValueError(
            f"label0_k + label1_k must equal top_k. Got {label0_k} + {label1_k} != {args.top_k}."
        )
    return {0: label0_k, 1: label1_k}


def select_balanced_pseudo_rows(predictions, previous_ids, args):
    label_counts = resolve_label_counts(args)
    candidates = predictions.loc[~predictions["row_id"].isin(previous_ids)].copy()

    if args.require_agreement:
        candidates = candidates.loc[
            candidates["llm_pred_label"].eq(candidates["combined_pred_label"])
            & candidates["model_pred_label"].eq(candidates["combined_pred_label"])
        ].copy()

    if args.min_cross_score is not None:
        candidates = candidates.loc[candidates["winning_cross_score"].ge(args.min_cross_score)].copy()

    selected_parts = []
    available_counts = candidates["combined_pred_label"].value_counts().to_dict()
    for label, label_k in label_counts.items():
        label_candidates = candidates.loc[candidates["combined_pred_label"].eq(label)].copy()
        if len(label_candidates) < label_k:
            raise ValueError(
                f"Only {len(label_candidates)} candidates remain for predicted label {label}, "
                f"but requested {label_k}. Available counts: {available_counts}"
            )
        selected_parts.append(
            label_candidates.sort_values(
                ["winning_cross_score", "combined_confidence", "row_id"],
                ascending=[False, False, True],
            )
            .head(label_k)
            .copy()
        )

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        .sort_values(
            ["winning_cross_score", "combined_confidence", "row_id"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["iteration"] = args.iteration
    return selected, candidates, label_counts


def main():
    args = parse_args()
    output_dir = args.output_dir or (
        f"raw_dataset/ihc_pure_toxicn_filtered_iter{args.iteration:02d}_cross_balanced_top{args.top_k}"
    )
    dataset_name = os.path.basename(output_dir.rstrip(os.sep))
    ensure_output_dir_is_new(output_dir, label="balanced iterative cross pseudo dataset output directory")

    ihc_valid = read_ihc_split(args.ihc_dir, "valid")
    ihc_test = read_ihc_split(args.ihc_dir, "test")
    previous_ids, previous_paths = collect_previous_row_ids(args)
    predictions = load_combined_predictions(args.combined_predictions)
    selected, candidates, label_counts = select_balanced_pseudo_rows(predictions, previous_ids, args)
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
        "Pseudo-only train set; selected by balanced per-label top-K winning raw cross score. "
        "IHC valid/test are copied only for pipeline compatibility."
    )
    metadata["requested_label_counts"] = {str(k): int(v) for k, v in label_counts.items()}
    metadata["commands"] = commands

    with open(os.path.join(output_dir, "mix_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "commands.sh"), "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for name, command in commands.items():
            f.write(f"# {name}\n{command}\n\n")

    print(f"Saved balanced iterative pseudo dataset to {output_dir}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
