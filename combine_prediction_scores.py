import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


DEFAULT_LLM_PREDICTIONS = "openai_batch/llm_predictions_final_all.csv"
DEFAULT_MODEL_PREDICTIONS = (
    "save/sbert-multi/sbert-multi_ihc_pure_c10/0/"
    "toxicn_filtered_trainpool_train_predictions_260530_v0.csv"
)
DEFAULT_OUTPUT = "openai_batch/combined_llm_ihc_cross_predictions.csv"


OUTPUT_COLUMNS = [
    "row_id",
    "post",
    "true_label",
    "llm_pred_label",
    "llm_confidence",
    "llm_prob_0",
    "llm_prob_1",
    "model_pred_label",
    "model_confidence",
    "model_prob_0",
    "model_prob_1",
    "cross_score_0",
    "cross_score_1",
    "combined_prob_0",
    "combined_prob_1",
    "combined_confidence",
    "combined_pred_label",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine LLM and model prediction CSVs by row_id using cross-product scores: "
            "combined_score_0 = llm_prob_0 * model_prob_0 and "
            "combined_score_1 = llm_prob_1 * model_prob_1."
        )
    )
    parser.add_argument("--llm_predictions", default=DEFAULT_LLM_PREDICTIONS)
    parser.add_argument("--model_predictions", default=DEFAULT_MODEL_PREDICTIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metrics_output",
        default=None,
        help="Optional metrics JSON path. Defaults to output path with _metrics.json suffix.",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Use raw cross scores for pred/confidence instead of normalizing score_0/score_1 to sum to 1.",
    )
    return parser.parse_args()


def load_predictions(path, source_name):
    df = pd.read_csv(path)
    required = {"row_id", "post", "true_label", "pred_label", "confidence", "prob_0", "prob_1"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source_name} prediction file is missing columns: {sorted(missing)}")

    df = df[["row_id", "post", "true_label", "pred_label", "confidence", "prob_0", "prob_1"]].copy()
    df["row_id"] = df["row_id"].astype(int)
    df["true_label"] = df["true_label"].astype(int)
    df["pred_label"] = df["pred_label"].astype(int)
    df["confidence"] = df["confidence"].astype(float)
    df["prob_0"] = df["prob_0"].astype(float)
    df["prob_1"] = df["prob_1"].astype(float)

    duplicate_mask = df["row_id"].duplicated(keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, "row_id"].head(20).tolist()
        raise ValueError(f"{source_name} has duplicate row_id values. Examples: {examples}")
    return df


def compute_metrics(df):
    y_true = df["true_label"]
    y_pred = df["combined_pred_label"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "rows": int(len(df)),
        "true_label_distribution": {
            str(int(k)): int(v)
            for k, v in y_true.value_counts().sort_index().to_dict().items()
        },
        "combined_pred_label_distribution": {
            str(int(k)): int(v)
            for k, v in y_pred.value_counts().sort_index().to_dict().items()
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix_true_rows_pred_cols_0_1": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "per_label": {
            str(label): {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate([0, 1])
        },
    }


def main():
    args = parse_args()
    llm = load_predictions(args.llm_predictions, "LLM")
    model = load_predictions(args.model_predictions, "model")

    merged = llm.merge(
        model,
        on="row_id",
        how="inner",
        suffixes=("_llm", "_model"),
    )
    if len(merged) != len(llm) or len(merged) != len(model):
        missing_llm = set(model["row_id"]) - set(llm["row_id"])
        missing_model = set(llm["row_id"]) - set(model["row_id"])
        raise ValueError(
            "Prediction files do not cover the same row_id set. "
            f"missing_in_llm={len(missing_llm)}, missing_in_model={len(missing_model)}"
        )

    post_mismatch = merged["post_llm"].astype(str) != merged["post_model"].astype(str)
    if post_mismatch.any():
        examples = merged.loc[post_mismatch, ["row_id", "post_llm", "post_model"]].head(5).to_dict(orient="records")
        raise ValueError(f"Posts differ for matching row_id values. Examples: {examples}")

    label_mismatch = merged["true_label_llm"].astype(int) != merged["true_label_model"].astype(int)
    if label_mismatch.any():
        examples = merged.loc[label_mismatch, ["row_id", "true_label_llm", "true_label_model"]].head(5).to_dict(orient="records")
        raise ValueError(f"true_label differs for matching row_id values. Examples: {examples}")

    output = pd.DataFrame(
        {
            "row_id": merged["row_id"].astype(int),
            "post": merged["post_llm"],
            "true_label": merged["true_label_llm"].astype(int),
            "llm_pred_label": merged["pred_label_llm"].astype(int),
            "llm_confidence": merged["confidence_llm"].astype(float),
            "llm_prob_0": merged["prob_0_llm"].astype(float),
            "llm_prob_1": merged["prob_1_llm"].astype(float),
            "model_pred_label": merged["pred_label_model"].astype(int),
            "model_confidence": merged["confidence_model"].astype(float),
            "model_prob_0": merged["prob_0_model"].astype(float),
            "model_prob_1": merged["prob_1_model"].astype(float),
        }
    )

    output["cross_score_0"] = output["llm_prob_0"] * output["model_prob_0"]
    output["cross_score_1"] = output["llm_prob_1"] * output["model_prob_1"]
    score_sum = output["cross_score_0"] + output["cross_score_1"]

    if args.no_normalize:
        output["combined_prob_0"] = output["cross_score_0"]
        output["combined_prob_1"] = output["cross_score_1"]
    else:
        output["combined_prob_0"] = output["cross_score_0"] / score_sum
        output["combined_prob_1"] = output["cross_score_1"] / score_sum

    tie_or_zero = score_sum.eq(0)
    if tie_or_zero.any():
        output.loc[tie_or_zero, "combined_prob_0"] = 0.5
        output.loc[tie_or_zero, "combined_prob_1"] = 0.5

    output["combined_pred_label"] = output["combined_prob_1"].gt(output["combined_prob_0"]).astype(int)
    output["combined_confidence"] = output[["combined_prob_0", "combined_prob_1"]].max(axis=1)
    output = output[OUTPUT_COLUMNS].sort_values("row_id").reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else output_path.with_name(output_path.stem + "_metrics.json")
    )
    metrics = compute_metrics(output)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Combined rows: {len(output)}")
    print(f"Saved combined CSV: {output_path}")
    print(f"Saved metrics JSON: {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
