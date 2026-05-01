import argparse
import json
from pathlib import Path

import pandas as pd


TOXIC_LABELS = {
    0: "non_toxic",
    1: "toxic",
}

TOXIC_TYPE_LABELS = {
    0: "non_toxic",
    1: "general_offensive_language",
    2: "hate_speech",
}

EXPRESSION_LABELS = {
    0: "non_hate",
    1: "explicit_hate_speech",
    2: "implicit_hate_speech",
    3: "reporting",
}

TARGET_LABELS = ["LGBTQ", "Region", "Sexism", "Racism", "Others"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze label and metadata distribution of a ToxiCN split."
    )
    parser.add_argument(
        "--input",
        default="raw_dataset/ToxiCN/data/test.json",
        help="Path to ToxiCN json split file.",
    )
    parser.add_argument(
        "--output",
        default="save/toxicn_test_distribution_analysis.json",
        help="Path to write the analysis JSON.",
    )
    return parser.parse_args()


def counts_with_labels(series, label_map):
    counts = series.value_counts().sort_index()
    result = {}
    total = int(len(series))
    for key, count in counts.items():
        label = label_map.get(int(key), str(key))
        result[str(int(key))] = {
            "label": label,
            "count": int(count),
            "ratio": float(count / total) if total else 0.0,
        }
    return result


def crosstab_to_records(frame, row_name):
    reset = frame.reset_index().rename(columns={frame.index.name or "index": row_name})
    records = []
    for record in reset.to_dict(orient="records"):
        normalized = {}
        for key, value in record.items():
            if hasattr(value, "item"):
                normalized[key] = value.item()
            else:
                normalized[key] = value
        records.append(normalized)
    return records


def length_stats(series):
    stats = series.describe()
    return {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in stats.to_dict().items()
    }


def analyze_targets(df):
    target_frame = pd.DataFrame(df["target"].tolist(), columns=TARGET_LABELS)
    total = len(target_frame)

    per_target = {}
    for column in TARGET_LABELS:
        count = int(target_frame[column].sum())
        per_target[column] = {
            "count": count,
            "ratio": float(count / total) if total else 0.0,
        }

    active_target_count = target_frame.sum(axis=1)
    active_distribution = active_target_count.value_counts().sort_index()

    toxic_mask = df["toxic"].astype(int).eq(1)
    toxic_target_frame = target_frame.loc[toxic_mask]
    toxic_total = len(toxic_target_frame)
    toxic_only = {}
    for column in TARGET_LABELS:
        count = int(toxic_target_frame[column].sum())
        toxic_only[column] = {
            "count": count,
            "ratio": float(count / toxic_total) if toxic_total else 0.0,
        }

    return {
        "per_target": per_target,
        "active_target_count_distribution": {
            str(int(k)): {
                "count": int(v),
                "ratio": float(v / total) if total else 0.0,
            }
            for k, v in active_distribution.items()
        },
        "toxic_only_per_target": toxic_only,
    }


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    topic_counts = df["topic"].value_counts().sort_index()
    platform_counts = df["platform"].value_counts().sort_index()

    toxic_topic = pd.crosstab(df["topic"], df["toxic"])
    toxic_topic = toxic_topic.rename(columns=TOXIC_LABELS)

    toxic_type_topic = pd.crosstab(df["topic"], df["toxic_type"])
    toxic_type_topic = toxic_type_topic.rename(columns=TOXIC_TYPE_LABELS)

    expression_topic = pd.crosstab(df["topic"], df["expression"])
    expression_topic = expression_topic.rename(columns=EXPRESSION_LABELS)

    toxic_expression = pd.crosstab(df["toxic"], df["expression"])
    toxic_expression.index = toxic_expression.index.map(TOXIC_LABELS)
    toxic_expression = toxic_expression.rename(columns=EXPRESSION_LABELS)

    analysis = {
        "file": str(input_path),
        "sample_count": int(len(df)),
        "columns": df.columns.tolist(),
        "topic_distribution": {
            topic: {
                "count": int(count),
                "ratio": float(count / len(df)) if len(df) else 0.0,
            }
            for topic, count in topic_counts.items()
        },
        "platform_distribution": {
            platform: {
                "count": int(count),
                "ratio": float(count / len(df)) if len(df) else 0.0,
            }
            for platform, count in platform_counts.items()
        },
        "toxic_distribution": counts_with_labels(df["toxic"].astype(int), TOXIC_LABELS),
        "toxic_type_distribution": counts_with_labels(df["toxic_type"].astype(int), TOXIC_TYPE_LABELS),
        "expression_distribution": counts_with_labels(df["expression"].astype(int), EXPRESSION_LABELS),
        "length_stats": length_stats(df["length"]),
        "crosstabs": {
            "topic_x_toxic": crosstab_to_records(toxic_topic, "topic"),
            "topic_x_toxic_type": crosstab_to_records(toxic_type_topic, "topic"),
            "topic_x_expression": crosstab_to_records(expression_topic, "topic"),
            "toxic_x_expression": crosstab_to_records(toxic_expression, "toxic"),
            "toxic_x_toxic_type": crosstab_to_records(
                pd.crosstab(df["toxic"], df["toxic_type"]).rename(index=TOXIC_LABELS, columns=TOXIC_TYPE_LABELS),
                "toxic",
            ),
        },
        "targets": analyze_targets(df),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
