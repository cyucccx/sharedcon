import argparse
import os
import random

import numpy as np
import pandas as pd
from angle_emb import AnglE
from simcse import SimCSE
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


np.random.seed(0)
random.seed(0)


def resolve_split_path(input_dir, split_name):
    candidates = []
    if split_name == "train":
        candidates = ["train.tsv", "train.csv"]
    elif split_name == "valid":
        candidates = ["valid.tsv", "valid.csv", "dev.tsv", "dev.csv"]
    elif split_name == "test":
        candidates = ["test.tsv", "test.csv"]
    else:
        raise NotImplementedError

    for candidate in candidates:
        path = os.path.join(input_dir, candidate)
        if os.path.exists(path):
            if candidate.endswith(".tsv"):
                return path, "\t", "tsv"
            return path, ",", "csv"

    raise FileNotFoundError(f"Cannot find split={split_name} under {input_dir}")


def load_sentence_encoder(model_name):
    print(f"LOAD_SENT_EMB_MODEL: {model_name}", flush=True)
    if model_name == "simcse":
        return SimCSE("princeton-nlp/sup-simcse-roberta-large")
    if model_name == "angle":
        return AnglE.from_pretrained(
            "SeanLee97/angle-bert-base-uncased-nli-en-v1",
            pooling_strategy="cls_avg",
        ).cuda()
    if model_name == "sbert":
        return SentenceTransformer("all-MiniLM-L6-v2")
    if model_name == "sbert-multi":
        return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    raise NotImplementedError(f"Unsupported sentence embedding model: {model_name}")


def cluster_n_select_global(dataset, sent_emb_model, input_col, cluster_num):
    posts = dataset[input_col].fillna("").astype(str).tolist()
    embeddings = sent_emb_model.encode(posts)
    embeddings = np.array(embeddings)

    if len(dataset) < cluster_num:
        raise ValueError(
            f"Number of rows ({len(dataset)}) is smaller than cluster_num ({cluster_num})."
        )

    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init="auto").fit(embeddings)
    cluster_labels = kmeans.labels_.tolist()
    centers = np.array(kmeans.cluster_centers_)

    closest_data = []
    for cluster_idx in range(cluster_num):
        center_vec = centers[cluster_idx].reshape(1, -1)
        data_idx_within_cluster = [
            idx for idx, assigned_cluster in enumerate(cluster_labels) if assigned_cluster == cluster_idx
        ]

        one_cluster_matrix = np.zeros((len(data_idx_within_cluster), centers.shape[1]))
        for row_num, data_idx in enumerate(data_idx_within_cluster):
            one_cluster_matrix[row_num] = embeddings[data_idx]

        closest, _ = pairwise_distances_argmin_min(center_vec, one_cluster_matrix)
        closest_idx_in_cluster = closest[0]
        closest_data_row_num = data_idx_within_cluster[closest_idx_in_cluster]
        closest_data.append(closest_data_row_num)

    centroid_sample = [dataset[input_col].iloc[closest_data[cluster_idx]] for cluster_idx in cluster_labels]

    clustered = dataset.copy()
    clustered["cluster"] = cluster_labels
    clustered["centroid_sample"] = centroid_sample
    return clustered


def process_split(input_dir, output_dir, split_name, sent_emb_model, input_col, cluster_num):
    split_path, sep, ext = resolve_split_path(input_dir, split_name)
    dataset = pd.read_csv(split_path, sep=sep)

    if input_col not in dataset.columns:
        raise KeyError(
            f"Column '{input_col}' does not exist in {split_path}. "
            f"Available columns: {dataset.columns.tolist()}"
        )

    print(f"Processing {split_name}: {split_path} (n={len(dataset)})")
    clustered = cluster_n_select_global(dataset, sent_emb_model, input_col, cluster_num)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}.{ext}")
    clustered.to_csv(output_path, sep=sep, index=False)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Global clustering without label split. Useful for mixed train sets with pseudo samples."
    )
    parser.add_argument(
        "--input_dir",
        default="raw_dataset/ihc_pure_cold_pseudo",
        type=str,
        help="Directory containing train/valid/test files.",
    )
    parser.add_argument(
        "--output_dir",
        default="clustered_dataset/sbert-multi/ihc_pure_cold_pseudo_c10",
        type=str,
        help="Directory to write clustered train/valid/test files.",
    )
    parser.add_argument(
        "--input_col",
        default="post",
        type=str,
        help="Text column to encode and cluster.",
    )
    parser.add_argument(
        "--cluster_num",
        default=10,
        type=int,
        help="Number of clusters.",
    )
    parser.add_argument(
        "--load_sent_emb_model",
        default="sbert-multi",
        type=str,
        help="Sentence embedding model name.",
    )
    args = parser.parse_args()

    sent_emb_model = load_sentence_encoder(args.load_sent_emb_model)

    for split_name in ["train", "valid", "test"]:
        process_split(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split_name=split_name,
            sent_emb_model=sent_emb_model,
            input_col=args.input_col,
            cluster_num=args.cluster_num,
        )


if __name__ == "__main__":
    main()
