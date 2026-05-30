import pandas as pd
import os
import numpy as np
import random
import argparse
import json
from angle_emb import AnglE
from simcse import SimCSE
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from util import ensure_output_dir_is_new

np.random.seed(0)
random.seed(0)


def split_train_valid_by_label(dataset, label_col, valid_ratio=0.1, random_state=0):
    train_parts = []
    valid_parts = []

    for _, label_df in dataset.groupby(label_col):
        label_df = label_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        if len(label_df) <= 1:
            train_parts.append(label_df)
            continue

        valid_count = max(1, int(round(len(label_df) * valid_ratio)))
        valid_count = min(valid_count, len(label_df) - 1)

        valid_parts.append(label_df.iloc[:valid_count].copy())
        train_parts.append(label_df.iloc[valid_count:].copy())

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    if valid_parts:
        valid_df = pd.concat(valid_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        valid_df = train_df.iloc[0:0].copy()

    return train_df, valid_df


def load_toxicn_splits(input_dir, valid_ratio=0.1, random_state=0):
    with open(os.path.join(input_dir, "train.json"), "r") as f:
        train_records = json.load(f)
    with open(os.path.join(input_dir, "test.json"), "r") as f:
        test_records = json.load(f)

    full_train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    train_df, valid_df = split_train_valid_by_label(
        full_train_df,
        label_col="toxic",
        valid_ratio=valid_ratio,
        random_state=random_state,
    )
    return train_df, valid_df, test_df


def load_sentence_encoder(model_name):
    local_files_only = os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"
    print(f'LOAD_SENT_EMB_MODEL: {model_name}', flush=True)
    if model_name == "simcse":
        return SimCSE("./sup-simcse-roberta-large")
    if model_name == "angle":
        return AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg').cuda()
    if model_name == "sbert":
        return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=local_files_only)
    if model_name == "sbert-multi":
        return SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=local_files_only,
        )
    raise NotImplementedError


def encode_posts(dataset, input_col, sent_emb_model):
    texts = dataset[input_col].fillna("").astype(str).to_list()
    return np.array(sent_emb_model.encode(texts))


def build_cluster_prototypes(dataset, sent_emb_model, input_col):
    if len(dataset) == 0:
        raise ValueError("Cannot build cluster prototypes from an empty dataset.")

    dataset = dataset.reset_index(drop=True).copy()
    embeddings = encode_posts(dataset, input_col, sent_emb_model)

    prototypes = []
    for cluster_id, cluster_df in dataset.groupby("cluster"):
        cluster_indices = cluster_df.index.to_list()
        cluster_embeddings = embeddings[cluster_indices]
        cluster_center = np.mean(cluster_embeddings, axis=0)
        closest, _ = pairwise_distances_argmin_min(cluster_center.reshape(1, -1), cluster_embeddings)
        closest_local_idx = int(closest[0])
        closest_global_idx = cluster_indices[closest_local_idx]

        if "centroid_sample" in cluster_df.columns:
            centroid_sample = cluster_df["centroid_sample"].iloc[0]
        else:
            centroid_sample = dataset.loc[closest_global_idx, input_col]

        prototypes.append(
            {
                "cluster": int(cluster_id),
                "centroid_embedding": cluster_center,
                "centroid_sample": centroid_sample,
            }
        )

    prototypes = pd.DataFrame(prototypes).sort_values("cluster").reset_index(drop=True)
    return prototypes


def attach_base_cluster_labels(mixed_train_dataset, base_clustered_train):
    base_metadata = base_clustered_train[["ID", "cluster", "centroid_sample"]].copy()
    english_rows = mixed_train_dataset.loc[mixed_train_dataset["is_pseudo"] != 1].copy()
    english_rows = english_rows.merge(base_metadata, on="ID", how="left", validate="one_to_one")

    if english_rows["cluster"].isna().any():
        missing_ids = english_rows.loc[english_rows["cluster"].isna(), "ID"].tolist()[:5]
        raise ValueError(
            f"Missing base cluster labels for some kept IHC rows. Example IDs: {missing_ids}"
        )

    english_rows["aligned_target_cluster"] = pd.NA
    english_rows["aligned_similarity"] = pd.NA
    return english_rows


def attach_stored_pseudo_alignment(pseudo_dataset):
    required_columns = ["aligned_target_cluster", "aligned_centroid_sample"]
    missing_columns = [col for col in required_columns if col not in pseudo_dataset.columns]
    if missing_columns:
        return None

    pseudo_dataset = pseudo_dataset.copy()
    missing_alignment = pseudo_dataset["aligned_target_cluster"].isna() | pseudo_dataset["aligned_centroid_sample"].isna()
    if missing_alignment.any():
        return None

    pseudo_dataset["cluster"] = pseudo_dataset["aligned_target_cluster"].astype(int)
    pseudo_dataset["centroid_sample"] = pseudo_dataset["aligned_centroid_sample"]
    if "aligned_similarity" not in pseudo_dataset.columns:
        pseudo_dataset["aligned_similarity"] = pd.NA
    return pseudo_dataset


def align_pseudo_clusters_to_base(pseudo_dataset, base_prototypes, sent_emb_model, input_col):
    if len(pseudo_dataset) == 0:
        empty = pseudo_dataset.copy()
        empty["cluster"] = pd.Series(dtype="Int64")
        empty["centroid_sample"] = pd.Series(dtype="object")
        empty["aligned_target_cluster"] = pd.Series(dtype="Int64")
        empty["aligned_similarity"] = pd.Series(dtype="float64")
        return empty

    if "pseudo_source_cluster" not in pseudo_dataset.columns:
        raise ValueError(
            "Expected pseudo_source_cluster column in mixed pseudo dataset. "
            "Rebuild the mixed raw pseudo dataset train.tsv with the updated build_pseudo_train_dataset.py first."
        )

    pseudo_dataset = pseudo_dataset.reset_index(drop=True).copy()
    pseudo_embeddings = encode_posts(pseudo_dataset, input_col, sent_emb_model)
    prototype_embeddings = np.stack(base_prototypes["centroid_embedding"].to_numpy())

    aligned_parts = []
    for pseudo_cluster_id, cluster_df in pseudo_dataset.groupby("pseudo_source_cluster", dropna=False):
        cluster_indices = cluster_df.index.to_list()
        cluster_embeddings = pseudo_embeddings[cluster_indices]
        pseudo_center = np.mean(cluster_embeddings, axis=0).reshape(1, -1)

        similarities = cosine_similarity(pseudo_center, prototype_embeddings)[0]
        best_idx = int(np.argmax(similarities))
        best_cluster = int(base_prototypes.iloc[best_idx]["cluster"])
        best_centroid_sample = base_prototypes.iloc[best_idx]["centroid_sample"]
        best_similarity = float(similarities[best_idx])

        cluster_df = cluster_df.copy()
        cluster_df["cluster"] = best_cluster
        cluster_df["centroid_sample"] = best_centroid_sample
        cluster_df["aligned_target_cluster"] = best_cluster
        cluster_df["aligned_similarity"] = best_similarity
        aligned_parts.append(cluster_df)

    return pd.concat(aligned_parts, ignore_index=True)


def build_aligned_mixed_ihc_dataset(
    mixed_train_dataset,
    base_clustered_train,
    base_clustered_valid,
    base_clustered_test,
    sent_emb_model,
    input_col,
):
    english_rows = attach_base_cluster_labels(mixed_train_dataset, base_clustered_train)
    pseudo_rows = mixed_train_dataset.loc[mixed_train_dataset["is_pseudo"] == 1].copy().reset_index(drop=True)
    pseudo_rows_with_alignment = attach_stored_pseudo_alignment(pseudo_rows)
    if pseudo_rows_with_alignment is not None:
        pseudo_rows = pseudo_rows_with_alignment
        print("use stored pseudo-to-IHC cluster alignment from raw mixed dataset")
    else:
        base_prototypes = build_cluster_prototypes(base_clustered_train, sent_emb_model, input_col)
        pseudo_rows = align_pseudo_clusters_to_base(pseudo_rows, base_prototypes, sent_emb_model, input_col)
        print("recomputed pseudo-to-IHC cluster alignment from pseudo source clusters")

    total_train_dataset = pd.concat([english_rows, pseudo_rows], ignore_index=True)
    total_train_dataset = total_train_dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    total_valid_dataset = base_clustered_valid.copy()
    total_test_dataset = base_clustered_test.copy()

    return total_train_dataset, total_valid_dataset, total_test_dataset



# given a dataset, compute clustering and select the closest data from each cluster
def cluster_n_select(dataset, sent_emb_model, input_col, is_not_hate):
    ## clustering
    ### encode models using the sentence embedding model
    embeddings = sent_emb_model.encode(dataset[input_col].to_list())

    # # if CUDA out of memory while using AnglE model, use this instead.
    # dummy = dataset.post.to_list()
    # block_size = 500
    # dummies = [dummy[i:i+block_size] for i in range(0,len(dummy), block_size)]
    # embeddingss = []
    # for dummy in dummies:
    #     embeddings = sent_emb_model.encode(dummy, to_numpy=True)
    #     embeddingss.append(embeddings)
    # embeddings = np.concatenate(tuple(embeddingss))

    all_data = [i for i in range(embeddings.shape[0])]

    
    ### kmeans clustering
    kmeans = KMeans(n_clusters=args.cluster_num, random_state=0, n_init="auto").fit(embeddings)
    m_clusters = kmeans.labels_.tolist()
    centers = np.array(kmeans.cluster_centers_)
    
    ## select the closest data from the cluster
    closest_data = []
    for i in range(args.cluster_num):
        center_vec = centers[i]
        center_vec = center_vec.reshape(1, -1) 
        
        data_idx_within_i_cluster = [idx for idx, clu_num in enumerate(m_clusters) if clu_num == i]

        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , centers.shape[1]))
        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = embeddings[data_idx]
            one_cluster_tf_matrix[row_num] = one_row
        
        closest, _ = pairwise_distances_argmin_min(center_vec, one_cluster_tf_matrix)
        closest_idx_in_one_cluster_tf_matrix = closest[0]
        closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
        data_id = all_data[closest_data_row_num]

        closest_data.append(data_id)
    assert len(closest_data) == args.cluster_num
    
    centroid_sample = [dataset[input_col][closest_data[i]] for i in m_clusters]

    ## distinguish the non-hate label
    if is_not_hate:
        m_clusters = [clu_num + 1000 for clu_num in m_clusters]
    
    dataset['cluster'] = m_clusters
    dataset['centroid_sample'] = centroid_sample
    
    return dataset

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_num', default=10,type=int, help='Enter the number of cluster')
    parser.add_argument('--load_dataset', default="ihc_pure",type=str, help='Enter the path of the dataset')
    parser.add_argument('--load_sent_emb_model', default="simcse",type=str, help='Enter the path/type of the sentence embedding model')
    parser.add_argument(
        '--align_pseudo_to_base',
        action='store_true',
        help='For mixed IHC+pseudo datasets, preserve base IHC cluster labels and align pseudo source clusters to them.',
    )
    parser.add_argument(
        '--base_cluster_dataset',
        default='ihc_pure_c10',
        type=str,
        help='Base clustered IHC dataset folder name under clustered_dataset/<model>/ used for pseudo cluster alignment.',
    )
    parser.add_argument(
        '--output_dataset_name',
        default=None,
        type=str,
        help='Optional output dataset folder name under clustered_dataset/<model>/. Refuses to overwrite existing directories.',
    )
    args = parser.parse_args()

    is_ihc_dataset = args.load_dataset.startswith("ihc_pure")
    is_mixed_ihc_dataset = args.load_dataset.startswith("ihc_pure_") and "_pseudo" in args.load_dataset

    # load raw dataset
    if is_ihc_dataset:
        train_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'train.tsv'), delimiter='\t', header=0)
        valid_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'valid.tsv'), delimiter='\t', header=0)
        test_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'test.tsv'), delimiter='\t', header=0)
        input_col = 'post'
        class_col = 'class'
        hate_class = "implicit_hate"
        not_hate_class = "not_hate"
    elif args.load_dataset == "sbic":
        train_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'train.csv'), delimiter=',', header=0)
        valid_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'dev.csv'), delimiter=',', header=0)
        test_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'test.csv'), delimiter=',', header=0)
        input_col = 'post'
        class_col = 'offensiveLABEL'
        hate_class = "offensive"
        not_hate_class = "not_offensive"
    elif args.load_dataset == "dynahate":
        train_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'train.csv'), delimiter=',', header=0)
        valid_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'dev.csv'), delimiter=',', header=0)
        test_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'test.csv'), delimiter=',', header=0)
        input_col = 'text'
        class_col = 'label'
        hate_class = "hate"
        not_hate_class = "nothate"
    elif args.load_dataset == "cold":
        train_dataset = pd.read_csv(os.path.join('raw_dataset', 'COLDataset', 'train.csv'), delimiter=',', header=0)
        valid_dataset = pd.read_csv(os.path.join('raw_dataset', 'COLDataset', 'dev.csv'), delimiter=',', header=0)
        test_dataset = pd.read_csv(os.path.join('raw_dataset', 'COLDataset', 'test.csv'), delimiter=',', header=0)
        input_col = 'TEXT'
        class_col = 'label'
        hate_class = 1
        not_hate_class = 0
    elif args.load_dataset in {"toxicn", "toxicn_filtered"}:
        toxicn_input_dir = (
            os.path.join("raw_dataset", "ToxiCN_filtered")
            if args.load_dataset == "toxicn_filtered"
            else os.path.join("raw_dataset", "ToxiCN", "data")
        )
        train_dataset, valid_dataset, test_dataset = load_toxicn_splits(
            toxicn_input_dir,
            valid_ratio=0.1,
            random_state=0,
        )
        input_col = "content"
        class_col = "toxic"
        hate_class = 1
        not_hate_class = 0
    else:
        raise NotImplementedError
    
    
    # load the sentence embedding model
    model = load_sentence_encoder(args.load_sent_emb_model)

    if is_mixed_ihc_dataset and args.align_pseudo_to_base:
        base_cluster_home = os.path.join("clustered_dataset", args.load_sent_emb_model, args.base_cluster_dataset)
        base_train_dataset = pd.read_csv(os.path.join(base_cluster_home, "train.tsv"), delimiter='\t', header=0)
        base_valid_dataset = pd.read_csv(os.path.join(base_cluster_home, "valid.tsv"), delimiter='\t', header=0)
        base_test_dataset = pd.read_csv(os.path.join(base_cluster_home, "test.tsv"), delimiter='\t', header=0)

        if "is_pseudo" not in train_dataset.columns:
            raise ValueError(
                "Expected is_pseudo column in raw mixed dataset. "
                f"Rebuild raw_dataset/{args.load_dataset}/train.tsv with build_pseudo_train_dataset.py first."
            )

        total_train_dataset, total_valid_dataset, total_test_dataset = build_aligned_mixed_ihc_dataset(
            mixed_train_dataset=train_dataset,
            base_clustered_train=base_train_dataset,
            base_clustered_valid=base_valid_dataset,
            base_clustered_test=base_test_dataset,
            sent_emb_model=model,
            input_col=input_col,
        )
        print("aligned pseudo source clusters to base IHC clusters")
    
    else:
        # processing each classes
        print("processing each classes...")

        ## 1) hate class
        mask_implicit_hate1 = train_dataset[class_col] == hate_class
        implicit_hate1 = train_dataset.loc[mask_implicit_hate1,:]
        implicit_hate1 = implicit_hate1.reset_index(drop=True)

        mask_implicit_hate2 = valid_dataset[class_col] == hate_class
        implicit_hate2 = valid_dataset.loc[mask_implicit_hate2,:]
        implicit_hate2 = implicit_hate2.reset_index(drop=True) 

        mask_implicit_hate3 = test_dataset[class_col] == hate_class
        implicit_hate3 = test_dataset.loc[mask_implicit_hate3,:]
        implicit_hate3 = implicit_hate3.reset_index(drop=True)

        ### cluster samples and and select the closest sample to the centroid per cluster
        implicit_hate1 = cluster_n_select(implicit_hate1, model, input_col, is_not_hate=False)
        implicit_hate2 = cluster_n_select(implicit_hate2, model, input_col, is_not_hate=False)
        implicit_hate3 = cluster_n_select(implicit_hate3, model, input_col, is_not_hate=False)
        print(f"class: implicit_hate DONE")
        
        ## 2) not_hate
        mask_not_hate1 = train_dataset[class_col] == not_hate_class
        not_hate1 = train_dataset.loc[mask_not_hate1,:]
        not_hate1 = not_hate1.reset_index(drop=True)

        mask_not_hate2 = valid_dataset[class_col] == not_hate_class
        not_hate2 = valid_dataset.loc[mask_not_hate2,:]
        not_hate2 = not_hate2.reset_index(drop=True)

        mask_not_hate3 = test_dataset[class_col] == not_hate_class
        not_hate3 = test_dataset.loc[mask_not_hate3,:]
        not_hate3 = not_hate3.reset_index(drop=True)

        ### cluster samples and and select the closest sample to the centroid per cluster
        not_hate1 = cluster_n_select(not_hate1, model, input_col, is_not_hate=True)
        not_hate2 = cluster_n_select(not_hate2, model, input_col, is_not_hate=True)
        not_hate3 = cluster_n_select(not_hate3, model, input_col, is_not_hate=True)
        print(f"class: not_hate DONE")
        
        
        # concat the samples of each class
        total_train_dataset = pd.concat([implicit_hate1, not_hate1])    
        total_train_dataset = total_train_dataset.sample(frac=1).reset_index(drop=True)

        total_valid_dataset = pd.concat([implicit_hate2, not_hate2])    
        total_valid_dataset = total_valid_dataset.sample(frac=1).reset_index(drop=True)

        total_test_dataset = pd.concat([implicit_hate3, not_hate3])    
        total_test_dataset = total_test_dataset.sample(frac=1).reset_index(drop=True)
    
    
    # save the dataset
    output_dataset_name = args.output_dataset_name or f"{args.load_dataset}_c{args.cluster_num}"
    output_dir = os.path.join("clustered_dataset", args.load_sent_emb_model, output_dataset_name)
    ensure_output_dir_is_new(output_dir, label="clustered dataset output directory")
    os.makedirs(output_dir, exist_ok=False)
    if is_ihc_dataset:
        total_train_dataset.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False)
        total_valid_dataset.to_csv(os.path.join(output_dir, "valid.tsv"), sep="\t", index=False)
        total_test_dataset.to_csv(os.path.join(output_dir, "test.tsv"), sep="\t", index=False)
    else:
        total_train_dataset.to_csv(os.path.join(output_dir, "train.csv"), sep=",", index=False)
        total_valid_dataset.to_csv(os.path.join(output_dir, "valid.csv"), sep=",", index=False)
        total_test_dataset.to_csv(os.path.join(output_dir, "test.csv"), sep=",", index=False)
    print(f"Clustered dataset is saved at {output_dir}")
