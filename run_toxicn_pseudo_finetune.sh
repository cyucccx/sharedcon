#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_TAG="${RUN_TAG:-$(date +%y%m%d)}"
if [[ -z "${RUN_VERSION:-}" ]]; then
  EXISTING_RUN_VERSION="$(find . -print | grep -Eo '_[0-9]{6}_v[0-9]+' | sed -E 's/.*_v([0-9]+)/\1/' | sort -n | tail -1 || true)"
  if [[ -n "$EXISTING_RUN_VERSION" ]]; then
    RUN_VERSION="$((EXISTING_RUN_VERSION + 1))"
  else
    RUN_VERSION="1"
  fi
fi

next_versioned_path() {
  local prefix="$1"
  local suffix="${2:-}"
  local candidate
  local duplicate_index=2
  candidate="${prefix}_${RUN_TAG}_v${RUN_VERSION}${suffix}"
  if [[ ! -e "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return
  fi
  while true; do
    candidate="${prefix}_${RUN_TAG}_v${RUN_VERSION}_r${duplicate_index}${suffix}"
    if [[ ! -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
    duplicate_index=$((duplicate_index + 1))
  done
}

next_versioned_label() {
  local parent_dir="$1"
  local stem="$2"
  local candidate
  local duplicate_index=2
  candidate="${stem}_${RUN_TAG}_v${RUN_VERSION}"
  if [[ ! -e "$parent_dir/$candidate" ]]; then
    printf '%s\n' "$candidate"
    return
  fi
  while true; do
    candidate="${stem}_${RUN_TAG}_v${RUN_VERSION}_r${duplicate_index}"
    if [[ ! -e "$parent_dir/$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
    duplicate_index=$((duplicate_index + 1))
  done
}

TERMINAL_LOG="${TERMINAL_LOG:-1}"
TERMINAL_LOG_DIR="${TERMINAL_LOG_DIR:-logs/terminal}"
TERMINAL_LOG_PATH="${TERMINAL_LOG_PATH:-}"
if [[ "$TERMINAL_LOG" == "1" ]]; then
  if [[ -z "$TERMINAL_LOG_PATH" ]]; then
    TERMINAL_LOG_PATH="$(next_versioned_path "$TERMINAL_LOG_DIR/run_pseudo_finetune" ".log")"
  elif [[ -e "$TERMINAL_LOG_PATH" ]]; then
    echo "Terminal log already exists: $TERMINAL_LOG_PATH" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$TERMINAL_LOG_PATH")"
  exec > >(tee "$TERMINAL_LOG_PATH") 2>&1
  echo "Terminal log: $TERMINAL_LOG_PATH"
fi

CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.99}"
MAX_PSEUDO_SAMPLES="${MAX_PSEUDO_SAMPLES:-}"
GLOBAL_CLUSTER_NUM="${GLOBAL_CLUSTER_NUM:-50}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-5}"
PURITY_THRESHOLD="${PURITY_THRESHOLD:-0.80}"
BALANCED_MARGIN="${BALANCED_MARGIN:-0.10}"
CLASS_RATIO_GAP_THRESHOLD="${CLASS_RATIO_GAP_THRESHOLD:-0.20}"
REMOVE_STRATEGY="${REMOVE_STRATEGY:-label_matched_random}"
SHARED_CLUSTER_NUM="${SHARED_CLUSTER_NUM:-10}"
SENT_EMB_MODEL="${SENT_EMB_MODEL:-sbert-multi}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-bert-base-multilingual-cased}"
ITERATION_ROUND="${ITERATION_ROUND:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_IHC_DATASET="${EVAL_IHC_DATASET:-ihc_pure_c10}"
EVAL_SOURCE_DATASET="${EVAL_SOURCE_DATASET:-toxicn}"
EVAL_MODEL_FILENAME="${EVAL_MODEL_FILENAME:-}"

DEFAULT_BASELINE_SAVE_DIR="save/sbert-multi/sbert-multi_ihc_pure_c10/0"
DEFAULT_SOURCE_DATASET_DIR="raw_dataset/ToxiCN/data"
SOURCE_SAVE_DIR="${SOURCE_SAVE_DIR:-$DEFAULT_BASELINE_SAVE_DIR}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-$DEFAULT_SOURCE_DATASET_DIR}"
SOURCE_DATASET_NAME="${SOURCE_DATASET_NAME:-toxicn}"
SOURCE_LABEL_FIELD="${SOURCE_LABEL_FIELD:-toxic}"
SOURCE_PREDICTION_DATASET_NAME="${SOURCE_PREDICTION_DATASET_NAME:-${SOURCE_DATASET_NAME}_source_pool}"
SOURCE_MIXED_RAW_DATASET_DIR="${SOURCE_MIXED_RAW_DATASET_DIR:-}"

if [[ -z "${MIXED_RAW_DATASET_BASE:-}" ]]; then
  MIXED_RAW_DATASET_BASE="ihc_pure_${SOURCE_DATASET_NAME}_pseudo"
fi

if [[ -z "$SOURCE_MIXED_RAW_DATASET_DIR" && "$ITERATION_ROUND" != "1" ]]; then
  PREVIOUS_VERSION="$((RUN_VERSION - 1))"
  PREVIOUS_MIXED_RAW_DATASET="raw_dataset/${MIXED_RAW_DATASET_BASE}_${RUN_TAG}_v${PREVIOUS_VERSION}"
  if [[ -d "$PREVIOUS_MIXED_RAW_DATASET" ]]; then
    SOURCE_MIXED_RAW_DATASET_DIR="$PREVIOUS_MIXED_RAW_DATASET"
  else
    echo "Missing previous mixed raw dataset: $PREVIOUS_MIXED_RAW_DATASET" >&2
    echo "Set SOURCE_MIXED_RAW_DATASET_DIR explicitly if the previous round uses a different name/date." >&2
    exit 1
  fi
fi

CONFIDENCE_TAG="${CONFIDENCE_THRESHOLD#0.}"
CONFIDENCE_TAG="${CONFIDENCE_TAG//./}"
if [[ -z "${PSEUDO_OUTPUT_DIR_BASE:-}" ]]; then
  PSEUDO_OUTPUT_DIR_BASE="pseudo_clusters/${SENT_EMB_MODEL}_${SOURCE_DATASET_NAME}_conf${CONFIDENCE_TAG}_global_k${GLOBAL_CLUSTER_NUM}_aligned"
fi

if [[ -z "${FILTERED_SOURCE_DATASET_BASE:-}" ]]; then
  FILTERED_SOURCE_DATASET_BASE="${SOURCE_DATASET_NAME}_no_pseudo"
fi

SOURCE_PREDICTION_PREFIX="${SOURCE_PREDICTION_PREFIX:-$SOURCE_PREDICTION_DATASET_NAME}"

MIXED_RAW_DATASET="$(next_versioned_label "raw_dataset" "$MIXED_RAW_DATASET_BASE")"
MIXED_CLUSTERED_DATASET="${MIXED_RAW_DATASET}_c${SHARED_CLUSTER_NUM}"
PSEUDO_OUTPUT_DIR="$(next_versioned_path "$PSEUDO_OUTPUT_DIR_BASE")"
FILTERED_SOURCE_DATASET_NAME="$(next_versioned_label "raw_dataset" "$FILTERED_SOURCE_DATASET_BASE")"
FILTERED_SOURCE_DATASET_DIR="raw_dataset/$FILTERED_SOURCE_DATASET_NAME"
FILTERED_SOURCE_PREPROCESSED="preprocessed_data/preprocessed_${FILTERED_SOURCE_DATASET_NAME}.pkl"
MIXED_PREPROCESSED_FILE="preprocessed_data/preprocessed_${SENT_EMB_MODEL}_${MIXED_CLUSTERED_DATASET}.pkl"
FINETUNE_SAVE_DIR="save/sbert-multi/${MIXED_CLUSTERED_DATASET}/0"
SOURCE_POOL_PREPROCESSED="preprocessed_data/preprocessed_${SOURCE_PREDICTION_DATASET_NAME}.pkl"

BASE_IHC_CHECKPOINT_DIR="$SOURCE_SAVE_DIR"
BASE_CLUSTERED_TRAIN="clustered_dataset/sbert-multi/ihc_pure_c10/train.tsv"
if [[ -n "$SOURCE_MIXED_RAW_DATASET_DIR" ]]; then
  BASE_RAW_TRAIN="$SOURCE_MIXED_RAW_DATASET_DIR/train.tsv"
  BASE_RAW_VALID="$SOURCE_MIXED_RAW_DATASET_DIR/valid.tsv"
  BASE_RAW_TEST="$SOURCE_MIXED_RAW_DATASET_DIR/test.tsv"
else
  BASE_RAW_TRAIN="raw_dataset/ihc_pure/train.tsv"
  BASE_RAW_VALID="raw_dataset/ihc_pure/valid.tsv"
  BASE_RAW_TEST="raw_dataset/ihc_pure/test.tsv"
fi

if compgen -G "$BASE_IHC_CHECKPOINT_DIR/model*.pt" > /dev/null; then
  BASE_IHC_CHECKPOINT="$(ls -1t "$BASE_IHC_CHECKPOINT_DIR"/model*.pt | head -n 1)"
else
  BASE_IHC_CHECKPOINT=""
fi

if [[ -z "$BASE_IHC_CHECKPOINT" ]]; then
  echo "Missing required baseline checkpoint under: $BASE_IHC_CHECKPOINT_DIR" >&2
  exit 1
fi

if compgen -G "$SOURCE_SAVE_DIR/${SOURCE_PREDICTION_PREFIX}_train_predictions*.csv" > /dev/null; then
  SOURCE_PREDICTIONS="$(ls -1t "$SOURCE_SAVE_DIR"/"${SOURCE_PREDICTION_PREFIX}"_train_predictions*.csv | head -n 1)"
elif compgen -G "$SOURCE_SAVE_DIR/${SOURCE_DATASET_NAME}_train_predictions*.csv" > /dev/null; then
  SOURCE_PREDICTIONS="$(ls -1t "$SOURCE_SAVE_DIR"/"${SOURCE_DATASET_NAME}"_train_predictions*.csv | head -n 1)"
  SOURCE_PREDICTION_PREFIX="$SOURCE_DATASET_NAME"
else
  SOURCE_PREDICTIONS=""
fi

echo "================ PIPELINE START ================"
echo "ROOT_DIR=$ROOT_DIR"
echo "HF_LOCAL_FILES_ONLY=$HF_LOCAL_FILES_ONLY"
echo "RUN_TAG=$RUN_TAG"
echo "RUN_VERSION=$RUN_VERSION"
echo "ITERATION_ROUND=$ITERATION_ROUND"
echo "CONFIDENCE_THRESHOLD=$CONFIDENCE_THRESHOLD"
echo "MAX_PSEUDO_SAMPLES=$MAX_PSEUDO_SAMPLES"
echo "REMOVE_STRATEGY=$REMOVE_STRATEGY"
echo "GLOBAL_CLUSTER_NUM=$GLOBAL_CLUSTER_NUM"
echo "SHARED_CLUSTER_NUM=$SHARED_CLUSTER_NUM"
echo "RUN_EVAL=$RUN_EVAL"
echo "EVAL_IHC_DATASET=$EVAL_IHC_DATASET"
echo "EVAL_SOURCE_DATASET=$EVAL_SOURCE_DATASET"
echo "EVAL_MODEL_FILENAME=$EVAL_MODEL_FILENAME"
echo "SOURCE_SAVE_DIR=$SOURCE_SAVE_DIR"
echo "SOURCE_DATASET_DIR=$SOURCE_DATASET_DIR"
echo "SOURCE_DATASET_NAME=$SOURCE_DATASET_NAME"
echo "SOURCE_LABEL_FIELD=$SOURCE_LABEL_FIELD"
echo "SOURCE_MIXED_RAW_DATASET_DIR=$SOURCE_MIXED_RAW_DATASET_DIR"
echo "SOURCE_PREDICTION_PREFIX=$SOURCE_PREDICTION_PREFIX"
echo "SOURCE_PREDICTION_DATASET_NAME=$SOURCE_PREDICTION_DATASET_NAME"
echo "SOURCE_CHECKPOINT=$BASE_IHC_CHECKPOINT"
echo "SOURCE_PREDICTIONS=$SOURCE_PREDICTIONS"
echo "PSEUDO_OUTPUT_DIR=$PSEUDO_OUTPUT_DIR"
echo "MIXED_RAW_DATASET=$MIXED_RAW_DATASET"
echo "MIXED_CLUSTERED_DATASET=$MIXED_CLUSTERED_DATASET"
echo "FILTERED_SOURCE_DATASET_DIR=$FILTERED_SOURCE_DATASET_DIR"
echo "MIXED_PREPROCESSED_FILE=$MIXED_PREPROCESSED_FILE"

for required_file in \
  "$BASE_IHC_CHECKPOINT" \
  "$BASE_CLUSTERED_TRAIN" \
  "$BASE_RAW_TRAIN" \
  "$BASE_RAW_VALID" \
  "$BASE_RAW_TEST"
do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

for required_json in "$SOURCE_DATASET_DIR/train.json" "$SOURCE_DATASET_DIR/test.json"
do
  if [[ ! -f "$required_json" ]]; then
    echo "Missing required source dataset file: $required_json" >&2
    exit 1
  fi
done

if [[ -z "$SOURCE_PREDICTIONS" ]]; then
  echo
  echo "[bootstrap] Generate baseline source-train predictions from $SOURCE_DATASET_NAME"
  if [[ ! -f "$SOURCE_POOL_PREPROCESSED" ]]; then
    python prepare_toxicn_eval.py \
      --input_dir "$SOURCE_DATASET_DIR" \
      --output "$SOURCE_POOL_PREPROCESSED" \
      --tokenizer "$TOKENIZER_TYPE" \
      --label-field "$SOURCE_LABEL_FIELD" \
      --local_files_only
  fi

  EVAL_DATASETS="$SOURCE_PREDICTION_DATASET_NAME" \
  EVAL_LOAD_DIR="$SOURCE_SAVE_DIR" \
  EVAL_MODEL_FILENAME="$(basename "$BASE_IHC_CHECKPOINT")" \
  EVAL_TRAIN_ONLY=1 \
  RUN_VERSION="$RUN_VERSION" \
  python eval.py

  if compgen -G "$SOURCE_SAVE_DIR/${SOURCE_PREDICTION_DATASET_NAME}_train_predictions*.csv" > /dev/null; then
    SOURCE_PREDICTIONS="$(ls -1t "$SOURCE_SAVE_DIR"/"${SOURCE_PREDICTION_DATASET_NAME}"_train_predictions*.csv | head -n 1)"
    SOURCE_PREDICTION_PREFIX="$SOURCE_PREDICTION_DATASET_NAME"
  else
    echo "Failed to generate bootstrap source predictions under: $SOURCE_SAVE_DIR" >&2
    exit 1
  fi
fi

echo
echo "[1/6] Select high-confidence pseudo clusters and align to IHC clusters"
python select_pseudo_clusters.py \
  --pseudo_predictions "$SOURCE_PREDICTIONS" \
  --base_clustered_train "$BASE_CLUSTERED_TRAIN" \
  --output_dir "$PSEUDO_OUTPUT_DIR" \
  --confidence_threshold "$CONFIDENCE_THRESHOLD" \
  --cluster_num "$GLOBAL_CLUSTER_NUM" \
  --min_cluster_size "$MIN_CLUSTER_SIZE" \
  --purity_threshold "$PURITY_THRESHOLD" \
  --balanced_margin "$BALANCED_MARGIN" \
  --class_ratio_gap_threshold "$CLASS_RATIO_GAP_THRESHOLD" \
  --load_sent_emb_model "$SENT_EMB_MODEL"

SELECTED_PSEUDO_CSV="$PSEUDO_OUTPUT_DIR/selected_cluster_samples.csv"
INJECTED_PSEUDO_CSV="raw_dataset/$MIXED_RAW_DATASET/selected_pseudo_samples.csv"
if [[ ! -f "$SELECTED_PSEUDO_CSV" ]]; then
  echo "Expected output not found: $SELECTED_PSEUDO_CSV" >&2
  exit 1
fi

echo
echo "[2/6] Replace part of IHC train set with selected pseudo samples"
BUILD_PSEUDO_CMD=(
  python build_pseudo_train_dataset.py
  --ihc_train "$BASE_RAW_TRAIN"
  --ihc_valid "$BASE_RAW_VALID"
  --ihc_test "$BASE_RAW_TEST"
  --pseudo_predictions "$SELECTED_PSEUDO_CSV"
  --confidence_threshold "$CONFIDENCE_THRESHOLD"
  --remove_strategy "$REMOVE_STRATEGY"
  --base_clustered_train "$BASE_CLUSTERED_TRAIN"
  --pseudo_source_name "$SOURCE_DATASET_NAME"
  --output_dir "raw_dataset/$MIXED_RAW_DATASET"
)
if [[ -n "$MAX_PSEUDO_SAMPLES" ]]; then
  BUILD_PSEUDO_CMD+=(--max_pseudo_samples "$MAX_PSEUDO_SAMPLES")
fi
if [[ -n "$SOURCE_MIXED_RAW_DATASET_DIR" ]]; then
  BUILD_PSEUDO_CMD+=(--cumulative)
fi
"${BUILD_PSEUDO_CMD[@]}"
if [[ ! -f "$INJECTED_PSEUDO_CSV" ]]; then
  echo "Expected injected pseudo output not found: $INJECTED_PSEUDO_CSV" >&2
  exit 1
fi

echo
echo "[3/6] Remove selected pseudo rows from future ${SOURCE_DATASET_NAME} train pool"
python prepare_toxicn_eval.py \
  --input_dir "$SOURCE_DATASET_DIR" \
  --exclude_samples "$INJECTED_PSEUDO_CSV" \
  --filtered_output_dir "$FILTERED_SOURCE_DATASET_DIR" \
  --output "$FILTERED_SOURCE_PREPROCESSED" \
  --tokenizer "$TOKENIZER_TYPE" \
  --label-field "$SOURCE_LABEL_FIELD" \
  --local_files_only

echo
echo "[4/6] Build clustered mixed dataset using stored pseudo-to-IHC alignment"
python shared_semantics.py \
  --cluster_num "$SHARED_CLUSTER_NUM" \
  --load_dataset "$MIXED_RAW_DATASET" \
  --load_sent_emb_model "$SENT_EMB_MODEL" \
  --align_pseudo_to_base \
  --base_cluster_dataset "ihc_pure_c10" \
  --output_dataset_name "$MIXED_CLUSTERED_DATASET"

echo
echo "[5/6] Preprocess clustered mixed dataset"
python preprocess_dataset.py \
  -m "$SENT_EMB_MODEL" \
  -d "$MIXED_CLUSTERED_DATASET" \
  -t "$TOKENIZER_TYPE" \
  -o "$MIXED_PREPROCESSED_FILE"

echo
echo "[6/6] Fine-tune from original IHC multilingual checkpoint"
echo "train.py will initialize from: $BASE_IHC_CHECKPOINT"
TRAIN_DATASET="$MIXED_CLUSTERED_DATASET" \
INIT_CHECKPOINT_DIR="$BASE_IHC_CHECKPOINT_DIR" \
INIT_CHECKPOINT_FILENAME="$(basename "$BASE_IHC_CHECKPOINT")" \
RUN_VERSION="$RUN_VERSION" \
python train.py

if [[ "$RUN_EVAL" == "1" ]]; then
  echo
  echo "[extra] Evaluate fine-tuned model on IHC and full ${SOURCE_DATASET_NAME} with eval.py"
  if [[ -z "$EVAL_MODEL_FILENAME" ]]; then
    TODAY_TAG="$(date +%y%m%d)"
    if compgen -G "$FINETUNE_SAVE_DIR/model_${TODAY_TAG}_v*.pt" > /dev/null; then
      LATEST_TODAY_MODEL="$(ls -1t "$FINETUNE_SAVE_DIR"/model_"${TODAY_TAG}"_v*.pt | head -n 1)"
    elif compgen -G "$FINETUNE_SAVE_DIR/model*.pt" > /dev/null; then
      LATEST_TODAY_MODEL="$(ls -1t "$FINETUNE_SAVE_DIR"/model*.pt | head -n 1)"
    else
      LATEST_TODAY_MODEL=""
    fi
    if [[ -z "$LATEST_TODAY_MODEL" ]]; then
      echo "No dated model for today found under $FINETUNE_SAVE_DIR" >&2
      exit 1
    fi
    EVAL_MODEL_FILENAME="$(basename "$LATEST_TODAY_MODEL")"
  fi
  echo "eval.py will use checkpoint: $EVAL_MODEL_FILENAME"

  if [[ "$EVAL_SOURCE_DATASET" == "toxicn" && ! -f "preprocessed_data/preprocessed_toxicn.pkl" ]]; then
    python prepare_toxicn_eval.py \
      --input "$SOURCE_DATASET_DIR/test.json" \
      --output "preprocessed_data/preprocessed_toxicn.pkl" \
      --tokenizer "$TOKENIZER_TYPE" \
      --label-field "$SOURCE_LABEL_FIELD" \
      --local_files_only
  fi

  EVAL_DATASETS="$EVAL_IHC_DATASET,$EVAL_SOURCE_DATASET" \
  EVAL_LOAD_DIR="$FINETUNE_SAVE_DIR" \
  EVAL_MODEL_FILENAME="$EVAL_MODEL_FILENAME" \
  EVAL_TEST_ONLY=1 \
  RUN_VERSION="$RUN_VERSION" \
  python eval.py

  echo
  echo "[extra] Generate next-round pseudo source predictions on filtered ${SOURCE_DATASET_NAME} train pool"
  EVAL_DATASETS="$FILTERED_SOURCE_DATASET_NAME" \
  EVAL_LOAD_DIR="$FINETUNE_SAVE_DIR" \
  EVAL_MODEL_FILENAME="$EVAL_MODEL_FILENAME" \
  EVAL_TRAIN_ONLY=1 \
  RUN_VERSION="$RUN_VERSION" \
  python eval.py
fi

echo
echo "================ PIPELINE DONE ================"
echo "Pseudo selection output: $PSEUDO_OUTPUT_DIR"
echo "Injected pseudo samples: $INJECTED_PSEUDO_CSV"
echo "Source save dir: $SOURCE_SAVE_DIR"
echo "Source dataset dir: $SOURCE_DATASET_DIR"
echo "Mixed raw dataset: raw_dataset/$MIXED_RAW_DATASET"
echo "Filtered source raw dataset: $FILTERED_SOURCE_DATASET_DIR"
echo "Mixed clustered dataset: clustered_dataset/$SENT_EMB_MODEL/$MIXED_CLUSTERED_DATASET"
echo "Preprocessed file: $MIXED_PREPROCESSED_FILE"
echo "Filtered source preprocessed file: $FILTERED_SOURCE_PREPROCESSED"
echo "Fine-tuned save dir: $FINETUNE_SAVE_DIR"
