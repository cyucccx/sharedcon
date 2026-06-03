import argparse
import csv
import http.client
import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd

from util import ensure_output_dir_is_new, ensure_output_path_is_new


API_HOST = "api.openai.com"
DEFAULT_MODEL = "gpt-4.1-mini"
PREDICTION_COLUMNS = [
    "row_id",
    "post",
    "true_label",
    "pred_label",
    "confidence",
    "prob_0",
    "prob_1",
]

MODEL_OUTPUT_COLUMNS = [
    "row_id",
    "true_label",
    "pred_label",
    "confidence",
    "prob_0",
    "prob_1",
]


SYSTEM_PROMPT = """You are an expert annotator for Chinese hate speech detection.

Classify one Chinese social media comment into an IHC-style binary label.

Labels:
0 = not_hate
1 = hate

Hate includes explicit hate speech and implicit hate speech toward identity-based groups, including LGBTQ, region/place of origin, sexism/gender, race/ethnicity/nationality, and other identity groups.

Do not label as hate if the comment is only general offensive language, general profanity, or a personal insult without an identity-based target.
Do not label as hate if the comment merely reports or quotes hate speech without endorsing it.

Return only one JSON object with exactly these keys:
row_id, true_label, pred_label, confidence, prob_0, prob_1

Rules:
- Keep row_id exactly the same as input.
- Keep true_label exactly the same as input.
- pred_label must be 0 or 1.
- confidence must be max(prob_0, prob_1).
- prob_0 and prob_1 must be numbers between 0 and 1.
- prob_0 + prob_1 must equal 1.0, allowing minor rounding.
- If pred_label is 0, prob_0 must be greater than or equal to prob_1.
- If pred_label is 1, prob_1 must be greater than or equal to prob_0.
"""


JSON_SCHEMA = {
    "name": "hate_prediction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "row_id": {"type": "integer"},
            "true_label": {"type": "integer", "enum": [0, 1]},
            "pred_label": {"type": "integer", "enum": [0, 1]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "prob_0": {"type": "number", "minimum": 0, "maximum": 1},
            "prob_1": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": MODEL_OUTPUT_COLUMNS,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenAI Batch API predictions for ToxiCN_filtered and convert results "
            "to the same CSV schema as eval.py prediction outputs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create OpenAI Batch JSONL input.")
    create_parser.add_argument("--input", default="raw_dataset/ToxiCN_filtered/train.json")
    create_parser.add_argument(
        "--input_csv",
        default=None,
        help="Optional CSV with row_id,post,true_label columns, useful for retrying missing rows.",
    )
    create_parser.add_argument("--output_dir", default="openai_batch/toxicn_filtered_train")
    create_parser.add_argument("--model", default=DEFAULT_MODEL)
    create_parser.add_argument(
        "--api",
        choices=["chat", "responses"],
        default="responses",
        help="OpenAI API endpoint to target. Use responses for GPT-5.5.",
    )
    create_parser.add_argument(
        "--reasoning_effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="none",
        help="Responses API reasoning effort. For this classification task, none is the cheapest starting point.",
    )
    create_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a smoke test before submitting the full train set.",
    )
    create_parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help=(
            "Rows per batch input shard. Use a smaller value if your organization "
            "hits the Batch API enqueued token limit."
        ),
    )

    submit_parser = subparsers.add_parser("submit", help="Upload JSONL and create a batch job.")
    submit_parser.add_argument("--input_jsonl", default="openai_batch/toxicn_filtered_train/batch_input.jsonl")
    submit_parser.add_argument("--output_dir", default="openai_batch/toxicn_filtered_train")

    status_parser = subparsers.add_parser("status", help="Fetch batch status.")
    status_parser.add_argument("--batch_id", default=None)
    status_parser.add_argument("--batch_json", default="openai_batch/toxicn_filtered_train/batch_job.json")

    download_parser = subparsers.add_parser("download", help="Download completed batch output and convert to CSV.")
    download_parser.add_argument("--batch_id", default=None)
    download_parser.add_argument("--batch_json", default="openai_batch/toxicn_filtered_train/batch_job.json")
    download_parser.add_argument("--manifest", default="openai_batch/toxicn_filtered_train/manifest.csv")
    download_parser.add_argument("--output_jsonl", default="openai_batch/toxicn_filtered_train/batch_output.jsonl")
    download_parser.add_argument("--output_csv", default="openai_batch/toxicn_filtered_train/llm_predictions.csv")
    download_parser.add_argument("--missing_csv", default="openai_batch/toxicn_filtered_train/missing_rows.csv")
    download_parser.add_argument("--error_jsonl", default="openai_batch/toxicn_filtered_train/batch_errors.jsonl")

    run_parser = subparsers.add_parser(
        "run-shards",
        help="Sequentially submit, wait for, download, and merge all shard batch jobs.",
    )
    run_parser.add_argument("--batch_dir", default="openai_batch/toxicn_filtered_train_gpt55_sharded_500")
    run_parser.add_argument("--start_shard", type=int, default=0)
    run_parser.add_argument(
        "--end_shard",
        type=int,
        default=None,
        help="Exclusive end shard index. Defaults to all shards in shards_manifest.csv.",
    )
    run_parser.add_argument("--poll_seconds", type=int, default=60)
    run_parser.add_argument(
        "--merged_csv",
        default=None,
        help="Final merged prediction CSV. Defaults to <batch_dir>/llm_predictions_merged.csv.",
    )
    run_parser.add_argument(
        "--missing_csv",
        default=None,
        help="Final missing-row CSV. Defaults to <batch_dir>/missing_rows_merged.csv.",
    )
    run_parser.add_argument(
        "--max_status_errors",
        type=int,
        default=5,
        help="Maximum transient status/download errors before stopping one shard.",
    )

    merge_parser = subparsers.add_parser(
        "merge-shards",
        help="Merge already-downloaded shard prediction CSV files without calling the API.",
    )
    merge_parser.add_argument("--batch_dir", default="openai_batch/toxicn_filtered_train_gpt55_sharded_500")
    merge_parser.add_argument(
        "--merged_csv",
        default=None,
        help="Final merged prediction CSV. Defaults to <batch_dir>/llm_predictions_merged.csv.",
    )
    merge_parser.add_argument(
        "--missing_csv",
        default=None,
        help="Final missing-row CSV. Defaults to <batch_dir>/missing_rows_merged.csv.",
    )
    return parser.parse_args()


def get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return api_key


def request_json(method, path, body=None):
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"

    conn = http.client.HTTPSConnection(API_HOST, timeout=120)
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    content = response.read()
    conn.close()

    if response.status >= 400:
        raise RuntimeError(f"{method} {path} failed: HTTP {response.status}: {content.decode('utf-8', errors='replace')}")
    if not content:
        return {}
    return json.loads(content.decode("utf-8"))


def request_bytes(method, path):
    conn = http.client.HTTPSConnection(API_HOST, timeout=120)
    conn.request(method, path, headers={"Authorization": f"Bearer {get_api_key()}"})
    response = conn.getresponse()
    content = response.read()
    conn.close()
    if response.status >= 400:
        raise RuntimeError(f"{method} {path} failed: HTTP {response.status}: {content.decode('utf-8', errors='replace')}")
    return content


def upload_file(path):
    boundary = f"----codex-openai-batch-{uuid.uuid4().hex}"
    file_bytes = Path(path).read_bytes()
    file_name = Path(path).name
    parts = [
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        "batch\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
        "Content-Type: application/jsonl\r\n\r\n",
    ]
    body = b"".join(part.encode("utf-8") for part in parts)
    body += file_bytes
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    conn = http.client.HTTPSConnection(API_HOST, timeout=120)
    conn.request("POST", "/v1/files", body=body, headers=headers)
    response = conn.getresponse()
    content = response.read()
    conn.close()
    if response.status >= 400:
        raise RuntimeError(f"File upload failed: HTTP {response.status}: {content.decode('utf-8', errors='replace')}")
    return json.loads(content.decode("utf-8"))


def load_toxicn_rows(path, limit=None):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if limit is not None:
        records = records[:limit]

    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "row_id": int(record.get("source_row_id", index)),
                "post": str(record["content"]),
                "true_label": int(record["toxic"]),
            }
        )
    return pd.DataFrame(rows)


def load_input_rows(json_path, csv_path=None, limit=None):
    if csv_path:
        rows = pd.read_csv(csv_path)
        required = {"row_id", "post", "true_label"}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"--input_csv is missing columns: {sorted(missing)}")
        rows = rows[["row_id", "post", "true_label"]].copy()
        rows["row_id"] = rows["row_id"].astype(int)
        rows["post"] = rows["post"].astype(str)
        rows["true_label"] = rows["true_label"].astype(int)
        if limit is not None:
            rows = rows.head(limit)
        return rows.reset_index(drop=True)
    return load_toxicn_rows(json_path, limit=limit)


def build_user_prompt(row):
    payload = {
        "row_id": int(row["row_id"]),
        "post": row["post"],
        "true_label": int(row["true_label"]),
    }
    return "Classify this row and return the required JSON object:\n" + json.dumps(payload, ensure_ascii=False)


def build_chat_batch_request(row, model):
    return {
        "custom_id": f"row-{int(row['row_id'])}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row)},
            ],
            "temperature": 0,
            "max_tokens": 160,
            "response_format": {
                "type": "json_schema",
                "json_schema": JSON_SCHEMA,
            },
        },
    }


def build_responses_batch_request(row, model, reasoning_effort):
    return {
        "custom_id": f"row-{int(row['row_id'])}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row)},
            ],
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": 160,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": JSON_SCHEMA["name"],
                    "strict": JSON_SCHEMA["strict"],
                    "schema": JSON_SCHEMA["schema"],
                }
            },
        },
    }


def build_batch_request(row, model, api, reasoning_effort):
    if api == "chat":
        return build_chat_batch_request(row, model)
    if api == "responses":
        return build_responses_batch_request(row, model, reasoning_effort)
    raise NotImplementedError(api)


def create_batch_input(args):
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be positive.")

    ensure_output_dir_is_new(args.output_dir, label="OpenAI batch output directory")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=False)

    rows = load_input_rows(args.input, csv_path=args.input_csv, limit=args.limit)
    manifest_path = output_dir / "manifest.csv"
    metadata_path = output_dir / "batch_metadata.json"

    rows.to_csv(manifest_path, index=False)

    shard_rows = []
    for shard_index, start in enumerate(range(0, len(rows), args.shard_size)):
        shard = rows.iloc[start : start + args.shard_size].copy()
        shard_path = shards_dir / f"batch_input_{shard_index:04d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for _, row in shard.iterrows():
                f.write(
                    json.dumps(
                        build_batch_request(row, args.model, args.api, args.reasoning_effort),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        shard_rows.append(
            {
                "shard_index": shard_index,
                "input_jsonl": str(shard_path),
                "rows": int(len(shard)),
                "start_position": int(start),
                "end_position_exclusive": int(start + len(shard)),
            }
        )

    shards_manifest_path = output_dir / "shards_manifest.csv"
    pd.DataFrame(shard_rows).to_csv(shards_manifest_path, index=False)

    metadata = {
        "input": args.input,
        "input_csv": args.input_csv,
        "model": args.model,
        "api": args.api,
        "endpoint": "/v1/responses" if args.api == "responses" else "/v1/chat/completions",
        "reasoning_effort": args.reasoning_effort if args.api == "responses" else None,
        "rows": int(len(rows)),
        "shard_size": int(args.shard_size),
        "shard_count": int(len(shard_rows)),
        "shards_manifest": str(shards_manifest_path),
        "manifest": str(manifest_path),
        "created_at": int(time.time()),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Created {len(shard_rows)} batch input shards under: {shards_dir}")
    print(f"Rows: {len(rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"Shards manifest: {shards_manifest_path}")


def submit_batch(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_json_path = output_dir / "batch_job.json"
    upload_json_path = output_dir / "uploaded_file.json"
    ensure_output_path_is_new(batch_json_path, label="batch job metadata")
    ensure_output_path_is_new(upload_json_path, label="uploaded file metadata")

    uploaded = upload_file(args.input_jsonl)
    with open(upload_json_path, "w", encoding="utf-8") as f:
        json.dump(uploaded, f, ensure_ascii=False, indent=2)

    with open(args.input_jsonl, encoding="utf-8") as f:
        first_request = json.loads(f.readline())
    endpoint = first_request["url"]

    batch = request_json(
        "POST",
        "/v1/batches",
        {
            "input_file_id": uploaded["id"],
            "endpoint": endpoint,
            "completion_window": "24h",
        },
    )
    with open(batch_json_path, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)

    print(f"Uploaded file: {uploaded['id']}")
    print(f"Created batch: {batch['id']}")
    print(f"Endpoint: {endpoint}")
    print(f"Status: {batch.get('status')}")
    print(f"Saved batch metadata: {batch_json_path}")


def resolve_batch_id(batch_id=None, batch_json=None):
    if batch_id:
        return batch_id
    if not batch_json:
        raise ValueError("Provide --batch_id or --batch_json.")
    with open(batch_json, encoding="utf-8") as f:
        return json.load(f)["id"]


def get_batch(batch_id):
    return request_json("GET", f"/v1/batches/{batch_id}")


def print_status(args):
    batch = get_batch(resolve_batch_id(args.batch_id, args.batch_json))
    print(json.dumps(batch, ensure_ascii=False, indent=2))


def extract_prediction_from_response(line):
    custom_id = line.get("custom_id")
    if line.get("error"):
        return None, {"custom_id": custom_id, "error": line["error"]}

    response = line.get("response") or {}
    body = response.get("body") or {}
    content = None

    choices = body.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content")
    elif body.get("output_text"):
        content = body.get("output_text")
    else:
        for output_item in body.get("output", []) or []:
            if output_item.get("type") != "message":
                continue
            for content_item in output_item.get("content", []) or []:
                if content_item.get("type") in {"output_text", "text"}:
                    content = content_item.get("text")
                    break
            if content:
                break

    if not content:
        return None, {"custom_id": custom_id, "error": "missing message content"}
    try:
        record = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, {"custom_id": custom_id, "error": f"invalid JSON content: {exc}", "content": content}

    return record, None


def normalize_predictions(records):
    frame = pd.DataFrame(records)
    missing_columns = set(MODEL_OUTPUT_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Prediction records are missing columns: {sorted(missing_columns)}")

    frame = frame[MODEL_OUTPUT_COLUMNS].copy()
    frame["row_id"] = frame["row_id"].astype(int)
    frame["true_label"] = frame["true_label"].astype(int)
    frame["pred_label"] = frame["pred_label"].astype(int)
    frame["prob_0"] = frame["prob_0"].astype(float)
    frame["prob_1"] = frame["prob_1"].astype(float)
    frame["confidence"] = frame[["prob_0", "prob_1"]].max(axis=1)

    invalid_labels = frame.loc[~frame["pred_label"].isin([0, 1])]
    if len(invalid_labels):
        raise ValueError(f"Invalid pred_label values: {invalid_labels.head(5).to_dict(orient='records')}")
    invalid_probs = frame.loc[
        frame["prob_0"].lt(0)
        | frame["prob_0"].gt(1)
        | frame["prob_1"].lt(0)
        | frame["prob_1"].gt(1)
    ]
    if len(invalid_probs):
        raise ValueError(f"Probabilities outside [0, 1]: {invalid_probs.head(5).to_dict(orient='records')}")

    return frame.sort_values("row_id").reset_index(drop=True)


def attach_manifest_fields(manifest, predictions):
    merged = manifest.merge(predictions, on="row_id", how="left", suffixes=("_expected", ""))
    post_column = "post_expected" if "post_expected" in merged.columns else "post"
    true_label_column = (
        "true_label_expected"
        if "true_label_expected" in merged.columns
        else "true_label"
    )
    if post_column not in merged.columns or true_label_column not in merged.columns:
        raise KeyError(
            "Could not find manifest post/true_label columns after merge. "
            f"Columns: {merged.columns.tolist()}"
        )
    return merged, post_column, true_label_column


def download_and_convert(args):
    batch = get_batch(resolve_batch_id(args.batch_id, args.batch_json))
    if batch.get("status") != "completed":
        raise RuntimeError(f"Batch is not completed. Current status: {batch.get('status')}")
    if not batch.get("output_file_id"):
        raise RuntimeError("Completed batch has no output_file_id.")

    ensure_output_path_is_new(args.output_jsonl, label="batch output JSONL")
    ensure_output_path_is_new(args.output_csv, label="prediction CSV")
    ensure_output_path_is_new(args.missing_csv, label="missing rows CSV")

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    output_bytes = request_bytes("GET", f"/v1/files/{batch['output_file_id']}/content")
    Path(args.output_jsonl).write_bytes(output_bytes)

    if batch.get("error_file_id"):
        if not os.path.exists(args.error_jsonl):
            error_bytes = request_bytes("GET", f"/v1/files/{batch['error_file_id']}/content")
            Path(args.error_jsonl).write_bytes(error_bytes)

    records = []
    parse_errors = []
    with open(args.output_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record, error = extract_prediction_from_response(json.loads(line))
            if error:
                parse_errors.append(error)
            else:
                records.append(record)

    predictions = normalize_predictions(records)
    manifest = pd.read_csv(args.manifest)
    merged, post_column, true_label_column = attach_manifest_fields(manifest, predictions)
    missing = merged.loc[merged["pred_label"].isna(), ["row_id", post_column, true_label_column]].copy()
    missing = missing.rename(columns={post_column: "post", true_label_column: "true_label"})

    completed = merged.loc[merged["pred_label"].notna()].copy()
    completed["post"] = completed[post_column]
    completed["true_label"] = completed[true_label_column].astype(int)
    completed = completed[PREDICTION_COLUMNS].sort_values("row_id").reset_index(drop=True)

    completed.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    missing.to_csv(args.missing_csv, index=False)

    if parse_errors:
        parse_error_path = Path(args.output_csv).with_name("parse_errors.json")
        with open(parse_error_path, "w", encoding="utf-8") as f:
            json.dump(parse_errors, f, ensure_ascii=False, indent=2)
        print(f"Parse errors: {len(parse_errors)} saved to {parse_error_path}")

    print(f"Downloaded JSONL: {args.output_jsonl}")
    print(f"Prediction CSV:   {args.output_csv}")
    print(f"Rows predicted:   {len(completed)}")
    print(f"Missing rows:     {len(missing)}")
    print(f"Missing CSV:      {args.missing_csv}")


def parse_batch_output_jsonl(path):
    records = []
    parse_errors = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record, error = extract_prediction_from_response(json.loads(line))
            if error:
                parse_errors.append(error)
            else:
                records.append(record)
    return records, parse_errors


def write_job_predictions(batch, job_dir):
    output_jsonl = job_dir / "batch_output.jsonl"
    output_csv = job_dir / "llm_predictions.csv"
    parse_errors_path = job_dir / "parse_errors.json"

    if not output_jsonl.exists():
        output_bytes = request_bytes("GET", f"/v1/files/{batch['output_file_id']}/content")
        output_jsonl.write_bytes(output_bytes)

    if batch.get("error_file_id"):
        error_jsonl = job_dir / "batch_errors.jsonl"
        if not error_jsonl.exists():
            error_bytes = request_bytes("GET", f"/v1/files/{batch['error_file_id']}/content")
            error_jsonl.write_bytes(error_bytes)

    if output_csv.exists():
        return pd.read_csv(output_csv)

    records, parse_errors = parse_batch_output_jsonl(output_jsonl)
    predictions = normalize_predictions(records)
    predictions.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    if parse_errors:
        with open(parse_errors_path, "w", encoding="utf-8") as f:
            json.dump(parse_errors, f, ensure_ascii=False, indent=2)

    return predictions


def submit_shard(input_jsonl, job_dir):
    job_dir.mkdir(parents=True, exist_ok=True)
    batch_json_path = job_dir / "batch_job.json"
    uploaded_json_path = job_dir / "uploaded_file.json"

    if batch_json_path.exists():
        with open(batch_json_path, encoding="utf-8") as f:
            return json.load(f)

    uploaded = upload_file(input_jsonl)
    with open(uploaded_json_path, "w", encoding="utf-8") as f:
        json.dump(uploaded, f, ensure_ascii=False, indent=2)

    with open(input_jsonl, encoding="utf-8") as f:
        first_request = json.loads(f.readline())
    endpoint = first_request["url"]

    batch = request_json(
        "POST",
        "/v1/batches",
        {
            "input_file_id": uploaded["id"],
            "endpoint": endpoint,
            "completion_window": "24h",
        },
    )
    with open(batch_json_path, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    return batch


def refresh_job_metadata(job_dir, batch):
    batch_json_path = job_dir / "batch_job.json"
    with open(batch_json_path, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)


def wait_for_batch(job_dir, initial_batch, poll_seconds, max_status_errors):
    batch = initial_batch
    status_errors = 0

    while True:
        status = batch.get("status")
        print(
            f"Batch {batch['id']} status={status} "
            f"completed={batch.get('request_counts', {}).get('completed')} "
            f"failed={batch.get('request_counts', {}).get('failed')}",
            flush=True,
        )

        if status == "completed":
            refresh_job_metadata(job_dir, batch)
            return batch
        if status in {"failed", "expired", "cancelled"}:
            refresh_job_metadata(job_dir, batch)
            raise RuntimeError(f"Batch {batch['id']} ended with status={status}: {batch.get('errors')}")

        time.sleep(poll_seconds)
        try:
            batch = get_batch(batch["id"])
            refresh_job_metadata(job_dir, batch)
            status_errors = 0
        except Exception as exc:
            status_errors += 1
            print(f"Transient status error {status_errors}/{max_status_errors}: {exc}", flush=True)
            if status_errors >= max_status_errors:
                raise


def merge_job_predictions(batch_dir, merged_csv, missing_csv):
    manifest_path = batch_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path)

    frames = []
    for csv_path in sorted(batch_dir.glob("job_*/llm_predictions.csv")):
        frame = pd.read_csv(csv_path)
        frame["source_job"] = csv_path.parent.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No job prediction CSV files found under {batch_dir}/job_*")

    predictions = pd.concat(frames, ignore_index=True)
    duplicate_mask = predictions["row_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = predictions.loc[duplicate_mask, "row_id"].head(20).tolist()
        raise ValueError(f"Duplicate row_id values across shard outputs. Examples: {duplicate_ids}")

    predictions = predictions[MODEL_OUTPUT_COLUMNS].sort_values("row_id").reset_index(drop=True)
    merged, post_column, true_label_column = attach_manifest_fields(manifest, predictions)
    missing = merged.loc[merged["pred_label"].isna(), ["row_id", post_column, true_label_column]].copy()
    missing = missing.rename(columns={post_column: "post", true_label_column: "true_label"})

    completed = merged.loc[merged["pred_label"].notna()].copy()
    completed["post"] = completed[post_column]
    completed["true_label"] = completed[true_label_column].astype(int)
    completed = completed[PREDICTION_COLUMNS].sort_values("row_id").reset_index(drop=True)

    completed.to_csv(merged_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    missing.to_csv(missing_csv, index=False)

    print(f"Merged predictions: {len(completed)}")
    print(f"Missing rows:       {len(missing)}")
    print(f"Merged CSV:         {merged_csv}")
    print(f"Missing CSV:        {missing_csv}")


def run_shards(args):
    batch_dir = Path(args.batch_dir)
    shards_manifest_path = batch_dir / "shards_manifest.csv"
    if not shards_manifest_path.is_file():
        raise FileNotFoundError(f"Shards manifest not found: {shards_manifest_path}")

    shards = pd.read_csv(shards_manifest_path)
    if args.end_shard is None:
        end_shard = int(shards["shard_index"].max()) + 1
    else:
        end_shard = args.end_shard

    selected_shards = shards.loc[
        shards["shard_index"].ge(args.start_shard)
        & shards["shard_index"].lt(end_shard)
    ].copy()
    if len(selected_shards) == 0:
        raise ValueError(f"No shards selected for range [{args.start_shard}, {end_shard}).")

    for _, shard in selected_shards.sort_values("shard_index").iterrows():
        shard_index = int(shard["shard_index"])
        input_jsonl = str(shard["input_jsonl"])
        job_dir = batch_dir / f"job_{shard_index:04d}"

        print(f"===== shard {shard_index:04d} rows={int(shard['rows'])} =====", flush=True)
        batch = submit_shard(input_jsonl, job_dir)
        batch = wait_for_batch(job_dir, batch, args.poll_seconds, args.max_status_errors)
        predictions = write_job_predictions(batch, job_dir)
        print(f"Shard {shard_index:04d} predictions: {len(predictions)} saved under {job_dir}", flush=True)

    merged_csv = Path(args.merged_csv) if args.merged_csv else batch_dir / "llm_predictions_merged.csv"
    missing_csv = Path(args.missing_csv) if args.missing_csv else batch_dir / "missing_rows_merged.csv"
    merge_job_predictions(batch_dir, merged_csv, missing_csv)


def main():
    args = parse_args()
    if args.command == "create":
        create_batch_input(args)
    elif args.command == "submit":
        submit_batch(args)
    elif args.command == "status":
        print_status(args)
    elif args.command == "download":
        download_and_convert(args)
    elif args.command == "run-shards":
        run_shards(args)
    elif args.command == "merge-shards":
        batch_dir = Path(args.batch_dir)
        merged_csv = Path(args.merged_csv) if args.merged_csv else batch_dir / "llm_predictions_merged.csv"
        missing_csv = Path(args.missing_csv) if args.missing_csv else batch_dir / "missing_rows_merged.csv"
        merge_job_predictions(batch_dir, merged_csv, missing_csv)
    else:
        raise NotImplementedError(args.command)


if __name__ == "__main__":
    main()
