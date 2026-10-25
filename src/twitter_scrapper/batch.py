from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError

from .config import Settings
from .prompt import SYSTEM_PROMPT, user_prompt
from .schema import ClassificationResult
from .video import process_video

TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}


class BatchWorkflowError(RuntimeError):
    """Raised for invalid job states or malformed API results."""


def prepare_job(video_path: Path, settings: Settings) -> dict[str, Any]:
    now = datetime.now(UTC)
    job_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    job_dir = settings.work_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    artifacts = process_video(video_path, job_dir, settings)
    custom_id = f"video-{uuid.uuid4().hex}"
    request = _batch_request(
        custom_id=custom_id,
        filename=artifacts.source.name,
        duration_seconds=artifacts.duration_seconds,
        sheet_paths=artifacts.sheet_paths,
        model=settings.model,
    )
    request_path = job_dir / "request.jsonl"
    request_path.write_text(
        json.dumps(request, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    request_size = request_path.stat().st_size
    if request_size > settings.max_batch_jsonl_bytes:
        raise BatchWorkflowError(
            f"Prepared JSONL is {request_size / 1024 / 1024:.1f} MB, above the configured "
            f"{settings.max_batch_jsonl_bytes / 1024 / 1024:.0f} MB safety limit. "
            "Shorten the video or reduce contact-sheet image dimensions/quality."
        )

    record: dict[str, Any] = {
        "job_id": job_id,
        "state": "prepared",
        "created_at": now.isoformat(),
        "source_video": str(artifacts.source),
        "filename": artifacts.source.name,
        "duration_seconds": artifacts.duration_seconds,
        "frame_count": len(artifacts.frame_paths),
        "sheet_count": len(artifacts.sheet_paths),
        "model": settings.model,
        "custom_id": custom_id,
        "request_file": str(request_path),
        "request_bytes": request_size,
        "batch_id": None,
        "input_file_id": None,
        "output_file_id": None,
        "error_file_id": None,
    }
    _write_record(job_dir, record)
    _remove_tree(job_dir / "frames")
    _remove_tree(job_dir / "sheets")
    return record


def prepare_many(video_paths: list[Path], settings: Settings) -> dict[str, Any]:
    if not video_paths:
        raise BatchWorkflowError("At least one video is required.")
    now = datetime.now(UTC)
    job_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    job_dir = settings.work_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    request_path = job_dir / "request.jsonl"
    videos: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    for index, video_path in enumerate(video_paths):
        video_dir = job_dir / "videos" / f"{index:04d}"
        video_dir.mkdir(parents=True, exist_ok=False)
        artifacts = process_video(video_path, video_dir, settings)
        custom_id = f"video-{uuid.uuid4().hex}"
        requests.append(
            _batch_request(
                custom_id=custom_id,
                filename=artifacts.source.name,
                duration_seconds=artifacts.duration_seconds,
                sheet_paths=artifacts.sheet_paths,
                model=settings.model,
            )
        )
        videos.append(
            {
                "source_video": str(artifacts.source),
                "filename": artifacts.source.name,
                "duration_seconds": artifacts.duration_seconds,
                "frame_count": len(artifacts.frame_paths),
                "sheet_count": len(artifacts.sheet_paths),
                "custom_id": custom_id,
            }
        )

    request_path.write_text(
        "".join(
            json.dumps(request, separators=(",", ":"), ensure_ascii=False) + "\n"
            for request in requests
        ),
        encoding="utf-8",
    )
    request_size = request_path.stat().st_size
    if request_size > settings.max_batch_jsonl_bytes:
        raise BatchWorkflowError(
            f"Prepared JSONL is {request_size / 1024 / 1024:.1f} MB, above the configured "
            f"{settings.max_batch_jsonl_bytes / 1024 / 1024:.0f} MB safety limit."
        )

    record: dict[str, Any] = {
        "job_id": job_id,
        "state": "prepared",
        "created_at": now.isoformat(),
        "video_count": len(videos),
        "videos": videos,
        "model": settings.model,
        "request_file": str(request_path),
        "request_bytes": request_size,
        "batch_id": None,
        "input_file_id": None,
        "output_file_id": None,
        "error_file_id": None,
    }
    _write_record(job_dir, record)
    _remove_tree(job_dir / "videos")
    return record


def submit_job(job_ref: str, settings: Settings, client: OpenAI | None = None) -> dict[str, Any]:
    job_dir, record = load_job(job_ref, settings)
    if record["state"] != "prepared":
        raise BatchWorkflowError(
            f"Job {record['job_id']} is {record['state']!r}; only prepared jobs can be submitted."
        )
    api = client or OpenAI(api_key=settings.require_api_key())
    with Path(record["request_file"]).open("rb") as request_file:
        uploaded = api.files.create(file=request_file, purpose="batch")
    batch = api.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"local_job_id": record["job_id"]},
    )
    record.update(
        {
            "state": batch.status,
            "submitted_at": datetime.now(UTC).isoformat(),
            "input_file_id": uploaded.id,
            "batch_id": batch.id,
            "request_uploaded_and_removed": True,
        }
    )
    _write_record(job_dir, record)
    Path(record["request_file"]).unlink(missing_ok=True)
    return record


def refresh_job(job_ref: str, settings: Settings, client: OpenAI | None = None) -> dict[str, Any]:
    job_dir, record = load_job(job_ref, settings)
    if not record.get("batch_id"):
        return record
    api = client or OpenAI(api_key=settings.require_api_key())
    batch = api.batches.retrieve(record["batch_id"])
    record.update(
        {
            "state": batch.status,
            "last_checked_at": datetime.now(UTC).isoformat(),
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "request_counts": _model_to_dict(batch.request_counts),
            "batch_errors": _model_to_dict(batch.errors),
        }
    )
    _write_record(job_dir, record)
    return record


def wait_for_job(
    job_ref: str,
    settings: Settings,
    poll_seconds: int = 30,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    if poll_seconds <= 0:
        raise BatchWorkflowError("poll_seconds must be greater than zero.")
    api = client or OpenAI(api_key=settings.require_api_key())
    while True:
        record = refresh_job(job_ref, settings, api)
        if record["state"] in TERMINAL_BATCH_STATES:
            return record
        time.sleep(poll_seconds)


def collect_job(job_ref: str, settings: Settings, client: OpenAI | None = None) -> Path:
    job_dir, record = load_job(job_ref, settings)
    api = client or OpenAI(api_key=settings.require_api_key())
    record = refresh_job(job_ref, settings, api)
    if record["state"] != "completed":
        details = f" Error file: {record['error_file_id']}." if record.get("error_file_id") else ""
        raise BatchWorkflowError(
            f"Job is {record['state']!r}, not 'completed'.{details} Run status and retry later."
        )
    if not record.get("output_file_id"):
        raise BatchWorkflowError("Completed batch has no output_file_id.")

    output_jsonl = job_dir / "output.jsonl"
    _download_file(api, record["output_file_id"], output_jsonl)
    if record.get("videos"):
        results = [
            parse_batch_output(output_jsonl, expected_custom_id=video["custom_id"])
            for video in record["videos"]
        ]
        result_payload: Any = _map_and_organize_results(
            results,
            [video["source_video"] for video in record["videos"]],
            record["job_id"],
            settings,
        )
        result_path = job_dir / "results.json"
    else:
        result = parse_batch_output(output_jsonl, expected_custom_id=record["custom_id"])
        result_payload = _map_and_organize_results(
            [result], [record["source_video"]], record["job_id"], settings
        )[0]
        result_path = job_dir / "result.json"
    result_path.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if record.get("error_file_id"):
        _download_file(api, record["error_file_id"], job_dir / "errors.jsonl")
    Path(record["request_file"]).unlink(missing_ok=True)
    output_jsonl.unlink(missing_ok=True)
    record.update(
        {
            "state": "collected",
            "collected_at": datetime.now(UTC).isoformat(),
            "result_file": str(result_path),
            "intermediate_artifacts_removed": True,
        }
    )
    _write_record(job_dir, record)
    return result_path


def load_job(job_ref: str, settings: Settings) -> tuple[Path, dict[str, Any]]:
    possible_path = Path(job_ref).expanduser()
    job_dir = possible_path if possible_path.is_dir() else settings.work_dir / "jobs" / job_ref
    record_path = job_dir.resolve() / "job.json"
    if not record_path.is_file():
        raise BatchWorkflowError(f"Job not found: {job_ref}")
    try:
        return record_path.parent, json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchWorkflowError(f"Job record is invalid JSON: {record_path}") from exc


def cleanup_job_artifacts(job_ref: str, settings: Settings) -> list[str]:
    job_dir, record = load_job(job_ref, settings)
    removed: list[str] = []
    for directory_name in ("frames", "sheets", "videos"):
        path = job_dir / directory_name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
    for file_name in ("request.jsonl", "output.jsonl"):
        path = job_dir / file_name
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    record["intermediate_artifacts_removed"] = True
    _write_record(job_dir, record)
    return removed


async def run_sync_job(
    job_ref: str,
    settings: Settings,
    concurrency: int = 3,
    client: AsyncOpenAI | None = None,
) -> Path:
    if concurrency <= 0:
        raise BatchWorkflowError("concurrency must be greater than zero.")
    job_dir, record = load_job(job_ref, settings)
    if record["state"] != "prepared":
        raise BatchWorkflowError(
            f"Job {record['job_id']} is {record['state']!r}; expected 'prepared'."
        )
    request_path = Path(record["request_file"])
    requests = [
        json.loads(line)
        for line in request_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(requests) != len(record.get("videos", [])):
        raise BatchWorkflowError("Prepared requests do not match the job's videos.")

    api = client or AsyncOpenAI(api_key=settings.require_api_key())
    semaphore = asyncio.Semaphore(concurrency)

    async def classify(request: dict[str, Any]) -> ClassificationResult:
        async with semaphore:
            for attempt in range(3):
                try:
                    response = await api.responses.create(**request["body"])
                    if not response.output_text:
                        raise BatchWorkflowError(
                            f"OpenAI returned no output text for {request['custom_id']}."
                        )
                    return ClassificationResult.model_validate_json(response.output_text)
                except (APIConnectionError, APITimeoutError, RateLimitError):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2**attempt)
            raise AssertionError("unreachable")

    record.update(
        {
            "state": "sync_in_progress",
            "sync_started_at": datetime.now(UTC).isoformat(),
            "sync_concurrency": concurrency,
        }
    )
    _write_record(job_dir, record)
    try:
        results = await asyncio.gather(*(classify(request) for request in requests))
    except Exception:
        record["state"] = "sync_failed"
        record["sync_failed_at"] = datetime.now(UTC).isoformat()
        _write_record(job_dir, record)
        raise

    mapped = _map_and_organize_results(
        results,
        [video["source_video"] for video in record["videos"]],
        record["job_id"],
        settings,
    )
    result_path = job_dir / "results.json"
    result_path.write_text(
        json.dumps(mapped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    request_path.unlink(missing_ok=True)
    record.update(
        {
            "state": "collected_sync",
            "sync_completed_at": datetime.now(UTC).isoformat(),
            "result_file": str(result_path),
            "intermediate_artifacts_removed": True,
        }
    )
    _write_record(job_dir, record)
    return result_path


def parse_batch_output(path: Path, expected_custom_id: str) -> ClassificationResult:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise BatchWorkflowError("Batch output is empty.")
    for raw_line in lines:
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise BatchWorkflowError("Batch output contains invalid JSONL.") from exc
        if item.get("custom_id") != expected_custom_id:
            continue
        if item.get("error"):
            raise BatchWorkflowError(f"Batch request failed: {item['error']}")
        response = item.get("response") or {}
        if response.get("status_code") != 200:
            raise BatchWorkflowError(
                f"Batch request returned HTTP {response.get('status_code')}: {response.get('body')}"
            )
        text = _extract_output_text(response.get("body") or {})
        try:
            return ClassificationResult.model_validate_json(text)
        except (ValueError, TypeError) as exc:
            raise BatchWorkflowError(f"Model returned an invalid classification: {exc}") from exc
    raise BatchWorkflowError(f"No result found for custom_id {expected_custom_id!r}.")


def _batch_request(
    custom_id: str,
    filename: str,
    duration_seconds: float,
    sheet_paths: tuple[Path, ...],
    model: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": user_prompt(filename, duration_seconds, len(sheet_paths)),
        }
    ]
    for sheet_path in sheet_paths:
        encoded = base64.b64encode(sheet_path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded}",
                "detail": "high",
            }
        )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "launch_video_classification",
                    "strict": True,
                    "schema": ClassificationResult.model_json_schema(),
                }
            },
        },
    }


def _extract_output_text(body: dict[str, Any]) -> str:
    for output in body.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
            if content.get("type") == "refusal":
                raise BatchWorkflowError(
                    f"Model refused the classification: {content.get('refusal')}"
                )
    raise BatchWorkflowError("OpenAI response did not contain output_text.")


def _download_file(client: OpenAI, file_id: str, destination: Path) -> None:
    response = client.files.content(file_id)
    if hasattr(response, "write_to_file"):
        response.write_to_file(destination)
        return
    data = response.read() if hasattr(response, "read") else response.content
    if isinstance(data, str):
        data = data.encode()
    destination.write_bytes(data)


def _write_record(job_dir: Path, record: dict[str, Any]) -> None:
    destination = job_dir / "job.json"
    temporary = job_dir / "job.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _model_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _map_and_organize_results(
    results: list[ClassificationResult],
    source_videos: list[str],
    job_id: str,
    settings: Settings,
) -> list[dict[str, Any]]:
    useful_dir = settings.useful_video_dir or (
        settings.work_dir.parent / "data" / "output" / "useful-videos"
    )
    useful_dir.mkdir(parents=True, exist_ok=True)
    mapped: list[dict[str, Any]] = []

    for result, source_value in zip(results, source_videos, strict=True):
        source = Path(source_value).resolve()
        destination = useful_dir / source.name
        useful_path: str | None = None
        if result.eligible_for_collection:
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            useful_path = str(destination.resolve())
        elif destination.is_file():
            destination.unlink()
        mapped.append(
            {
                "source_video": str(source),
                "useful_video": useful_path,
                "classification": result.model_dump(mode="json"),
            }
        )

    catalog_path = useful_dir / f"{job_id}-results.json"
    catalog_path.write_text(
        json.dumps(mapped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return mapped


def restore_useful_videos(result_path: Path, settings: Settings) -> dict[str, int]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else [payload]
    useful_dir = settings.useful_video_dir or (
        settings.work_dir.parent / "data" / "output" / "useful-videos"
    )
    useful_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    already_present = 0
    missing_sources = 0

    for item in items:
        classification = item.get("classification", {})
        if not classification.get("eligible_for_collection"):
            continue
        source = Path(item["source_video"]).resolve()
        if not source.is_file():
            missing_sources += 1
            continue
        destination = useful_dir / source.name
        if destination.is_file():
            already_present += 1
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        restored += 1

    return {
        "restored": restored,
        "already_present": already_present,
        "missing_sources": missing_sources,
    }


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
