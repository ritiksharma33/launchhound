from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

from openai import OpenAIError

from .batch import (
    BatchWorkflowError,
    cleanup_job_artifacts,
    collect_job,
    prepare_job,
    prepare_many,
    refresh_job,
    restore_useful_videos,
    run_sync_job,
    submit_job,
    wait_for_job,
)
from .config import ConfigurationError, Settings
from .video import VideoProcessingError
from .x_ingest import (
    XIngestError,
    add_cookie_account,
    clear_downloaded_videos,
    downloaded_video_status,
    fetch_account_videos,
    fetch_multiple_accounts,
    list_accounts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launch-classifier",
        description="Classify local product-launch videos through OpenAI Batch.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Extract frames and create Batch JSONL.")
    prepare.add_argument("video", type=Path)

    prepare_many_parser = subparsers.add_parser(
        "prepare-many", help="Create one Batch job containing multiple videos."
    )
    prepare_many_parser.add_argument("videos", type=Path, nargs="+")

    prepare_manifest = subparsers.add_parser(
        "prepare-manifest", help="Create one Batch job from an X video manifest."
    )
    prepare_manifest.add_argument("manifest", type=Path)

    submit = subparsers.add_parser("submit", help="Upload and submit a prepared job.")
    submit.add_argument("job")

    status = subparsers.add_parser("status", help="Refresh and print a job's status.")
    status.add_argument("job")

    collect = subparsers.add_parser("collect", help="Download and validate a completed result.")
    collect.add_argument("job")

    wait_collect = subparsers.add_parser(
        "wait-collect", help="Wait for a submitted Batch and save its final JSON."
    )
    wait_collect.add_argument("job")
    wait_collect.add_argument("--poll-seconds", type=int, default=30)

    cleanup = subparsers.add_parser("cleanup", help="Remove temporary artifacts from jobs.")
    cleanup.add_argument("jobs", nargs="+")

    merge_results = subparsers.add_parser(
        "merge-results", help="Merge result JSON files into one deduplicated master file."
    )
    merge_results.add_argument("output", type=Path)
    merge_results.add_argument("inputs", type=Path, nargs="+")

    restore_videos = subparsers.add_parser(
        "restore-useful-videos",
        help="Restore approved videos from a result JSON file.",
    )
    restore_videos.add_argument("results", type=Path)

    accepted_results = subparsers.add_parser(
        "accepted-results",
        help="Write only collection-eligible entries from a result JSON file.",
    )
    accepted_results.add_argument("input", type=Path)
    accepted_results.add_argument("output", type=Path)

    run = subparsers.add_parser("run", help="Prepare, submit, wait, and collect one video.")
    run.add_argument("video", type=Path)
    run.add_argument("--poll-seconds", type=int, default=30)

    run_manifest = subparsers.add_parser(
        "run-manifest", help="Prepare, submit, wait for, and collect an X manifest."
    )
    run_manifest.add_argument("manifest", type=Path)
    run_manifest.add_argument("--poll-seconds", type=int, default=30)

    run_manifest_fast = subparsers.add_parser(
        "run-manifest-fast",
        help="Analyze one or more manifests synchronously with limited concurrency.",
    )
    run_manifest_fast.add_argument("manifests", type=Path, nargs="+")
    run_manifest_fast.add_argument("--skip-results", type=Path)
    run_manifest_fast.add_argument("--concurrency", type=int, default=3)
    run_manifest_fast.add_argument("--useful-dir", type=Path)

    add_cookie = subparsers.add_parser(
        "x-add-cookie", help="Securely add an X browser session to twscrape."
    )
    add_cookie.add_argument("alias", help="Local name for this X session.")

    subparsers.add_parser("x-accounts", help="List local twscrape account sessions.")
    subparsers.add_parser(
        "x-clear-videos",
        help="Delete downloaded X videos while preserving account folders and manifests.",
    )
    subparsers.add_parser("x-media-status", help="Count currently downloaded X videos by account.")

    x_fetch = subparsers.add_parser(
        "x-fetch", help="Download recent video posts from an X account."
    )
    x_fetch.add_argument("handle", help="X handle, with or without @.")
    x_fetch.add_argument("--post-limit", type=int, default=20)
    x_fetch.add_argument("--post-offset", type=int, default=0)
    x_fetch.add_argument("--max-videos", type=int, default=5)
    x_fetch.add_argument("--min-duration-seconds", type=float, default=0)
    x_fetch.add_argument(
        "--refresh", action="store_true", help="Ignore the six-hour manifest cache."
    )

    x_fetch_founders = subparsers.add_parser(
        "x-fetch-founders",
        help="Sequentially fetch videos for every enabled founder in a JSON file.",
    )
    x_fetch_founders.add_argument("founders", type=Path)
    x_fetch_founders.add_argument("--post-limit", type=int, default=200)
    x_fetch_founders.add_argument("--max-videos", type=int, default=200)
    x_fetch_founders.add_argument("--min-duration-seconds", type=float, default=20)
    x_fetch_founders.add_argument("--delay-seconds", type=float, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.load()
    current_job_id: str | None = None

    try:
        if args.command == "prepare":
            record = prepare_job(args.video, settings)
            _print_record(record)
        elif args.command == "prepare-many":
            record = prepare_many(args.videos, settings)
            _print_record(record)
        elif args.command == "prepare-manifest":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            videos = [Path(item["local_path"]) for item in manifest.get("videos", [])]
            record = prepare_many(videos, settings)
            _print_record(record)
        elif args.command == "submit":
            record = submit_job(args.job, settings)
            _print_record(record)
        elif args.command == "status":
            record = refresh_job(args.job, settings)
            _print_record(record)
        elif args.command == "collect":
            result_path = collect_job(args.job, settings)
            print(result_path.read_text(encoding="utf-8"), end="")
            print(f"\nSaved result to {result_path}", file=sys.stderr)
        elif args.command == "wait-collect":
            record = wait_for_job(args.job, settings, poll_seconds=args.poll_seconds)
            if record["state"] != "completed":
                raise BatchWorkflowError(
                    f"Batch ended in state {record['state']!r}; no result was collected."
                )
            result_path = collect_job(args.job, settings)
            print(f"Saved result to {result_path}")
        elif args.command == "cleanup":
            removed = [path for job in args.jobs for path in cleanup_job_artifacts(job, settings)]
            print(json.dumps({"removed": removed}, indent=2))
        elif args.command == "merge-results":
            merged: dict[str, dict[str, object]] = {}
            for input_path in args.inputs:
                payload = json.loads(input_path.read_text(encoding="utf-8"))
                items = payload if isinstance(payload, list) else [payload]
                for item in items:
                    source = item.get("source_video")
                    if not source:
                        source = item.get("classification", {}).get("filename")
                    if not source:
                        raise BatchWorkflowError(
                            f"Result in {input_path} has no source video or filename."
                        )
                    merged[str(source)] = item
            args.output.write_text(
                json.dumps(list(merged.values()), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {"videos": len(merged), "output": str(args.output.resolve())},
                    indent=2,
                )
            )
        elif args.command == "restore-useful-videos":
            restored = restore_useful_videos(args.results, settings)
            print(
                json.dumps(
                    {**restored, "output": str(settings.useful_video_dir)},
                    indent=2,
                )
            )
        elif args.command == "accepted-results":
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else [payload]
            accepted = [
                item
                for item in items
                if item.get("classification", {}).get("eligible_for_collection") is True
            ]
            args.output.write_text(
                json.dumps(accepted, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {"accepted": len(accepted), "output": str(args.output.resolve())},
                    indent=2,
                )
            )
        elif args.command == "run":
            record = prepare_job(args.video, settings)
            current_job_id = record["job_id"]
            print(f"Prepared job {current_job_id}.", file=sys.stderr)
            record = submit_job(current_job_id, settings)
            print(f"Submitted batch {record['batch_id']}; waiting for completion.", file=sys.stderr)
            record = wait_for_job(current_job_id, settings, poll_seconds=args.poll_seconds)
            print(f"Batch finished with state {record['state']}.", file=sys.stderr)
            result_path = collect_job(current_job_id, settings)
            print(result_path.read_text(encoding="utf-8"), end="")
        elif args.command == "run-manifest":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            videos = [Path(item["local_path"]) for item in manifest.get("videos", [])]
            record = prepare_many(videos, settings)
            current_job_id = record["job_id"]
            print(
                f"Prepared {record['video_count']} videos as job {current_job_id}.",
                file=sys.stderr,
            )
            record = submit_job(current_job_id, settings)
            print(f"Submitted batch {record['batch_id']}; waiting.", file=sys.stderr)
            record = wait_for_job(current_job_id, settings, poll_seconds=args.poll_seconds)
            if record["state"] != "completed":
                raise BatchWorkflowError(
                    f"Batch ended in state {record['state']!r}; no result was collected."
                )
            result_path = collect_job(current_job_id, settings)
            print(f"Saved result to {result_path}")
        elif args.command == "run-manifest-fast":
            run_settings = (
                replace(settings, useful_video_dir=args.useful_dir.expanduser().resolve())
                if args.useful_dir
                else settings
            )
            skip_names: set[str] = set()
            if args.skip_results:
                previous = json.loads(args.skip_results.read_text(encoding="utf-8"))
                skip_names = {
                    Path(item["source_video"]).name for item in previous if item.get("source_video")
                }
            unique_videos: dict[str, Path] = {}
            for manifest_path in args.manifests:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for item in manifest.get("videos", []):
                    video = Path(item["local_path"])
                    if video.name not in skip_names and video.is_file():
                        unique_videos.setdefault(video.name, video)
            videos = list(unique_videos.values())
            record = prepare_many(videos, settings)
            current_job_id = record["job_id"]
            print(
                f"Prepared {record['video_count']} videos as job {current_job_id}; "
                f"running {args.concurrency} at a time.",
                file=sys.stderr,
            )
            result_path = asyncio.run(
                run_sync_job(
                    current_job_id,
                    run_settings,
                    concurrency=args.concurrency,
                )
            )
            print(f"Saved result to {result_path}")
        elif args.command == "x-add-cookie":
            asyncio.run(add_cookie_account(args.alias, settings))
            print(f"Saved active X session as {args.alias!r}.")
        elif args.command == "x-accounts":
            print(json.dumps(asyncio.run(list_accounts(settings)), indent=2))
        elif args.command == "x-clear-videos":
            count, removed_bytes = clear_downloaded_videos(settings)
            print(
                json.dumps(
                    {
                        "removed_videos": count,
                        "freed_bytes": removed_bytes,
                        "freed_gb": round(removed_bytes / 1024**3, 2),
                    },
                    indent=2,
                )
            )
        elif args.command == "x-media-status":
            status = downloaded_video_status(settings)
            print(
                json.dumps(
                    {
                        "accounts": status,
                        "total_videos": sum(item["videos"] for item in status),
                        "total_bytes": sum(item["bytes"] for item in status),
                    },
                    indent=2,
                )
            )
        elif args.command == "x-fetch":
            manifest, videos = asyncio.run(
                fetch_account_videos(
                    args.handle,
                    settings,
                    post_limit=args.post_limit,
                    max_videos=args.max_videos,
                    refresh=args.refresh,
                    post_offset=args.post_offset,
                    min_duration_seconds=args.min_duration_seconds,
                )
            )
            print(
                json.dumps(
                    {
                        "downloaded": len(videos),
                        "manifest": str(manifest),
                        "videos": [video.local_path for video in videos],
                    },
                    indent=2,
                )
            )
        elif args.command == "x-fetch-founders":
            founders = json.loads(args.founders.read_text(encoding="utf-8"))
            handles = [item["handle"] for item in founders if item.get("enabled", True)]
            summaries = asyncio.run(
                fetch_multiple_accounts(
                    handles,
                    settings,
                    post_limit=args.post_limit,
                    max_videos=args.max_videos,
                    min_duration_seconds=args.min_duration_seconds,
                    delay_seconds=args.delay_seconds,
                )
            )
            print(json.dumps(summaries, indent=2))
        return 0
    except KeyboardInterrupt:
        suffix = (
            f" Resume with: launch-classifier status {current_job_id}" if current_job_id else ""
        )
        print(f"\nInterrupted.{suffix}", file=sys.stderr)
        return 130
    except (
        BatchWorkflowError,
        ConfigurationError,
        VideoProcessingError,
        XIngestError,
        OpenAIError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _print_record(record: dict[str, object]) -> None:
    visible = {
        key: record.get(key)
        for key in (
            "job_id",
            "state",
            "filename",
            "video_count",
            "duration_seconds",
            "frame_count",
            "sheet_count",
            "request_bytes",
            "batch_id",
            "request_counts",
            "output_file_id",
            "error_file_id",
            "result_file",
        )
        if record.get(key) is not None
    }
    print(json.dumps(visible, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
