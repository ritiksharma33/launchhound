from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from getpass import getpass
from pathlib import Path
from typing import Any

import httpx
from twscrape import API, gather
from twscrape.accounts_pool import NoAccountError

from .config import Settings


class XIngestError(RuntimeError):
    """Raised when unofficial X ingestion cannot continue."""


@dataclass(frozen=True)
class DownloadedVideo:
    tweet_id: str
    tweet_url: str
    posted_at: str
    text: str
    views: int | None
    likes: int
    media_index: int
    duration_ms: int
    bitrate: int
    source_url: str
    local_path: str


def account_database(settings: Settings) -> Path:
    path = settings.work_dir / "twscrape" / "accounts.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clear_downloaded_videos(settings: Settings) -> tuple[int, int]:
    media_root = settings.work_dir / "x-media"
    removed_count = 0
    removed_bytes = 0
    if not media_root.is_dir():
        return removed_count, removed_bytes
    for video in media_root.rglob("*.mp4"):
        if not video.is_file():
            continue
        removed_bytes += video.stat().st_size
        video.unlink()
        removed_count += 1
    return removed_count, removed_bytes


def downloaded_video_status(settings: Settings) -> list[dict[str, Any]]:
    media_root = settings.work_dir / "x-media"
    if not media_root.is_dir():
        return []
    status: list[dict[str, Any]] = []
    for account_dir in sorted(path for path in media_root.iterdir() if path.is_dir()):
        videos = [path for path in account_dir.glob("*.mp4") if path.is_file()]
        if videos:
            status.append(
                {
                    "handle": account_dir.name,
                    "videos": len(videos),
                    "bytes": sum(path.stat().st_size for path in videos),
                }
            )
    return status


async def add_cookie_account(alias: str, settings: Settings, cookies: str | None = None) -> None:
    cookie_value = cookies or getpass(
        "Paste X cookies containing auth_token and ct0 (input is hidden): "
    )
    if not cookie_value.strip():
        raise XIngestError("Cookie string cannot be empty.")
    api = API(str(account_database(settings)), raise_when_no_account=True)
    try:
        await api.pool.add_account_cookies(alias, cookie_value)
    except ValueError as exc:
        raise XIngestError(str(exc)) from exc


async def list_accounts(settings: Settings) -> list[dict[str, Any]]:
    api = API(str(account_database(settings)))
    accounts = await api.pool.get_all()
    return [
        {
            "alias": account.username,
            "active": account.active,
            "error": account.error_msg,
        }
        for account in accounts
    ]


async def fetch_account_videos(
    handle: str,
    settings: Settings,
    post_limit: int = 20,
    max_videos: int = 5,
    refresh: bool = False,
    post_offset: int = 0,
    min_duration_seconds: float = 0,
) -> tuple[Path, list[DownloadedVideo]]:
    if post_limit <= 0 or max_videos <= 0 or post_offset < 0 or min_duration_seconds < 0:
        raise XIngestError(
            "Limits must be positive; offsets and minimum duration cannot be negative."
        )
    min_duration_ms = round(min_duration_seconds * 1000)
    clean_handle = handle.strip().removeprefix("@").lower()
    if not re.fullmatch(r"[a-z0-9_]{1,15}", clean_handle):
        raise XIngestError(f"Invalid X handle: {handle!r}")

    destination = settings.work_dir / "x-media" / clean_handle
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "manifest.json"
    if not refresh:
        cached = _load_cached_manifest(
            manifest,
            post_limit,
            max_videos,
            post_offset,
            min_duration_ms,
        )
        if cached is not None:
            return manifest, cached

    api = API(
        str(account_database(settings)),
        raise_when_no_account=True,
        wait_timeout=15 * 60,
    )
    try:
        user = await api.user_by_login(clean_handle)
        if user is None:
            raise XIngestError(f"X account @{clean_handle} was not found.")
        timeline = await gather(api.user_tweets(user.id, limit=post_offset + post_limit))
        tweets = timeline[post_offset : post_offset + post_limit]
    except NoAccountError as exc:
        raise XIngestError(
            "No X account became available within 15 minutes. Check `x-accounts`; "
            "refresh cookies only if the account is inactive or has an authentication error."
        ) from exc

    downloaded = _existing_downloads(manifest, min_duration_ms)
    seen_tweet_ids: set[str] = set()
    seen_media = {(item.tweet_id, item.media_index) for item in downloaded}

    async with httpx.AsyncClient(follow_redirects=True, timeout=90) as client:
        for tweet in tweets:
            if len(downloaded) >= max_videos:
                break
            if tweet.retweetedTweet is not None:
                continue
            candidates = [tweet]
            if tweet.quotedTweet is not None:
                candidates.append(tweet.quotedTweet)
            for candidate in candidates:
                if candidate.id_str in seen_tweet_ids:
                    continue
                seen_tweet_ids.add(candidate.id_str)
                for media_index, video in enumerate(candidate.media.videos):
                    if video.duration < min_duration_ms:
                        continue
                    media_key = (candidate.id_str, media_index)
                    if media_key in seen_media:
                        continue
                    variant = best_mp4_variant(video.variants)
                    if variant is None:
                        continue
                    output = destination / f"{candidate.id_str}_{media_index}.mp4"
                    if not output.is_file():
                        await _download(client, variant.url, output)
                    downloaded.append(
                        DownloadedVideo(
                            tweet_id=candidate.id_str,
                            tweet_url=candidate.url,
                            posted_at=candidate.date.isoformat(),
                            text=candidate.rawContent,
                            views=candidate.viewCount,
                            likes=candidate.likeCount,
                            media_index=media_index,
                            duration_ms=video.duration,
                            bitrate=variant.bitrate,
                            source_url=variant.url,
                            local_path=str(output.resolve()),
                        )
                    )
                    seen_media.add(media_key)
                    if len(downloaded) >= max_videos:
                        break
                if len(downloaded) >= max_videos:
                    break

    previous_ranges = _manifest_ranges(manifest)
    current_range = [post_offset, post_offset + post_limit]
    if current_range not in previous_ranges:
        previous_ranges.append(current_range)
    manifest.write_text(
        json.dumps(
            {
                "handle": clean_handle,
                "profile_url": f"https://x.com/{clean_handle}",
                "posts_requested": max(
                    [end for _, end in previous_ranges], default=post_offset + post_limit
                ),
                "scanned_post_ranges": previous_ranges,
                "min_duration_seconds": min_duration_seconds,
                "videos": [asdict(item) for item in downloaded],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, downloaded


async def fetch_multiple_accounts(
    handles: list[str],
    settings: Settings,
    post_limit: int,
    max_videos: int,
    min_duration_seconds: float,
    delay_seconds: float = 15,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, handle in enumerate(handles):
        try:
            manifest, videos = await fetch_account_videos(
                handle,
                settings,
                post_limit=post_limit,
                max_videos=max_videos,
                refresh=True,
                min_duration_seconds=min_duration_seconds,
            )
            summaries.append(
                {
                    "handle": handle,
                    "videos": len(videos),
                    "manifest": str(manifest),
                }
            )
        except Exception as exc:
            summaries.append({"handle": handle, "error": str(exc)})
        if index < len(handles) - 1 and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
    return summaries


def _load_cached_manifest(
    manifest: Path,
    post_limit: int,
    max_videos: int,
    post_offset: int,
    min_duration_ms: int,
    cache_seconds: int = 6 * 60 * 60,
) -> list[DownloadedVideo] | None:
    if not manifest.is_file() or time.time() - manifest.stat().st_mtime > cache_seconds:
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        requested_range = [post_offset, post_offset + post_limit]
        scanned_ranges = payload.get("scanned_post_ranges", [])
        if requested_range not in scanned_ranges:
            return None
        if payload.get("min_duration_seconds", 0) * 1000 < min_duration_ms:
            return None
        videos = [
            DownloadedVideo(**item)
            for item in payload.get("videos", [])
            if item.get("duration_ms", 0) >= min_duration_ms
        ][:max_videos]
        if any(not Path(video.local_path).is_file() for video in videos):
            return None
        return videos
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _existing_downloads(manifest: Path, min_duration_ms: int = 0) -> list[DownloadedVideo]:
    if not manifest.is_file():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return [
            DownloadedVideo(**item)
            for item in payload.get("videos", [])
            if item.get("duration_ms", 0) >= min_duration_ms and Path(item["local_path"]).is_file()
        ]
    except (json.JSONDecodeError, KeyError, OSError, TypeError):
        return []


def _manifest_ranges(manifest: Path) -> list[list[int]]:
    if not manifest.is_file():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        ranges = payload.get("scanned_post_ranges", [])
        return [item for item in ranges if isinstance(item, list) and len(item) == 2]
    except (json.JSONDecodeError, OSError, TypeError):
        return []


def best_mp4_variant(variants: list[Any]) -> Any | None:
    mp4_variants = [
        variant for variant in variants if variant.contentType == "video/mp4" and variant.url
    ]
    return max(mp4_variants, key=lambda variant: variant.bitrate, default=None)


async def _download(client: httpx.AsyncClient, url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                async for chunk in response.aiter_bytes():
                    output.write(chunk)
        partial.replace(destination)
    except (httpx.HTTPError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise XIngestError(f"Could not download video to {destination}: {exc}") from exc
