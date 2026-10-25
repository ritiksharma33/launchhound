import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from twitter_scrapper.config import Settings
from twitter_scrapper.x_ingest import (
    XIngestError,
    add_cookie_account,
    best_mp4_variant,
    clear_downloaded_videos,
    fetch_account_videos,
    list_accounts,
)


def settings(tmp_path: Path) -> Settings:
    return Settings(
        model="test-model",
        work_dir=tmp_path,
        max_video_seconds=60,
        max_batch_jsonl_bytes=1_000_000,
    )


def test_selects_highest_bitrate_mp4() -> None:
    variants = [
        SimpleNamespace(contentType="application/x-mpegURL", bitrate=0, url="playlist"),
        SimpleNamespace(contentType="video/mp4", bitrate=256_000, url="low"),
        SimpleNamespace(contentType="video/mp4", bitrate=2_000_000, url="high"),
    ]

    selected = best_mp4_variant(variants)

    assert selected.url == "high"


def test_cookie_account_is_stored_locally(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    asyncio.run(
        add_cookie_account(
            "x-test",
            app_settings,
            cookies="auth_token=test-auth-token; ct0=test-csrf-token",
        )
    )

    accounts = asyncio.run(list_accounts(app_settings))

    assert accounts == [{"alias": "x-test", "active": True, "error": None}]


def test_rejects_invalid_handle_before_network_access(tmp_path: Path) -> None:
    with pytest.raises(XIngestError, match="Invalid X handle"):
        asyncio.run(fetch_account_videos("https://x.com/not-a-handle", settings(tmp_path)))


def test_clear_videos_preserves_account_folders_and_manifests(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    account_dir = app_settings.work_dir / "x-media" / "founder"
    account_dir.mkdir(parents=True)
    video = account_dir / "post.mp4"
    video.write_bytes(b"video")
    manifest = account_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    count, removed_bytes = clear_downloaded_videos(app_settings)

    assert count == 1
    assert removed_bytes == 5
    assert account_dir.is_dir()
    assert manifest.is_file()
    assert not video.exists()
