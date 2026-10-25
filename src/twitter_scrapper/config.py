from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when a required local or API setting is unavailable."""


@dataclass(frozen=True)
class Settings:
    model: str
    work_dir: Path
    max_video_seconds: int
    max_batch_jsonl_bytes: int
    useful_video_dir: Path | None = None
    frame_width: int = 384
    frame_height: int = 216
    sheet_columns: int = 4
    sheet_rows: int = 4
    jpeg_quality: int = 75

    @classmethod
    def load(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or Path.cwd()).resolve()
        # Project credentials must win over Cursor/terminal gateway variables.
        load_dotenv(root / ".env", override=True)
        work_dir = Path(os.getenv("CLASSIFIER_WORK_DIR", ".launch-classifier"))
        if not work_dir.is_absolute():
            work_dir = root / work_dir
        useful_video_dir = Path(os.getenv("USEFUL_VIDEO_DIR", "data/output/useful-videos"))
        if not useful_video_dir.is_absolute():
            useful_video_dir = root / useful_video_dir
        return cls(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            work_dir=work_dir.resolve(),
            max_video_seconds=_positive_int("MAX_VIDEO_SECONDS", 600),
            max_batch_jsonl_bytes=_positive_int("MAX_BATCH_JSONL_MB", 190) * 1024 * 1024,
            useful_video_dir=useful_video_dir.resolve(),
        )

    def require_local_tools(self) -> None:
        missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
        if missing:
            raise ConfigurationError(
                f"Missing required executable(s): {', '.join(missing)}. Install FFmpeg first."
            )

    @staticmethod
    def require_api_key() -> str:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key == "sk-proj-replace-me":
            raise ConfigurationError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and add a newly created key."
            )
        return key


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return value
