from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import Settings


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be probed or transformed."""


@dataclass(frozen=True)
class VideoArtifacts:
    source: Path
    duration_seconds: float
    frame_paths: tuple[Path, ...]
    sheet_paths: tuple[Path, ...]


def probe_duration(video_path: Path) -> float:
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise VideoProcessingError(f"Video does not exist: {video_path}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = _run(command, "probe video")
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if not streams:
            raise VideoProcessingError(f"No video stream found in {video_path.name}.")
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoProcessingError(f"Could not determine duration for {video_path.name}.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise VideoProcessingError(f"Video duration must be positive, got {duration!r}.")
    return duration


def process_video(video_path: Path, job_dir: Path, settings: Settings) -> VideoArtifacts:
    settings.require_local_tools()
    source = video_path.expanduser().resolve()
    duration = probe_duration(source)
    if duration > settings.max_video_seconds:
        raise VideoProcessingError(
            f"Video is {duration:.1f}s; configured limit is {settings.max_video_seconds}s."
        )

    frames_dir = job_dir / "frames"
    sheets_dir = job_dir / "sheets"
    frames_dir.mkdir(parents=True, exist_ok=False)
    sheets_dir.mkdir(parents=True, exist_ok=False)

    frame_pattern = frames_dir / "frame_%06d.jpg"
    filter_graph = (
        "fps=1,"
        f"scale={settings.frame_width}:{settings.frame_height}:force_original_aspect_ratio=decrease,"
        f"pad={settings.frame_width}:{settings.frame_height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            filter_graph,
            "-q:v",
            "4",
            "-start_number",
            "0",
            str(frame_pattern),
        ],
        "extract one frame per second",
    )

    frames = tuple(sorted(frames_dir.glob("frame_*.jpg")))
    if not frames:
        raise VideoProcessingError("FFmpeg produced no frames.")
    sheets = _create_contact_sheets(frames, sheets_dir, settings)
    return VideoArtifacts(source, duration, frames, sheets)


def timestamp_for_frame(frame_path: Path) -> int:
    try:
        return int(frame_path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise VideoProcessingError(f"Unexpected frame filename: {frame_path.name}") from exc


def _create_contact_sheets(
    frames: tuple[Path, ...], sheets_dir: Path, settings: Settings
) -> tuple[Path, ...]:
    frames_per_sheet = settings.sheet_columns * settings.sheet_rows
    caption_height = 28
    tile_height = settings.frame_height + caption_height
    font = ImageFont.load_default(size=16)
    output: list[Path] = []

    for sheet_index, offset in enumerate(range(0, len(frames), frames_per_sheet)):
        sheet = Image.new(
            "RGB",
            (settings.sheet_columns * settings.frame_width, settings.sheet_rows * tile_height),
            "black",
        )
        for position, frame_path in enumerate(frames[offset : offset + frames_per_sheet]):
            x = (position % settings.sheet_columns) * settings.frame_width
            y = (position // settings.sheet_columns) * tile_height
            with Image.open(frame_path) as frame:
                sheet.paste(frame.convert("RGB"), (x, y))
            timestamp = timestamp_for_frame(frame_path)
            ImageDraw.Draw(sheet).text(
                (x + 8, y + settings.frame_height + 5),
                f"t={timestamp:02d}s",
                fill="white",
                font=font,
            )

        destination = sheets_dir / f"sheet_{sheet_index:04d}.jpg"
        sheet.save(destination, "JPEG", quality=settings.jpeg_quality, optimize=True)
        output.append(destination)

    return tuple(output)


def _run(command: list[str], action: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise VideoProcessingError(
            f"Cannot {action}: executable {command[0]!r} not found."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or f"exit status {exc.returncode}"
        raise VideoProcessingError(f"Failed to {action}: {detail}") from exc
