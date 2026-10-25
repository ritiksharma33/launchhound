from pathlib import Path

from PIL import Image

from twitter_scrapper.config import Settings
from twitter_scrapper.video import _create_contact_sheets, timestamp_for_frame


def settings(work_dir: Path) -> Settings:
    return Settings(
        model="test-model",
        work_dir=work_dir,
        max_video_seconds=60,
        max_batch_jsonl_bytes=10_000_000,
        frame_width=80,
        frame_height=45,
        sheet_columns=4,
        sheet_rows=4,
        jpeg_quality=70,
    )


def test_timestamp_comes_from_frame_filename(tmp_path: Path) -> None:
    assert timestamp_for_frame(tmp_path / "frame_000042.jpg") == 42


def test_contact_sheets_keep_all_frames_in_order(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sheets_dir = tmp_path / "sheets"
    frames_dir.mkdir()
    sheets_dir.mkdir()
    frames = []
    for index in range(17):
        path = frames_dir / f"frame_{index:06d}.jpg"
        Image.new("RGB", (80, 45), (index, 0, 0)).save(path)
        frames.append(path)

    sheets = _create_contact_sheets(tuple(frames), sheets_dir, settings(tmp_path))

    assert [path.name for path in sheets] == ["sheet_0000.jpg", "sheet_0001.jpg"]
    assert all(path.stat().st_size > 0 for path in sheets)
