import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from twitter_scrapper import batch
from twitter_scrapper.batch import BatchWorkflowError, _batch_request, parse_batch_output
from twitter_scrapper.config import Settings
from twitter_scrapper.schema import ClassificationResult
from twitter_scrapper.video import VideoArtifacts


def result_payload() -> dict[str, object]:
    return {
        "filename": "demo.mp4",
        "is_launch_video": False,
        "launch_confidence": 0.2,
        "is_saas_motion": False,
        "contains_talking_head": True,
        "contains_multi_video_showcase": False,
        "eligible_for_collection": False,
        "category": "other",
        "subcategory": "unknown",
        "visual_style": "talking_head",
        "production_quality": "basic",
        "summary": "A founder talks to the camera.",
        "evidence_timestamps": [
            {"timestamp_seconds": 0, "observation": "A person fills the frame."}
        ],
        "rejection_reasons": ["No visible product launch or polished product motion."],
    }


def test_batch_request_contains_one_responses_request_and_schema(tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.jpg"
    Image.new("RGB", (20, 20), "blue").save(sheet)

    request = _batch_request("video-123", "demo.mp4", 12.0, (sheet,), "gpt-4o-mini")

    assert request["url"] == "/v1/responses"
    assert request["custom_id"] == "video-123"
    assert request["body"]["text"]["format"]["strict"] is True
    content = request["body"]["input"][0]["content"]
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    json.dumps(request)


def test_parse_batch_output_validates_result(tmp_path: Path) -> None:
    output = {
        "custom_id": "video-123",
        "response": {
            "status_code": 200,
            "body": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(result_payload())}],
                    }
                ]
            },
        },
        "error": None,
    }
    path = tmp_path / "output.jsonl"
    path.write_text(json.dumps(output) + "\n", encoding="utf-8")

    result = parse_batch_output(path, "video-123")

    assert isinstance(result, ClassificationResult)
    assert result.eligible_for_collection is False


def test_parse_batch_output_surfaces_request_error(tmp_path: Path) -> None:
    path = tmp_path / "output.jsonl"
    path.write_text(
        json.dumps({"custom_id": "video-123", "error": {"message": "bad request"}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BatchWorkflowError, match="bad request"):
        parse_batch_output(path, "video-123")


def test_prepare_many_creates_one_request_per_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    for video in videos:
        video.write_bytes(b"video")

    def fake_process(video_path: Path, job_dir: Path, settings: Settings) -> VideoArtifacts:
        sheet = job_dir / "sheet.jpg"
        Image.new("RGB", (20, 20), "blue").save(sheet)
        return VideoArtifacts(video_path.resolve(), 1.0, (sheet,), (sheet,))

    monkeypatch.setattr(batch, "process_video", fake_process)
    settings = Settings(
        model="gpt-4o",
        work_dir=tmp_path / "work",
        max_video_seconds=60,
        max_batch_jsonl_bytes=1_000_000,
    )

    record = batch.prepare_many(videos, settings)
    lines = Path(record["request_file"]).read_text(encoding="utf-8").splitlines()

    assert record["video_count"] == 2
    assert len(lines) == 2
    assert len({json.loads(line)["custom_id"] for line in lines}) == 2
    assert not (Path(record["request_file"]).parent / "videos").exists()


def test_eligible_result_is_mapped_and_organized(tmp_path: Path) -> None:
    source = tmp_path / "launch.mp4"
    source.write_bytes(b"video")
    result = ClassificationResult.model_validate(
        {
            "filename": source.name,
            "is_launch_video": True,
            "launch_confidence": 0.9,
            "is_saas_motion": True,
            "contains_talking_head": False,
            "contains_multi_video_showcase": False,
            "eligible_for_collection": True,
            "category": "developer_tools",
            "subcategory": "api",
            "visual_style": "product_motion",
            "production_quality": "premium",
            "summary": "A polished API launch.",
            "evidence_timestamps": [{"timestamp_seconds": 0, "observation": "Product reveal."}],
            "rejection_reasons": [],
        }
    )
    settings = Settings(
        model="gpt-4o",
        work_dir=tmp_path / "work",
        max_video_seconds=60,
        max_batch_jsonl_bytes=1_000_000,
    )

    mapped = batch._map_and_organize_results([result], [str(source)], "job-1", settings)

    useful_path = Path(mapped[0]["useful_video"])
    assert useful_path.read_bytes() == b"video"
    assert mapped[0]["classification"]["eligible_for_collection"] is True
    assert (useful_path.parent / "job-1-results.json").is_file()


def test_sync_job_runs_prepared_requests_with_limited_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")

    def fake_process(video_path: Path, job_dir: Path, settings: Settings) -> VideoArtifacts:
        sheet = job_dir / "sheet.jpg"
        Image.new("RGB", (20, 20), "blue").save(sheet)
        return VideoArtifacts(video_path.resolve(), 1.0, (sheet,), (sheet,))

    class FakeResponses:
        async def create(self, **body: object) -> SimpleNamespace:
            return SimpleNamespace(output_text=json.dumps(result_payload()))

    fake_client = SimpleNamespace(responses=FakeResponses())
    monkeypatch.setattr(batch, "process_video", fake_process)
    settings = Settings(
        model="gpt-4o",
        work_dir=tmp_path / "work",
        max_video_seconds=60,
        max_batch_jsonl_bytes=1_000_000,
    )
    record = batch.prepare_many([video], settings)

    result_path = asyncio.run(
        batch.run_sync_job(record["job_id"], settings, concurrency=1, client=fake_client)
    )
    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert len(saved) == 1
    assert saved[0]["classification"]["eligible_for_collection"] is False
    assert not Path(record["request_file"]).exists()
