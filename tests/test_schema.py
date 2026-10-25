import pytest
from pydantic import ValidationError

from twitter_scrapper.schema import ClassificationResult


def valid_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "filename": "launch.mp4",
        "is_launch_video": True,
        "launch_confidence": 0.95,
        "is_saas_motion": True,
        "contains_talking_head": False,
        "contains_multi_video_showcase": False,
        "eligible_for_collection": True,
        "category": "fintech",
        "subcategory": "payments",
        "visual_style": "product_motion",
        "production_quality": "premium",
        "summary": "A polished payments product reveal.",
        "evidence_timestamps": [
            {"timestamp_seconds": 2, "observation": "The product name and launch appear."}
        ],
        "rejection_reasons": [],
    }
    result.update(overrides)
    return result


def test_accepts_consistent_eligible_result() -> None:
    parsed = ClassificationResult.model_validate(valid_result())
    assert parsed.eligible_for_collection is True
    assert parsed.category.value == "fintech"


def test_rejects_inconsistent_collection_decision() -> None:
    with pytest.raises(ValidationError, match="eligible_for_collection"):
        ClassificationResult.model_validate(valid_result(is_launch_video=False))


def test_rejects_eligibility_when_any_talking_head_is_present() -> None:
    with pytest.raises(ValidationError, match="eligible_for_collection"):
        ClassificationResult.model_validate(valid_result(contains_talking_head=True))


def test_rejects_eligibility_for_multi_video_showcase() -> None:
    with pytest.raises(ValidationError, match="eligible_for_collection"):
        ClassificationResult.model_validate(valid_result(contains_multi_video_showcase=True))


def test_rejected_result_requires_reason() -> None:
    with pytest.raises(ValidationError, match="rejection reason"):
        ClassificationResult.model_validate(
            valid_result(
                is_launch_video=False,
                eligible_for_collection=False,
                rejection_reasons=[],
            )
        )
