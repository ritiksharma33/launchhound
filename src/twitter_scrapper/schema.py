from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Category(StrEnum):
    FINTECH = "fintech"
    AI = "ai"
    DEVELOPER_TOOLS = "developer_tools"
    PRODUCTIVITY = "productivity"
    DESIGN = "design"
    MARKETING = "marketing"
    ECOMMERCE = "ecommerce"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    OTHER = "other"


class VisualStyle(StrEnum):
    PRODUCT_MOTION = "product_motion"
    TALKING_HEAD = "talking_head"
    SCREEN_RECORDING = "screen_recording"
    ANIMATION = "animation"
    MIXED = "mixed"
    OTHER = "other"


class ProductionQuality(StrEnum):
    PREMIUM = "premium"
    GOOD = "good"
    BASIC = "basic"
    POOR = "poor"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_seconds: int = Field(ge=0)
    observation: str = Field(min_length=1)


class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    is_launch_video: bool
    launch_confidence: float = Field(ge=0, le=1)
    is_saas_motion: bool
    contains_talking_head: bool
    contains_multi_video_showcase: bool
    eligible_for_collection: bool
    category: Category
    subcategory: str = Field(min_length=1)
    visual_style: VisualStyle
    production_quality: ProductionQuality
    summary: str = Field(min_length=1)
    evidence_timestamps: list[Evidence]
    rejection_reasons: list[str]

    @model_validator(mode="after")
    def collection_decision_is_consistent(self) -> ClassificationResult:
        expected = (
            self.is_launch_video
            and self.is_saas_motion
            and not self.contains_talking_head
            and not self.contains_multi_video_showcase
            and self.visual_style != VisualStyle.TALKING_HEAD
            and self.production_quality in {ProductionQuality.PREMIUM, ProductionQuality.GOOD}
        )
        if self.eligible_for_collection != expected:
            raise ValueError(
                "eligible_for_collection must match launch, SaaS motion, style, and quality fields"
            )
        if self.eligible_for_collection and self.rejection_reasons:
            raise ValueError("eligible results must have no rejection reasons")
        if not self.eligible_for_collection and not self.rejection_reasons:
            raise ValueError("ineligible results must explain at least one rejection reason")
        return self
