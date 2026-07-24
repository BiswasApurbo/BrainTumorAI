"""Request schemas for the BrainTumorAI REST API.

This module contains Pydantic models for validating request bodies and common
request parameters. The schemas are intentionally limited to validation and
serialization concerns.
"""

from uuid import UUID

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Request schema for creating an inference request.

    Attributes:
        upload_id: Unique identifier for a previously uploaded medical image.
        model_name: Optional model name to use for future inference.
        generate_report: Whether a diagnostic report should be generated.
        save_visualization: Whether inference visualizations should be saved.
    """

    upload_id: UUID = Field(
        ...,
        description="Unique identifier for a previously uploaded medical image.",
    )
    model_name: str | None = Field(
        default=None,
        description="Optional model name to use for future inference.",
    )
    generate_report: bool = Field(
        default=True,
        description="Whether a diagnostic report should be generated.",
    )
    save_visualization: bool = Field(
        default=True,
        description="Whether inference visualizations should be saved.",
    )


class ReportRequest(BaseModel):
    """Request schema for retrieving or referencing a report.

    Attributes:
        report_id: Unique identifier for a diagnostic report.
    """

    report_id: UUID = Field(
        ...,
        description="Unique identifier for a diagnostic report.",
    )


class PaginationRequest(BaseModel):
    """Request schema for paginated API queries.

    Attributes:
        page: Page number to retrieve.
        page_size: Number of items to include per page.
    """

    page: int = Field(
        ...,
        ge=1,
        description="Page number to retrieve.",
    )
    page_size: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of items to include per page.",
    )
