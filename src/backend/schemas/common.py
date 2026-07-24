"""Common Pydantic schemas shared across the BrainTumorAI backend.

This module defines small reusable response and metadata models used by API
endpoints and infrastructure layers. The models are intentionally free of
business logic.
"""

from datetime import datetime

from pydantic import AwareDatetime, BaseModel, Field


class StatusResponse(BaseModel):
    """Generic status response schema.

    Attributes:
        status: Machine-readable status value.
        message: Human-readable status message.
    """

    status: str = Field(
        ...,
        min_length=1,
        description="Machine-readable status value.",
        examples=["success"],
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable status message.",
        examples=["Operation completed successfully."],
    )


class ErrorResponse(BaseModel):
    """Generic error response schema.

    Attributes:
        error: Machine-readable error category or code.
        detail: Human-readable error details.
        timestamp: UTC-aware timestamp describing when the error occurred.
    """

    error: str = Field(
        ...,
        min_length=1,
        description="Machine-readable error category or code.",
        examples=["validation_error"],
    )
    detail: str = Field(
        ...,
        min_length=1,
        description="Human-readable error details.",
        examples=["The submitted request could not be validated."],
    )
    timestamp: AwareDatetime = Field(
        ...,
        description="UTC-aware timestamp describing when the error occurred.",
        examples=[datetime.fromisoformat("2026-01-01T00:00:00+00:00")],
    )


class Pagination(BaseModel):
    """Pagination metadata schema.

    Attributes:
        page: Current page number.
        page_size: Number of items requested per page.
        total: Total number of available items.
    """

    page: int = Field(
        ...,
        ge=1,
        description="Current page number.",
        examples=[1],
    )
    page_size: int = Field(
        ...,
        ge=1,
        description="Number of items requested per page.",
        examples=[20],
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of available items.",
        examples=[0],
    )


class Metadata(BaseModel):
    """Application metadata response schema.

    Attributes:
        application: Application name.
        version: Application version.
        timestamp: UTC-aware timestamp describing when metadata was generated.
    """

    application: str = Field(
        ...,
        min_length=1,
        description="Application name.",
        examples=["BrainTumorAI"],
    )
    version: str = Field(
        ...,
        min_length=1,
        description="Application version.",
        examples=["0.1.0"],
    )
    timestamp: AwareDatetime = Field(
        ...,
        description=(
            "UTC-aware timestamp describing when metadata was generated."
        ),
        examples=[datetime.fromisoformat("2026-01-01T00:00:00+00:00")],
    )
