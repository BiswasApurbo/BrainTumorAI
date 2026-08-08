"""Response schemas for the BrainTumorAI REST API.

This module defines reusable Pydantic response models for API endpoints across
the backend. The schemas describe response structure only and do not contain
endpoint or business logic.
"""

from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for application health checks.

    Attributes:
        status: Current health status.
        application: Application name.
        version: Application version.
        environment: Runtime environment name.
        timestamp: UTC-aware timestamp when the response was generated.
    """

    status: str = Field(..., description="Current health status.")
    application: str = Field(..., description="Application name.")
    version: str = Field(..., description="Application version.")
    environment: str = Field(..., description="Runtime environment name.")
    timestamp: AwareDatetime = Field(
        ...,
        description="UTC-aware timestamp when the response was generated.",
    )


class UploadResponse(BaseModel):
    """Response schema for uploaded medical imaging files.

    Attributes:
        upload_id: Unique identifier assigned to the upload.
        original_filename: Original filename supplied by the client.
        stored_filename: Server-side filename used to store the upload.
        file_size_bytes: Uploaded file size in bytes.
        upload_directory: Directory where the upload was stored.
        upload_timestamp: UTC-aware timestamp when the upload was stored.
        status: Current upload status.
    """

    upload_id: UUID = Field(
        ...,
        description="Unique identifier assigned to the upload.",
    )
    upload_directory: str = Field(
        ...,
        description="Directory where the uploaded modalities were stored.",
    )
    upload_timestamp: AwareDatetime = Field(
        ...,
        description="UTC-aware timestamp when the upload was stored.",
    )
    status: str = Field(..., description="Current upload status.")


class InferenceResponse(BaseModel):
    """Response schema for queued or completed inference requests.

    Attributes:
        request_id: Unique identifier assigned to the inference request.
        upload_id: Unique identifier for the uploaded medical image.
        status: Current inference request status.
        message: Human-readable status message.
        created_at: UTC-aware timestamp when the request was created.
        report_id: Optional unique identifier for the generated diagnostic report.
        visualization_path: Optional path to the 3D Plotly HTML visualization.
        tumor_mask_path: Optional path to the tumor segmentation mask.
        anatomy_mask_path: Optional path to the anatomy segmentation mask.
        volumetric_analysis: Optional quantitative volumetric measurements dictionary.
    """

    request_id: UUID = Field(
        ...,
        description="Unique identifier assigned to the inference request.",
    )
    upload_id: UUID = Field(
        ...,
        description="Unique identifier for the uploaded medical image.",
    )
    status: str = Field(
        ...,
        description="Current inference request status.",
    )
    message: str = Field(
        ...,
        description="Human-readable status message.",
    )
    created_at: AwareDatetime = Field(
        ...,
        description="UTC-aware timestamp when the request was created.",
    )
    report_id: UUID | None = Field(
        default=None,
        description="Optional unique identifier for the generated report.",
    )
    visualization_path: str | None = Field(
        default=None,
        description="Optional path to the 3D Plotly HTML visualization.",
    )
    tumor_mask_path: str | None = Field(
        default=None,
        description="Optional path to the tumor segmentation mask.",
    )
    anatomy_mask_path: str | None = Field(
        default=None,
        description="Optional path to the anatomy segmentation mask.",
    )
    volumetric_analysis: dict[str, float] | None = Field(
        default=None,
        description="Optional quantitative volumetric measurements dictionary.",
    )


class ReportSummary(BaseModel):
    """Response schema for report summaries.

    Attributes:
        report_id: Unique identifier for the diagnostic report.
        status: Current report status.
        created_at: UTC-aware timestamp when the report became available.
    """

    report_id: UUID = Field(
        ...,
        description="Unique identifier for the diagnostic report.",
    )
    status: str = Field(..., description="Current report status.")
    created_at: AwareDatetime = Field(
        ...,
        description="UTC-aware timestamp when the report became available.",
    )


class ReportDetail(BaseModel):
    """Response schema for detailed report metadata.

    Attributes:
        report_id: Unique identifier for the diagnostic report.
        status: Current report status.
        path: Filesystem path to the report.
        created_at: UTC-aware timestamp when the report became available.
    """

    report_id: UUID = Field(
        ...,
        description="Unique identifier for the diagnostic report.",
    )
    status: str = Field(..., description="Current report status.")
    path: str = Field(..., description="Filesystem path to the report.")
    created_at: AwareDatetime = Field(
        ...,
        description="UTC-aware timestamp when the report became available.",
    )


class ReportListResponse(BaseModel):
    """Response schema for paginated report listings.

    Attributes:
        reports: Report summaries for the current page.
        total: Total number of available reports.
        page: Current page number.
        page_size: Number of reports included per page.
        status: Current reports API status.
    """

    reports: list[ReportSummary] = Field(
        ...,
        description="Report summaries for the current page.",
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of available reports.",
    )
    page: int = Field(
        ...,
        ge=1,
        description="Current page number.",
    )
    page_size: int = Field(
        ...,
        ge=1,
        description="Number of reports included per page.",
    )
    status: str = Field(..., description="Current reports API status.")
