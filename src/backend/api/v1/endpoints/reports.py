"""Reports endpoints for the BrainTumorAI API.

This module exposes read-only API routes for future AI-generated diagnostic
reports. It does not generate reports, run inference, or modify files.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ....core.config import settings


router = APIRouter()


class ReportsListResponse(BaseModel):
    """Response body for the paginated reports list.

    Attributes:
        reports: Reports available for the requested page.
        total: Total number of available reports.
        page: Current page number.
        page_size: Number of reports requested per page.
        status: Current reports API status.
    """

    reports: list[dict[str, str]]
    total: int
    page: int
    page_size: int
    status: str


class ReportResponse(BaseModel):
    """Response body for a single available report.

    Attributes:
        report_id: Identifier for the requested report.
        status: Availability status for the report.
        path: Filesystem path to the report.
        created_at: UTC ISO 8601 timestamp derived from file metadata.
    """

    report_id: str
    status: str
    path: str
    created_at: str


@router.get("", response_model=ReportsListResponse)
async def list_reports() -> ReportsListResponse:
    """Return a paginated list of available reports.

    Report generation is not implemented yet, so the endpoint returns an empty
    first page while preserving the response shape expected by clients.

    Returns:
        A JSON-serializable paginated reports response.
    """

    return ReportsListResponse(
        reports=[],
        total=0,
        page=1,
        page_size=20,
        status="ready",
    )


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str) -> ReportResponse:
    """Return metadata for an available report.

    Args:
        report_id: Unique report identifier supplied in the route path.

    Returns:
        A JSON-serializable payload describing the available report.

    Raises:
        HTTPException: If the report identifier is invalid or the report does
        not exist in the configured reports directory.
    """

    normalized_report_id = _normalize_report_id(report_id)
    report_path = _find_report_path(normalized_report_id)
    if report_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report was not found.",
        )

    return ReportResponse(
        report_id=normalized_report_id,
        status="available",
        path=str(report_path),
        created_at=_get_file_created_at(report_path),
    )


def _normalize_report_id(report_id: str) -> str:
    """Validate and normalize a report identifier.

    Args:
        report_id: Raw report identifier from the path parameter.

    Returns:
        Canonical string representation of the report UUID.

    Raises:
        HTTPException: If the report identifier is blank or invalid.
    """

    candidate = report_id.strip()
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="report_id must be provided.",
        )

    try:
        return str(UUID(candidate))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="report_id must be a valid UUID.",
        ) from exc


def _find_report_path(report_id: str) -> Path | None:
    """Find a report file for a report identifier.

    Args:
        report_id: Canonical report UUID string.

    Returns:
        Matching report path when present, otherwise ``None``.
    """

    reports_directory = Path(settings.reports_directory)
    exact_report_path = reports_directory / report_id
    if exact_report_path.is_file():
        return exact_report_path

    return next(
        (
            path
            for path in reports_directory.glob(f"{report_id}.*")
            if path.is_file()
        ),
        None,
    )


def _get_file_created_at(path: Path) -> str:
    """Return an ISO 8601 UTC timestamp for a report file.

    Args:
        path: Existing report file path.

    Returns:
        UTC ISO 8601 timestamp derived from the file metadata.
    """

    created_at = datetime.fromtimestamp(path.stat().st_ctime, tz=timezone.utc)
    return created_at.isoformat()
