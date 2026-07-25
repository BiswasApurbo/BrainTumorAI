"""Reports endpoints for the BrainTumorAI API.

This module exposes API routes for managing diagnostic reports, delegating all
storage, retrieval, listing, and deletion logic to the injected ``ReportService``.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ....dependencies import get_report_service
from ....services.report_service import ReportService


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
        created_at: UTC ISO 8601 timestamp derived from report metadata.
    """

    report_id: str
    status: str
    path: str
    created_at: str


@router.get("", response_model=ReportsListResponse)
async def list_reports(
    report_service: Annotated[ReportService, Depends(get_report_service)],
) -> ReportsListResponse:
    """Return a paginated list of available reports.

    Args:
        report_service: Injected report management service instance.

    Returns:
        A JSON-serializable paginated reports response.
    """

    metadatas = report_service.list_reports()
    reports = [
        {
            "report_id": str(m.report_id),
            "upload_id": str(m.upload_id),
            "status": m.status,
            "created_at": m.created_at.isoformat(),
        }
        for m in metadatas
    ]

    return ReportsListResponse(
        reports=reports,
        total=len(reports),
        page=1,
        page_size=20,
        status="ready",
    )


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    report_service: Annotated[ReportService, Depends(get_report_service)],
) -> ReportResponse:
    """Return metadata for an available report.

    Args:
        report_id: Unique report identifier supplied in the route path.
        report_service: Injected report management service instance.

    Returns:
        A JSON-serializable payload describing the available report.

    Raises:
        HTTPException: If the report identifier is invalid or the report does
        not exist.
    """

    try:
        report = report_service.get_report(report_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report was not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    path_val = (
        str(report.content.get("path", ""))
        if isinstance(report.content, dict)
        else ""
    )

    return ReportResponse(
        report_id=str(report.report_id),
        status=report.status,
        path=path_val,
        created_at=report.created_at.isoformat(),
    )


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(
    report_id: str,
    report_service: Annotated[ReportService, Depends(get_report_service)],
) -> None:
    """Delete an available report.

    Args:
        report_id: Unique report identifier supplied in the route path.
        report_service: Injected report management service instance.

    Raises:
        HTTPException: If the report identifier is invalid or the report does
        not exist.
    """

    try:
        report_service.delete_report(report_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report was not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
