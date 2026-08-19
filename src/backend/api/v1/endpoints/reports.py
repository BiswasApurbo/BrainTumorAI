"""Reports endpoints for the BrainTumorAI API.

This module exposes API routes for managing and downloading diagnostic reports,
delegating storage, retrieval, listing, and automatic artifact deletion to the
injected ``ReportService``.
"""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.logging import get_logger
from ....dependencies import get_report_service
from ....services.report_service import ReportService

logger = get_logger(__name__)

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
        download_url: Endpoint URL to securely download the analysis report.
        created_at: UTC ISO 8601 timestamp derived from report metadata.
    """

    report_id: str
    status: str
    download_url: str
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
            "download_url": f"/api/v1/reports/{m.report_id}/download",
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


@router.get("/{report_id}/download")
async def download_report(
    report_id: str,
    background_tasks: BackgroundTasks,
    report_service: Annotated[ReportService, Depends(get_report_service)],
) -> FileResponse:
    """Download the 3D HTML analysis report and purge all associated artifacts after download.

    Args:
        report_id: Unique report identifier.
        background_tasks: FastAPI background tasks context for post-response cleanup.
        report_service: Injected report management service instance.

    Returns:
        FileResponse delivering the standalone 3D HTML analysis.

    Raises:
        HTTPException: 404 if the report does not exist or was already downloaded.
    """

    try:
        report, file_path = report_service.claim_report_for_download(report_id)
    except (KeyError, FileNotFoundError) as exc:
        logger.warning("Report %s not found, currently downloading, or already purged: %s", report_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found or no longer available.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Schedule complete artifact and metadata cleanup immediately after successful file transmission
    background_tasks.add_task(
        report_service.delete_report_and_artifacts,
        report.report_id,
    )

    filename = f"BrainTumorAI_Report_{report.report_id}.html"

    logger.info("Serving one-time download for report %s as %s", report.report_id, filename)

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/html",
        background=background_tasks,
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
        HTTPException: If the report identifier is invalid or does not exist.
    """

    try:
        report = report_service.get_report(report_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found or no longer available.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ReportResponse(
        report_id=str(report.report_id),
        status=report.status,
        download_url=f"/api/v1/reports/{report.report_id}/download",
        created_at=report.created_at.isoformat(),
    )


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(
    report_id: str,
    report_service: Annotated[ReportService, Depends(get_report_service)],
) -> None:
    """Delete an available report and its artifacts manually.

    Args:
        report_id: Unique report identifier supplied in the route path.
        report_service: Injected report management service instance.

    Raises:
        HTTPException: If the report identifier is invalid or does not exist.
    """

    try:
        report_service.delete_report_and_artifacts(report_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found or no longer available.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
