"""Diagnostic report management service for the BrainTumorAI backend.

This module provides in-memory report storage, retrieval, listing, deletion,
and existence checking for AI-generated diagnostic reports. It does not perform
inference, preprocessing, visualization, or any other pipeline operations.
Report content is supplied by external callers and stored as-is.

The public API is designed so that the in-memory store can be replaced by a
persistent database backend without changing method signatures or return types.
"""

import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Report:
    """Internal representation of a diagnostic report."""

    report_id: UUID
    upload_id: UUID
    status: str
    content: dict[str, Any]
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass(frozen=True, slots=True)
class ReportMetadata:
    """Public metadata returned when a report is created or listed."""

    report_id: UUID
    upload_id: UUID
    status: str
    created_at: datetime


class ReportService:
    """Manage diagnostic reports and their transient filesystem artifacts."""

    def __init__(self) -> None:
        """Initialize the report management service."""

        self._reports: dict[UUID, Report] = {}
        self._upload_index: dict[UUID, UUID] = {}
        self._lock = threading.Lock()
        logger.info("ReportService initialized.")

    def store_report(
        self,
        upload_id: UUID | str,
        content: dict[str, Any],
        status: str = "completed",
    ) -> ReportMetadata:
        """Store a new diagnostic report."""

        normalized_upload_id = self._normalize_uuid(upload_id, "upload_id")

        with self._lock:
            if normalized_upload_id in self._upload_index:
                raise ValueError(
                    f"A report already exists for upload {normalized_upload_id}."
                )

            report = Report(
                report_id=uuid4(),
                upload_id=normalized_upload_id,
                status=status,
                content=content,
            )
            self._reports[report.report_id] = report
            self._upload_index[normalized_upload_id] = report.report_id

        logger.info(
            "Report %s created for upload %s.",
            report.report_id,
            normalized_upload_id,
        )

        return ReportMetadata(
            report_id=report.report_id,
            upload_id=report.upload_id,
            status=report.status,
            created_at=report.created_at,
        )

    def get_report(self, identifier: UUID | str) -> Report:
        """Retrieve a report by report_id or upload_id."""

        normalized_id = self._normalize_uuid(identifier, "report_id")
        with self._lock:
            if normalized_id in self._reports:
                return self._reports[normalized_id]
            if normalized_id in self._upload_index:
                report_id = self._upload_index[normalized_id]
                return self._reports[report_id]

            raise KeyError(f"No report found for identifier {normalized_id}.")

    def get_report_path(self, identifier: UUID | str) -> Path:
        """Locate the standalone 3D HTML visualization file for a report."""

        report = self.get_report(identifier)
        vis_path_raw = (
            report.content.get("visualization_path")
            if isinstance(report.content, dict)
            else None
        )

        if vis_path_raw:
            path = Path(vis_path_raw)
            if path.is_file() and path.exists():
                return path

        # Fallback to standard convention in outputs/visualizations/
        fallback_path = settings.outputs_directory / "visualizations" / f"{report.upload_id}_3d.html"
        if fallback_path.is_file() and fallback_path.exists():
            return fallback_path

        raise FileNotFoundError(
            f"HTML visualization file for report {report.report_id} was not found on disk."
        )

    def claim_report_for_download(self, identifier: UUID | str) -> tuple[Report, Path]:
        """Atomically claim a report for download.

        Guarantees that if multiple concurrent requests (e.g. two browser tabs)
        attempt to download the report simultaneously, only ONE request can succeed
        and claim the report.
        """

        normalized_id = self._normalize_uuid(identifier, "report_id")
        with self._lock:
            report = None
            if normalized_id in self._reports:
                report = self._reports[normalized_id]
            elif normalized_id in self._upload_index:
                rep_id = self._upload_index[normalized_id]
                report = self._reports.get(rep_id)

            if report is None or report.status in ("downloading", "purged"):
                raise KeyError(f"Report {normalized_id} not found or already downloaded.")

            # Resolve file path
            vis_path_raw = (
                report.content.get("visualization_path")
                if isinstance(report.content, dict)
                else None
            )
            path = (
                Path(vis_path_raw)
                if vis_path_raw
                else (settings.outputs_directory / "visualizations" / f"{report.upload_id}_3d.html")
            )

            if not path.is_file() or not path.exists():
                raise FileNotFoundError(
                    f"Report HTML file missing on disk for {normalized_id}."
                )

            # Atomically transition status to downloading to prevent concurrent downloads
            report.status = "downloading"
            return report, path

    def list_reports(self) -> list[ReportMetadata]:
        """List all active diagnostic reports."""

        with self._lock:
            return [
                ReportMetadata(
                    report_id=report.report_id,
                    upload_id=report.upload_id,
                    status=report.status,
                    created_at=report.created_at,
                )
                for report in self._reports.values()
                if report.status != "purged"
            ]

    def delete_report_and_artifacts(self, identifier: UUID | str) -> bool:
        """Safely delete the report metadata and all associated disk artifacts."""

        normalized_id = self._normalize_uuid(identifier, "report_id")

        with self._lock:
            report = None
            if normalized_id in self._reports:
                report = self._reports.pop(normalized_id)
                self._upload_index.pop(report.upload_id, None)
            elif normalized_id in self._upload_index:
                rep_id = self._upload_index.pop(normalized_id)
                report = self._reports.pop(rep_id, None)

        if report is None:
            logger.debug("Report %s already cleaned up or does not exist.", normalized_id)
            return False

        logger.info(
            "Report %s downloaded. Beginning automatic artifact cleanup...",
            report.report_id,
        )

        # 1. Delete HTML visualization file
        try:
            vis_path_raw = report.content.get("visualization_path") if isinstance(report.content, dict) else None
            if vis_path_raw:
                html_path = Path(vis_path_raw)
                if html_path.exists():
                    html_path.unlink(missing_ok=True)
                    logger.info("Deleted visualization artifact: %s", html_path.name)
        except Exception as exc:
            logger.error("Failed to delete visualization artifact for %s: %s", report.report_id, exc)

        # 2. Delete Uploaded files directory
        try:
            upload_dir = settings.uploads_directory / str(report.upload_id)
            if upload_dir.exists() and upload_dir.is_dir():
                shutil.rmtree(upload_dir, ignore_errors=True)
                logger.info("Deleted uploaded patient directory: %s", upload_dir.name)
        except Exception as exc:
            logger.error("Failed to delete upload directory for %s: %s", report.upload_id, exc)

        # 3. Clean any residual workspaces
        try:
            workspace_dir = settings.outputs_directory / "workspaces" / str(report.upload_id)
            if workspace_dir.exists() and workspace_dir.is_dir():
                shutil.rmtree(workspace_dir, ignore_errors=True)
        except Exception as exc:
            logger.debug("No residual workspace for %s: %s", report.upload_id, exc)

        logger.info("Successfully completed automatic cleanup for report %s.", report.report_id)
        return True

    def delete_report(self, identifier: UUID | str) -> bool:
        """Alias for delete_report_and_artifacts."""
        return self.delete_report_and_artifacts(identifier)

    def report_exists(self, identifier: UUID | str) -> bool:
        """Check if report exists by report_id or upload_id."""

        try:
            normalized_id = self._normalize_uuid(identifier, "report_id")
        except ValueError:
            return False

        with self._lock:
            return normalized_id in self._reports or normalized_id in self._upload_index

    def cleanup_expired_reports(self, max_age_hours: int = 24) -> int:
        """Purge all reports and orphan artifacts older than max_age_hours."""

        cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).timestamp()
        expired_ids: list[UUID] = []

        with self._lock:
            for report in list(self._reports.values()):
                if report.created_at.timestamp() < cutoff_time:
                    expired_ids.append(report.report_id)

        purged_count = 0
        for rep_id in expired_ids:
            logger.info("Purging expired report %s (created > %d hours ago)", rep_id, max_age_hours)
            if self.delete_report_and_artifacts(rep_id):
                purged_count += 1

        # Also sweep disk directories for any orphan artifacts from prior restarts
        for folder_name in ("visualizations", "uploads", "workspaces"):
            target_dir = settings.outputs_directory / folder_name
            if not target_dir.exists() or not target_dir.is_dir():
                continue
            for item in target_dir.iterdir():
                try:
                    if item.stat().st_mtime < cutoff_time:
                        if item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                        else:
                            item.unlink(missing_ok=True)
                        logger.info("Swept orphan disk artifact: %s/%s", folder_name, item.name)
                        purged_count += 1
                except Exception as exc:
                    logger.debug("Failed sweeping %s: %s", item, exc)

        return purged_count

    @staticmethod
    def _normalize_uuid(value: UUID | str, field_name: str) -> UUID:
        """Normalize a UUID value."""

        if isinstance(value, UUID):
            return value

        candidate = value.strip()
        if not candidate:
            raise ValueError(f"{field_name} must be provided.")

        return UUID(candidate)
