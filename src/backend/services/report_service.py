"""Diagnostic report management service for the BrainTumorAI backend.

This module provides in-memory report storage, retrieval, listing, deletion,
and existence checking for AI-generated diagnostic reports. It does not perform
inference, preprocessing, visualization, or any other pipeline operations.
Report content is supplied by external callers and stored as-is.

The public API is designed so that the in-memory store can be replaced by a
persistent database backend without changing method signatures or return types.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

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
    """Manage diagnostic reports for the BrainTumorAI backend."""

    def __init__(self) -> None:
        """Initialize the report management service."""

        self._reports: dict[UUID, Report] = {}
        self._upload_index: dict[UUID, UUID] = {}
        self._lock = threading.Lock()
        logger.info("ReportService initialised.")

    def store_report(
        self,
        upload_id: UUID | str,
        content: dict[str, Any],
        status: str = "completed",
    ) -> ReportMetadata:
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
            "Report %s stored for upload %s.",
            report.report_id,
            normalized_upload_id,
        )

        return ReportMetadata(
            report_id=report.report_id,
            upload_id=report.upload_id,
            status=report.status,
            created_at=report.created_at,
        )

    def get_report(self, upload_id: UUID | str) -> Report:
        normalized_upload_id = self._normalize_uuid(upload_id, "upload_id")
        with self._lock:
            report_id = self._upload_index.get(normalized_upload_id)
            if report_id is None:
                raise KeyError(
                    f"No report found for upload {normalized_upload_id}."
                )
            return self._reports[report_id]

    def list_reports(self) -> list[ReportMetadata]:
        with self._lock:
            return [
                ReportMetadata(
                    report_id=report.report_id,
                    upload_id=report.upload_id,
                    status=report.status,
                    created_at=report.created_at,
                )
                for report in self._reports.values()
            ]

    def delete_report(self, upload_id: UUID | str) -> bool:
        normalized_upload_id = self._normalize_uuid(upload_id, "upload_id")
        with self._lock:
            report_id = self._upload_index.get(normalized_upload_id)
            if report_id is None:
                raise KeyError(
                    f"No report found for upload {normalized_upload_id}."
                )
            del self._reports[report_id]
            del self._upload_index[normalized_upload_id]

        logger.info(
            "Report %s deleted for upload %s.",
            report_id,
            normalized_upload_id,
        )

        return True

    def report_exists(self, upload_id: UUID | str) -> bool:
        normalized_upload_id = self._normalize_uuid(upload_id, "upload_id")
        with self._lock:
            return normalized_upload_id in self._upload_index

    @staticmethod
    def _normalize_uuid(value: UUID | str, field_name: str) -> UUID:
        """Normalize a UUID value.

        Args:
            value: UUID or UUID string to normalize.
            field_name: Field name used in validation error messages.

        Returns:
            Normalized UUID value.

        Raises:
            ValueError: If the supplied value is blank or not a valid UUID.
        """

        if isinstance(value, UUID):
            return value

        candidate = value.strip()
        if not candidate:
            raise ValueError(f"{field_name} must be provided.")

        return UUID(candidate)
