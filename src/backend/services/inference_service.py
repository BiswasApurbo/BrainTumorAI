"""Inference workflow orchestration service.

This module coordinates the BrainTumorAI inference workflow without
implementing medical image preprocessing, model inference, segmentation,
postprocessing, visualization, quantitative analysis, or report generation.
Future pipeline services can be injected behind small runtime protocols.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from backend.services.upload_service import UploadService


class PipelineStatus(str, Enum):
    """Supported inference request lifecycle states."""

    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"
    VISUALIZATION = "visualization"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStep(Protocol):
    """Protocol for injectable inference pipeline collaborators."""

    def run(self, job: "InferenceJob") -> None:
        """Run a pipeline step for an inference job.

        Args:
            job: Inference job metadata shared across pipeline steps.
        """


@dataclass(slots=True)
class InferenceJob:
    """Metadata for an inference workflow request.

    Attributes:
        request_id: Unique identifier assigned to the inference request.
        upload_id: Unique identifier for the uploaded medical image.
        input_path: Filesystem path to the uploaded input file.
        status: Current lifecycle state for the request.
        created_at: UTC-aware timestamp for request creation.
        updated_at: UTC-aware timestamp for the latest status update.
        error_message: Optional failure reason for the request.
    """

    request_id: UUID
    upload_id: UUID
    input_path: Path
    status: PipelineStatus = PipelineStatus.QUEUED
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceMetadata:
    """Public metadata returned when an inference request is created.

    Attributes:
        request_id: Unique identifier assigned to the inference request.
        upload_id: Unique identifier for the uploaded medical image.
        status: Current lifecycle state for the request.
        input_path: Filesystem path to the uploaded input file.
        created_at: UTC-aware timestamp for request creation.
    """

    request_id: UUID
    upload_id: UUID
    status: PipelineStatus
    input_path: Path
    created_at: datetime


class InferenceService:
    """Coordinate BrainTumorAI inference workflow requests.

    The service owns orchestration state and delegates file validation to an
    injected ``UploadService``. Future preprocessing, segmentation,
    postprocessing, visualization, and report services can be injected through
    the constructor without changing the public API.
    """

    def __init__(
        self,
        upload_service: UploadService | None = None,
        preprocessing_service: PipelineStep | None = None,
        segmentation_service: PipelineStep | None = None,
        postprocessing_service: PipelineStep | None = None,
        visualization_service: PipelineStep | None = None,
        report_service: PipelineStep | None = None,
    ) -> None:
        """Initialize the inference orchestration service.

        Args:
            upload_service: Service used to locate and validate uploaded files.
            preprocessing_service: Optional future preprocessing collaborator.
            segmentation_service: Optional future segmentation collaborator.
            postprocessing_service: Optional future postprocessing collaborator.
            visualization_service: Optional future visualization collaborator.
            report_service: Optional future report generation collaborator.
        """

        self.upload_service = upload_service or UploadService()
        self.preprocessing_service = preprocessing_service
        self.segmentation_service = segmentation_service
        self.postprocessing_service = postprocessing_service
        self.visualization_service = visualization_service
        self.report_service = report_service
        self._jobs: dict[UUID, InferenceJob] = {}

    def create_inference_request(self, upload_id: UUID | str) -> InferenceMetadata:
        """Create a queued inference job for an uploaded medical image.

        Args:
            upload_id: Upload UUID or UUID string to process.

        Returns:
            Metadata for the newly queued inference request.

        Raises:
            ValueError: If the upload identifier or file format is invalid.
            FileNotFoundError: If the uploaded file cannot be found.
        """

        normalized_upload_id, input_path = self.validate_input(upload_id)
        job = InferenceJob(
            request_id=uuid4(),
            upload_id=normalized_upload_id,
            input_path=input_path,
        )
        self._jobs[job.request_id] = job

        return InferenceMetadata(
            request_id=job.request_id,
            upload_id=job.upload_id,
            status=job.status,
            input_path=job.input_path,
            created_at=job.created_at,
        )

    def validate_input(self, upload_id: UUID | str) -> tuple[UUID, Path]:
        """Validate that an upload exists and uses a supported file format.

        Args:
            upload_id: Upload UUID or UUID string to validate.

        Returns:
            A tuple containing the normalized upload UUID and uploaded file
            path.

        Raises:
            ValueError: If the upload identifier or file extension is invalid.
            FileNotFoundError: If the uploaded file cannot be found.
        """

        normalized_upload_id = self._normalize_uuid(upload_id, "upload_id")
        input_path = self.upload_service.get_uploaded_file(normalized_upload_id)
        self.upload_service.validate_extension(input_path.name)
        return normalized_upload_id, input_path

    def run_pipeline(self, request_id: UUID | str) -> PipelineStatus:
        """Coordinate the complete inference workflow for a request.

        This method updates job status and invokes any injected pipeline
        collaborators. It does not implement preprocessing, segmentation,
        postprocessing, visualization, quantitative analysis, or reporting.

        Args:
            request_id: Inference request UUID or UUID string to run.

        Returns:
            Final pipeline status for the request.

        Raises:
            ValueError: If the request identifier is invalid.
            KeyError: If no inference job exists for the request.
        """

        job = self._get_job(request_id)
        if job.status is not PipelineStatus.QUEUED:
            return job.status

        try:
            self._run_step(
                job,
                PipelineStatus.PREPROCESSING,
                self.preprocessing_service,
            )
            self._run_step(
                job,
                PipelineStatus.INFERENCE,
                self.segmentation_service,
            )
            self._run_step(
                job,
                PipelineStatus.POSTPROCESSING,
                self.postprocessing_service,
            )
            self._run_step(
                job,
                PipelineStatus.VISUALIZATION,
                self.visualization_service,
            )
            self._run_step(
                job,
                PipelineStatus.REPORT_GENERATION,
                self.report_service,
            )
            self._set_status(job, PipelineStatus.COMPLETED)
        except Exception as exc:
            self._set_status(job, PipelineStatus.FAILED)
            job.error_message = str(exc)
            raise

        return job.status

    def get_status(self, request_id: UUID | str) -> PipelineStatus:
        """Return the current status for an inference request.

        Args:
            request_id: Inference request UUID or UUID string.

        Returns:
            Current lifecycle state for the inference request.

        Raises:
            ValueError: If the request identifier is invalid.
            KeyError: If no inference job exists for the request.
        """

        return self._get_job(request_id).status

    def cancel_request(self, request_id: UUID | str) -> PipelineStatus:
        """Cancel a queued inference request.

        Args:
            request_id: Inference request UUID or UUID string.

        Returns:
            Updated request status.

        Raises:
            ValueError: If the request identifier is invalid or the request is
            not queued.
            KeyError: If no inference job exists for the request.
        """

        job = self._get_job(request_id)
        if job.status is not PipelineStatus.QUEUED:
            raise ValueError("Only queued inference requests can be cancelled.")

        self._set_status(job, PipelineStatus.CANCELLED)
        return job.status

    def _get_job(self, request_id: UUID | str) -> InferenceJob:
        """Return an inference job by request identifier.

        Args:
            request_id: Inference request UUID or UUID string.

        Returns:
            Matching inference job.

        Raises:
            ValueError: If the request identifier is invalid.
            KeyError: If no inference job exists for the request.
        """

        normalized_request_id = self._normalize_uuid(request_id, "request_id")
        try:
            return self._jobs[normalized_request_id]
        except KeyError as exc:
            raise KeyError("Inference request was not found.") from exc

    def _run_step(
        self,
        job: InferenceJob,
        status: PipelineStatus,
        service: PipelineStep | None,
    ) -> None:
        """Set a pipeline status and invoke an injected service if present.

        Args:
            job: Inference job being coordinated.
            status: Lifecycle status represented by the pipeline step.
            service: Optional collaborator for the pipeline step.
        """

        self._set_status(job, status)
        if service is not None:
            service.run(job)

    @staticmethod
    def _set_status(job: InferenceJob, status: PipelineStatus) -> None:
        """Update an inference job status and timestamp.

        Args:
            job: Inference job to update.
            status: New lifecycle status.
        """

        job.status = status
        job.updated_at = datetime.now(timezone.utc)

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
