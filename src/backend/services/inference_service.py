"""Inference workflow orchestration service.

This module coordinates the complete BrainTumorAI inference workflow from upload
validation through preprocessing, nnU-Net segmentation, SynthSeg anatomy
segmentation, volumetric postprocessing, 3D visualization, and report generation.
"""

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from backend.core.logging import get_logger
from backend.services.upload_service import UploadService
from ai.exceptions import (
    ModelInferenceError,
    PostProcessingError,
    PreprocessingError,
    ReportGenerationError,
    VisualizationError,
)
from ai.pipeline import AIPipeline

logger = get_logger(__name__)


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
        report_id: Unique identifier for the generated diagnostic report.
        visualization_path: Path to the generated 3D Plotly HTML visualization.
        tumor_mask_path: Path to the generated tumor segmentation mask.
        anatomy_mask_path: Path to the generated anatomy segmentation mask.
        volumetric_analysis: Quantitative volumetric measurements dictionary.
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
    report_id: UUID | None = None
    visualization_path: Path | None = None
    tumor_mask_path: Path | None = None
    anatomy_mask_path: Path | None = None
    volumetric_analysis: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class InferenceMetadata:
    """Public metadata returned when an inference request is created.

    Attributes:
        request_id: Unique identifier assigned to the inference request.
        upload_id: Unique identifier for the uploaded medical image.
        status: Current lifecycle state for the request.
        input_path: Filesystem path to the uploaded input file.
        created_at: UTC-aware timestamp for request creation.
        report_id: Optional unique identifier for the generated report.
        visualization_path: Optional path to the 3D Plotly HTML visualization.
        tumor_mask_path: Optional path to the tumor segmentation mask.
        anatomy_mask_path: Optional path to the anatomy segmentation mask.
        volumetric_analysis: Optional quantitative volumetric measurements dictionary.
    """

    request_id: UUID
    upload_id: UUID
    status: PipelineStatus
    input_path: Path
    created_at: datetime
    report_id: UUID | None = None
    visualization_path: Path | None = None
    tumor_mask_path: Path | None = None
    anatomy_mask_path: Path | None = None
    volumetric_analysis: dict[str, Any] | None = None


class InferenceService:
    """Coordinate BrainTumorAI inference workflow requests.

    The service owns orchestration state and delegates execution to the injected
    ``AIPipeline`` or individual pipeline collaborators in a thread-safe manner.
    """

    def __init__(
        self,
        upload_service: UploadService | None = None,
        preprocessing_service: PipelineStep | None = None,
        segmentation_service: PipelineStep | None = None,
        postprocessing_service: PipelineStep | None = None,
        visualization_service: PipelineStep | None = None,
        report_service: PipelineStep | None = None,
        ai_pipeline: AIPipeline | None = None,
    ) -> None:
        """Initialize the inference orchestration service.

        Args:
            upload_service: Service used to locate and validate uploaded files.
            preprocessing_service: Optional collaborator for preprocessing.
            segmentation_service: Optional collaborator for segmentation.
            postprocessing_service: Optional collaborator for postprocessing.
            visualization_service: Optional collaborator for 3D visualization.
            report_service: Optional collaborator for report generation.
            ai_pipeline: Integrated end-to-end AIPipeline instance.
        """

        self.upload_service = upload_service or UploadService()
        self.preprocessing_service = preprocessing_service
        self.segmentation_service = segmentation_service
        self.postprocessing_service = postprocessing_service
        self.visualization_service = visualization_service
        self.report_service = report_service
        self.ai_pipeline = ai_pipeline
        self._jobs: dict[UUID, InferenceJob] = {}
        self._lock = threading.Lock()

    def create_inference_request(self, upload_id: UUID | str) -> InferenceMetadata:
        """Create a queued inference job for an uploaded medical image and execute pipeline.

        Args:
            upload_id: Upload UUID or UUID string to process.

        Returns:
            Metadata for the newly created inference request.

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
        with self._lock:
            self._jobs[job.request_id] = job

        logger.info("Created inference job %s for upload %s", job.request_id, job.upload_id)

        # Auto-execute pipeline if integrated AIPipeline is available
        if self.ai_pipeline is not None:
            self.run_pipeline(job.request_id)

        return InferenceMetadata(
            request_id=job.request_id,
            upload_id=job.upload_id,
            status=job.status,
            input_path=job.input_path,
            created_at=job.created_at,
            report_id=job.report_id,
            visualization_path=job.visualization_path,
            tumor_mask_path=job.tumor_mask_path,
            anatomy_mask_path=job.anatomy_mask_path,
            volumetric_analysis=job.volumetric_analysis,
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

    def create_inference_request_from_path(
        self,
        input_path: Path,
        upload_id: UUID | None = None,
    ) -> InferenceMetadata:
        """Create an inference job from a filesystem path (file or BraTS directory).

        This method supports both:
        - A single NIfTI file path (legacy behavior).
        - A BraTS patient directory containing FLAIR, T1, T1CE, and T2 modalities.

        Args:
            input_path: Path to a NIfTI file or a BraTS patient directory.
            upload_id: Optional upload UUID. Generated if not supplied.

        Returns:
            Metadata for the newly created inference request.

        Raises:
            AIProcessingError: If the path is unreadable, the directory is empty,
                or required BraTS modalities are missing.
            FileNotFoundError: If the input path does not exist.
        """

        from ai.exceptions import AIProcessingError

        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_path.is_dir():
            self._validate_brats_directory(input_path)
        elif not input_path.is_file():
            raise AIProcessingError(
                f"Input path is neither a file nor a directory: {input_path}",
                detail=str(input_path),
            )

        resolved_upload_id = upload_id or uuid4()

        job = InferenceJob(
            request_id=uuid4(),
            upload_id=resolved_upload_id,
            input_path=input_path,
        )
        with self._lock:
            self._jobs[job.request_id] = job

        logger.info(
            "Created inference job %s from path %s (upload: %s)",
            job.request_id,
            input_path,
            job.upload_id,
        )

        if self.ai_pipeline is not None:
            self.run_pipeline(job.request_id)

        return InferenceMetadata(
            request_id=job.request_id,
            upload_id=job.upload_id,
            status=job.status,
            input_path=job.input_path,
            created_at=job.created_at,
            report_id=job.report_id,
            visualization_path=job.visualization_path,
            tumor_mask_path=job.tumor_mask_path,
            anatomy_mask_path=job.anatomy_mask_path,
            volumetric_analysis=job.volumetric_analysis,
        )

    @staticmethod
    def _validate_brats_directory(directory: Path) -> None:
        """Validate that a directory is a readable BraTS patient folder.

        Checks:
        - Directory is readable.
        - Directory is not empty.
        - All four required BraTS modalities (flair, t1, t1ce, t2) are present.

        Args:
            directory: Path to a candidate BraTS patient directory.

        Raises:
            AIProcessingError: If validation fails.
        """

        from ai.exceptions import AIProcessingError

        if not os.access(directory, os.R_OK):
            raise AIProcessingError(
                f"BraTS patient directory is not readable: {directory}",
                detail=str(directory),
            )

        children = list(directory.iterdir())
        if not children:
            raise AIProcessingError(
                f"BraTS patient directory is empty: {directory}",
                detail=str(directory),
            )

        required_modalities = ("flair", "t1", "t1ce", "t2")
        missing = []
        for modality in required_modalities:
            found = any(
                child.name.endswith(f"_{modality}.nii")
                or child.name.endswith(f"_{modality}.nii.gz")
                for child in children
                if child.is_file()
            )
            if not found:
                missing.append(modality.upper())

        if missing:
            raise AIProcessingError(
                f"BraTS patient directory {directory.name} is missing required modalities: {', '.join(missing)}.",
                detail=f"Directory: {directory}, Missing: {missing}",
            )

    def run_pipeline(self, request_id: UUID | str) -> PipelineStatus:
        """Coordinate the complete inference workflow for a request.

        Args:
            request_id: Inference request UUID or UUID string to run.

        Returns:
            Final pipeline status for the request.

        Raises:
            ValueError: If the request identifier is invalid.
            KeyError: If no inference job exists for the request.
        """

        job = self._get_job(request_id)
        if job.status is PipelineStatus.COMPLETED:
            return job.status

        logger.info("Executing inference workflow for job %s", job.request_id)

        if self.ai_pipeline is not None:
            try:
                self._set_status(job, PipelineStatus.PREPROCESSING)
                pipeline_result = self.ai_pipeline.run(job)

                job.report_id = pipeline_result.report_id
                job.visualization_path = pipeline_result.visualization_path
                job.volumetric_analysis = {
                    "tumor_volumes_cm3": pipeline_result.volumetric_analysis.tumor_volumes_cm3,
                    "total_brain_volume_cm3": pipeline_result.volumetric_analysis.total_brain_volume_cm3,
                    "tumor_to_brain_ratio_percent": pipeline_result.volumetric_analysis.tumor_to_brain_ratio_percent,
                    "affected_structures": pipeline_result.volumetric_analysis.affected_structures,
                }
                self._set_status(job, PipelineStatus.COMPLETED)
                logger.info("Successfully completed AI pipeline execution for job %s", job.request_id)
            except (
                PreprocessingError,
                ModelInferenceError,
                PostProcessingError,
                VisualizationError,
                ReportGenerationError,
            ) as exc:
                logger.error("Domain pipeline failure for job %s: %s", job.request_id, exc)
                self._set_status(job, PipelineStatus.FAILED)
                job.error_message = str(exc)
                raise
            except Exception as exc:
                logger.error("Unexpected error during AI pipeline for job %s: %s", job.request_id, exc)
                self._set_status(job, PipelineStatus.FAILED)
                job.error_message = f"Unexpected pipeline failure: {exc}"
                raise
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
            logger.info("Successfully completed modular pipeline steps for job %s", job.request_id)
        except Exception as exc:
            logger.error("Error executing modular pipeline step for job %s: %s", job.request_id, exc)
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
        logger.info("Cancelled inference job %s", job.request_id)
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
        with self._lock:
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

    def _set_status(self, job: InferenceJob, status: PipelineStatus) -> None:
        """Update an inference job status and timestamp under thread-safe lock.

        Args:
            job: Inference job to update.
            status: New lifecycle status.
        """

        with self._lock:
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

        candidate = str(value).strip()
        if not candidate:
            raise ValueError(f"{field_name} must be provided.")

        return UUID(candidate)
