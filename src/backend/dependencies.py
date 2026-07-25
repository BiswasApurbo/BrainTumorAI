"""Dependency injection factories for the BrainTumorAI backend.

This module provides cached factory functions that return singleton-compatible
service instances for use with FastAPI ``Depends()``. Each factory function
is decorated with ``@lru_cache`` so that repeated calls within a single
process return the same instance, matching the caching pattern established in
``backend.core.settings.get_settings``.

The factories contain no business logic. They exist solely to create and wire
service instances, keeping the concrete construction details out of endpoint
modules.
"""

from functools import lru_cache

from backend.services.inference_service import InferenceService
from backend.services.report_service import ReportService
from backend.services.upload_service import UploadService
from ai.pipeline import AIPipeline


@lru_cache
def get_upload_service() -> UploadService:
    """Return the shared upload service instance.

    The instance uses the default upload directory derived from application
    settings.

    Returns:
        A cached upload service instance.
    """

    return UploadService()


@lru_cache
def get_report_service() -> ReportService:
    """Return the shared report management service instance.

    Returns:
        A cached report service instance.
    """

    return ReportService()


@lru_cache
def get_inference_service() -> InferenceService:
    """Return the shared inference orchestration service instance.

    The inference service receives the shared upload service and integrated
    AIPipeline so that inference requests trigger end-to-end AI execution.

    Returns:
        A cached inference service instance.
    """

    report_service = get_report_service()
    ai_pipeline = AIPipeline(report_service=report_service)
    return InferenceService(
        upload_service=get_upload_service(),
        ai_pipeline=ai_pipeline,
    )
