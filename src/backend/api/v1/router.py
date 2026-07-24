"""Central API v1 router for the BrainTumorAI backend.

This module aggregates all version 1 endpoint routers into a single
``APIRouter`` instance for registration by the application entry point.
"""

from fastapi import APIRouter

from .endpoints import health, inference, reports, upload


router = APIRouter()

router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)
router.include_router(
    upload.router,
    prefix="/upload",
    tags=["Upload"],
)
router.include_router(
    inference.router,
    prefix="/inference",
    tags=["Inference"],
)
router.include_router(
    reports.router,
    prefix="/reports",
    tags=["Reports"],
)
