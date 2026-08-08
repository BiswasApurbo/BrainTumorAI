"""FastAPI application entry point for the BrainTumorAI backend.

This module creates the backend application instance, attaches the application
lifespan manager, and exposes lightweight root and health endpoints.
"""

from typing import Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.api.v1.router import router as api_v1_router
from backend.core.config import settings
from backend.core.lifespan import lifespan
from backend.exceptions.handlers import register_exception_handlers
from backend.middleware.cors import setup_cors


ResponsePayload = dict[str, Any]


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug,
    lifespan=lifespan,
)

setup_cors(app)
register_exception_handlers(app)
app.include_router(api_v1_router, prefix=settings.api_prefix)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
templates = Jinja2Templates(directory=FRONTEND_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """Render the BrainTumorAI frontend.

    Returns:
        HTML template response for the main interface.
    """
    return templates.TemplateResponse(name="index.html", context={"request": request}, request=request)


@app.get("/health")
async def health() -> ResponsePayload:
    """Return the current application health status.

    Returns:
        A JSON-serializable payload with health status and application
        metadata.
    """

    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.version,
    }
