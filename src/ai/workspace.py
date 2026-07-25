"""Job workspace directory lifecycle manager.

This module provides thread-safe directory creation, path validation, and
isolated temporary file cleanup for individual AI inference jobs.
"""

import shutil
import threading
from pathlib import Path
from typing import Any, Self
from uuid import UUID

from backend.core.config import settings
from backend.core.logging import get_logger
from ai.exceptions import AIProcessingError

logger = get_logger(__name__)


class WorkspaceError(AIProcessingError):
    """Raised when workspace directory creation, path resolution, or cleanup fails."""


class WorkspaceManager:
    """Manage job-specific filesystem workspace lifecycle."""

    def __init__(
        self,
        request_id: UUID,
        base_workspace_dir: Path | None = None,
        visualization_dir: Path | None = None,
    ) -> None:
        """Initialize the workspace manager for a specific request.

        Args:
            request_id: Unique request UUID for the job.
            base_workspace_dir: Optional base directory for temporary workspaces.
            visualization_dir: Optional base directory for persistent 3D HTMLs.
        """

        if not request_id:
            raise WorkspaceError("Invalid request_id: UUID must be provided.")

        self.request_id = request_id
        self._lock = threading.Lock()

        self.base_workspace_dir = Path(
            base_workspace_dir or settings.outputs_directory / "workspaces"
        ).resolve()

        self.workspace_dir = (self.base_workspace_dir / str(request_id)).resolve()
        self.visualization_dir = Path(
            visualization_dir or settings.outputs_directory / "visualizations"
        ).resolve()

        self.normalized_input_path = self.workspace_dir / "input_normalized.nii.gz"
        self.tumor_mask_path = self.workspace_dir / "tumor_mask.nii.gz"
        self.anatomy_mask_path = self.workspace_dir / "anatomy_mask.nii.gz"
        self.fusion_mask_path = self.workspace_dir / "fusion_mask.nii.gz"
        self.metrics_json_path = self.workspace_dir / "metrics.json"

        self.visualization_html_path = (
            self.visualization_dir / f"{request_id}_3d.html"
        )

    def create(self) -> Path:
        """Create the job workspace and persistent visualization directories safely.

        Returns:
            Path to the job workspace directory.

        Raises:
            WorkspaceError: If directory creation fails.
        """

        with self._lock:
            try:
                self.workspace_dir.mkdir(parents=True, exist_ok=True)
                self.visualization_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Created isolated workspace directory: %s for request %s",
                    self.workspace_dir,
                    self.request_id,
                )
                logger.info(
                    "Initialized persistent visualization artifact path: %s",
                    self.visualization_html_path,
                )
                return self.workspace_dir
            except Exception as exc:
                raise WorkspaceError(
                    f"Failed to create workspace directory at {self.workspace_dir}.",
                    detail=str(exc),
                ) from exc

    def cleanup(self) -> None:
        """Clean up intermediate scratch files and remove workspace directory safely.

        This method is idempotent and safe against accidental deletion outside root.
        """

        with self._lock:
            if not self.workspace_dir.exists():
                logger.debug(
                    "Workspace directory %s already cleaned up or does not exist.",
                    self.workspace_dir,
                )
                return

            if not self._is_safe_subpath(self.workspace_dir, self.base_workspace_dir):
                logger.error(
                    "Refusing to delete unsafe workspace path: %s (not a strict subpath of %s)",
                    self.workspace_dir,
                    self.base_workspace_dir,
                )
                raise WorkspaceError(
                    f"Refusing to delete unsafe path {self.workspace_dir}.",
                    detail=f"Target path is not a strict child of {self.base_workspace_dir}.",
                )

            logger.info("Starting workspace cleanup for request %s...", self.request_id)
            try:
                shutil.rmtree(self.workspace_dir)
                logger.info("Completed workspace cleanup for request %s", self.request_id)
            except Exception as exc:
                logger.error(
                    "Failed to clean up workspace directory %s: %s",
                    self.workspace_dir,
                    exc,
                )
                raise WorkspaceError(
                    f"Failed to clean up workspace directory {self.workspace_dir}.",
                    detail=str(exc),
                ) from exc

    def __enter__(self) -> Self:
        """Context manager entry."""
        self.create()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit automatically triggers cleanup."""
        self.cleanup()

    @staticmethod
    def _is_safe_subpath(target: Path, base: Path) -> bool:
        """Verify target path is a strict descendant of base directory."""

        try:
            target_resolved = target.resolve()
            base_resolved = base.resolve()
            return target_resolved != base_resolved and base_resolved in target_resolved.parents
        except Exception:
            return False
