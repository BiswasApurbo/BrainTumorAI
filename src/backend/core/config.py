"""Configuration helpers for the BrainTumorAI backend.

This module exposes the shared application settings instance and lightweight
helpers for common configuration tasks. It intentionally contains no business
logic and derives all configurable values from ``settings.py``.
"""

from pathlib import Path

from backend.core.settings import Settings, get_settings


settings: Settings = get_settings()


def ensure_project_directories() -> tuple[Path, ...]:
    """Ensure configured project directories exist.

    The directories are derived from the shared settings object, making this
    helper safe to call repeatedly during application startup or tests.

    Returns:
        A tuple containing the configured directories that were ensured.
    """

    directories = (
        settings.model_directory,
        settings.checkpoint_directory,
        settings.dataset_directory,
        settings.uploads_directory,
        settings.outputs_directory,
        settings.reports_directory,
        settings.log_directory,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def get_application_version() -> str:
    """Return the configured application version.

    Returns:
        The semantic version string configured for the application.
    """

    return settings.version


def get_application_name() -> str:
    """Return the configured application name.

    Returns:
        The human-readable application name from configuration.
    """

    return settings.app_name
