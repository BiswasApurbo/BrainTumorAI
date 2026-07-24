"""Application configuration powered by Pydantic Settings.

This module defines the runtime settings used by the backend application.
Configuration values are loaded from environment variables and an optional
``.env`` file while retaining sensible defaults for local development.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the BrainTumorAI backend application.

    Settings are organized by application area and can be overridden with
    environment variables or values defined in the project ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="BrainTumorAI", description="Application name.")
    version: str = Field(default="0.1.0", description="Application version.")
    debug: bool = Field(default=False, description="Enable debug mode.")
    environment: str = Field(
        default="development",
        description="Runtime environment name.",
    )

    # API
    api_prefix: str = Field(default="/api/v1", description="Base API route prefix.")
    host: str = Field(default="0.0.0.0", description="API bind host.")
    port: int = Field(default=8000, description="API bind port.")

    # CORS
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed CORS origins.",
    )
    allowed_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        description="Allowed CORS methods.",
    )
    allowed_headers: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS headers.",
    )

    # Model
    model_directory: Path = Field(
        default=Path("models"),
        description="Directory containing model artifacts.",
    )
    checkpoint_directory: Path = Field(
        default=Path("models/checkpoints"),
        description="Directory containing model checkpoints.",
    )
    device: str = Field(default="cpu", description="Model inference device.")
    batch_size: int = Field(default=1, description="Default model batch size.")

    # Data
    dataset_directory: Path = Field(
        default=Path("datasets"),
        description="Directory containing datasets.",
    )
    uploads_directory: Path = Field(
        default=Path("outputs/uploads"),
        description="Directory for uploaded files.",
    )
    outputs_directory: Path = Field(
        default=Path("outputs"),
        description="Directory for generated outputs.",
    )
    reports_directory: Path = Field(
        default=Path("outputs/reports"),
        description="Directory for generated reports.",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Application log level.")
    log_directory: Path = Field(
        default=Path("outputs/logs"),
        description="Directory for application logs.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings.

    Caching prevents repeated parsing of environment variables and the ``.env``
    file while providing a single configuration instance for the application.
    """

    return Settings()
