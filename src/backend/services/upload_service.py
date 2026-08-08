"""Uploaded medical image file management service.

This module provides upload storage, lookup, extension validation, and safe
deletion helpers for medical imaging files. It is independent of API routing
and does not perform preprocessing, inference, segmentation, or reporting.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from backend.core.config import settings


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd"}
)
CHUNK_SIZE_BYTES: int = 1024 * 1024


@dataclass(frozen=True, slots=True)
class UploadMetadata:
    """Metadata describing a stored upload.

    Attributes:
        upload_id: Unique identifier assigned to the upload.
        upload_directory: Directory where the upload was stored.
    """

    upload_id: UUID
    upload_directory: Path


class UploadService:
    """Service responsible for uploaded medical image file management.

    The service validates supported medical image extensions, stores uploads
    with UUID-based filenames, retrieves stored files by upload identifier, and
    deletes uploaded files safely from the configured uploads directory.
    """

    def __init__(self, upload_directory: Path | None = None) -> None:
        """Initialize the upload service.

        Args:
            upload_directory: Optional upload directory override. When omitted,
            the directory configured in application settings is used.
        """

        self.upload_directory = Path(
            upload_directory or settings.uploads_directory
        )

    async def save_upload(
        self,
        flair: UploadFile,
        t1: UploadFile,
        t1ce: UploadFile,
        t2: UploadFile,
    ) -> UploadMetadata:
        """Validate and persist four uploaded BraTS medical imaging files.

        Args:
            flair: FLAIR modality file.
            t1: T1 modality file.
            t1ce: T1CE modality file.
            t2: T2 modality file.

        Returns:
            Metadata describing the stored upload directory.

        Raises:
            ValueError: If a file has an unsupported extension or is empty.
            OSError: If files cannot be written to storage.
        """

        upload_id = uuid4()
        case_dir = self.upload_directory / str(upload_id)
        case_dir.mkdir(parents=True, exist_ok=True)

        for modality_name, file in [("flair", flair), ("t1", t1), ("t1ce", t1ce), ("t2", t2)]:
            original_filename = self._get_safe_original_filename(file)
            extension = self.validate_extension(original_filename)
            stored_filename = f"{upload_id}_{modality_name}{extension}"
            destination = case_dir / stored_filename
            
            file_size_bytes = await self._write_upload(file, destination)
            if file_size_bytes == 0:
                destination.unlink(missing_ok=True)
                raise ValueError(f"Uploaded file {modality_name} is empty.")

        return UploadMetadata(
            upload_id=upload_id,
            upload_directory=case_dir,
        )

    def validate_extension(self, filename: str) -> str:
        """Validate a medical image filename extension.

        Args:
            filename: Filename whose extension should be validated.

        Returns:
            Normalized supported extension, preserving compound extensions such
            as ``.nii.gz``.

        Raises:
            ValueError: If the filename extension is unsupported.
        """

        normalized_filename = filename.lower()
        for extension in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
            if normalized_filename.endswith(extension):
                return extension

        raise ValueError("Unsupported file extension.")

    def get_uploaded_directory(self, upload_id: UUID | str) -> Path:
        """Locate a stored uploaded directory by upload identifier.

        Args:
            upload_id: Upload UUID or UUID string to locate.

        Returns:
            Path to the stored uploaded directory.

        Raises:
            ValueError: If the upload identifier is invalid.
            FileNotFoundError: If no matching upload exists.
        """

        normalized_upload_id = self._normalize_upload_id(upload_id)
        upload_path = self.upload_directory / normalized_upload_id
        if not upload_path.is_dir():
            raise FileNotFoundError("Uploaded directory was not found.")

        return upload_path

    def delete_upload(self, upload_id: UUID | str) -> bool:
        """Delete a stored uploaded file safely.

        Args:
            upload_id: Upload UUID or UUID string to delete.

        Returns:
            ``True`` when a stored upload was deleted.

        Raises:
            ValueError: If the upload identifier is invalid.
            FileNotFoundError: If no matching upload exists.
            OSError: If the file cannot be deleted.
        """

        upload_path = self.get_uploaded_directory(upload_id)
        import shutil
        shutil.rmtree(upload_path)
        return True

    async def _write_upload(
        self,
        file: UploadFile,
        destination: Path,
    ) -> int:
        """Write an uploaded file to disk and return its byte size.

        Args:
            file: FastAPI upload file object to persist.
            destination: Destination path inside the uploads directory.

        Returns:
            Number of bytes written to disk.
        """

        file_size_bytes = 0

        stored_file = await asyncio.to_thread(destination.open, "wb")
        try:
            while chunk := await file.read(CHUNK_SIZE_BYTES):
                file_size_bytes += len(chunk)
                await asyncio.to_thread(stored_file.write, chunk)
        finally:
            await asyncio.to_thread(stored_file.close)

        return file_size_bytes

    def _find_upload_path(self, upload_id: str) -> Path | None:
        """Find a stored upload path for a normalized upload identifier.

        Args:
            upload_id: Canonical upload UUID string.

        Returns:
            Matching upload path when present, otherwise ``None``.
        """

        return next(
            (
                path
                for path in self.upload_directory.glob(f"{upload_id}.*")
                if path.is_file()
            ),
            None,
        )

    @staticmethod
    def _get_safe_original_filename(file: UploadFile) -> str:
        """Return a sanitized original filename from an uploaded file.

        Args:
            file: FastAPI upload file object.

        Returns:
            Filename without client-supplied directory components.

        Raises:
            ValueError: If the uploaded file does not include a valid filename.
        """

        if not file.filename:
            raise ValueError("Uploaded file must include a filename.")

        original_filename = Path(file.filename.replace("\\", "/")).name
        if not original_filename:
            raise ValueError("Uploaded file must include a valid filename.")

        return original_filename

    @staticmethod
    def _normalize_upload_id(upload_id: UUID | str) -> str:
        """Normalize an upload identifier to a canonical UUID string.

        Args:
            upload_id: Upload UUID or UUID string.

        Returns:
            Canonical upload UUID string.

        Raises:
            ValueError: If the upload identifier is invalid.
        """

        if isinstance(upload_id, UUID):
            return str(upload_id)

        candidate = upload_id.strip()
        if not candidate:
            raise ValueError("upload_id must be provided.")

        return str(UUID(candidate))
