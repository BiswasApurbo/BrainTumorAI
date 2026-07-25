"""Unit and integration tests for the full AIPipeline end-to-end flow."""

from pathlib import Path
from uuid import uuid4

from backend.services.inference_service import InferenceJob
from backend.services.report_service import ReportService
from ai.pipeline import AIPipeline


def test_ai_pipeline_end_to_end(tmp_path: Path) -> None:
    """Test executing the complete AIPipeline end-to-end."""

    # Create dummy raw input file
    input_file = tmp_path / "brain_scan.nii.gz"
    input_file.write_bytes(b"dummy_raw_brain_scan_nifti_bytes")

    job = InferenceJob(
        request_id=uuid4(),
        upload_id=uuid4(),
        input_path=input_file,
    )

    report_service = ReportService()
    pipeline = AIPipeline(report_service=report_service)

    result = pipeline.run(job)

    assert result.request_id == job.request_id
    assert result.upload_id == job.upload_id
    assert result.visualization_path.exists()
    assert report_service.report_exists(job.upload_id) is True

    report = report_service.get_report(job.upload_id)
    assert report.content["upload_id"] == str(job.upload_id)
    assert "whole_tumor_cm3" in report.content["volumetric_analysis"]


def test_ai_pipeline_brats_case_end_to_end(tmp_path: Path) -> None:
    """Test executing the complete AIPipeline with a BraTS patient directory."""

    patient_dir = tmp_path / "BraTS20_Training_999"
    patient_dir.mkdir()

    for suffix in ("flair", "t1", "t1ce", "t2", "seg"):
        (patient_dir / f"BraTS20_Training_999_{suffix}.nii").write_bytes(
            b"dummy_nifti_data"
        )

    job = InferenceJob(
        request_id=uuid4(),
        upload_id=uuid4(),
        input_path=patient_dir,
    )

    report_service = ReportService()
    pipeline = AIPipeline(report_service=report_service)

    result = pipeline.run(job)

    assert result.request_id == job.request_id
    assert result.upload_id == job.upload_id
    assert result.visualization_path.exists()
    assert report_service.report_exists(job.upload_id) is True

