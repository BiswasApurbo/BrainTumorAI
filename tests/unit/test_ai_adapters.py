"""Unit tests for AI model adapters (nnUNet and SynthSeg)."""

from pathlib import Path

import pytest

from ai.adapters.nnunet_adapter import NNUNetAdapter
from ai.adapters.synthseg_adapter import SynthSegAdapter
from ai.contracts import NormalizedBraTSCase, NormalizedScan
from ai.exceptions import ModelInferenceError


def test_nnunet_adapter_brats_case_success(tmp_path: Path) -> None:
    """Test NNUNetAdapter stages all four BraTS modalities and produces segmentation."""

    flair = tmp_path / "normalized_flair.nii.gz"
    t1 = tmp_path / "normalized_t1.nii.gz"
    t1ce = tmp_path / "normalized_t1ce.nii.gz"
    t2 = tmp_path / "normalized_t2.nii.gz"

    for p in (flair, t1, t1ce, t2):
        p.write_bytes(b"scan_data")

    case = NormalizedBraTSCase(
        patient_id="BraTS20_Training_001",
        normalized_flair=flair,
        normalized_t1=t1,
        normalized_t1ce=t1ce,
        normalized_t2=t2,
        dimensions=(240, 240, 155),
        voxel_spacing=(1.0, 1.0, 1.0),
    )

    mask_path = tmp_path / "output" / "tumor_mask.nii.gz"
    adapter = NNUNetAdapter()
    result = adapter.predict(case, mask_path)

    assert result.mask_path == mask_path
    assert "et" in result.subregion_voxel_counts
    assert mask_path.exists()


def test_nnunet_adapter_brats_case_missing_modality(tmp_path: Path) -> None:
    """Test NNUNetAdapter raises ModelInferenceError when a BraTS modality file is missing."""

    flair = tmp_path / "normalized_flair.nii.gz"
    t1 = tmp_path / "normalized_t1.nii.gz"
    t1ce = tmp_path / "normalized_t1ce.nii.gz"
    t2 = tmp_path / "normalized_t2.nii.gz"

    # Only create three of four modalities — omit T1CE
    for p in (flair, t1, t2):
        p.write_bytes(b"scan_data")

    case = NormalizedBraTSCase(
        patient_id="BraTS20_Training_001",
        normalized_flair=flair,
        normalized_t1=t1,
        normalized_t1ce=t1ce,
        normalized_t2=t2,
        dimensions=(240, 240, 155),
        voxel_spacing=(1.0, 1.0, 1.0),
    )

    mask_path = tmp_path / "output" / "tumor_mask.nii.gz"
    adapter = NNUNetAdapter()

    with pytest.raises(ModelInferenceError, match="Missing required BraTS modalities"):
        adapter.predict(case, mask_path)


def test_nnunet_adapter_success(tmp_path: Path) -> None:
    """Test NNUNetAdapter produces expected tumor segmentation mask."""

    input_path = tmp_path / "input.nii.gz"
    input_path.write_bytes(b"scan_data")

    scan = NormalizedScan(
        input_path=input_path,
        original_path=input_path,
        dimensions=(240, 240, 155),
        voxel_spacing=(1.0, 1.0, 1.0),
    )

    mask_path = tmp_path / "tumor_mask.nii.gz"
    adapter = NNUNetAdapter()
    result = adapter.predict(scan, mask_path)

    assert result.mask_path == mask_path
    assert "et" in result.subregion_voxel_counts
    assert mask_path.exists()


def test_nnunet_adapter_missing_input(tmp_path: Path) -> None:
    """Test NNUNetAdapter raises ModelInferenceError when input scan does not exist."""

    input_path = tmp_path / "non_existent.nii.gz"
    scan = NormalizedScan(
        input_path=input_path,
        original_path=input_path,
        dimensions=(240, 240, 155),
        voxel_spacing=(1.0, 1.0, 1.0),
    )

    mask_path = tmp_path / "tumor_mask.nii.gz"
    adapter = NNUNetAdapter()

    with pytest.raises(ModelInferenceError, match="Input scan for nnUNet segmentation does not exist"):
        adapter.predict(scan, mask_path)


def test_synthseg_adapter_success(tmp_path: Path) -> None:
    """Test SynthSegAdapter produces expected brain anatomy segmentation mask."""

    input_path = tmp_path / "input.nii.gz"
    input_path.write_bytes(b"scan_data")

    scan = NormalizedScan(
        input_path=input_path,
        original_path=input_path,
        dimensions=(240, 240, 155),
        voxel_spacing=(1.0, 1.0, 1.0),
    )

    mask_path = tmp_path / "anatomy_mask.nii.gz"
    adapter = SynthSegAdapter()
    result = adapter.predict(scan, mask_path)

    assert result.mask_path == mask_path
    assert len(result.structure_voxel_counts) > 0
    assert mask_path.exists()
