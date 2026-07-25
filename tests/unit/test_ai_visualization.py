"""Unit tests for 3D reconstruction, Plotly HTML renderer, and workspace manager."""

from pathlib import Path
from uuid import uuid4

from ai.contracts import Mesh3DData
from ai.visualization.plotly_renderer import PlotlyRenderer
from ai.visualization.reconstruction_3d import Reconstruction3D
from ai.workspace import WorkspaceManager


def test_workspace_manager_lifecycle(tmp_path: Path) -> None:
    """Test workspace creation and cleanup lifecycle."""

    request_id = uuid4()
    base_ws = tmp_path / "workspaces"
    base_vis = tmp_path / "vis"

    wm = WorkspaceManager(
        request_id=request_id,
        base_workspace_dir=base_ws,
        visualization_dir=base_vis,
    )

    created_dir = wm.create()
    assert created_dir.exists()
    assert wm.visualization_dir.exists()

    wm.cleanup()
    assert not created_dir.exists()


def test_reconstruction_3d_extraction(tmp_path: Path) -> None:
    """Test 3D surface mesh extraction returns expected surface layers."""

    tumor_mask = tmp_path / "tumor.nii"
    anatomy_mask = tmp_path / "anatomy.nii"

    recon = Reconstruction3D()
    meshes = recon.extract_meshes(tumor_mask, anatomy_mask)

    assert len(meshes) == 4
    labels = {m.label for m in meshes}
    assert "Enhancing Tumor (ET)" in labels
    assert "Peritumoral Edema (ED)" in labels


def test_plotly_renderer_html(tmp_path: Path) -> None:
    """Test PlotlyRenderer generates a valid standalone HTML file."""

    output_html = tmp_path / "visualization_3d.html"
    mesh = Mesh3DData(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces=[[0, 1, 2]],
        label="Test Region",
        color="#FF0000",
        opacity=0.8,
    )

    renderer = PlotlyRenderer()
    res_path = renderer.render_html([mesh], output_html, title="Test 3D Title")

    assert res_path == output_html
    assert output_html.exists()
    content = output_html.read_text(encoding="utf-8")
    assert "Test 3D Title" in content or "Test Region" in content
