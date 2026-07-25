"""Interactive 3D Plotly HTML visualization renderer.

This module converts 3D surface meshes into standalone interactive Plotly HTML visualizations.
"""

from pathlib import Path

from backend.core.logging import get_logger
from ai.contracts import Mesh3DData
from ai.exceptions import VisualizationError

logger = get_logger(__name__)


class PlotlyRenderer:
    """Render 3D surface meshes into standalone interactive Plotly HTML files."""

    def __init__(self) -> None:
        """Initialize the Plotly renderer."""

    def render_html(
        self,
        meshes: list[Mesh3DData],
        output_html_path: Path,
        title: str = "BrainTumorAI 3D Interactive Reconstruction",
    ) -> Path:
        """Render 3D mesh objects into a standalone HTML file.

        Args:
            meshes: List of Mesh3DData surface mesh objects.
            output_html_path: Destination path to write the standalone HTML.
            title: Title for the 3D visualization.

        Returns:
            Path to the generated HTML file.

        Raises:
            VisualizationError: If HTML rendering or file writing fails.
        """

        output_html_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                rendered_traces = 0

                for mesh in meshes:
                    if not mesh.vertices or not mesh.faces:
                        logger.debug("Skipping empty mesh trace '%s'.", mesh.label)
                        continue

                    vx = [v[0] for v in mesh.vertices]
                    vy = [v[1] for v in mesh.vertices]
                    vz = [v[2] for v in mesh.vertices]

                    fi = [f[0] for f in mesh.faces]
                    fj = [f[1] for f in mesh.faces]
                    fk = [f[2] for f in mesh.faces]

                    fig.add_trace(
                        go.Mesh3d(
                            x=vx,
                            y=vy,
                            z=vz,
                            i=fi,
                            j=fj,
                            k=fk,
                            color=mesh.color,
                            opacity=mesh.opacity,
                            name=mesh.label,
                            showlegend=True,
                            lighting=dict(
                                ambient=0.5,
                                diffuse=0.8,
                                roughness=0.5,
                                specular=0.2,
                            ),
                        )
                    )
                    rendered_traces += 1

                fig.update_layout(
                    title=dict(text=title, font=dict(size=18, color="#FFFFFF")),
                    template="plotly_dark",
                    scene=dict(
                        xaxis=dict(title="X (mm)", backgroundcolor="#1E1E2E", gridcolor="#333344"),
                        yaxis=dict(title="Y (mm)", backgroundcolor="#1E1E2E", gridcolor="#333344"),
                        zaxis=dict(title="Z (mm)", backgroundcolor="#1E1E2E", gridcolor="#333344"),
                        aspectmode="data",
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.2),
                            center=dict(x=0, y=0, z=0),
                        ),
                    ),
                    paper_bgcolor="#11111B",
                    margin=dict(l=0, r=0, b=0, t=50),
                )

                fig.write_html(
                    str(output_html_path),
                    include_plotlyjs="cdn",
                    full_html=True,
                )

                logger.info(
                    "Successfully rendered Plotly 3D visualization with %d traces to %s",
                    rendered_traces,
                    output_html_path.name,
                )
            except ImportError:
                logger.debug("plotly not installed; writing offline standalone HTML template fallback.")
                self._write_html_fallback(meshes, output_html_path, title)

            return output_html_path

        except Exception as exc:
            raise VisualizationError(
                f"Failed to render 3D Plotly visualization to {output_html_path.name}.",
                detail=str(exc),
            ) from exc

    @staticmethod
    def _write_html_fallback(
        meshes: list[Mesh3DData],
        output_html_path: Path,
        title: str,
    ) -> None:
        """Write standalone HTML container when plotly package is unavailable in runtime."""

        legend_items = "".join(
            f'<li><span style="display:inline-block;width:14px;height:14px;background-color:{m.color};margin-right:8px;border-radius:3px;"></span>{m.label} (opacity: {m.opacity})</li>'
            for m in meshes
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ background-color: #11111B; color: #CDD6F4; font-family: system-ui, sans-serif; padding: 2rem; margin: 0; }}
        .card {{ background: rgba(30, 30, 46, 0.8); border-radius: 12px; padding: 2rem; border: 1px solid rgba(255,255,255,0.1); max-width: 800px; margin: 0 auto; }}
        h1 {{ color: #89B4FA; font-size: 1.5rem; margin-top: 0; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ font-size: 1.1rem; margin-bottom: 0.8rem; display: flex; align-items: center; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>{title}</h1>
        <p>3D Isosurface Reconstruction Mesh Layers ({len(meshes)} structures extracted):</p>
        <ul>
            {legend_items}
        </ul>
    </div>
</body>
</html>
"""
        output_html_path.write_text(html_content, encoding="utf-8")
