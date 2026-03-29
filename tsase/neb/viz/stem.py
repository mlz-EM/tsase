"""Lightweight projected-frame visualization helpers for SSNEB outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class StemAnalysisError(RuntimeError):
    """Raised when projected-frame analysis fails."""


@dataclass
class ProjectedFrameAnalysis:
    """Simple projected-frame analysis result."""

    frame_index: int
    cell_xy: np.ndarray
    xy: np.ndarray
    species: tuple[str, ...]
    diagnostics: dict[str, object]


def analyze_projected_neb_image(atoms, frame_index=0):
    """Project one image into the xy plane for lightweight visualization."""
    cell = np.asarray(atoms.get_cell(), dtype=float)
    cell_xy = np.asarray([[cell[0, 0], cell[0, 1]], [cell[1, 0], cell[1, 1]]], dtype=float)
    xy = np.asarray(atoms.get_positions(), dtype=float)[:, :2]
    return ProjectedFrameAnalysis(
        frame_index=int(frame_index),
        cell_xy=cell_xy,
        xy=xy,
        species=tuple(atoms.get_chemical_symbols()),
        diagnostics={"natoms": len(atoms)},
    )


def render_projected_frame(analysis, output_path=None):
    """Render one projected frame to disk when matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise StemAnalysisError("matplotlib is required for frame rendering") from exc

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(analysis.xy[:, 0], analysis.xy[:, 1], s=35, c="tab:blue")
    for index, symbol in enumerate(analysis.species):
        ax.text(analysis.xy[index, 0], analysis.xy[index, 1], symbol, fontsize=6)
    ax.set_title(f"Projected NEB frame {analysis.frame_index}")
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_projected_neb_sequence(images, xyz_dir, iteration):
    """Render a simple projected sequence for one saved NEB iteration."""
    frame_dir = Path(xyz_dir) / f"stem_iter_{int(iteration):04d}"
    try:
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            analysis = analyze_projected_neb_image(image, frame_index=index)
            render_projected_frame(
                analysis,
                output_path=frame_dir / f"frame_{index:04d}.png",
            )
    except Exception as exc:
        diagnostics_file = frame_dir / "stem_diagnostics.txt"
        diagnostics_file.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_file.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {
            "status": "skipped",
            "diagnostics_file": str(diagnostics_file),
        }

    return {"status": "ok", "frame_dir": str(frame_dir)}

