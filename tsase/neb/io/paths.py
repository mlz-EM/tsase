"""Output-path helpers for SSNEB workflows."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunLayout:
    """Normalized run directory layout for SSNEB runs."""

    run_dir: Path
    public_dir: Path
    xyz_dir: Path
    diagnostics_file: Path
    log_file: Path
    manifest_file: Path
    prepared_structures_dir: Path
    inputs_dir: Path
    images_dir: Path

    @classmethod
    def from_inputs(cls, output_dir=None, xyz_dir=None, diagnostics_file=None, log_file=None):
        run_dir = Path.cwd().resolve() if output_dir is None else Path(output_dir).expanduser().resolve()
        public_dir = run_dir
        xyz_dir = (Path("neb_xyz") if xyz_dir is None and output_dir is None else run_dir / "xyz") if xyz_dir is None else Path(xyz_dir).expanduser().resolve()
        diagnostics_file = (
            Path("neb_diagnostics.csv").resolve()
            if diagnostics_file is None and output_dir is None
            else run_dir / "diagnostics.csv"
        ) if diagnostics_file is None else Path(diagnostics_file).expanduser().resolve()
        log_file = (
            Path("fe.out").resolve()
            if log_file is None and output_dir is None
            else run_dir / "fe.out"
        ) if log_file is None else Path(log_file).expanduser().resolve()
        return cls(
            run_dir=run_dir,
            public_dir=public_dir,
            xyz_dir=xyz_dir,
            diagnostics_file=diagnostics_file,
            log_file=log_file,
            manifest_file=run_dir / "run_manifest.json",
            prepared_structures_dir=run_dir / "prepared_structures",
            inputs_dir=run_dir / "inputs",
            images_dir=run_dir / "images",
        )

    def image_workdir(self, index):
        return self.images_dir / f"image_{int(index):04d}"


def resolve_output_paths(output_dir=None, xyz_dir=None, diagnostics_file=None, log_file=None):
    """Return normalized public output paths for a NEB run."""
    layout = RunLayout.from_inputs(
        output_dir=output_dir,
        xyz_dir=xyz_dir,
        diagnostics_file=diagnostics_file,
        log_file=log_file,
    )
    return {
        "output_dir": str(layout.public_dir),
        "xyz_dir": str(layout.xyz_dir),
        "diagnostics_file": str(layout.diagnostics_file),
        "log_file": str(layout.log_file),
    }
