"""Output-path helpers for SSNEB workflows."""

from pathlib import Path


def resolve_output_paths(output_dir=None, xyz_dir=None, diagnostics_file=None, log_file=None):
    """Return normalized output paths for a NEB run."""
    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        if xyz_dir is None:
            xyz_dir = output_dir / "xyz"
        if diagnostics_file is None:
            diagnostics_file = output_dir / "diagnostics.csv"
        if log_file is None:
            log_file = output_dir / "fe.out"

    xyz_dir = Path("neb_xyz") if xyz_dir is None else Path(xyz_dir).expanduser()
    diagnostics_file = (
        Path("neb_diagnostics.csv")
        if diagnostics_file is None
        else Path(diagnostics_file).expanduser()
    )
    log_file = Path("fe.out") if log_file is None else Path(log_file).expanduser()

    return {
        "output_dir": None if output_dir is None else str(output_dir),
        "xyz_dir": str(xyz_dir),
        "diagnostics_file": str(diagnostics_file),
        "log_file": str(log_file),
    }

