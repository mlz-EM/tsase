"""Helpers for reproducible SSNEB run directories and output files."""

import hashlib
import json
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ase.io import write

from .paths import resolve_output_paths


def _sha256sum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot(cwd):
    root = Path(cwd)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        ).stdout.strip()
    except Exception:
        commit = None
    try:
        status_short = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        ).stdout.splitlines()
    except Exception:
        status_short = []
    return {"commit": commit, "status_short": status_short}


class RunArtifacts:
    """Manage a self-contained run directory for SSNEB workflows."""

    def __init__(
        self,
        run_dir,
        xyz_dir=None,
        diagnostics_file=None,
        log_file=None,
        manifest_name="run_manifest.json",
    ):
        paths = resolve_output_paths(
            output_dir=run_dir,
            xyz_dir=xyz_dir,
            diagnostics_file=diagnostics_file,
            log_file=log_file,
        )
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.output_dir = self.run_dir
        self.xyz_dir = Path(paths["xyz_dir"])
        self.diagnostics_file = Path(paths["diagnostics_file"])
        self.log_file = Path(paths["log_file"])
        self.manifest_file = self.run_dir / manifest_name

    @classmethod
    def create(
        cls,
        base_dir="neb_runs",
        run_name="ssneb",
        timestamp=True,
        create_subdir=True,
        xyz_subdir="xyz",
        diagnostics_name="diagnostics.csv",
        log_name="fe.out",
    ):
        base_dir = Path(base_dir).expanduser().resolve()
        if create_subdir:
            suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if timestamp else ""
            candidate = base_dir / f"{run_name}{suffix}"
            run_dir = candidate
            counter = 1
            while run_dir.exists():
                run_dir = base_dir / f"{run_name}{suffix}_{counter:02d}"
                counter += 1
        else:
            run_dir = base_dir

        (run_dir / xyz_subdir).mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            xyz_dir=run_dir / xyz_subdir,
            diagnostics_file=run_dir / diagnostics_name,
            log_file=run_dir / log_name,
        )

    def as_neb_paths(self):
        return {
            "output_dir": str(self.output_dir),
            "xyz_dir": str(self.xyz_dir),
            "diagnostics_file": str(self.diagnostics_file),
            "log_file": str(self.log_file),
        }

    def copy_inputs(self, files, subdir="inputs"):
        destination_dir = self.run_dir / subdir
        destination_dir.mkdir(parents=True, exist_ok=True)
        records = []

        if isinstance(files, dict):
            items = files.items()
        else:
            items = [(None, entry) for entry in files]

        for label, source in items:
            source_path = Path(source).expanduser().resolve()
            destination_name = source_path.name if label is None else f"{label}_{source_path.name}"
            destination_path = destination_dir / destination_name
            shutil.copy2(source_path, destination_path)
            records.append(
                {
                    "label": label,
                    "source": str(source_path),
                    "copied_to": str(destination_path),
                    "sha256": _sha256sum(destination_path),
                }
            )
        return records

    def snapshot_script(self, script_path, destination_name=None):
        source_path = Path(script_path).expanduser().resolve()
        destination_path = self.run_dir / (destination_name or source_path.name)
        shutil.copy2(source_path, destination_path)
        return str(destination_path)

    def write_structures(
        self,
        structures,
        indices=None,
        subdir="prepared_structures",
        prefix="image",
        format="extxyz",
    ):
        destination_dir = self.run_dir / subdir
        destination_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for offset, atoms in enumerate(structures):
            image_index = offset if indices is None else indices[offset]
            destination_path = destination_dir / f"{prefix}_{int(image_index):04d}.{format}"
            snapshot = atoms.copy()
            snapshot.calc = None
            write(destination_path, snapshot, format=format)
            output_paths.append(str(destination_path))
        return output_paths

    def write_manifest(
        self,
        *,
        config=None,
        inputs=None,
        prepared_structures=None,
        script_path=None,
        extra_metadata=None,
        git_cwd=None,
    ):
        manifest = {
            "created_at": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "python_executable": sys.executable,
            "run_directory": str(self.run_dir),
            "outputs": self.as_neb_paths(),
        }
        if git_cwd is not None:
            manifest["git"] = _git_snapshot(git_cwd)
        if script_path is not None:
            manifest["script"] = str(Path(script_path).expanduser().resolve())
        if config is not None:
            manifest["config"] = config
        if inputs is not None:
            manifest["inputs"] = inputs
        if prepared_structures is not None:
            manifest["prepared_structures"] = prepared_structures
        if extra_metadata:
            manifest.update(extra_metadata)

        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_file.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return str(self.manifest_file)

