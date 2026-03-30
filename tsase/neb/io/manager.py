"""Unified output management for SSNEB workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from ase import io

from .diagnostics import append_diagnostics_rows, initialize_diagnostics_file
from .energy_profile import ENTRY_METADATA


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


def _unique_run_dir(base_dir, run_name, timestamp=True):
    base_dir = Path(base_dir).expanduser().resolve()
    suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if timestamp else ""
    candidate = base_dir / f"{run_name}{suffix}"
    run_dir = candidate
    counter = 1
    while run_dir.exists():
        run_dir = base_dir / f"{run_name}{suffix}_{counter:02d}"
        counter += 1
    return run_dir


@dataclass(frozen=True)
class RunPaths:
    """Normalized run-directory layout for SSNEB outputs."""

    run_dir: Path
    config_dir: Path
    outputs_dir: Path
    stages_dir: Path
    transitions_dir: Path
    work_dir: Path
    path_dir: Path
    energy_dir: Path
    stem_dir: Path
    diagnostics_file: Path
    log_file: Path
    manifest_file: Path
    input_config_file: Path
    resolved_config_file: Path

    @classmethod
    def from_run_dir(cls, run_dir, *, manifest_file=None):
        root = Path(run_dir).expanduser().resolve()
        config_dir = root / "config"
        outputs_dir = root / "outputs"
        manifest_path = config_dir / "manifest.json" if manifest_file is None else Path(manifest_file)
        return cls(
            run_dir=root,
            config_dir=config_dir,
            outputs_dir=outputs_dir,
            stages_dir=root / "stages",
            transitions_dir=root / "transitions",
            work_dir=root / "work",
            path_dir=outputs_dir / "path",
            energy_dir=outputs_dir / "energy",
            stem_dir=outputs_dir / "stem",
            diagnostics_file=outputs_dir / "diagnostics.csv",
            log_file=outputs_dir / "fe.out",
            manifest_file=manifest_path.expanduser().resolve(),
            input_config_file=config_dir / "input.yaml",
            resolved_config_file=config_dir / "resolved.yaml",
        )

    def image_workdir(self, index):
        return self.work_dir / f"image_{int(index):04d}"


class OutputManager:
    """Own the user-facing output layout and emitted artifacts for one run."""

    def __init__(self, paths, *, active=True, settings=None):
        self.paths = paths
        self.is_active = bool(active)
        self._log_handle = None
        defaults = {
            "diagnostics": True,
            "path_snapshots": True,
            "energy_profile": True,
            "energy_profile_csv": True,
            "energy_profile_plot": True,
            "energy_profile_entries": ["enthalpy_adjusted", "intrinsic_energy", "field_energy", "polarization_mag"],
            "stem": False,
            "final_path_snapshot": True,
        }
        if settings:
            defaults.update(dict(settings))
        self.settings = defaults

    @classmethod
    def create(cls, *, base_dir="neb_runs", run_name="ssneb", timestamp=True, active=True, settings=None):
        run_dir = _unique_run_dir(base_dir, run_name, timestamp=timestamp)
        return cls(RunPaths.from_run_dir(run_dir), active=active, settings=settings)

    @classmethod
    def from_run_dir(cls, run_dir, *, active=True, settings=None, manifest_file=None):
        return cls(
            RunPaths.from_run_dir(run_dir, manifest_file=manifest_file),
            active=active,
            settings=settings,
        )

    def child(self, run_dir, *, manifest_file=None):
        return type(self).from_run_dir(
            run_dir,
            active=self.is_active,
            settings=self.settings,
            manifest_file=manifest_file,
        )

    @property
    def run_dir(self):
        return self.paths.run_dir

    @property
    def manifest_file(self):
        return self.paths.manifest_file

    @property
    def diagnostics_file(self):
        return self.paths.diagnostics_file

    @property
    def log_file(self):
        return self.paths.log_file

    @property
    def path_dir(self):
        return self.paths.path_dir

    @property
    def energy_dir(self):
        return self.paths.energy_dir

    @property
    def stem_dir(self):
        return self.paths.stem_dir

    @property
    def xyz_dir(self):
        return self.paths.path_dir

    def as_public_paths(self):
        return {
            "run_dir": str(self.paths.run_dir),
            "outputs_dir": str(self.paths.outputs_dir),
            "path_dir": str(self.paths.path_dir),
            "energy_dir": str(self.paths.energy_dir),
            "stem_dir": str(self.paths.stem_dir),
            "diagnostics_file": str(self.paths.diagnostics_file),
            "log_file": str(self.paths.log_file),
        }

    def write_text(self, path, text):
        destination = Path(path)
        if not self.is_active:
            return str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(text), encoding="utf-8")
        return str(destination)

    def write_json(self, path, payload):
        destination = Path(path)
        if not self.is_active:
            return str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return str(destination)

    def copy_files(self, files, *, subdir="inputs"):
        destination_dir = self.paths.run_dir / subdir
        if not self.is_active:
            return []
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

    def copy_input_config(self, source_path):
        source = Path(source_path).expanduser().resolve()
        if not self.is_active:
            return str(self.paths.input_config_file)
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, self.paths.input_config_file)
        return str(self.paths.input_config_file)

    def snapshot_script(self, script_path, destination_name=None):
        source_path = Path(script_path).expanduser().resolve()
        destination_path = self.paths.config_dir / (destination_name or source_path.name)
        if not self.is_active:
            return str(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return str(destination_path)

    def write_manifest(self, *, config=None, inputs=None, script_path=None, extra_metadata=None, git_cwd=None):
        if not self.is_active:
            return str(self.paths.manifest_file)
        manifest = {
            "created_at": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "python_executable": sys.executable,
            "run_directory": str(self.paths.run_dir),
            "outputs": self.as_public_paths(),
        }
        if git_cwd is not None:
            manifest["git"] = _git_snapshot(git_cwd)
        if script_path is not None:
            manifest["script"] = str(Path(script_path).expanduser().resolve())
        if config is not None:
            manifest["config"] = config
        if inputs is not None:
            manifest["inputs"] = inputs
        if extra_metadata:
            manifest.update(extra_metadata)
        return self.write_json(self.paths.manifest_file, manifest)

    def initialize_diagnostics(self):
        if self.is_active and self.settings["diagnostics"]:
            initialize_diagnostics_file(str(self.paths.diagnostics_file))

    def write_diagnostics(self, iteration, images, frozen_images):
        if self.is_active and self.settings["diagnostics"]:
            append_diagnostics_rows(str(self.paths.diagnostics_file), iteration, images, frozen_images)

    def make_snapshot_images(self, path):
        images = []
        for index, image in enumerate(path):
            snapshot = image.copy()
            snapshot.calc = None
            snapshot.info = deepcopy(getattr(image, "info", {}))
            snapshot.info.pop("spatial_map_order", None)
            snapshot.info["neb_image"] = index
            if hasattr(image, "u"):
                try:
                    snapshot.info["energy"] = float(image.u)
                except Exception:
                    pass
            if hasattr(image, "polarization_c_per_m2"):
                try:
                    snapshot.info["polarization_c_per_m2"] = [
                        float(value) for value in image.polarization_c_per_m2
                    ]
                except Exception:
                    pass
            if hasattr(image, "f"):
                try:
                    snapshot.arrays["forces"] = image.f.copy()
                except Exception:
                    pass
            images.append(snapshot)
        return images

    def write_path_snapshot(self, path, iteration, stem_sequence_writer):
        if not self.is_active or not self.settings["path_snapshots"]:
            return None
        self.paths.path_dir.mkdir(parents=True, exist_ok=True)
        images = self.make_snapshot_images(path)
        outfile = self.paths.path_dir / f"iter_{int(iteration):04d}.xyz"
        io.write(str(outfile), images, format="extxyz")
        if self.settings["stem"]:
            result = stem_sequence_writer(
                xyz_file=outfile,
                output_dir=self.paths.stem_dir,
                iteration=iteration,
                emit_npy=False,
            )
            if isinstance(result, dict) and result.get("status") != "ok":
                diagnostics_file = result.get("diagnostics_file")
                print(
                    "Projected STEM visualization skipped for "
                    f"iter_{int(iteration):04d}; see {diagnostics_file}"
                )
        return str(outfile)

    def write_final_path_snapshot(self, path):
        if not self.is_active or not self.settings["final_path_snapshot"]:
            return None
        self.paths.path_dir.mkdir(parents=True, exist_ok=True)
        outfile = self.paths.path_dir / "final.xyz"
        io.write(str(outfile), self.make_snapshot_images(path), format="extxyz")
        return str(outfile)

    def write_energy_profile_csv(self, iteration, entries, rows):
        if (
            not self.is_active
            or not self.settings["energy_profile"]
            or not self.settings["energy_profile_csv"]
            or not entries
        ):
            return None
        self.paths.energy_dir.mkdir(parents=True, exist_ok=True)
        outfile = self.paths.energy_dir / f"profile_iter_{int(iteration):04d}.csv"
        header = ["image"] + [ENTRY_METADATA[entry]["csv_header"] for entry in entries]
        lines = [",".join(header)]
        for row in rows:
            values = [str(int(row["image"]))]
            for entry in entries:
                values.append(f"{float(row[entry]):.16e}")
            lines.append(",".join(values))
        outfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(outfile)

    def save_energy_plot(self, fig, iteration):
        if (
            not self.is_active
            or not self.settings["energy_profile"]
            or not self.settings["energy_profile_plot"]
        ):
            return None
        self.paths.energy_dir.mkdir(parents=True, exist_ok=True)
        outfile = self.paths.energy_dir / f"profile_iter_{int(iteration):04d}.png"
        fig.tight_layout()
        fig.savefig(str(outfile), dpi=150)
        return str(outfile)

    def open_log(self, header, separator):
        if not self.is_active:
            return
        self.paths.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.paths.log_file.open("a", encoding="utf-8")
        self._log_handle.write(header + "\n")
        self._log_handle.write(separator + "\n")

    def write_status_block(self, header, output, separator):
        if not self.is_active:
            return
        print("-------------------------SSNEB------------------------------")
        print(header)
        print(output)
        print(separator)
        if self._log_handle is not None:
            self._log_handle.write(header + "\n")
            self._log_handle.write(output + "\n")
            self._log_handle.write(separator + "\n")

    def write_summary_header(self, header, separator):
        if not self.is_active:
            return
        print("-----------------------SSNEB Finished------------------------------")
        print(header)
        print(separator)
        if self._log_handle is not None:
            self._log_handle.write("-----------------------SSNEB Finished------------------------------\n")
            self._log_handle.write(header + "\n")
            self._log_handle.write(separator + "\n")

    def write_summary_line(self, line):
        if not self.is_active:
            return
        print(line)
        if self._log_handle is not None:
            self._log_handle.write(line + "\n")

    def close(self):
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def resolve_output_paths(output_dir=None):
    """Return normalized public output paths for a run directory."""
    run_dir = Path.cwd().resolve() if output_dir is None else Path(output_dir).expanduser().resolve()
    return OutputManager.from_run_dir(run_dir).as_public_paths()
