#!/usr/bin/env python

"""Example field-coupled SSNEB run using the local PbZrO3 CIFs and MACE model."""

import hashlib
import json
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from mace.calculators import MACECalculator
from tsase import neb
from ase.io import read, write
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import get_symmetrized_atoms


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "example" / "field_ssneb_runs"

# Input structures and model available in this repository.
STRUCTURE_POINTS = {
    0: ROOT / "example" / "Si_Pbam.cif",
    # 20: ROOT / "example" / "interpolated" / "Config#06.cif",
    # 21: ROOT / "example" / "interpolated" / "Config#05.cif",
    # 22: ROOT / "example" / "interpolated" / "Config#04.cif",
    # 23: ROOT / "example" / "interpolated" / "Config#03.cif",
    # 24: ROOT / "example" / "interpolated" / "Config#02.cif",
    # 25: ROOT / "example" / "interpolated" / "Config#01.cif",
    26: ROOT / "example" / "R3c_40atom_rot_aligned_1.cif",
}
NUM_IMAGES = 27
MODEL_PATH = ROOT / "example" / "MACE_model.model"
POLARIZATION_REFERENCE_PATH = ROOT / "example" / "Pm-3m.cif"

# Fixed ionic charges used by the enthalpy wrapper.
CHARGE_MAP = {"Pb": 2.0, "Zr": 4.0, "O": -2.0}

# Electric field in eV / (e Angstrom). Adjust as needed.
FIELD = None
FIELD_CRYSTAL = np.array([0.002, -0.001, 0.00], dtype=float)

# Cell relaxation settings.
FILTER_NAME = "ExpCellFilter"
FILTER_MASK = [1, 1, 1, 1, 1, 1]
HYDROSTATIC_STRAIN = False
CONSTANT_VOLUME = False

# NEB / optimizer settings.
SPRING = 5.0
SPRING_MIN = 5.0
SPRING_MAX = 20.0
ADAPTIVE_SPRINGS = False
METHOD = "ci"
CI_ACTIVATION_FORCE = 0.1
FMAX = 0.01
MAX_STEPS = 2000
OUTPUT_INTERVAL = 50
PLOT_PROPERTY = "P_mag"
IMAGE_UPDATE_FACTORS = None
IMAGE_UPDATE_ITERATIONS = 0
# {
#     19: 0.9,
#     20: 0.6,
#     21: 0.4,
#     22: 0.3,
#     23: 0.2,
#     24: 0.1,
#     25: 0.1,
# }
RESTART_FROM_XYZ = None #'/home/gridsan/mzhu/Tools/STEM_TOOL/tsase/example/field_ssneb_xyz/iter_0600.xyz'

def build_filter_factory():
    from ase.filters import ExpCellFilter, UnitCellFilter

    if FILTER_NAME == "ExpCellFilter":
        return lambda image: ExpCellFilter(
            image,
            mask=FILTER_MASK,
            hydrostatic_strain=HYDROSTATIC_STRAIN,
            constant_volume=CONSTANT_VOLUME,
        )
    if FILTER_NAME == "UnitCellFilter":
        return lambda image: UnitCellFilter(
            image,
            mask=FILTER_MASK,
            hydrostatic_strain=HYDROSTATIC_STRAIN,
            constant_volume=CONSTANT_VOLUME,
        )
    return None


def sha256sum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = RUNS_DIR / f"{Path(__file__).stem}_{timestamp}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix:02d}")
        suffix += 1
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "prepared_structures").mkdir()
    (run_dir / "xyz").mkdir()
    return run_dir


def snapshot_git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip()
    except Exception:
        commit = None
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.splitlines()
    except Exception:
        status = []
    return {"commit": commit, "status_short": status}


def copy_run_inputs(run_dir, structure_indices):
    copied_structures = {}
    copied_inputs = []
    inputs_dir = run_dir / "inputs"
    for index in structure_indices:
        source = STRUCTURE_POINTS[index]
        destination = inputs_dir / f"image_{index:04d}_{source.name}"
        shutil.copy2(source, destination)
        copied_structures[index] = destination
        copied_inputs.append(
            {
                "kind": "structure",
                "index": index,
                "source": str(source),
                "copied_to": str(destination),
                "sha256": sha256sum(destination),
            }
        )

    model_copy = inputs_dir / MODEL_PATH.name
    shutil.copy2(MODEL_PATH, model_copy)
    copied_inputs.append(
        {
            "kind": "model",
            "source": str(MODEL_PATH),
            "copied_to": str(model_copy),
            "sha256": sha256sum(model_copy),
        }
    )

    restart_copy = None
    if RESTART_FROM_XYZ is not None:
        restart_source = Path(RESTART_FROM_XYZ)
        restart_copy = inputs_dir / restart_source.name
        shutil.copy2(restart_source, restart_copy)
        copied_inputs.append(
            {
                "kind": "restart",
                "source": str(restart_source),
                "copied_to": str(restart_copy),
                "sha256": sha256sum(restart_copy),
            }
        )

    script_snapshot = run_dir / Path(__file__).name
    shutil.copy2(Path(__file__), script_snapshot)

    return {
        "structure_points": copied_structures,
        "model": model_copy,
        "restart": restart_copy,
        "script_snapshot": script_snapshot,
        "copied_inputs": copied_inputs,
    }


def write_prepared_structures(run_dir, structure_indices, structures):
    prepared_dir = run_dir / "prepared_structures"
    prepared_paths = []
    for index, atoms in zip(structure_indices, structures):
        output_path = prepared_dir / f"image_{index:04d}_prepared.extxyz"
        snapshot = atoms.copy()
        snapshot.calc = None
        write(output_path, snapshot, format="extxyz")
        prepared_paths.append(str(output_path))
    return prepared_paths


def write_run_manifest(
    run_dir,
    diagnostics_file,
    log_file,
    polarization_reference_path,
    field_vector,
    structure_indices,
    copied_inputs,
    prepared_paths,
):
    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_snapshot": str(run_dir / Path(__file__).name),
        "run_directory": str(run_dir),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "git": snapshot_git_state(),
        "inputs": copied_inputs,
        "prepared_structures": prepared_paths,
        "outputs": {
            "xyz_dir": str(run_dir / "xyz"),
            "diagnostics_file": str(diagnostics_file),
            "log_file": str(log_file),
            "polarization_reference": str(polarization_reference_path),
        },
        "parameters": {
            "structure_indices": structure_indices,
            "num_images": NUM_IMAGES,
            "charge_map": CHARGE_MAP,
            "field": None if FIELD is None else FIELD.tolist(),
            "field_crystal": None if FIELD_CRYSTAL is None else FIELD_CRYSTAL.tolist(),
            "field_cartesian": None if field_vector is None else field_vector.tolist(),
            "filter_name": FILTER_NAME,
            "filter_mask": FILTER_MASK,
            "hydrostatic_strain": HYDROSTATIC_STRAIN,
            "constant_volume": CONSTANT_VOLUME,
            "spring": SPRING,
            "spring_min": SPRING_MIN,
            "spring_max": SPRING_MAX,
            "adaptive_springs": ADAPTIVE_SPRINGS,
            "method": METHOD,
            "ci_activation_force": CI_ACTIVATION_FORCE,
            "fmax": FMAX,
            "max_steps": MAX_STEPS,
            "output_interval": OUTPUT_INTERVAL,
            "plot_property": PLOT_PROPERTY,
            "image_update_factors": IMAGE_UPDATE_FACTORS,
            "image_update_iterations": IMAGE_UPDATE_ITERATIONS,
            "restart_from_xyz": None if RESTART_FROM_XYZ is None else str(RESTART_FROM_XYZ),
        },
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def main():
    run_dir = make_run_directory()
    xyz_dir = run_dir / "xyz"
    diagnostics_file = run_dir / "diagnostics.csv"
    log_file = run_dir / "fe.out"
    polarization_reference_path = run_dir / "polarization_reference.cif"

    structure_indices = sorted(STRUCTURE_POINTS)
    copied = copy_run_inputs(run_dir, structure_indices)
    structures = [read(copied["structure_points"][index]) for index in structure_indices]
    print(f"Run directory: {run_dir}")

    try:
        base_calc = MACECalculator(model_paths=str(copied["model"]), device="cuda")
    except RuntimeError as exc:
        if "Found no NVIDIA driver" in str(exc):
            raise RuntimeError(
                "this example expects a CUDA-capable GPU node; submit it with "
                "`LLsub -i full -g volta:2 -T 02:00:00` and run inside the "
                "`NEB` conda environment"
            ) from exc
        raise

    for i, atoms in enumerate(structures):
        if i == 0 or i == len(structures) - 1:
            atoms.set_calculator(base_calc)
            relax_log = run_dir / f"endpoint_relax_{structure_indices[i]:04d}.log"
            relax_traj = run_dir / f"endpoint_relax_{structure_indices[i]:04d}.traj"
            optimizer = BFGS(atoms, logfile=str(relax_log), trajectory=str(relax_traj))
            optimizer.run(fmax=0.02)
        neb.attach_field_charges(atoms, CHARGE_MAP)

    atoms = structures[0]
    refined_atoms, dataset = get_symmetrized_atoms(atoms, symprec=1.0)

    print(f"Refined Space Group: {dataset.international}")
    refined_atoms.set_cell(atoms.cell, scale_atoms=False)
    refined_atoms.write(polarization_reference_path)

    polarization_reference = structures[0]
    if polarization_reference_path is not None:
        polarization_reference = neb.spatial_map(
            structures[0],
            read(polarization_reference_path),
        )

    field_vector = (
        neb.crystal_field_to_cartesian(structures[0].get_cell(), FIELD_CRYSTAL)
        if FIELD_CRYSTAL is not None
        else FIELD
    )

    prepared_paths = write_prepared_structures(run_dir, structure_indices, structures)
    write_run_manifest(
        run_dir=run_dir,
        diagnostics_file=diagnostics_file,
        log_file=log_file,
        polarization_reference_path=polarization_reference_path,
        field_vector=field_vector,
        structure_indices=structure_indices,
        copied_inputs=copied["copied_inputs"],
        prepared_paths=prepared_paths,
    )

    calc = neb.EnthalpyWrapper(
        base_calc,
        field=field_vector,
        reference_atoms=polarization_reference,
        charges=neb.build_charge_array(structures[0], charge_map=CHARGE_MAP),
    )

    for atoms in structures:
        atoms.calc = calc

    image_update_schedule = None
    if IMAGE_UPDATE_FACTORS and IMAGE_UPDATE_ITERATIONS > 0:
        image_update_schedule = [
            {
                "factors": IMAGE_UPDATE_FACTORS,
                "iterations": IMAGE_UPDATE_ITERATIONS,
            }
        ]

    band = neb.ssneb(
        structures,
        structure_indices,
        numImages=NUM_IMAGES,
        k=SPRING,
        method=METHOD,
        xyz_dir=str(xyz_dir),
        filter_factory=build_filter_factory(),
        adaptive_springs=ADAPTIVE_SPRINGS,
        kmin=SPRING_MIN,
        kmax=SPRING_MAX,
        diagnostics_file=str(diagnostics_file),
        image_update_schedule=image_update_schedule,
    )
    if copied["restart"] is not None:
        neb.load_band_configuration_from_xyz(band, str(copied["restart"]))

    optimizer = neb.fire_ssneb(
        band,
        maxmove=0.10,
        dt=0.10,
        dtmax=0.10,
        xyz_dir=str(xyz_dir),
        log_file=str(log_file),
        output_interval=OUTPUT_INTERVAL,
        plot_property=PLOT_PROPERTY,
        ci_activation_force=CI_ACTIVATION_FORCE,
    )
    optimizer.minimize(forceConverged=FMAX, maxIterations=MAX_STEPS)


if __name__ == "__main__":
    main()
