#!/usr/bin/env python

"""Example field-coupled SSNEB run using the local PbZrO3 CIFs and MACE model."""

from pathlib import Path

import numpy as np
from tsase import neb
from tsase.neb.runtime import load_mace_calculator
from ase.io import read
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import get_symmetrized_atoms


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "example" / "field_ssneb_runs"

# Input structures and model available in this repository.
STRUCTURE_POINTS = {
    0: ROOT / "example" / "Pbam_SC_NC.cif",
    # 20: ROOT / "example" / "interpolated" / "Config#06.cif",
    # 21: ROOT / "example" / "interpolated" / "Config#05.cif",
    # 22: ROOT / "example" / "interpolated" / "Config#04.cif",
    # 23: ROOT / "example" / "interpolated" / "Config#03.cif",
    # 24: ROOT / "example" / "interpolated" / "Config#02.cif",
    # 25: ROOT / "example" / "interpolated" / "Config#01.cif",
    26: ROOT / "example" / "Cluster#2_SC.cif",
}
NUM_IMAGES = 27
MODEL_PATH = ROOT / "example" / "MACE_model.model"
POLARIZATION_REFERENCE_PATH = ROOT / "example" / "Pm-3m.cif"

# Fixed ionic charges used by the enthalpy wrapper.
CHARGE_MAP = {"Pb": 2.0, "Zr": 4.0, "O": -2.0}

# Electric field in eV / (e Angstrom). Adjust as needed.
FIELD = None
FIELD_CRYSTAL = np.array([-0.001, 0.0, 0.00], dtype=float)

# Cell relaxation settings.
FILTER_NAME = "ExpCellFilter"
# ASE Voigt mask order: [xx, yy, zz, yz, xz, xy].
# Fix the ac plane by freezing a, c, and beta (xz shear).
FILTER_MASK = [0, 1, 0, 1, 0, 1]
HYDROSTATIC_STRAIN = False
CONSTANT_VOLUME = False

# NEB / optimizer settings.
SPRING = 5.0
SPRING_MIN = 5.0
SPRING_MAX = 20.0
ADAPTIVE_SPRINGS = False
METHOD = "ci"
CI_ACTIVATION_FORCE = 0.5
FMAX = 0.01
MAX_STEPS = 5000
OUTPUT_INTERVAL = 200
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
RESTART_FROM_XYZ = None #'/home/gridsan/mzhu/Tools/STEM_TOOL/tsase/example/iter_2000.xyz'

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


def standardize_cell_for_ssneb(atoms):
    """Rotate the structure into TSASE's expected lower-triangular cell form."""
    reduced_cell, transform = atoms.cell.standard_form(form="lower")
    atoms.positions = atoms.positions @ transform.T
    atoms.set_cell(reduced_cell, scale_atoms=False)
    return atoms


def validate_ssneb_cell_orientation(atoms, label):
    cell = atoms.cell.array
    upper_terms = np.array([cell[0, 1], cell[0, 2], cell[1, 2]])
    if np.dot(upper_terms, upper_terms) > 1e-3:
        raise ValueError(
            f"{label} still violates TSASE cell orientation: "
            f"cell[0,1]={cell[0,1]:.6f}, cell[0,2]={cell[0,2]:.6f}, cell[1,2]={cell[1,2]:.6f}"
        )


def main():
    MACECalculator = load_mace_calculator()
    artifacts = neb.RunArtifacts.create(
        base_dir=RUNS_DIR,
        run_name=Path(__file__).stem,
    )
    run_dir = artifacts.run_dir
    polarization_reference_path = run_dir / "polarization_reference.cif"

    structure_indices = sorted(STRUCTURE_POINTS)
    copied_structure_records = artifacts.copy_inputs(
        {
            f"image_{index:04d}": STRUCTURE_POINTS[index]
            for index in structure_indices
        }
    )
    structures = [read(record["copied_to"]) for record in copied_structure_records]
    input_records = list(copied_structure_records)
    model_record = artifacts.copy_inputs({"model": MODEL_PATH})[0]
    input_records.append(model_record)
    restart_path = None
    if RESTART_FROM_XYZ is not None:
        restart_records = artifacts.copy_inputs({"restart": RESTART_FROM_XYZ})
        input_records.extend(restart_records)
        restart_path = Path(restart_records[0]["copied_to"])

    print(f"Run directory: {run_dir}")

    try:
        base_calc = MACECalculator(model_paths=model_record["copied_to"], device="cuda")
    except RuntimeError as exc:
        if "Found no NVIDIA driver" in str(exc):
            raise RuntimeError(
                "this example expects a CUDA-capable GPU node; submit it with "
                "`LLsub -i full -g volta:2 -T 02:00:00` and run inside the "
                "`NEB` conda environment"
            ) from exc
        raise

    filter_factory = build_filter_factory()

    for i, atoms in enumerate(structures):
        if i == 0 or i == len(structures) - 1:
            atoms.set_calculator(base_calc)
            # relax_log = run_dir / f"endpoint_relax_{structure_indices[i]:04d}.log"
            # relax_traj = run_dir / f"endpoint_relax_{structure_indices[i]:04d}.traj"
            relax_target = (
                filter_factory(atoms)
                if filter_factory is not None
                else atoms
            )
            optimizer = BFGS(relax_target)
            optimizer.run(fmax=0.02)
        neb.attach_field_charges(atoms, CHARGE_MAP)
        # standardize_cell_for_ssneb(atoms)
        atoms.write(run_dir / f"prepared_image_{structure_indices[i]:04d}.cif")

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

    prepared_paths = artifacts.write_structures(
        structures,
        indices=structure_indices,
    )
    artifacts.snapshot_script(__file__)
    artifacts.write_manifest(
        script_path=__file__,
        git_cwd=ROOT,
        inputs=input_records,
        prepared_structures=prepared_paths,
        config={
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
        extra_metadata={
            "outputs": {
                **artifacts.as_neb_paths(),
                "polarization_reference": str(polarization_reference_path),
            }
        },
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

    for i, atoms in enumerate(structures):
        standardize_cell_for_ssneb(atoms)
        validate_ssneb_cell_orientation(atoms, f"structure {structure_indices[i]}")

    band = neb.ssneb(
        structures,
        structure_indices,
        numImages=NUM_IMAGES,
        k=SPRING,
        method=METHOD,
        output_dir=str(artifacts.output_dir),
        filter_factory=build_filter_factory(),
        adaptive_springs=ADAPTIVE_SPRINGS,
        kmin=SPRING_MIN,
        kmax=SPRING_MAX,
        image_update_schedule=image_update_schedule,
    )
    if restart_path is not None:
        neb.load_band_configuration_from_xyz(band, str(restart_path))

    optimizer = neb.fire_ssneb(
        band,
        maxmove=0.10,
        dt=0.10,
        dtmax=0.10,
        output_interval=OUTPUT_INTERVAL,
        plot_property=PLOT_PROPERTY,
        ci_activation_force=CI_ACTIVATION_FORCE,
    )
    optimizer.minimize(forceConverged=FMAX, maxIterations=MAX_STEPS)


if __name__ == "__main__":
    main()
