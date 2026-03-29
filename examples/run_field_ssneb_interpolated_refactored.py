#!/usr/bin/env python

"""Cleanup-phase field-SSNEB example using the normalized workflow interface."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsase.neb.workflows import FieldSSNEBConfig, run_field_ssneb


DEFAULT_RUN_DIR = ROOT / "examples" / "runs" / "run_field_ssneb_interpolated_refactored"


def build_structures():
    cell = np.diag([6.0, 6.0, 6.0])
    common = dict(symbols="Cu2", cell=cell, pbc=[True, True, True])
    start = Atoms(positions=[[1.6, 3.0, 3.0], [4.4, 3.0, 3.0]], **common)
    midpoint = Atoms(positions=[[1.8, 2.4, 3.0], [4.2, 3.6, 3.0]], **common)
    end = Atoms(positions=[[2.1, 2.0, 3.0], [3.9, 4.0, 3.0]], **common)
    return [start, midpoint, end]


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--num-images", type=int, default=7)
    parser.add_argument("--fmax", type=float, default=0.2)
    parser.add_argument("--spring", type=float, default=1.5)
    parser.add_argument("--field", type=float, nargs=3, default=None)
    parser.add_argument(
        "--field-crystal",
        type=float,
        nargs=3,
        default=None,
        help="Optional crystal-axis field components. Omit both --field and --field-crystal for zero field.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    structures = build_structures()
    structure_indices = [0, args.num_images // 2, args.num_images - 1]
    charges = np.array([1.0, -1.0], dtype=float)

    config = FieldSSNEBConfig.from_inputs(
        structures=structures,
        structure_indices=structure_indices,
        num_images=args.num_images,
        calculator=EMT(),
        charge_map=charges,
        field=args.field,
        field_crystal=args.field_crystal,
        reference_atoms=structures[0],
        run_dir=Path(args.output_dir),
        spring=args.spring,
        method="normal",
        band_kwargs={"ss": False},
        optimizer_kwargs={
            "maxmove": 0.05,
            "dt": 0.05,
            "dtmax": 0.05,
            "output_interval": 1,
            "plot_property": "px",
        },
        minimize_kwargs={"forceConverged": args.fmax, "maxIterations": args.max_steps},
        script_path=__file__,
        manifest_config={
            "structure_indices": structure_indices,
            "num_images": args.num_images,
            "field": None if args.field is None else list(args.field),
            "field_crystal": None if args.field_crystal is None else list(args.field_crystal),
            "spring": args.spring,
            "fmax": args.fmax,
            "max_steps": args.max_steps,
        },
    )
    result = run_field_ssneb(
        structures=config.structures,
        structure_indices=config.structure_indices,
        num_images=config.num_images,
        calculator=config.calculator,
        charge_map=config.charge_map,
        field=config.field_vector,
        reference_atoms=config.reference_atoms,
        run_dir=config.run_dir,
        restart_xyz=config.restart_xyz,
        spring=config.spring,
        method=config.method,
        filter_factory=config.filter_factory,
        remesh_stages=config.remesh_stages,
        image_mobility_rates=config.image_mobility_rates,
        ci_activation_iteration=config.ci_activation_iteration,
        ci_activation_force=config.ci_activation_force,
        band_kwargs=config.band_kwargs,
        optimizer_kwargs=config.optimizer_kwargs,
        minimize_kwargs=config.minimize_kwargs,
        script_path=config.script_path,
        manifest_config=config.manifest_config,
    )
    workflow_artifacts = result["workflow_artifacts"]
    artifacts = result["artifacts"]
    print(f"Run directory: {workflow_artifacts.run_dir}")
    print(f"Final stage diagnostics: {artifacts.diagnostics_file}")
    print(f"Final stage XYZ output: {artifacts.xyz_dir}")
    return result


if __name__ == "__main__":
    main()
