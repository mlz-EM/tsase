#!/usr/bin/env python

"""Production driver for field-coupled SSNEB with multi-point interpolation."""

import argparse
from pathlib import Path
import numpy as np
from ase.constraints import FixAtoms
from ase.filters import ExpCellFilter, Filter, StrainFilter, UnitCellFilter
from ase.io import read
from tsase import neb
from tsase.neb.runtime import load_mace_calculator


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"cannot parse boolean value {value!r}")


def parse_charge_map(entries):
    charge_map = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"charge assignment must have the form Element=value, got {entry!r}")
        key, value = entry.split("=", 1)
        charge_map[key] = float(value)
    return charge_map


def parse_image_update_map(entries):
    update_map = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"image update entries must have the form index=factor, got {entry!r}"
            )
        key, value = entry.split("=", 1)
        update_map[int(key)] = float(value)
    return update_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures", nargs="+", required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--num_images", type=int, required=True)
    parser.add_argument("--model", default="example/MACE_model.model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--field", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--field-crystal", nargs=3, type=float)
    parser.add_argument("--polarization-reference")
    parser.add_argument("--charge-map", nargs="*", default=["Pb=2", "Zr=4", "O=-2"])
    parser.add_argument("--filter", choices=["Filter", "StrainFilter", "UnitCellFilter", "ExpCellFilter"])
    parser.add_argument("--filter-mask", nargs="+", type=float)
    parser.add_argument("--hydrostatic-strain", action="store_true")
    parser.add_argument("--constant-volume", action="store_true")
    parser.add_argument("--freeze-symbols", nargs="*", default=[])
    parser.add_argument("--freeze-indices", nargs="*", type=int, default=[])
    parser.add_argument("--adaptive", type=str2bool, default=False)
    parser.add_argument("--spring", type=float, default=5.0)
    parser.add_argument("--spring-min", type=float, default=1.0)
    parser.add_argument("--spring-max", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--method", choices=["normal", "ci"], default="normal")
    parser.add_argument("--ci-activation-force", type=float, default=None)
    parser.add_argument("--xyz-dir", default="neb_xyz")
    parser.add_argument("--diagnostics-file", default="neb_diagnostics.csv")
    parser.add_argument("--output-interval", type=int, default=1)
    parser.add_argument("--plot-property", default="none")
    parser.add_argument("--image-update", nargs="*", default=[])
    parser.add_argument("--image-update-iterations", type=int, default=0)
    parser.add_argument("--restart-from-xyz")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-dir-base")
    parser.add_argument("--run-name", default="ssneb")
    parser.add_argument("--stress", nargs=6, type=float, default=[0.0] * 6)
    return parser.parse_args()


def stress_matrix_from_voigt(values):
    xx, yy, zz, yz, xz, xy = values
    return np.array([[xx, 0.0, 0.0], [xy, yy, 0.0], [xz, yz, zz]], dtype=float)


def maybe_apply_freeze_constraints(atoms, freeze_symbols, freeze_indices):
    if not freeze_symbols and not freeze_indices:
        return
    mask = np.zeros(len(atoms), dtype=bool)
    if freeze_symbols:
        symbols = np.array(atoms.get_chemical_symbols())
        for symbol in freeze_symbols:
            mask |= symbols == symbol
    if freeze_indices:
        mask[np.array(freeze_indices, dtype=int)] = True
    atoms.set_constraint(FixAtoms(mask=mask))


def build_filter_factory(args):
    if args.filter is None:
        return None

    mask = args.filter_mask
    if args.filter == "Filter":
        def factory(image):
            movable = np.ones(len(image), dtype=bool)
            if args.freeze_symbols:
                symbols = np.array(image.get_chemical_symbols())
                for symbol in args.freeze_symbols:
                    movable[symbols == symbol] = False
            if args.freeze_indices:
                movable[np.array(args.freeze_indices, dtype=int)] = False
            return Filter(image, mask=movable)
        return factory

    if args.filter == "StrainFilter":
        strain_mask = mask if mask is not None else [1, 1, 1, 1, 1, 1]
        return lambda image: StrainFilter(image, mask=strain_mask)

    if args.filter == "UnitCellFilter":
        cell_mask = mask if mask is not None else [1, 1, 1, 1, 1, 1]
        return lambda image: UnitCellFilter(
            image,
            mask=cell_mask,
            hydrostatic_strain=args.hydrostatic_strain,
            constant_volume=args.constant_volume,
        )

    if args.filter == "ExpCellFilter":
        cell_mask = mask if mask is not None else [1, 1, 1, 1, 1, 1]
        return lambda image: ExpCellFilter(
            image,
            mask=cell_mask,
            hydrostatic_strain=args.hydrostatic_strain,
            constant_volume=args.constant_volume,
        )

    raise ValueError(f"unsupported filter type {args.filter!r}")


def main():
    args = parse_args()
    MACECalculator = load_mace_calculator()
    if len(args.structures) != len(args.indices):
        raise ValueError("--structures and --indices must have the same length")
    if args.indices[0] != 0 or args.indices[-1] != args.num_images - 1:
        raise ValueError("--indices must start at 0 and end at num_images - 1")

    artifacts = None
    input_records = []
    if args.output_dir:
        artifacts = neb.RunArtifacts(args.output_dir)
        artifacts.xyz_dir.mkdir(parents=True, exist_ok=True)
    elif args.run_dir_base:
        artifacts = neb.RunArtifacts.create(
            base_dir=args.run_dir_base,
            run_name=args.run_name,
        )

    structure_paths = list(args.structures)
    model_path = args.model
    polarization_reference_path = args.polarization_reference
    restart_path = args.restart_from_xyz
    if artifacts is not None:
        copied_structure_records = artifacts.copy_inputs(
            {
                f"image_{index:04d}": path
                for index, path in zip(args.indices, structure_paths)
            }
        )
        input_records.extend(copied_structure_records)
        structure_paths = [record["copied_to"] for record in copied_structure_records]

        model_record = artifacts.copy_inputs({"model": model_path})[0]
        input_records.append(model_record)
        model_path = model_record["copied_to"]

        if polarization_reference_path:
            reference_record = artifacts.copy_inputs({"polarization_reference": polarization_reference_path})[0]
            input_records.append(reference_record)
            polarization_reference_path = reference_record["copied_to"]

        if restart_path:
            restart_record = artifacts.copy_inputs({"restart": restart_path})[0]
            input_records.append(restart_record)
            restart_path = restart_record["copied_to"]

    structures = [read(path) for path in structure_paths]
    polarization_reference = structures[0]
    if polarization_reference_path:
        polarization_reference = neb.spatial_map(
            structures[0],
            read(polarization_reference_path),
        )
    if args.field_crystal is not None:
        field_vector = neb.crystal_field_to_cartesian(structures[0].get_cell(), args.field_crystal)
    else:
        field_vector = np.array(args.field, dtype=float)
    charge_map = parse_charge_map(args.charge_map)
    image_update_map = parse_image_update_map(args.image_update)
    for atoms in structures:
        neb.attach_field_charges(atoms, charge_map)
        maybe_apply_freeze_constraints(atoms, args.freeze_symbols, args.freeze_indices)

    try:
        base_calc = MACECalculator(model_paths=model_path, device=args.device)
    except RuntimeError as exc:
        if "Found no NVIDIA driver" in str(exc):
            raise RuntimeError(
                "the selected MACE model requires a CUDA-capable GPU runtime; "
                "launch this workflow on a GPU node (for example via "
                "`LLsub -i full -g volta:2 -T 02:00:00`) and set `--device cuda`"
            ) from exc
        raise
    calc = neb.EnthalpyWrapper(
        base_calc,
        field=field_vector,
        reference_atoms=polarization_reference,
        charges=neb.build_charge_array(structures[0], charge_map=charge_map),
    )
    for atoms in structures:
        atoms.calc = calc

    filter_factory = build_filter_factory(args)
    image_update_schedule = None
    if image_update_map and args.image_update_iterations > 0:
        image_update_schedule = [
            {
                "factors": image_update_map,
                "iterations": args.image_update_iterations,
            }
        ]
    if artifacts is not None:
        prepared_paths = artifacts.write_structures(structures, indices=args.indices)
        artifacts.write_manifest(
            script_path=__file__,
            git_cwd=Path(__file__).resolve().parents[0],
            inputs=input_records,
            prepared_structures=prepared_paths,
            config=vars(args),
        )
    band = neb.ssneb(
        structures,
        args.indices,
        numImages=args.num_images,
        k=args.spring,
        method=args.method,
        express=stress_matrix_from_voigt(args.stress),
        output_dir=None if artifacts is None else str(artifacts.output_dir),
        xyz_dir=args.xyz_dir if artifacts is None else None,
        filter_factory=filter_factory,
        adaptive_springs=args.adaptive,
        kmin=args.spring_min,
        kmax=args.spring_max,
        diagnostics_file=args.diagnostics_file if artifacts is None else None,
        image_update_schedule=image_update_schedule,
    )
    if restart_path:
        neb.load_band_configuration_from_xyz(band, restart_path)
    opt = neb.fire_ssneb(
        band,
        maxmove=0.1,
        dt=0.1,
        dtmax=0.1,
        ci_activation_force=args.ci_activation_force,
        xyz_dir=args.xyz_dir if artifacts is None else None,
        output_interval=args.output_interval,
        plot_property=args.plot_property,
    )
    opt.minimize(forceConverged=args.fmax, maxIterations=args.max_steps)


if __name__ == "__main__":
    main()
