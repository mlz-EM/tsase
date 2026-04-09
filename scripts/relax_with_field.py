#!/usr/bin/env python

"""Relax Pbam_SC_NC.cif across an electric-field sweep with a MACE-field model.

The script performs an inclusive field sweep from ``[0, 0, 0]`` to
``[0, 20000, 0]`` kV/cm by default. Each relaxation starts from the relaxed
structure of the previous field, records the final energy and polarization, and
saves the relaxed structure. After the sweep, it runs the STEM visualization
pipeline on the saved multi-frame trajectory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import site
import sys
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS, FIRE


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "example" / "Pbam_SC_NC.cif"
DEFAULT_OUTPUT_DIR = ROOT / "example" / "relax_with_field_run"
FIELD_KV_PER_CM_TO_V_PER_ANGSTROM = 1.0e-5
FIELD_V_PER_ANGSTROM_TO_KV_PER_CM = 1.0e5
POLARIZATION_E_PER_ANG2_TO_UC_PER_CM2 = 1602.176634


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--model-path", default="MACEField.model")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--field-start",
        nargs=3,
        type=float,
        default=[-75, 300, 0.0],
        help="Inclusive sweep start in kV/cm.",
    )
    parser.add_argument(
        "--field-end",
        nargs=3,
        type=float,
        default=[-1000, 4000.0, 0.0],
        help="Inclusive sweep end in kV/cm.",
    )
    parser.add_argument("--num-fields", type=int, default=30)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head", default="pt_head")
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--optimizer", choices=("fire", "bfgs"), default="bfgs")
    parser.add_argument(
        "--cell-mask",
        nargs="+",
        type=float,
        help=(
            "Optional cell-relaxation mask. Pass 3 values to control the a/b/c axes "
            "(shear fixed automatically), or 6 ASE strain-mask values."
        ),
    )
    parser.add_argument(
        "--no-stem",
        action="store_true",
        help="Skip the final STEM analysis step.",
    )
    return parser.parse_args(argv)


def resolve_path(path_like, *, base_dir=ROOT):
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (base_dir / path).resolve()


def build_field_grid(start, end, count):
    if count < 2:
        raise ValueError("--num-fields must be at least 2 for an inclusive sweep.")
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return np.linspace(start, end, count)


def field_kv_per_cm_to_v_per_angstrom(value):
    return float(value) * FIELD_KV_PER_CM_TO_V_PER_ANGSTROM


def field_v_per_angstrom_to_kv_per_cm(value):
    return float(value) * FIELD_V_PER_ANGSTROM_TO_KV_PER_CM


def polarization_e_per_ang2_to_uc_per_cm2(value):
    return float(value) * POLARIZATION_E_PER_ANG2_TO_UC_PER_CM2


def normalize_cell_mask(mask_values):
    if mask_values is None:
        return None
    mask = [float(value) for value in mask_values]
    if len(mask) == 3:
        return [mask[0], mask[1], mask[2], 0.0, 0.0, 0.0]
    if len(mask) == 6:
        return mask
    raise ValueError("--cell-mask must contain either 3 values (a/b/c axes) or 6 ASE strain-mask values.")


def _user_site_paths():
    try:
        paths = site.getusersitepackages()
    except Exception:
        return []
    if isinstance(paths, str):
        paths = [paths]
    normalized = []
    for entry in paths:
        try:
            normalized.append(str(Path(entry).expanduser().resolve()))
        except Exception:
            normalized.append(str(entry))
    return normalized


def _clear_import_state():
    for name in list(sys.modules):
        if name == "mace" or name.startswith("mace."):
            sys.modules.pop(name, None)
        if name == "torch" or name.startswith("torch."):
            sys.modules.pop(name, None)


def _remove_user_site_from_sys_path():
    user_sites = _user_site_paths()
    if not user_sites:
        return []

    removed = []
    kept = []
    for entry in sys.path:
        try:
            resolved = str(Path(entry).expanduser().resolve())
        except Exception:
            resolved = str(entry)
        if resolved in user_sites:
            removed.append(entry)
        else:
            kept.append(entry)
    if removed:
        sys.path[:] = kept
        _clear_import_state()
    return removed


def load_mace_calculator():
    try:
        from mace.calculators import MACECalculator

        return MACECalculator
    except Exception as exc:
        removed = _remove_user_site_from_sys_path()
        if removed:
            try:
                from mace.calculators import MACECalculator

                return MACECalculator
            except ModuleNotFoundError as retry_exc:
                if retry_exc.name == "torch":
                    raise RuntimeError(
                        "MACE is installed in the active environment, but PyTorch is "
                        "missing from the env or is being shadowed by user-site packages."
                    ) from exc
                raise
            except Exception as retry_exc:
                raise RuntimeError(
                    "Failed to import MACE after removing user-site packages from sys.path."
                ) from retry_exc

        raise RuntimeError(
            "Failed to import MACECalculator. If the traceback mentions CUDA or a user-site "
            "PyTorch under ~/.local, try PYTHONNOUSERSITE=1 and confirm PyTorch is installed "
            "inside the active environment."
        ) from exc


def load_stem_analyzer():
    module_path = ROOT / "tsase" / "neb" / "viz" / "stem.py"
    spec = importlib.util.spec_from_file_location("tsase_local_stem", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load STEM visualization module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module.analyze_stem_sequence_from_xyz


def build_calculator(*, model_path, field_vector, device, head, default_dtype):
    MACECalculator = load_mace_calculator()
    field_vector_v_per_angstrom = [
        field_kv_per_cm_to_v_per_angstrom(value)
        for value in np.asarray(field_vector, dtype=float)
    ]
    return MACECalculator(
        model_paths=str(model_path),
        model_type="MACEField",
        electric_field=field_vector_v_per_angstrom,
        device=device,
        head=head,
        default_dtype=default_dtype,
    )


def build_relaxation_target(atoms, mask=None):
    return FrechetCellFilter(atoms, mask=mask)


def get_optimizer_class(optimizer_name):
    optimizer_name = str(optimizer_name).lower()
    if optimizer_name == "fire":
        return FIRE
    if optimizer_name == "bfgs":
        return BFGS
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def max_force_norm(forces):
    forces = np.asarray(forces, dtype=float)
    if forces.ndim == 1:
        forces = forces.reshape(1, -1)
    if forces.size == 0:
        return 0.0
    return float(np.linalg.norm(forces, axis=1).max())


def extract_polarization(atoms):
    polarization = atoms.calc.results.get("polarization")
    if polarization is None:
        raise ValueError("MACEField calculator did not populate results['polarization'].")
    polarization = np.asarray(polarization, dtype=float)
    if polarization.shape != (3,):
        raise ValueError(
            "Expected results['polarization'] to have shape (3,), "
            f"got {polarization.shape}."
        )
    return polarization


def relax_structure(
    atoms,
    *,
    model_path,
    field_vector,
    device,
    head,
    default_dtype,
    optimizer_name,
    cell_mask,
    fmax,
    max_steps,
    logfile,
    trajectory,
):
    relaxed = atoms.copy()
    relaxed.calc = build_calculator(
        model_path=model_path,
        field_vector=field_vector,
        device=device,
        head=head,
        default_dtype=default_dtype,
    )

    target = build_relaxation_target(relaxed, mask=cell_mask)
    optimizer_class = get_optimizer_class(optimizer_name)
    optimizer = optimizer_class(target, logfile=str(logfile), trajectory=str(trajectory))
    converged = bool(optimizer.run(fmax=fmax, steps=max_steps))

    energy = float(relaxed.get_potential_energy())
    atomic_forces = np.asarray(relaxed.get_forces(), dtype=float)
    filtered_forces = np.asarray(target.get_forces(), dtype=float)
    polarization = extract_polarization(relaxed)

    result = {
        "atoms": relaxed,
        "converged": converged,
        "steps": int(getattr(optimizer, "nsteps", 0)),
        "energy_ev": energy,
        "polarization": polarization,
        "polarization_magnitude": float(np.linalg.norm(polarization)),
        "atomic_fmax": max_force_norm(atomic_forces),
        "filtered_fmax": max_force_norm(filtered_forces),
    }
    return result


def annotate_atoms(atoms, *, field_index, field_vector, energy_ev, polarization):
    atoms.info["field_index"] = int(field_index)
    atoms.info["electric_field_x_kv_per_cm"] = float(field_vector[0])
    atoms.info["electric_field_y_kv_per_cm"] = float(field_vector[1])
    atoms.info["electric_field_z_kv_per_cm"] = float(field_vector[2])
    atoms.info["energy_ev"] = float(energy_ev)
    atoms.info["polarization_x_uc_per_cm2"] = float(polarization[0])
    atoms.info["polarization_y_uc_per_cm2"] = float(polarization[1])
    atoms.info["polarization_z_uc_per_cm2"] = float(polarization[2])
    atoms.info["polarization_magnitude_uc_per_cm2"] = float(np.linalg.norm(polarization))


def run_sweep(args):
    input_path = resolve_path(args.input)
    model_path = resolve_path(args.model_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input structure not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Pass --model-path to point to your MACEField model."
        )

    field_vectors = build_field_grid(args.field_start, args.field_end, args.num_fields)
    cell_mask = normalize_cell_mask(args.cell_mask)

    structures_dir = output_dir / "structures"
    logs_dir = output_dir / "logs"
    structures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    sweep_cif = output_dir / "relaxed_field_sweep.cif"

    current = read(str(input_path))
    relaxed_images = []
    records = []

    for field_index, field_vector in enumerate(field_vectors):
        log_path = logs_dir / f"field_{field_index:04d}.log"
        trajectory_path = logs_dir / f"field_{field_index:04d}.traj"
        cif_path = structures_dir / f"field_{field_index:04d}.cif"
        xyz_path = structures_dir / f"field_{field_index:04d}.extxyz"

        result = relax_structure(
            current,
            model_path=model_path,
            field_vector=field_vector,
            device=args.device,
            head=args.head,
            default_dtype=args.default_dtype,
            optimizer_name=args.optimizer,
            cell_mask=cell_mask,
            fmax=args.fmax,
            max_steps=args.max_steps,
            logfile=log_path,
            trajectory=trajectory_path,
        )
        if not result["converged"]:
            raise RuntimeError(
                "Relaxation did not converge for "
                f"field index {field_index} ({field_vector.tolist()}) within {args.max_steps} steps."
            )

        polarization_uc_per_cm2 = np.asarray(
            [polarization_e_per_ang2_to_uc_per_cm2(value) for value in result["polarization"]],
            dtype=float,
        )
        relaxed = result["atoms"]
        annotate_atoms(
            relaxed,
            field_index=field_index,
            field_vector=field_vector,
            energy_ev=result["energy_ev"],
            polarization=polarization_uc_per_cm2,
        )
        stored = relaxed.copy()
        stored.calc = None
        write(str(cif_path), stored)
        write(str(xyz_path), stored, format="extxyz")

        record = {
            "field_index": field_index,
            "field_vector": [float(value) for value in field_vector],
            "energy_ev": result["energy_ev"],
            "polarization": [float(value) for value in polarization_uc_per_cm2],
            "polarization_magnitude": float(np.linalg.norm(polarization_uc_per_cm2)),
            "atomic_fmax": result["atomic_fmax"],
            "filtered_fmax": result["filtered_fmax"],
            "optimizer_steps": result["steps"],
            "structure_cif": str(cif_path),
            "structure_extxyz": str(xyz_path),
            "trajectory": str(trajectory_path),
            "logfile": str(log_path),
        }
        records.append(record)
        relaxed_images.append(stored.copy())
        current = stored.copy()

    write(str(sweep_cif), relaxed_images, format="cif")

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "field_index",
                "field_x_kv_per_cm",
                "field_y_kv_per_cm",
                "field_z_kv_per_cm",
                "energy_ev",
                "polarization_x_uc_per_cm2",
                "polarization_y_uc_per_cm2",
                "polarization_z_uc_per_cm2",
                "polarization_magnitude_uc_per_cm2",
                "atomic_fmax",
                "filtered_fmax",
                "optimizer_steps",
                "structure_cif",
                "structure_extxyz",
                "trajectory",
                "logfile",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["field_index"],
                    record["field_vector"][0],
                    record["field_vector"][1],
                    record["field_vector"][2],
                    record["energy_ev"],
                    record["polarization"][0],
                    record["polarization"][1],
                    record["polarization"][2],
                    record["polarization_magnitude"],
                    record["atomic_fmax"],
                    record["filtered_fmax"],
                    record["optimizer_steps"],
                    record["structure_cif"],
                    record["structure_extxyz"],
                    record["trajectory"],
                    record["logfile"],
                ]
            )

    stem_result = None
    stem_input_xyz = None
    if not args.no_stem:
        stem_input_xyz = logs_dir / "relaxed_field_sweep_stem.extxyz"
        write(str(stem_input_xyz), relaxed_images, format="extxyz")
        analyze_stem_sequence_from_xyz = load_stem_analyzer()
        stem_result = analyze_stem_sequence_from_xyz(
            stem_input_xyz,
            output_dir=output_dir / "stem",
            emit_png=True,
            emit_gif=True,
            emit_npy=True,
        )

    summary_payload = {
        "input_structure": str(input_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "logs_dir": str(logs_dir),
        "field_start": [float(value) for value in np.asarray(args.field_start, dtype=float)],
        "field_end": [float(value) for value in np.asarray(args.field_end, dtype=float)],
        "num_fields": int(args.num_fields),
        "fmax": float(args.fmax),
        "max_steps": int(args.max_steps),
        "device": str(args.device),
        "head": str(args.head),
        "default_dtype": str(args.default_dtype),
        "optimizer": str(args.optimizer),
        "cell_mask": None if cell_mask is None else [float(value) for value in cell_mask],
        "field_units": "kV/cm",
        "polarization_units": "uC/cm^2",
        "sweep_cif": str(sweep_cif),
        "stem_input_xyz": None if stem_input_xyz is None else str(stem_input_xyz),
        "summary_csv": str(summary_csv),
        "records": records,
        "stem": stem_result,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    return summary_payload


def main(argv=None):
    args = parse_args(argv)
    summary = run_sweep(args)
    print(f"Output directory: {summary['output_dir']}")
    print(f"Summary CSV: {summary['summary_csv']}")
    print(f"Relaxed sweep: {summary['sweep_cif']}")
    if summary["stem"] is not None:
        print(f"STEM status: {summary['stem']['status']}")
        print(f"STEM output: {summary['stem']['output_dir']}")
    return summary


if __name__ == "__main__":
    main()
