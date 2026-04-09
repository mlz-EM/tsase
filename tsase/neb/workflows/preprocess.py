"""Preprocess raw control-point structures for a YAML-driven field SSNEB run."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from ase.io import read
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import get_symmetrized_atoms

from tsase.neb.core.mapping import spatial_map
from tsase.neb.models.charges import attach_field_charges
from tsase.neb.util import sPBC
from tsase.neb.viz import save_projected_neb_sequence

from .config import dump_yaml, load_field_ssneb_config, load_yaml_file


def standardize_cell_for_ssneb(atoms):
    """Rotate the structure into TSASE's expected lower-triangular cell form."""

    reduced_cell, transform = atoms.cell.standard_form(form="lower")
    atoms.positions = atoms.positions @ transform.T
    atoms.set_cell(reduced_cell, scale_atoms=False)
    return atoms


def validate_ssneb_cell_orientation(atoms, label):
    """Reject cells that still violate TSASE's expected orientation."""

    cell = atoms.cell.array
    upper_terms = np.array([cell[0, 1], cell[0, 2], cell[1, 2]])
    if np.dot(upper_terms, upper_terms) > 1e-3:
        raise ValueError(
            f"{label} still violates TSASE cell orientation: "
            f"cell[0,1]={cell[0,1]:.6f}, cell[0,2]={cell[0,2]:.6f}, cell[1,2]={cell[1,2]:.6f}"
        )


def align_translation_only(reference, candidate):
    """Align ``candidate`` to ``reference`` using only a periodic translation."""

    aligned = spatial_map(reference, candidate)

    reference_frac = np.asarray(reference.get_scaled_positions(), dtype=float)
    candidate_frac = np.asarray(aligned.get_scaled_positions(), dtype=float)
    shift_frac = -np.mean(sPBC(candidate_frac - reference_frac), axis=0)

    aligned.set_scaled_positions(candidate_frac + shift_frac)

    residual_frac = sPBC(np.asarray(aligned.get_scaled_positions(), dtype=float) - reference_frac)
    residual_cart = residual_frac @ np.asarray(reference.get_cell(), dtype=float)
    aligned.info["translation_alignment_shift_fractional"] = [float(value) for value in shift_frac]
    aligned.info["translation_alignment_rms_cart"] = float(
        np.sqrt(np.mean(np.sum(residual_cart**2, axis=1)))
    )
    return aligned


def _copy_calculator(calculator):
    try:
        return deepcopy(calculator)
    except Exception:
        return calculator


def _preprocess_settings(raw_config):
    preprocess = dict(raw_config.get("preprocess", {}))
    relax = dict(preprocess.get("relax", {}))
    reference = dict(preprocess.get("reference", {}))
    outputs = dict(preprocess.get("outputs", {}))
    return {
        "output_dir": preprocess.get("output_dir"),
        "relax_control_points": bool(relax.get("enabled", True)),
        "relax_fmax": float(relax.get("fmax", 0.02)),
        "symmetrize_reference": bool(reference.get("symmetrize", True)),
        "reference_symprec": float(reference.get("symprec", 1.0)),
        "processed_config_name": str(
            outputs.get("config_name", "run_field_ssneb_interpolated_preprocessed.yaml")
        ),
    }


def _resolved_output_dir(config, settings, config_path):
    if settings["output_dir"] is not None:
        return (config_path.parent / settings["output_dir"]).resolve()
    return (config.run_dir / "preprocessed_control_points").resolve()


def _build_polarization_reference(atoms, *, symmetrize, symprec):
    if not symmetrize:
        return atoms.copy(), None
    refined_atoms, dataset = get_symmetrized_atoms(atoms, symprec=symprec)
    refined_atoms.set_cell(atoms.cell, scale_atoms=False)
    return refined_atoms, getattr(dataset, "international", None)


def _render_control_point_stem_visualization(processed_structures, destination_dir):
    """Render a STEM visualization across all processed control points."""

    return save_projected_neb_sequence(
        [atoms.copy() for atoms in processed_structures],
        xyz_dir=Path(destination_dir) / "stem_endpoints",
        iteration=0,
    )


def preprocess_field_ssneb_control_points(config_path, *, output_dir=None):
    """Generate NEB-ready control-point CIFs and a derived run config."""

    config_path = Path(config_path).expanduser().resolve()
    raw_config = load_yaml_file(config_path)
    settings = _preprocess_settings(raw_config)
    config = load_field_ssneb_config(config_path)

    source = config.resolved_config["path"]["source"]
    if source["kind"] != "control_points":
        raise ValueError("preprocessing currently supports only path.source.kind=control_points")

    destination_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _resolved_output_dir(config, settings, config_path)
    )
    destination_dir.mkdir(parents=True, exist_ok=True)

    processed_structures = [atoms.copy() for atoms in config.structures]
    filter_factory = config.filter_factory

    for index, atoms in enumerate(processed_structures):
        if settings["relax_control_points"]:
            atoms.calc = _copy_calculator(config.calculator)
            relax_target = filter_factory(atoms) if filter_factory is not None else atoms
            optimizer = BFGS(relax_target)
            optimizer.run(fmax=settings["relax_fmax"])
        attach_field_charges(atoms, config.charge_map)
        standardize_cell_for_ssneb(atoms)
        validate_ssneb_cell_orientation(
            atoms,
            f"control point {config.structure_indices[index]}",
        )

    for index in range(1, len(processed_structures)):
        processed_structures[index] = align_translation_only(
            processed_structures[index - 1],
            processed_structures[index],
        )

    processed_paths = []
    for structure_index, atoms in zip(config.structure_indices, processed_structures):
        path = destination_dir / f"control_point_{int(structure_index):04d}.cif"
        snapshot = atoms.copy()
        snapshot.calc = None
        snapshot.write(path)
        processed_paths.append(path)

    polarization_reference, space_group = _build_polarization_reference(
        processed_structures[0],
        symmetrize=settings["symmetrize_reference"],
        symprec=settings["reference_symprec"],
    )
    polarization_reference_path = destination_dir / "polarization_reference.cif"
    polarization_reference.write(polarization_reference_path)

    endpoint_stem = _render_control_point_stem_visualization(processed_structures, destination_dir)

    derived_config = deepcopy(raw_config)
    derived_config.setdefault("path", {}).setdefault("source", {})["files"] = [
        str(path) for path in processed_paths
    ]
    derived_config["path"]["source"]["kind"] = "control_points"
    derived_config["path"]["source"]["indices"] = list(config.structure_indices)
    derived_config.setdefault("model", {}).setdefault("reference", {})
    derived_config["model"]["reference"]["kind"] = "file"
    derived_config["model"]["reference"]["file"] = str(polarization_reference_path)

    derived_config_path = destination_dir / settings["processed_config_name"]
    derived_config_path.write_text(dump_yaml(derived_config) + "\n", encoding="utf-8")

    return {
        "output_dir": destination_dir,
        "processed_config": derived_config_path,
        "processed_control_points": processed_paths,
        "polarization_reference": polarization_reference_path,
        "endpoint_stem": endpoint_stem,
        "space_group": space_group,
    }
