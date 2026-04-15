"""Maintained workflow helpers for dimer-based saddle searches."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

import numpy as np
from ase.calculators.emt import EMT
from ase.io import read, write
from ase.optimize import BFGS, FIRE

from tsase.dimer.lanczos import LanczosDimer
from tsase.dimer.ssdimer import SSDimer
from tsase.neb.core.mapping import spatial_map
from tsase.neb.models.charges import attach_field_charges, build_charge_array
from tsase.neb.models.field import EnthalpyWrapper, resolve_field_vector
from tsase.neb.runtime import load_mace_calculator
from tsase.neb.util import vunit
from tsase.neb.viz.stem import analyze_stem_sequence_from_xyz
from tsase.neb.workflows.config import dump_yaml, load_yaml_file


def _copy_atoms_with_calculator(atoms):
    snapshot = atoms.copy()
    snapshot.calc = atoms.calc
    snapshot.info = dict(atoms.info)
    return snapshot


def _deep_update(base, updates):
    result = dict(base)
    for key, value in dict(updates).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(base_dir, value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_charge_map(charges_config):
    charges_config = dict(charges_config or {})
    if not charges_config:
        return None
    kind = str(charges_config.get("kind", "species_map")).lower()
    values = charges_config.get("values")
    if kind == "species_map":
        return {str(key): float(value) for key, value in dict(values).items()}
    if kind == "array":
        return np.array(values, dtype=float)
    raise ValueError("model.charges.kind must be one of: species_map, array")


def _resolve_field_settings(field_config):
    field_config = dict(field_config or {})
    kind = str(field_config.get("kind", "none")).lower()
    value = field_config.get("value")
    if kind == "none":
        return {"field": None, "field_crystal": None}
    if kind == "cartesian":
        return {"field": value, "field_crystal": None}
    if kind == "crystal":
        return {"field": None, "field_crystal": value}
    raise ValueError("model.field.kind must be one of: none, cartesian, crystal")


def _build_calculator(calculator_config, base_dir, *, field_vector):
    calculator_config = dict(calculator_config or {})
    kind = str(calculator_config.get("kind", "emt")).lower()
    kwargs = {key: value for key, value in calculator_config.items() if key != "kind"}
    if kind == "emt":
        return EMT(), "emt"
    if kind == "mace":
        if "model_path" in kwargs:
            kwargs["model_paths"] = str(_resolve_path(base_dir, kwargs.pop("model_path")))
        return load_mace_calculator()(**kwargs), "mace"
    if kind == "mace_field":
        if "model_path" in kwargs:
            kwargs["model_paths"] = str(_resolve_path(base_dir, kwargs.pop("model_path")))
        kwargs["electric_field"] = [float(value) for value in np.asarray(field_vector, dtype=float)]
        return load_mace_calculator()(model_type="MACEField", **kwargs), "mace_field"
    raise ValueError("model.calculator.kind must be one of: emt, mace, mace_field")


def _resolve_reference_atoms(reference_config, base_dir, structure):
    reference_config = dict(reference_config or {})
    kind = str(reference_config.get("kind", "first_structure")).lower()
    if kind == "first_structure":
        return structure.copy()
    if kind == "file":
        return read(str(_resolve_path(base_dir, reference_config["file"])))
    raise ValueError("model.reference.kind must be one of: first_structure, file")


def _compute_jacobian(atoms, weight):
    natom = len(atoms)
    volume = float(atoms.get_volume())
    avglen = (volume / natom) ** (1.0 / 3.0)
    return avglen * natom ** 0.5 * float(weight)


def _mode_from_difference(structure, target, *, include_cell, weight):
    reference = _copy_atoms_with_calculator(structure)
    mapped_target = spatial_map(reference, target)
    atomic_mode = mapped_target.get_positions() - reference.get_positions()
    if not include_cell:
        return atomic_mode
    jacobian = _compute_jacobian(reference, weight)
    dcell = np.array(mapped_target.get_cell(), dtype=float) - np.array(reference.get_cell(), dtype=float)
    cell_mode = np.dot(np.linalg.inv(reference.get_cell()), dcell) * jacobian
    return np.vstack((atomic_mode, cell_mode))


def _resolve_mode(mode_config, base_dir, structure, *, ss, weight):
    mode_config = dict(mode_config or {})
    kind = str(mode_config.get("kind", "random")).lower()
    if kind == "random":
        return None
    if kind == "array":
        return np.array(mode_config["values"], dtype=float)
    if kind == "file":
        path = _resolve_path(base_dir, mode_config["file"])
        if path.suffix.lower() != ".npy":
            raise ValueError("search.mode.kind=file currently requires a .npy file")
        return np.array(np.load(path), dtype=float)
    if kind == "difference":
        target = read(str(_resolve_path(base_dir, mode_config["file"])))
        return _mode_from_difference(
            structure,
            target,
            include_cell=bool(mode_config.get("include_cell", ss)),
            weight=weight,
        )
    raise ValueError("search.mode.kind must be one of: random, array, file, difference")


def _normalize_method(method):
    normalized = str(method).lower()
    if normalized not in {"ssdimer", "lanczos"}:
        raise ValueError("search.method must be one of: ssdimer, lanczos")
    return normalized


def _normalize_downhill_settings(settings):
    settings = dict(settings or {})
    if not settings:
        return {
            "enabled": False,
            "step_size": 0.01,
            "optimizer": "BFGS",
            "fmax": 0.01,
            "max_steps": 200,
        }
    optimizer = str(settings.get("optimizer", "BFGS")).upper()
    if optimizer not in {"BFGS", "FIRE"}:
        raise ValueError("postprocess.downhill.optimizer must be one of: BFGS, FIRE")
    return {
        "enabled": bool(settings.get("enabled", False)),
        "step_size": float(settings.get("step_size", 0.01)),
        "optimizer": optimizer,
        "fmax": float(settings.get("fmax", 0.01)),
        "max_steps": int(settings.get("max_steps", 200)),
    }


def _normalize_output_settings(settings):
    settings = dict(settings or {})
    return {
        "write_structures": bool(settings.get("write_structures", True)),
        "write_mode": bool(settings.get("write_mode", True)),
        "stem": bool(settings.get("stem", False)),
        "stem_interval": int(settings.get("stem_interval", 10)),
        "output_interval": int(
            settings.get(
                "output_interval",
                settings.get("stem_interval", settings.get("movie_interval", 10)),
            )
        ),
        "progress_log": bool(settings.get("progress_log", True)),
        "live_plot": bool(settings.get("live_plot", True)),
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(payload) + "\n", encoding="utf-8")
    return path


def _read_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _state_artifacts(run_dir):
    run_dir = Path(run_dir)
    state_dir = run_dir / "state"
    return {
        "state_dir": state_dir,
        "current_structure": state_dir / "current.cif",
        "current_mode": state_dir / "current_mode.npy",
        "current_velocity": state_dir / "current_velocity.npy",
        "runtime_state": state_dir / "runtime_state.json",
    }


def _load_progress_history(path):
    path = Path(path)
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return []
    columns = lines[0].split("\t")
    history = []
    for line in lines[1:]:
        values = line.split("\t")
        row = dict(zip(columns, values))
        history.append(
            {
                "step": int(row["step"]),
                "region": row["region"],
                "fmax": float(row["fmax_ev_per_a"]),
                "curvature": float(row["curvature"]),
                "delta_e_mev_per_atom": float(row["delta_e_mev_per_atom"]),
                "step_norm": float(row["step_norm"]),
                "force_calls_step": int(row["force_calls_step"]),
                "force_calls_total": int(row["force_calls_total"]),
                "alpha": float(row["alpha"]),
                "rotation_angle_deg": float(row["rotation_angle_deg"]),
                "rotation_substeps": int(row["rotation_substeps"]),
                "ftrans_norm": float(row["ftrans_norm"]),
                "mode_status": row["mode_status"],
                "translation_mode": row["translation_mode"],
                "converged": row["converged"].lower() == "true",
            }
        )
    return history


@dataclass(frozen=True)
class RelaxationResult:
    """Result from relaxing one downhill branch."""

    atoms: object
    converged: bool
    energy: float
    steps: int
    optimizer: str


@dataclass(frozen=True)
class DownhillConnectionResult:
    """Result from relaxing both downhill branches from a saddle."""

    positive: RelaxationResult
    negative: RelaxationResult
    mode: np.ndarray
    step_size: float


@dataclass(frozen=True)
class StructureMatch:
    """Best-match summary between one structure and a set of references."""

    label: str
    position_rms: float
    cell_rms: float


@dataclass(frozen=True)
class DimerConfig:
    """Resolved configuration for maintained dimer and Lanczos workflows."""

    structure: object
    calculator: object
    calculator_mode: str
    charge_map: object
    field_vector: np.ndarray
    reference_atoms: object
    run_dir: Path
    method: str
    copy_atoms: bool = True
    mode: object = None
    dimer_kwargs: dict = field(default_factory=dict)
    search_kwargs: dict = field(default_factory=dict)
    downhill_settings: dict = field(default_factory=dict)
    output_settings: dict = field(default_factory=dict)
    config_path: Optional[Path] = None
    resolved_config: dict = field(default_factory=dict)

    @classmethod
    def from_inputs(
        cls,
        *,
        structure,
        calculator,
        calculator_mode="emt",
        charge_map=None,
        field=None,
        field_crystal=None,
        reference_atoms=None,
        run_dir=None,
        method="ssdimer",
        copy_atoms=True,
        mode=None,
        dimer_kwargs=None,
        search_kwargs=None,
        downhill_settings=None,
        output_settings=None,
        resolved_config=None,
    ):
        structure = _copy_atoms_with_calculator(structure)
        field_vector = resolve_field_vector(
            structure.get_cell(),
            field=field,
            field_crystal=field_crystal,
        )
        run_dir = Path("dimer_runs") if run_dir is None else Path(run_dir)
        method = _normalize_method(method)
        return cls(
            structure=structure,
            calculator=calculator,
            calculator_mode=str(calculator_mode),
            charge_map=charge_map,
            field_vector=np.array(field_vector, dtype=float),
            reference_atoms=None if calculator_mode == "mace_field" else reference_atoms,
            run_dir=run_dir,
            method=method,
            copy_atoms=bool(copy_atoms),
            mode=None if mode is None else np.array(mode, dtype=float),
            dimer_kwargs={} if dimer_kwargs is None else dict(dimer_kwargs),
            search_kwargs={} if search_kwargs is None else dict(search_kwargs),
            downhill_settings=_normalize_downhill_settings(downhill_settings),
            output_settings=_normalize_output_settings(output_settings),
            resolved_config={} if resolved_config is None else dict(resolved_config),
        )

    @classmethod
    def from_mapping(cls, mapping, *, source_path=None):
        source_path = None if source_path is None else Path(source_path).expanduser().resolve()
        base_dir = Path.cwd().resolve() if source_path is None else source_path.parent

        run_config = dict(mapping.get("run", {}))
        run_root = run_config.get("root")
        run_name = str(run_config.get("name", "dimer_search"))
        run_dir = (
            _resolve_path(base_dir, run_root)
            if run_root is not None
            else (base_dir / "runs" / run_name).resolve()
        )

        structure_config = dict(mapping.get("structure", {}))
        structure = read(str(_resolve_path(base_dir, structure_config["file"])))

        model_config = dict(mapping.get("model", {}))
        charge_map = _resolve_charge_map(model_config.get("charges"))
        field_settings = _resolve_field_settings(model_config.get("field"))
        field_vector = resolve_field_vector(
            structure.get_cell(),
            field=field_settings["field"],
            field_crystal=field_settings["field_crystal"],
        )
        calculator, calculator_mode = _build_calculator(
            model_config.get("calculator"),
            base_dir,
            field_vector=field_vector,
        )

        search_config = dict(mapping.get("search", {}))
        method = _normalize_method(search_config.get("method", "ssdimer"))
        copy_atoms = bool(search_config.get("copy_atoms", True))
        dimer_kwargs = dict(search_config.get("dimer", {}))
        ss = bool(dimer_kwargs.get("ss", False))
        weight = float(dimer_kwargs.get("weight", 1.0))
        mode = _resolve_mode(search_config.get("mode"), base_dir, structure, ss=ss, weight=weight)

        reference_atoms = None
        if calculator_mode != "mace_field":
            reference_atoms = _resolve_reference_atoms(model_config.get("reference"), base_dir, structure)

        convergence = dict(mapping.get("convergence", {}))
        outputs = dict(mapping.get("outputs", {}))
        search_kwargs = {
            "minForce": float(convergence.get("minForce", 0.01)),
            "maxForceCalls": int(convergence.get("maxForceCalls", 100000)),
            "quiet": bool(search_config.get("quiet", outputs.get("quiet", False))),
            "movie": None if outputs.get("movie") is None else str((run_dir / outputs["movie"]).resolve()),
            "interval": int(outputs.get("movie_interval", 50)),
        }

        downhill_settings = _normalize_downhill_settings(dict(mapping.get("postprocess", {})).get("downhill"))
        output_settings = _normalize_output_settings(outputs)

        resolved_config = {
            "run": {"root": str(run_dir), "name": run_dir.name},
            "structure": {"file": str(_resolve_path(base_dir, structure_config["file"]))},
            "model": {
                "calculator": {"kind": calculator_mode},
                "charges": None if charge_map is None else model_config.get("charges"),
                "field": {"kind": "cartesian", "value": [float(value) for value in field_vector]},
            },
            "search": {
                "method": method,
                "copy_atoms": copy_atoms,
                "quiet": search_kwargs["quiet"],
                "mode": search_config.get("mode"),
                "dimer": dict(dimer_kwargs),
            },
            "convergence": {
                "minForce": search_kwargs["minForce"],
                "maxForceCalls": search_kwargs["maxForceCalls"],
            },
            "outputs": {
                "movie": search_kwargs["movie"],
                "movie_interval": search_kwargs["interval"],
                "write_structures": output_settings["write_structures"],
                "write_mode": output_settings["write_mode"],
                "stem": output_settings["stem"],
                "stem_interval": output_settings["stem_interval"],
                "output_interval": output_settings["output_interval"],
                "progress_log": output_settings["progress_log"],
                "live_plot": output_settings["live_plot"],
            },
            "postprocess": {"downhill": dict(downhill_settings)},
        }
        if reference_atoms is not None:
            resolved_config["model"]["reference"] = dict(model_config.get("reference", {"kind": "first_structure"}))

        return cls(
            structure=_copy_atoms_with_calculator(structure),
            calculator=calculator,
            calculator_mode=calculator_mode,
            charge_map=charge_map,
            field_vector=np.array(field_vector, dtype=float),
            reference_atoms=reference_atoms,
            run_dir=run_dir,
            method=method,
            copy_atoms=copy_atoms,
            mode=mode,
            dimer_kwargs=dimer_kwargs,
            search_kwargs=search_kwargs,
            downhill_settings=downhill_settings,
            output_settings=output_settings,
            config_path=source_path,
            resolved_config=resolved_config,
        )

    @classmethod
    def from_yaml(cls, path, *, overrides=None):
        raw = load_yaml_file(path)
        if overrides:
            raw = _deep_update(raw, overrides)
        return cls.from_mapping(raw, source_path=path)


def _prepare_runtime_atoms(config):
    atoms = _copy_atoms_with_calculator(config.structure) if config.copy_atoms else config.structure
    atoms.info["tsase_calculator_mode"] = config.calculator_mode
    if config.charge_map is not None:
        attach_field_charges(atoms, config.charge_map)

    if config.calculator_mode == "mace_field":
        atoms.calc = config.calculator
        return atoms

    polarization_reference = spatial_map(atoms, config.reference_atoms)
    atoms.calc = EnthalpyWrapper(
        config.calculator,
        field=config.field_vector,
        reference_atoms=polarization_reference,
        charges=build_charge_array(atoms, charge_map=config.charge_map),
    )
    return atoms


def _normalize_mode_array(mode, natoms):
    mode_array = np.array(mode, dtype=float, copy=True)
    if len(mode_array) == natoms:
        return np.vstack((mode_array, np.zeros((3, 3))))
    if len(mode_array) == natoms + 3:
        return mode_array
    raise ValueError("mode must have shape (natom, 3) or (natom + 3, 3)")


def apply_mode_displacement(atoms, mode, *, step_size, ss=False, weight=1.0):
    """Return a copy displaced along the provided generalized mode."""

    displaced = _copy_atoms_with_calculator(atoms)
    normalized_mode = vunit(_normalize_mode_array(mode, len(displaced)))
    step_size = float(step_size)

    if ss:
        jacobian = _compute_jacobian(displaced, weight)
        displacement = step_size * normalized_mode
        cell = np.array(displaced.get_cell(), dtype=float)
        displaced_cell = cell + np.dot(cell, displacement[-3:]) / jacobian
        displaced.set_cell(displaced_cell, scale_atoms=True)
        scaled_positions = atoms.get_scaled_positions()
        positions = np.dot(scaled_positions, displaced_cell) + displacement[:-3]
        displaced.set_positions(positions)
        return displaced

    positions = np.array(displaced.get_positions(), dtype=float) + step_size * normalized_mode[:-3]
    displaced.set_positions(positions)
    return displaced


def relax_structure(atoms, *, optimizer="BFGS", fmax=0.01, max_steps=200):
    """Relax one structure with a small maintained optimizer surface."""

    working = _copy_atoms_with_calculator(atoms)
    optimizer_name = str(optimizer).upper()
    if optimizer_name == "BFGS":
        dyn = BFGS(working, logfile=None)
    elif optimizer_name == "FIRE":
        dyn = FIRE(working, logfile=None)
    else:
        raise ValueError("optimizer must be one of: BFGS, FIRE")
    converged = dyn.run(fmax=float(fmax), steps=int(max_steps))
    energy = float(working.get_potential_energy())
    return RelaxationResult(
        atoms=working,
        converged=bool(converged),
        energy=energy,
        steps=int(dyn.get_number_of_steps()),
        optimizer=optimizer_name,
    )


def relax_downhill_from_saddle(
    saddle_atoms,
    mode,
    *,
    step_size=0.01,
    ss=False,
    weight=1.0,
    optimizer="BFGS",
    fmax=0.01,
    max_steps=200,
):
    """Relax both downhill branches by displacing along ``+/- mode`` from a saddle."""

    positive_seed = apply_mode_displacement(
        saddle_atoms,
        mode,
        step_size=abs(float(step_size)),
        ss=ss,
        weight=weight,
    )
    negative_seed = apply_mode_displacement(
        saddle_atoms,
        mode,
        step_size=-abs(float(step_size)),
        ss=ss,
        weight=weight,
    )
    positive = relax_structure(
        positive_seed,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
    )
    negative = relax_structure(
        negative_seed,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
    )
    return DownhillConnectionResult(
        positive=positive,
        negative=negative,
        mode=np.array(mode, dtype=float, copy=True),
        step_size=float(step_size),
    )


def compare_structures(reference, candidate):
    """Compare two structures after spatial atom remapping."""

    mapped = spatial_map(reference, candidate)
    position_delta = np.array(mapped.get_positions(), dtype=float) - np.array(reference.get_positions(), dtype=float)
    cell_delta = np.array(mapped.get_cell(), dtype=float) - np.array(reference.get_cell(), dtype=float)
    position_rms = float(np.sqrt(np.mean(np.sum(position_delta ** 2, axis=1))))
    cell_rms = float(np.sqrt(np.mean(cell_delta ** 2)))
    return {
        "position_rms": position_rms,
        "cell_rms": cell_rms,
    }


def match_structure_to_references(candidate, references, *, cell_weight=1.0):
    """Return the closest reference label under a simple position-plus-cell metric."""

    best_match = None
    for label, reference in dict(references).items():
        metrics = compare_structures(reference, candidate)
        score = metrics["position_rms"] + float(cell_weight) * metrics["cell_rms"]
        if best_match is None or score < best_match[0]:
            best_match = (
                score,
                StructureMatch(
                    label=str(label),
                    position_rms=metrics["position_rms"],
                    cell_rms=metrics["cell_rms"],
                ),
            )
    if best_match is None:
        raise ValueError("references must contain at least one labeled structure")
    return best_match[1]


def identify_downhill_connections(downhill_result, references, *, cell_weight=1.0):
    """Match each downhill branch to the closest labeled reference minimum."""

    return {
        "positive": match_structure_to_references(
            downhill_result.positive.atoms,
            references,
            cell_weight=cell_weight,
        ),
        "negative": match_structure_to_references(
            downhill_result.negative.atoms,
            references,
            cell_weight=cell_weight,
        ),
    }


def _write_workflow_outputs(config, search, search_result, downhill_result):
    run_dir = config.run_dir
    config_dir = run_dir / "config"
    saddle_dir = run_dir / "saddle"
    connections_dir = run_dir / "connections"
    config_dir.mkdir(parents=True, exist_ok=True)
    saddle_dir.mkdir(parents=True, exist_ok=True)
    connections_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "run_dir": run_dir,
        "resolved_config": _write_yaml(config_dir / "resolved.yaml", config.resolved_config),
        "summary_file": None,
        "saddle_structure": saddle_dir / "saddle.cif",
        "saddle_mode": saddle_dir / "mode.npy",
        "iteration_dir": run_dir / "iterations",
        "stem_dir": run_dir / "stem",
        "log_dir": run_dir / "logs",
        "state_dir": run_dir / "state",
        "current_structure": run_dir / "state" / "current.cif",
        "current_mode": run_dir / "state" / "current_mode.npy",
        "current_velocity": run_dir / "state" / "current_velocity.npy",
        "runtime_state": run_dir / "state" / "runtime_state.json",
        "progress_log": run_dir / "logs" / "dimer_progress.tsv",
        "live_plot": run_dir / "logs" / "live_progress.png",
        "positive_structure": None,
        "negative_structure": None,
        "connections_summary": None,
    }

    if config.output_settings.get("write_structures", True):
        write(str(artifacts["saddle_structure"]), search.R0)
    if config.output_settings.get("write_mode", True):
        np.save(artifacts["saddle_mode"], np.array(search.get_mode(), dtype=float))

    summary = {
        "method": config.method,
        "calculator_mode": config.calculator_mode,
        "converged": bool(search_result.converged),
        "steps": int(search_result.steps),
        "force_calls": int(search_result.force_calls),
        "curvature": float(search_result.curvature),
        "energy": None if search_result.energy is None else float(search_result.energy),
        "max_force": float(search_result.max_force),
        "downhill": None,
    }

    if downhill_result is not None:
        positive_path = connections_dir / "downhill_positive.cif"
        negative_path = connections_dir / "downhill_negative.cif"
        if config.output_settings.get("write_structures", True):
            write(str(positive_path), downhill_result.positive.atoms)
            write(str(negative_path), downhill_result.negative.atoms)
        artifacts["positive_structure"] = positive_path
        artifacts["negative_structure"] = negative_path
        downhill_summary = {
            "step_size": float(downhill_result.step_size),
            "positive": {
                "converged": bool(downhill_result.positive.converged),
                "energy": float(downhill_result.positive.energy),
                "steps": int(downhill_result.positive.steps),
                "optimizer": downhill_result.positive.optimizer,
            },
            "negative": {
                "converged": bool(downhill_result.negative.converged),
                "energy": float(downhill_result.negative.energy),
                "steps": int(downhill_result.negative.steps),
                "optimizer": downhill_result.negative.optimizer,
            },
        }
        artifacts["connections_summary"] = _write_json(connections_dir / "summary.json", downhill_summary)
        summary["downhill"] = downhill_summary

    artifacts["summary_file"] = _write_json(config_dir / "summary.json", summary)
    return artifacts


def _write_progress_header(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "step\tregion\tfmax_ev_per_a\tcurvature\tdelta_e_mev_per_atom\tstep_norm\t"
        "force_calls_step\tforce_calls_total\talpha\trotation_angle_deg\t"
        "rotation_substeps\tftrans_norm\tmode_status\ttranslation_mode\tconverged\n"
    )
    path.write_text(header, encoding="utf-8")


def _append_progress_row(path, diagnostics):
    path = Path(path)
    if not path.exists():
        _write_progress_header(path)
    row = (
        f"{int(diagnostics['step'])}\t{diagnostics['region']}\t"
        f"{float(diagnostics['fmax']):.10f}\t{float(diagnostics['curvature']):.10f}\t"
        f"{float(diagnostics['delta_e_mev_per_atom']):.10f}\t{float(diagnostics['step_norm']):.10f}\t"
        f"{int(diagnostics['force_calls_step'])}\t{int(diagnostics['force_calls_total'])}\t"
        f"{float(diagnostics['alpha']):.10f}\t{float(diagnostics['rotation_angle_deg']):.10f}\t"
        f"{int(diagnostics['rotation_substeps'])}\t{float(diagnostics['ftrans_norm']):.10f}\t"
        f"{diagnostics['mode_status']}\t{diagnostics['translation_mode']}\t"
        f"{str(bool(diagnostics['converged'])).lower()}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def _update_live_progress_plot(path, history, *, min_force):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not history:
        return None

    colors = {
        "fmax": "#1f77b4",
        "curvature": "#d62728",
        "delta_e": "#2ca02c",
        "force_calls_step": "#9467bd",
    }
    metrics = [
        ("fmax", "Fmax (eV/A)", [entry["fmax"] for entry in history]),
        ("curvature", "|Curvature|", [abs(entry["curvature"]) for entry in history]),
        ("delta_e", "dE (meV/atom)", [entry["delta_e_mev_per_atom"] for entry in history]),
        ("force_calls_step", "FC/step", [entry["force_calls_step"] for entry in history]),
    ]
    steps = [entry["step"] for entry in history]

    fig, host = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(right=0.78)
    host.set_xlabel("Step")
    host.set_yticks([])
    host.spines["left"].set_visible(False)
    host.spines["right"].set_visible(False)

    axes = []
    lines = []
    for index, (key, label, values) in enumerate(metrics):
        axis = host.twinx()
        axis.spines["right"].set_position(("axes", 1.0 + 0.12 * index))
        axis.spines["right"].set_visible(True)
        axis.yaxis.set_label_position("right")
        axis.yaxis.tick_right()
        axis.tick_params(axis="y", colors=colors[key], labelsize=9)
        axis.spines["right"].set_color(colors[key])
        axis.set_ylabel(label, color=colors[key], fontsize=10)
        line, = axis.plot(steps, values, color=colors[key], linewidth=1.8, label=label)
        if key == "fmax":
            positive_values = [max(value, 1.0e-12) for value in values]
            line.set_ydata(positive_values)
            axis.set_yscale("log")
            axis.axhline(max(min_force, 1.0e-12), color=colors[key], linestyle="--", linewidth=1.0, alpha=0.7)
        if key == "curvature":
            positive_values = [max(value, 1.0e-12) for value in values]
            line.set_ydata(positive_values)
            axis.set_yscale("log")
        axes.append(axis)
        lines.append(line)

    latest = history[-1]
    title = (
        f"Step {latest['step']} | {latest['region']} | "
        f"Fmax={latest['fmax']:.4e} eV/A | Curv={latest['curvature']:.4e}"
    )
    host.set_title(title)
    host.grid(True, axis="x", alpha=0.25)
    host.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def _write_runtime_checkpoint(search, run_dir):
    state = _state_artifacts(run_dir)
    state["state_dir"].mkdir(parents=True, exist_ok=True)
    write(str(state["current_structure"]), search.R0)
    np.save(state["current_mode"], np.array(search.get_mode(), dtype=float))
    velocity = getattr(search, "V", None)
    if velocity is None:
        shape = (search.natom + 3, 3) if search.ss else (search.natom, 3)
        velocity = np.zeros(shape, dtype=float)
    np.save(state["current_velocity"], np.array(velocity, dtype=float))
    payload = {
        "method": type(search).__name__,
        "steps": int(search.steps),
        "force_calls": int(search.forceCalls),
        "curvature": float(search.curvature),
        "initial_energy": None if search.initial_energy is None else float(search.initial_energy),
        "energy": None if search.E is None else float(search.E),
        "alpha": float(getattr(search, "last_alpha", 0.0)),
        "rotation_angle_deg": float(getattr(search, "last_rotation_angle_deg", 0.0)),
        "rotation_substeps": int(getattr(search, "last_rotation_substeps", 0)),
        "mode_status": str(getattr(search, "last_mode_status", "rotating")),
        "translation_mode": str(getattr(search, "last_translation_mode", "drag_up")),
        "ftrans_norm": float(getattr(search, "last_ftrans_norm", 0.0)),
        "step_diagnostics": dict(getattr(search, "step_diagnostics", {})),
    }
    _write_json(state["runtime_state"], payload)
    return state


def _restore_runtime_checkpoint(search, run_dir):
    state = _state_artifacts(run_dir)
    required = (
        state["current_structure"],
        state["current_mode"],
        state["current_velocity"],
        state["runtime_state"],
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot resume dimer run; missing checkpoint files: " + ", ".join(missing)
        )

    checkpoint_atoms = read(str(state["current_structure"]))
    search.R0.set_cell(checkpoint_atoms.get_cell(), scale_atoms=False)
    search.R0.set_positions(checkpoint_atoms.get_positions())
    search.N = search._normalize_mode(np.load(state["current_mode"]))
    search.V = np.array(np.load(state["current_velocity"]), dtype=float)

    payload = _read_json(state["runtime_state"])
    search.steps = int(payload.get("steps", 0))
    search.forceCalls = int(payload.get("force_calls", 0))
    search.curvature = float(payload.get("curvature", 1.0))
    search.initial_energy = payload.get("initial_energy")
    if search.initial_energy is not None:
        search.initial_energy = float(search.initial_energy)
    search.E = payload.get("energy")
    if search.E is not None:
        search.E = float(search.E)
    search.last_alpha = float(payload.get("alpha", 0.0))
    search.last_rotation_angle_deg = float(payload.get("rotation_angle_deg", 0.0))
    search.last_rotation_substeps = int(payload.get("rotation_substeps", 0))
    search.last_mode_status = str(payload.get("mode_status", "rotating"))
    search.last_translation_mode = str(payload.get("translation_mode", "drag_up"))
    search.last_ftrans_norm = float(payload.get("ftrans_norm", 0.0))
    search.step_diagnostics = dict(payload.get("step_diagnostics", {}))
    search.R1 = search._copy_runtime_atoms(search.R0)
    search.R1_prime = search._copy_runtime_atoms(search.R0)
    return payload


def _make_snapshot_callback(config, *, resume=False):
    if not (
        True
        or config.output_settings.get("progress_log", True)
        or config.output_settings.get("live_plot", True)
    ):
        return None

    iteration_dir = config.run_dir / "iterations"
    stem_dir = config.run_dir / "stem"
    log_dir = config.run_dir / "logs"
    progress_log = log_dir / "dimer_progress.tsv"
    live_plot = log_dir / "live_progress.png"
    interval = max(1, int(config.output_settings.get("output_interval", 10)))
    history = _load_progress_history(progress_log) if resume else []
    if not resume and config.output_settings.get("progress_log", True):
        _write_progress_header(progress_log)

    def callback(search):
        diagnostics = dict(search.step_diagnostics)
        _write_runtime_checkpoint(search, config.run_dir)
        if config.output_settings.get("progress_log", True):
            _append_progress_row(progress_log, diagnostics)
        if config.output_settings.get("live_plot", True):
            log_dir.mkdir(parents=True, exist_ok=True)
            history.append(diagnostics)
            _update_live_progress_plot(
                live_plot,
                history,
                min_force=float(search.target_min_force or 0.0),
            )
        if search.steps == 1 or search.steps % interval == 0:
            snapshot_path = iteration_dir / f"iter_{int(search.steps):04d}.cif"
            mode_snapshot_path = iteration_dir / f"mode_{int(search.steps):04d}.npy"
            if (
                config.output_settings.get("write_structures", True)
                or config.output_settings.get("stem", False)
                or config.output_settings.get("write_mode", True)
            ):
                iteration_dir.mkdir(parents=True, exist_ok=True)
            if config.output_settings.get("write_structures", True) or config.output_settings.get("stem", False):
                write(str(snapshot_path), search.R0)
            if config.output_settings.get("write_mode", True):
                np.save(mode_snapshot_path, np.array(search.get_mode(), dtype=float))
            if config.output_settings.get("stem", False):
                result = analyze_stem_sequence_from_xyz(
                    snapshot_path,
                    output_dir=stem_dir,
                    iteration=search.steps,
                    emit_npy=False,
                )
                if isinstance(result, dict) and result.get("status") != "ok":
                    diagnostics_file = result.get("diagnostics_file")
                    print(
                        "Projected STEM visualization skipped for "
                        f"iter_{int(search.steps):04d}; see {diagnostics_file}"
                    )

    return callback


def load_dimer_config(path, *, overrides=None):
    """Load and resolve a YAML-backed dimer workflow configuration."""

    return DimerConfig.from_yaml(path, overrides=overrides)


def load_dimer_resume_config(run_dir, *, overrides=None):
    """Load a resolved run directory and point the workflow at its checkpoint state."""

    run_dir = Path(run_dir).expanduser().resolve()
    resolved_path = run_dir / "config" / "resolved.yaml"
    if not resolved_path.exists():
        raise FileNotFoundError(f"Cannot resume dimer run; missing {resolved_path}")

    state = _state_artifacts(run_dir)
    raw = load_yaml_file(resolved_path)
    raw = _deep_update(
        raw,
        {
            "run": {"root": str(run_dir), "name": run_dir.name},
            "structure": {"file": str(state["current_structure"])},
            "search": {"mode": {"kind": "file", "file": str(state["current_mode"])}},
        },
    )
    if overrides:
        raw = _deep_update(raw, overrides)
    return DimerConfig.from_mapping(raw, source_path=resolved_path)


def run_dimer(*, config=None, resume=False, **unexpected_kwargs):
    """Run a maintained dimer/Lanczos workflow from a resolved config."""

    if unexpected_kwargs:
        unexpected = ", ".join(sorted(unexpected_kwargs))
        raise TypeError(
            "run_dimer no longer accepts workflow keyword arguments "
            f"({unexpected}). Pass config=DimerConfig(...) instead, or "
            "use run_dimer_from_yaml(...) for YAML-driven runs."
        )
    if not isinstance(config, DimerConfig):
        raise TypeError(
            "run_dimer expects config=DimerConfig(...). "
            "Use load_dimer_config(...) or run_dimer_from_yaml(...) for YAML-driven runs."
        )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    atoms = _prepare_runtime_atoms(config)
    search_cls = LanczosDimer if config.method == "lanczos" else SSDimer
    search = search_cls.from_atoms(
        atoms,
        calculator_mode=config.calculator_mode,
        copy_atoms=False,
        mode=config.mode,
        **dict(config.dimer_kwargs),
    )
    if resume:
        _restore_runtime_checkpoint(search, config.run_dir)
    search_kwargs = dict(config.search_kwargs)
    search_kwargs["output_callback"] = _make_snapshot_callback(config, resume=bool(resume))
    search_result = search.search(**search_kwargs)

    downhill_result = None
    if config.downhill_settings.get("enabled", False):
        downhill_result = relax_downhill_from_saddle(
            search.R0,
            search.get_mode(),
            step_size=config.downhill_settings["step_size"],
            ss=bool(config.dimer_kwargs.get("ss", False)),
            weight=float(config.dimer_kwargs.get("weight", 1.0)),
            optimizer=config.downhill_settings["optimizer"],
            fmax=config.downhill_settings["fmax"],
            max_steps=config.downhill_settings["max_steps"],
        )

    artifacts = _write_workflow_outputs(config, search, search_result, downhill_result)
    return {
        "result": search_result,
        "search": search,
        "atoms": search.R0,
        "config": config,
        "run_dir": config.run_dir,
        "downhill_result": downhill_result,
        "artifacts": artifacts,
    }


def run_dimer_from_yaml(path, *, overrides=None):
    """Load, resolve, and run a YAML-backed dimer/Lanczos workflow."""

    return run_dimer(config=load_dimer_config(path, overrides=overrides))


def resume_dimer_from_run_dir(run_dir, *, overrides=None):
    """Resume a checkpointed dimer workflow from an existing run directory."""

    return run_dimer(
        config=load_dimer_resume_config(run_dir, overrides=overrides),
        resume=True,
    )
