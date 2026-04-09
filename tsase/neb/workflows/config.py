"""Resolved YAML-backed workflow configuration for field-coupled SSNEB."""

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
from ase.calculators.emt import EMT
from ase.io import read

from tsase.neb.core.interfaces import PathSpec
from tsase.neb.core.mapping import ensure_atom_ids, reorder_by_atom_ids, spatial_map
from tsase.neb.models.field import resolve_field_vector
from tsase.neb.optimize import build_optimizer_kwargs, normalize_optimizer_kind
from tsase.neb.runtime import load_mace_calculator


_ENERGY_PROFILE_ENTRY_ALIASES = {
    "enthalpy": "enthalpy_adjusted",
    "enthalpy_adjusted": "enthalpy_adjusted",
    "intrinsic_energy": "intrinsic_energy",
    "base_energy": "intrinsic_energy",
    "field_energy": "field_energy",
    "polarization_mag": "polarization_mag",
    "polarization_magnitude": "polarization_mag",
    "polarization_x": "polarization_x",
    "polarization_y": "polarization_y",
    "polarization_z": "polarization_z",
    "px": "polarization_x",
    "py": "polarization_y",
    "pz": "polarization_z",
    "p_mag": "polarization_mag",
    "pmag": "polarization_mag",
}


@dataclass(frozen=True)
class CalculatorMode:
    """Resolved calculator semantics for maintained field-aware workflows."""

    kind: str
    energy_semantics: str
    polarization_source: str
    requires_reference_polarization: bool
    supports_field_decomposition: bool
    supported_output_entries: tuple[str, ...]

    @classmethod
    def intrinsic(cls, kind):
        return cls(
            kind=str(kind),
            energy_semantics="intrinsic",
            polarization_source="wrapper",
            requires_reference_polarization=True,
            supports_field_decomposition=True,
            supported_output_entries=(
                "enthalpy_adjusted",
                "intrinsic_energy",
                "field_energy",
                "polarization_mag",
                "polarization_x",
                "polarization_y",
                "polarization_z",
            ),
        )

    @classmethod
    def mace_field(cls):
        return cls(
            kind="mace_field",
            energy_semantics="enthalpy_adjusted",
            polarization_source="results.polarization",
            requires_reference_polarization=False,
            supports_field_decomposition=False,
            supported_output_entries=(
                "enthalpy_adjusted",
                "polarization_mag",
                "polarization_x",
                "polarization_y",
                "polarization_z",
            ),
        )


def _normalize_energy_profile_entries(entries):
    normalized = []
    for entry in list(entries or []):
        key = str(entry).strip().lower()
        if key not in _ENERGY_PROFILE_ENTRY_ALIASES:
            choices = ", ".join(sorted(set(_ENERGY_PROFILE_ENTRY_ALIASES.values())))
            raise ValueError(f"unsupported energy profile entry {entry!r}; expected one of: {choices}")
        canonical = _ENERGY_PROFILE_ENTRY_ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _default_energy_profile_entries(calculator_mode):
    if calculator_mode is not None and calculator_mode.kind == "mace_field":
        return ["enthalpy_adjusted", "polarization_mag"]
    return ["enthalpy_adjusted", "intrinsic_energy", "field_energy", "polarization_mag"]


def _resolve_energy_profile_settings(energy_profile_config, calculator_mode):
    energy_profile_config = dict(energy_profile_config or {})
    enabled = bool(energy_profile_config.get("enabled", True))
    if "plot_property" in energy_profile_config:
        raise ValueError(
            "outputs.energy_profile.plot_property is no longer supported; "
            "use outputs.energy_profile.entries instead"
        )

    entries = energy_profile_config.get("entries")
    requested_entries = (
        _default_energy_profile_entries(calculator_mode)
        if entries is None
        else _normalize_energy_profile_entries(entries)
    )
    strict = bool(energy_profile_config.get("strict", False))
    supported_entries = set(
        _default_energy_profile_entries(calculator_mode)
        if calculator_mode is None
        else calculator_mode.supported_output_entries
    )
    unsupported_entries = [entry for entry in requested_entries if entry not in supported_entries]
    if unsupported_entries and strict:
        raise ValueError(
            "outputs.energy_profile.entries requested unsupported values for "
            f"{calculator_mode.kind if calculator_mode is not None else 'this calculator'}: {unsupported_entries}"
        )
    if unsupported_entries:
        warnings.warn(
            "Skipping unsupported outputs.energy_profile.entries for "
            f"{calculator_mode.kind if calculator_mode is not None else 'this calculator'}: {unsupported_entries}",
            stacklevel=3,
        )
    filtered_entries = [entry for entry in requested_entries if entry in supported_entries]
    return {
        "enabled": enabled,
        "emit_csv": bool(energy_profile_config.get("emit_csv", enabled)),
        "emit_png": bool(energy_profile_config.get("emit_png", enabled)),
        "entries": filtered_entries,
        "strict": strict,
    }


def _normalize_express_tensor(tensor):
    matrix = np.asarray(tensor, dtype=float)
    if matrix.shape == (6,):
        xx, yy, zz, yz, xz, xy = matrix.tolist()
        matrix = np.array(
            [
                [xx, xy, xz],
                [xy, yy, yz],
                [xz, yz, zz],
            ],
            dtype=float,
        )
    if matrix.shape != (3, 3):
        raise ValueError("band.express.tensor must be a 3x3 matrix or 6-value Voigt-like list")
    normalized = matrix.copy()
    if abs(normalized[0, 1]) > 1.0e-12 or abs(normalized[0, 2]) > 1.0e-12 or abs(normalized[1, 2]) > 1.0e-12:
        warnings.warn(
            "band.express upper-triangular terms are ignored in the maintained SSNEB path",
            stacklevel=3,
        )
    normalized[0, 1] = 0.0
    normalized[0, 2] = 0.0
    normalized[1, 2] = 0.0
    return normalized


def _resolve_band_express(band_config):
    express_config = band_config.get("express")
    if express_config is None:
        return None
    if isinstance(express_config, dict):
        units = str(express_config.get("units", "gpa")).lower()
        tensor = express_config.get("tensor")
    else:
        units = "gpa"
        tensor = express_config
    if tensor is None:
        raise ValueError("band.express must provide a tensor value")
    if units != "gpa":
        raise ValueError("band.express.units must be GPa in the maintained YAML workflow")
    return _normalize_express_tensor(tensor)


def _strip_comment(line):
    in_single = False
    in_double = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_inline_collection(token):
    translated = (
        token.replace(": true", ": True")
        .replace(": false", ": False")
        .replace(": null", ": None")
        .replace("[true", "[True")
        .replace("[false", "[False")
        .replace("[null", "[None")
        .replace(", true", ", True")
        .replace(", false", ", False")
        .replace(", null", ", None")
    )
    return literal_eval(translated)


def _parse_scalar(token):
    token = token.strip()
    if token == "":
        return ""
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if token[0] in "[{" and token[-1] in "]}":
        return _parse_inline_collection(token)
    if token[0] in {'"', "'"} and token[-1] == token[0]:
        return literal_eval(token)
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _clean_yaml_lines(text):
    cleaned = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw).rstrip()
        if stripped.strip():
            cleaned.append(stripped)
    return cleaned


def _parse_block(lines, start_index=0, indent=0):
    if start_index >= len(lines):
        return None, start_index
    if len(lines[start_index]) - len(lines[start_index].lstrip(" ")) < indent:
        return None, start_index
    if lines[start_index].lstrip().startswith("-"):
        return _parse_sequence(lines, start_index, indent)
    return _parse_mapping(lines, start_index, indent)


def _parse_sequence(lines, start_index, indent):
    items = []
    index = start_index
    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent != indent or not line.lstrip().startswith("-"):
            raise ValueError(f"invalid YAML sequence line: {line!r}")
        stripped = line.lstrip()
        remainder = stripped[1:].strip()
        index += 1
        item = None
        if remainder:
            if ":" in remainder and not remainder.startswith(("[", "{")):
                key, raw_value = remainder.split(":", 1)
                raw_value = raw_value.strip()
                item = {key.strip(): None if raw_value == "" else _parse_scalar(raw_value)}
            else:
                item = _parse_scalar(remainder)
        if index < len(lines):
            next_indent = len(lines[index]) - len(lines[index].lstrip(" "))
            if next_indent > indent:
                nested, index = _parse_block(lines, index, next_indent)
                if item is None:
                    item = nested
                elif isinstance(item, dict) and isinstance(nested, dict):
                    item.update(nested)
                else:
                    raise ValueError("cannot merge nested YAML content into a scalar sequence item")
        items.append(item)
    return items, index


def _parse_mapping(lines, start_index, indent):
    mapping = {}
    index = start_index
    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(f"invalid YAML mapping indentation: {line!r}")
        content = line.strip()
        if ":" not in content:
            raise ValueError(f"invalid YAML mapping entry: {line!r}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value == "":
            nested = {}
            if index < len(lines):
                next_indent = len(lines[index]) - len(lines[index].lstrip(" "))
                if next_indent > indent:
                    nested, index = _parse_block(lines, index, next_indent)
            mapping[key] = nested
        else:
            mapping[key] = _parse_scalar(raw_value)
    return mapping, index


def load_yaml_file(path):
    """Load a restricted YAML subset without requiring PyYAML."""

    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    cleaned = _clean_yaml_lines(text)
    if not cleaned:
        return {}
    parsed, index = _parse_block(cleaned, 0, 0)
    if index != len(cleaned):
        raise ValueError(f"unexpected trailing YAML content in {source}")
    return parsed


def _yaml_scalar(value):
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        if value == "" or any(char in value for char in ":#[]{}-,"):
            return repr(value)
        return value
    return str(value)


def dump_yaml(data, indent=0):
    """Serialize basic Python data structures as YAML."""

    prefix = " " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(dump_yaml(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for value in data:
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(dump_yaml(value, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(value)}")
        return "\n".join(lines)
    return f"{prefix}{_yaml_scalar(data)}"


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


def _normalize_loaded_path(images, remap_mode):
    if not images:
        raise ValueError("path source did not yield any images")
    normalized = [images[0].copy()]
    reference = normalized[0]
    reference_ids = ensure_atom_ids(reference).copy()
    for image in images[1:]:
        loaded = image.copy()
        if remap_mode == "none":
            mapped = loaded
        elif remap_mode == "atom_ids":
            mapped = reorder_by_atom_ids(reference, loaded)
            if mapped is None:
                raise ValueError("restart/full-path image is missing atom IDs needed for atom_ids remapping")
        elif remap_mode == "spatial":
            mapped = spatial_map(reference, loaded)
        else:
            raise ValueError("remap_on_restart must be one of: atom_ids, spatial, none")
        mapped.arrays["neb_atom_id"] = reference_ids.copy()
        normalized.append(mapped)
    ensure_atom_ids(normalized, reference=reference)
    return normalized


def _resolve_path_source(path_config, base_dir):
    source = dict(path_config.get("source", {}))
    kind = str(source.get("kind", "control_points"))
    num_images = int(path_config.get("num_images", 0))
    remap_value = path_config.get("remap_on_restart", source.get("remap_on_restart", "atom_ids"))
    remap_mode = "none" if remap_value is None else str(remap_value)

    if kind == "control_points":
        file_entries = list(source.get("files", []))
        if not file_entries:
            raise ValueError("control_points path source requires one or more files")
        files = [_resolve_path(base_dir, entry) for entry in file_entries]
        structures = [read(str(path)) for path in files]
        indices = source.get("indices")
        if indices is None:
            indices = [0, num_images - 1] if len(structures) == 2 else list(range(len(structures)))
        indices = [int(value) for value in indices]
        return {
            "kind": kind,
            "files": [str(path) for path in files],
            "structures": structures,
            "indices": indices,
            "num_images": num_images,
            "remap_on_restart": remap_mode,
        }

    file_value = source.get("file")
    if file_value is None:
        files = list(source.get("files", []))
        if len(files) != 1:
            raise ValueError(f"{kind} path source requires exactly one file")
        file_value = files[0]
    source_file = _resolve_path(base_dir, file_value)
    images = read(str(source_file), ":")
    if kind == "restart_xyz":
        raise ValueError(
            "path.source.kind=restart_xyz is no longer part of the maintained runtime; "
            "use path.source.kind=full_path_xyz instead"
        )
    if kind == "full_path_xyz":
        if remap_mode == "none":
            ensure_atom_ids(images)
        else:
            images = _normalize_loaded_path(images, remap_mode)
    else:
        raise ValueError("path.source.kind must be one of: control_points, full_path_xyz")

    resolved_num_images = len(images) if num_images == 0 else num_images
    if resolved_num_images != len(images):
        raise ValueError(
            f"{kind} source provides {len(images)} images, but path.num_images={resolved_num_images}"
        )
    indices = list(range(resolved_num_images))
    return {
        "kind": kind,
        "file": str(source_file),
        "structures": [image.copy() for image in images],
        "indices": indices,
        "num_images": resolved_num_images,
        "remap_on_restart": remap_mode,
    }


def _resolve_charge_map(charges_config):
    charges_config = dict(charges_config)
    kind = str(charges_config.get("kind", "species_map")).lower()
    values = charges_config.get("values")
    if kind == "species_map":
        return {str(key): float(value) for key, value in dict(values).items()}
    if kind == "array":
        return np.array(values, dtype=float)
    raise ValueError("model.charges.kind must be one of: species_map, array")


def _resolve_field_settings(field_config):
    field_config = dict(field_config)
    kind = str(field_config.get("kind", "none")).lower()
    value = field_config.get("value")
    if kind == "none":
        return {"field": None, "field_crystal": None}
    if kind == "cartesian":
        return {"field": value, "field_crystal": None}
    if kind == "crystal":
        return {"field": None, "field_crystal": value}
    raise ValueError("model.field.kind must be one of: none, cartesian, crystal")


def _build_calculator(calculator_config, base_dir, *, field_vector=None):
    calculator_config = dict(calculator_config)
    kind = str(calculator_config.get("kind", "emt")).lower()
    kwargs = {
        key: value
        for key, value in calculator_config.items()
        if key
        not in {
            "kind",
            "energy_semantics",
            "polarization_source",
            "requires_reference_polarization",
        }
    }
    if kind == "emt":
        return EMT(), CalculatorMode.intrinsic(kind)
    if kind == "mace":
        if "model_path" in kwargs:
            kwargs["model_paths"] = str(_resolve_path(base_dir, kwargs.pop("model_path")))
        MACECalculator = load_mace_calculator()
        return MACECalculator(**kwargs), CalculatorMode.intrinsic(kind)
    if kind == "mace_field":
        if "model_path" in kwargs:
            kwargs["model_paths"] = str(_resolve_path(base_dir, kwargs.pop("model_path")))
        if calculator_config.get("energy_semantics", "enthalpy_adjusted") != "enthalpy_adjusted":
            raise ValueError("model.calculator.energy_semantics must be enthalpy_adjusted for mace_field")
        if calculator_config.get("polarization_source", "results.polarization") != "results.polarization":
            raise ValueError("model.calculator.polarization_source must be results.polarization for mace_field")
        if bool(calculator_config.get("requires_reference_polarization", False)):
            raise ValueError("model.calculator.requires_reference_polarization must be false for mace_field")
        if field_vector is None:
            field_vector = np.zeros(3, dtype=float)
        kwargs["electric_field"] = [float(value) for value in np.asarray(field_vector, dtype=float)]
        MACECalculator = load_mace_calculator()
        return MACECalculator(**kwargs), CalculatorMode.mace_field()
    raise ValueError("model.calculator.kind must be one of: emt, mace, mace_field")


def _build_filter_factory(filter_config):
    filter_config = dict(filter_config or {})
    kind = str(filter_config.get("kind", "none"))
    if kind.lower() == "none":
        return None

    mask = filter_config.get("mask")
    hydrostatic_strain = bool(filter_config.get("hydrostatic_strain", False))
    constant_volume = bool(filter_config.get("constant_volume", False))
    move_atoms = bool(filter_config.get("move_atoms", True))
    if not move_atoms and kind in {"ExpCellFilter", "UnitCellFilter"}:
        raise ValueError(f"{kind} does not support move_atoms=false in the maintained YAML workflow")

    from ase.filters import ExpCellFilter, StrainFilter, UnitCellFilter

    if kind == "ExpCellFilter":
        return lambda image: ExpCellFilter(
            image,
            mask=mask,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
        )
    if kind == "UnitCellFilter":
        return lambda image: UnitCellFilter(
            image,
            mask=mask,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
        )
    if kind == "StrainFilter":
        return lambda image: StrainFilter(image, mask=mask)
    raise ValueError("constraints.filter.kind must be one of: none, ExpCellFilter, UnitCellFilter, StrainFilter")


def _resolve_reference_atoms(reference_config, base_dir, structures):
    reference_config = dict(reference_config or {})
    kind = str(reference_config.get("kind", "first_structure"))
    if kind == "first_structure":
        return structures[0]
    if kind == "file":
        return read(str(_resolve_path(base_dir, reference_config["file"])))
    raise ValueError("model.reference.kind must be one of: first_structure, file")


def _resolve_trigger(trigger_config):
    if trigger_config is None:
        return None
    trigger_config = dict(trigger_config)
    kind = str(trigger_config.pop("kind", "stabilized_perp_force"))
    if kind != "stabilized_perp_force":
        raise ValueError("staging.remesh.trigger.kind must be stabilized_perp_force")
    from .staged import StabilizedPerpForce

    return StabilizedPerpForce(**trigger_config)


def _resolve_remesh_stages(staging_config):
    remesh_entries = list(dict(staging_config or {}).get("remesh", []))
    from .staged import RemeshStage

    stages = []
    for entry in remesh_entries:
        payload = dict(entry)
        payload["trigger"] = _resolve_trigger(payload.get("trigger"))
        stages.append(RemeshStage(**payload))
    return stages


def _resolve_output_settings(outputs_config, calculator_mode=None):
    outputs_config = dict(outputs_config or {})
    path_snapshot = dict(outputs_config.get("path_snapshot", {}))
    diagnostics = dict(outputs_config.get("diagnostics", {}))
    energy_profile = _resolve_energy_profile_settings(outputs_config.get("energy_profile", {}), calculator_mode)
    stem = dict(outputs_config.get("stem", {}))
    settings = {
        "diagnostics": bool(diagnostics.get("enabled", True)),
        "path_snapshots": bool(path_snapshot.get("enabled", True)),
        "energy_profile": bool(energy_profile.get("enabled", True)),
        "energy_profile_csv": bool(energy_profile.get("emit_csv", energy_profile.get("enabled", True))),
        "energy_profile_plot": bool(energy_profile.get("emit_png", energy_profile.get("enabled", True))),
        "energy_profile_entries": list(energy_profile.get("entries", [])),
        "stem": bool(stem.get("enabled", False)),
        "final_path_snapshot": bool(dict(path_snapshot.get("schedule", {})).get("include_final", True)),
    }
    return settings


def _serialize_output_settings(output_settings, *, output_interval):
    settings = dict(output_settings or {})
    energy_profile = {
        "enabled": bool(settings.get("energy_profile", True)),
        "emit_csv": bool(settings.get("energy_profile_csv", settings.get("energy_profile", True))),
        "emit_png": bool(settings.get("energy_profile_plot", settings.get("energy_profile", True))),
        "entries": list(settings.get("energy_profile_entries", [])),
    }
    return {
        "path_snapshot": {
            "enabled": bool(settings.get("path_snapshots", True)),
            "schedule": {
                "every": int(output_interval),
                "include_final": bool(settings.get("final_path_snapshot", True)),
            },
        },
        "diagnostics": {
            "enabled": bool(settings.get("diagnostics", True)),
        },
        "energy_profile": energy_profile,
        "stem": {
            "enabled": bool(settings.get("stem", False)),
        },
    }


def _serialize_charge_map(charge_map):
    if isinstance(charge_map, dict):
        return {"kind": "species_map", "values": dict(charge_map)}
    return {"kind": "array", "values": [float(value) for value in np.asarray(charge_map, dtype=float)]}


@dataclass
class FieldSSNEBConfig:
    """Resolved field-SSNEB workflow options."""

    structures: list
    structure_indices: list
    num_images: int
    calculator: object
    calculator_mode: object
    charge_map: object
    field_vector: object
    reference_atoms: object
    run_dir: Path
    spring: float = 5.0
    method: str = "ci"
    filter_factory: object = None
    remesh_stages: object = None
    image_mobility_rates: object = None
    ci_activation_iteration: Optional[int] = None
    ci_activation_force: Optional[float] = None
    optimizer_kind: str = "fire"
    band_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    minimize_kwargs: dict = field(default_factory=dict)
    output_settings: dict = field(default_factory=dict)
    script_path: Optional[str] = None
    manifest_config: Optional[dict] = None
    config_path: Optional[Path] = None
    resolved_config: dict = field(default_factory=dict)

    @classmethod
    def from_inputs(
        cls,
        *,
        structures,
        structure_indices,
        num_images,
        calculator,
        calculator_mode=None,
        charge_map,
        field=None,
        field_crystal=None,
        reference_atoms=None,
        run_dir=None,
        spring=5.0,
        method="ci",
        filter_factory=None,
        remesh_stages=None,
        image_mobility_rates=None,
        ci_activation_iteration=None,
        ci_activation_force=None,
        optimizer_kind="fire",
        band_kwargs=None,
        optimizer_kwargs=None,
        minimize_kwargs=None,
        output_settings=None,
        script_path=None,
        manifest_config=None,
    ):
        structures = [atoms.copy() for atoms in structures]
        calculator_mode = CalculatorMode.intrinsic("programmatic") if calculator_mode is None else calculator_mode
        reference_atoms = (
            None
            if not calculator_mode.requires_reference_polarization
            else (structures[0] if reference_atoms is None else reference_atoms)
        )
        run_dir = Path("field_ssneb_runs") if run_dir is None else Path(run_dir)
        optimizer_kind = normalize_optimizer_kind(optimizer_kind)
        field_vector = resolve_field_vector(
            structures[0].get_cell(),
            field=field,
            field_crystal=field_crystal,
        )
        resolved_config = {
            "run": {"root": str(run_dir)},
            "path": {
                "source": {"kind": "programmatic"},
                "num_images": int(num_images),
                "structure_indices": list(structure_indices),
            },
            "model": {
                "calculator": {
                    "kind": calculator_mode.kind,
                    "energy_semantics": calculator_mode.energy_semantics,
                    "polarization_source": calculator_mode.polarization_source,
                    "requires_reference_polarization": calculator_mode.requires_reference_polarization,
                },
                "charges": _serialize_charge_map(charge_map),
                "field": {"kind": "cartesian", "value": [float(value) for value in np.asarray(field_vector)]},
            },
            "band": {
                "spring": float(spring),
                "method": str(method),
            },
            "optimizer": {
                "kind": optimizer_kind,
                "ci_activation": {
                    "iteration": ci_activation_iteration,
                    "force": ci_activation_force,
                },
                "runtime": dict(optimizer_kwargs or {}),
                "convergence": {
                    "fmax": dict(minimize_kwargs or {}).get("forceConverged", 0.01),
                    "max_steps": dict(minimize_kwargs or {}).get("maxIterations", 1000),
                },
            },
            "outputs": dict(output_settings or {}),
        }
        return cls(
            structures=structures,
            structure_indices=list(structure_indices),
            num_images=int(num_images),
            calculator=calculator,
            calculator_mode=calculator_mode,
            charge_map=charge_map,
            field_vector=field_vector,
            reference_atoms=reference_atoms,
            run_dir=run_dir,
            spring=float(spring),
            method=str(method),
            filter_factory=filter_factory,
            remesh_stages=remesh_stages,
            image_mobility_rates=image_mobility_rates,
            ci_activation_iteration=ci_activation_iteration,
            ci_activation_force=ci_activation_force,
            optimizer_kind=optimizer_kind,
            band_kwargs={} if band_kwargs is None else dict(band_kwargs),
            optimizer_kwargs={} if optimizer_kwargs is None else dict(optimizer_kwargs),
            minimize_kwargs={} if minimize_kwargs is None else dict(minimize_kwargs),
            output_settings={} if output_settings is None else dict(output_settings),
            script_path=script_path,
            manifest_config=manifest_config,
            resolved_config=resolved_config,
        )

    @classmethod
    def from_mapping(cls, mapping, *, source_path=None):
        source_path = None if source_path is None else Path(source_path).expanduser().resolve()
        base_dir = Path.cwd().resolve() if source_path is None else source_path.parent

        run_config = dict(mapping.get("run", {}))
        run_root = run_config.get("root")
        run_name = str(run_config.get("name", "field_ssneb"))
        run_dir = (
            _resolve_path(base_dir, run_root)
            if run_root is not None
            else (base_dir / "runs" / run_name).resolve()
        )

        path_info = _resolve_path_source(dict(mapping.get("path", {})), base_dir)
        structures = [atoms.copy() for atoms in path_info["structures"]]
        structure_indices = list(path_info["indices"])
        num_images = int(path_info["num_images"])
        path_spec = PathSpec(structures, structure_indices, num_images)
        if len(path_spec.build_path()) != num_images:
            raise ValueError("resolved path definition did not produce the requested number of images")

        model_config = dict(mapping.get("model", {}))
        charge_map = _resolve_charge_map(dict(model_config.get("charges", {})))
        field_settings = _resolve_field_settings(dict(model_config.get("field", {})))
        field_vector = resolve_field_vector(
            structures[0].get_cell(),
            field=field_settings["field"],
            field_crystal=field_settings["field_crystal"],
        )
        calculator, calculator_mode = _build_calculator(
            dict(model_config.get("calculator", {})),
            base_dir,
            field_vector=field_vector,
        )
        reference_atoms = (
            None
            if not calculator_mode.requires_reference_polarization
            else _resolve_reference_atoms(model_config.get("reference"), base_dir, structures)
        )

        band_config = dict(mapping.get("band", {}))
        optimizer_config = dict(mapping.get("optimizer", {}))
        optimizer_kind = normalize_optimizer_kind(optimizer_config.get("kind", "fire"))
        convergence_config = dict(optimizer_config.get("convergence", {}))
        ci_activation = dict(optimizer_config.get("ci_activation", {}))
        output_settings = _resolve_output_settings(mapping.get("outputs"), calculator_mode)

        output_interval = optimizer_config.get("output_interval")
        if output_interval is None:
            output_interval = dict(dict(mapping.get("outputs", {})).get("path_snapshot", {}).get("schedule", {})).get("every", 1)

        band_kwargs = {
            "ss": bool(band_config.get("ss", True)),
            "weight": float(band_config.get("weight", 1.0)),
        }
        if "tangent" in band_config:
            band_kwargs["tangent"] = band_config["tangent"]
        express = _resolve_band_express(band_config)
        if express is not None:
            band_kwargs["express"] = express

        if "plot_property" in optimizer_config:
            raise ValueError(
                "optimizer.plot_property is no longer supported; "
                "move it to outputs.energy_profile.entries"
            )
        optimizer_kwargs = build_optimizer_kwargs(
            optimizer_kind,
            optimizer_config,
            output_interval=output_interval,
            energy_profile_entries=output_settings.get("energy_profile_entries", []),
        )

        resolved_config = _deep_update(
            dict(mapping),
            {
                "run": {
                    "name": run_name,
                    "root": str(run_dir),
                },
                "path": {
                    "source": {
                        key: value
                        for key, value in path_info.items()
                        if key not in {"structures", "indices", "num_images"}
                    },
                    "num_images": num_images,
                    "structure_indices": structure_indices,
                },
                "model": {
                    "calculator": _deep_update(
                        dict(model_config.get("calculator", {})),
                        {
                            "kind": calculator_mode.kind,
                            "energy_semantics": calculator_mode.energy_semantics,
                            "polarization_source": calculator_mode.polarization_source,
                            "requires_reference_polarization": calculator_mode.requires_reference_polarization,
                        },
                    ),
                    "charges": _serialize_charge_map(charge_map),
                    "field": {
                        "kind": dict(model_config.get("field", {})).get("kind", "none"),
                        "value": [float(value) for value in np.asarray(field_vector, dtype=float)],
                    },
                },
                "band": _deep_update(
                    dict(mapping.get("band", {})),
                    {}
                    if express is None
                    else {
                        "express": {
                            "units": "GPa",
                            "tensor": [[float(value) for value in row] for row in express],
                        }
                    },
                ),
                "optimizer": _deep_update(
                    dict(mapping.get("optimizer", {})),
                    {
                        "kind": optimizer_kind,
                    },
                ),
                "outputs": _serialize_output_settings(
                    output_settings,
                    output_interval=output_interval,
                ),
            },
        )

        return cls(
            structures=structures,
            structure_indices=structure_indices,
            num_images=num_images,
            calculator=calculator,
            calculator_mode=calculator_mode,
            charge_map=charge_map,
            field_vector=field_vector,
            reference_atoms=reference_atoms,
            run_dir=run_dir,
            spring=float(band_config.get("spring", 5.0)),
            method=str(band_config.get("method", "ci")),
            filter_factory=_build_filter_factory(dict(dict(mapping.get("constraints", {})).get("filter", {}))),
            remesh_stages=_resolve_remesh_stages(mapping.get("staging")),
            image_mobility_rates=dict(optimizer_config.get("image_mobility_rates", {})),
            ci_activation_iteration=ci_activation.get("iteration"),
            ci_activation_force=ci_activation.get("force"),
            optimizer_kind=optimizer_kind,
            band_kwargs=band_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            minimize_kwargs={
                "forceConverged": float(convergence_config.get("fmax", 0.01)),
                "maxIterations": int(convergence_config.get("max_steps", 1000)),
            },
            output_settings=output_settings,
            manifest_config=mapping,
            config_path=source_path,
            resolved_config=resolved_config,
        )

    @classmethod
    def from_yaml(cls, path, *, overrides=None):
        raw = load_yaml_file(path)
        if overrides:
            raw = _deep_update(raw, overrides)
        return cls.from_mapping(raw, source_path=path)


def load_field_ssneb_config(path, *, overrides=None):
    """Load and resolve a YAML workflow config file."""

    return FieldSSNEBConfig.from_yaml(path, overrides=overrides)
