"""Energy-profile entry helpers for maintained SSNEB outputs."""

from __future__ import annotations

from tsase.neb.models.field import POLARIZATION_E_A2_TO_C_M2


ENTRY_METADATA = {
    "enthalpy_adjusted": {
        "axis": "energy",
        "csv_header": "enthalpy_adjusted_mev_per_atom",
        "label": "Relative enthalpy (meV/atom)",
    },
    "intrinsic_energy": {
        "axis": "energy",
        "csv_header": "intrinsic_energy_mev_per_atom",
        "label": "Relative intrinsic energy (meV/atom)",
    },
    "field_energy": {
        "axis": "energy",
        "csv_header": "field_energy_mev_per_atom",
        "label": "Field energy (meV/atom)",
    },
    "polarization_mag": {
        "axis": "polarization",
        "csv_header": "polarization_magnitude_c_per_m2",
        "label": "Polarization |P| (C/m^2)",
    },
    "polarization_x": {
        "axis": "polarization",
        "csv_header": "polarization_x_c_per_m2",
        "label": "Polarization Px (C/m^2)",
    },
    "polarization_y": {
        "axis": "polarization",
        "csv_header": "polarization_y_c_per_m2",
        "label": "Polarization Py (C/m^2)",
    },
    "polarization_z": {
        "axis": "polarization",
        "csv_header": "polarization_z_c_per_m2",
        "label": "Polarization Pz (C/m^2)",
    },
}

ENTRY_ALIASES = {
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


def normalize_energy_profile_entries(entries):
    normalized = []
    for entry in list(entries or []):
        key = str(entry).strip().lower()
        if key not in ENTRY_ALIASES:
            choices = ", ".join(sorted(ENTRY_METADATA))
            raise ValueError(
                f"energy_profile_entries contains unsupported value {entry!r}; "
                f"expected one of: {choices}"
            )
        canonical = ENTRY_ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _polarization_c_per_m2(image):
    polarization = getattr(image, "polarization_c_per_m2", None)
    if polarization is not None and len(polarization) == 3:
        return polarization
    polarization = getattr(image, "polarization", None)
    if polarization is None or len(polarization) != 3:
        return None
    return POLARIZATION_E_A2_TO_C_M2 * polarization


def _relative_energy_series(images, attribute):
    values = []
    for image in images:
        value = getattr(image, attribute, None)
        if value is None:
            return None
        values.append(float(value))
    natoms = max(1, len(images[0])) if images else 1
    reference = values[0] if values else 0.0
    return [1000.0 * (value - reference) / natoms for value in values]


def _field_energy_series(images):
    values = []
    for image in images:
        value = getattr(image, "field_u", None)
        if value is None:
            return None
        values.append(float(value))
    natoms = max(1, len(images[0])) if images else 1
    return [1000.0 * value / natoms for value in values]


def _polarization_series(images, component):
    values = []
    for image in images:
        polarization = _polarization_c_per_m2(image)
        if polarization is None:
            return None
        if component == "mag":
            values.append(float((polarization[0] ** 2 + polarization[1] ** 2 + polarization[2] ** 2) ** 0.5))
        elif component == "x":
            values.append(float(polarization[0]))
        elif component == "y":
            values.append(float(polarization[1]))
        else:
            values.append(float(polarization[2]))
    return values


def get_entry_series(images, entry):
    if entry == "enthalpy_adjusted":
        return _relative_energy_series(images, "u")
    if entry == "intrinsic_energy":
        return _relative_energy_series(images, "base_u")
    if entry == "field_energy":
        return _field_energy_series(images)
    if entry == "polarization_mag":
        return _polarization_series(images, "mag")
    if entry == "polarization_x":
        return _polarization_series(images, "x")
    if entry == "polarization_y":
        return _polarization_series(images, "y")
    if entry == "polarization_z":
        return _polarization_series(images, "z")
    raise KeyError(entry)


def build_energy_profile_rows(images, requested_entries):
    active_entries = []
    series_map = {}
    skipped_entries = []
    for entry in requested_entries:
        values = get_entry_series(images, entry)
        if values is None:
            skipped_entries.append(entry)
            continue
        active_entries.append(entry)
        series_map[entry] = values

    rows = []
    for index in range(len(images)):
        row = {"image": int(index)}
        for entry in active_entries:
            row[entry] = float(series_map[entry][index])
        rows.append(row)
    return active_entries, rows, skipped_entries
