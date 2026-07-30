"""Charge helper functions for field-coupled SSNEB."""

import numpy as np


def build_charge_array(atoms, charge_map=None, array_name="field_charges"):
    """Return a per-atom charge array in units of |e|."""
    if array_name in atoms.arrays:
        charges = np.array(atoms.arrays[array_name], dtype=float)
    else:
        charges = np.array(atoms.get_initial_charges(), dtype=float)
        if len(charges) != len(atoms):
            charges = np.zeros(len(atoms), dtype=float)
        if np.allclose(charges, 0.0) and charge_map is not None:
            try:
                charges = np.array(
                    [float(charge_map[symbol]) for symbol in atoms.get_chemical_symbols()],
                    dtype=float,
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing charge assignment for element {exc.args[0]!r}"
                ) from exc
    if len(charges) != len(atoms):
        raise ValueError("charge array length must match the number of atoms")
    return charges


def attach_field_charges(atoms, charges, array_name="field_charges"):
    """Attach fixed charges used by the field-coupled enthalpy wrapper."""
    if isinstance(charges, dict):
        charge_array = build_charge_array(atoms, charge_map=charges, array_name=array_name)
    else:
        charge_array = np.array(charges, dtype=float)
        if len(charge_array) != len(atoms):
            raise ValueError("charge array length must match the number of atoms")
    atoms.arrays[array_name] = charge_array.copy()
    atoms.set_initial_charges(charge_array)
    return charge_array

