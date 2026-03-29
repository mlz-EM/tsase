"""Backward-compatible import shim for field-coupling helpers."""

from tsase.neb.models.charges import attach_field_charges, build_charge_array
from tsase.neb.models.field import (
    EnthalpyWrapper,
    POLARIZATION_E_A2_TO_C_M2,
    crystal_field_to_cartesian,
)

