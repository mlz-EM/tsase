"""Electric-field coupling utilities for SSNEB workflows."""

import numpy as np
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.geometry import find_mic
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress


POLARIZATION_E_A2_TO_C_M2 = 16.02176634


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


def crystal_field_to_cartesian(cell, field_crystal):
    """Convert field components given in lattice-axis coordinates to Cartesian."""
    field_crystal = np.array(field_crystal, dtype=float)
    if field_crystal.shape != (3,):
        raise ValueError("field_crystal must contain exactly 3 components")
    return np.dot(field_crystal, np.array(cell, dtype=float))


class EnthalpyWrapper(Calculator):
    """ASE calculator wrapper for ionic field coupling on H = U - E·p."""

    implemented_properties = ["energy", "free_energy", "forces", "stress", "dipole"]

    def __init__(
        self,
        base_calculator,
        field=(0.0, 0.0, 0.0),
        reference_atoms=None,
        charges=None,
        charge_array_name="field_charges",
        use_mic=True,
        include_field_stress=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_calculator = base_calculator
        self.field = np.array(field, dtype=float)
        self.charge_array_name = charge_array_name
        self.use_mic = use_mic
        self.include_field_stress = include_field_stress
        self.reference_positions = None
        self.reference_pbc = None
        self.fixed_charges = None if charges is None else np.array(charges, dtype=float)
        if reference_atoms is not None:
            self.set_reference_atoms(reference_atoms)

    def set_field(self, field):
        self.field = np.array(field, dtype=float)
        self.results = {}

    def set_reference_atoms(self, atoms):
        self.reference_positions = np.array(atoms.get_positions(), dtype=float)
        self.reference_pbc = np.array(atoms.get_pbc(), dtype=bool)
        if self.fixed_charges is None:
            self.fixed_charges = build_charge_array(
                atoms, array_name=self.charge_array_name
            )

    def _charges_for_atoms(self, atoms):
        if self.fixed_charges is not None:
            if len(self.fixed_charges) != len(atoms):
                raise ValueError("fixed charges do not match the number of atoms")
            return self.fixed_charges
        return build_charge_array(atoms, array_name=self.charge_array_name)

    def _ionic_dipole(self, atoms, charges):
        if self.reference_positions is None:
            raise ValueError("reference_atoms must be set before evaluating field coupling")
        displacements = np.array(atoms.get_positions(), dtype=float) - self.reference_positions
        if self.use_mic:
            displacements, _ = find_mic(
                displacements,
                cell=atoms.get_cell(),
                pbc=self.reference_pbc if self.reference_pbc is not None else atoms.get_pbc(),
            )
        return np.einsum("i,ij->j", charges, displacements)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms if atoms is None else atoms

        charges = self._charges_for_atoms(atoms)
        base_energy = float(self.base_calculator.get_potential_energy(atoms))
        base_forces = np.array(self.base_calculator.get_forces(atoms), dtype=float)

        base_stress = None
        try:
            base_stress = np.array(self.base_calculator.get_stress(atoms), dtype=float)
        except PropertyNotImplementedError:
            base_stress = None

        dipole = self._ionic_dipole(atoms, charges)
        field_energy = -float(np.dot(self.field, dipole))
        total_energy = base_energy + field_energy
        total_forces = base_forces + charges[:, None] * self.field[None, :]

        volume = float(atoms.get_volume())
        if volume <= 1e-16:
            polarization = np.zeros(3, dtype=float)
            stress_field = np.zeros((3, 3), dtype=float)
        else:
            polarization = dipole / volume
            virial_field = 0.5 * (
                np.outer(self.field, dipole) + np.outer(dipole, self.field)
            )
            # ASE uses virial = -V * stress, so the stress contribution carries a minus sign.
            stress_field = -virial_field / volume

        self.results = {
            "energy": total_energy,
            "free_energy": total_energy,
            "forces": total_forces,
            "dipole": dipole,
            "base_energy": base_energy,
            "field_energy": field_energy,
            "polarization": polarization,
            "polarization_c_per_m2": polarization * POLARIZATION_E_A2_TO_C_M2,
            "charges": charges.copy(),
            "stress_field": stress_field.copy(),
        }

        if base_stress is None:
            if self.include_field_stress and np.linalg.norm(self.field) > 0.0:
                self.results["stress"] = full_3x3_to_voigt_6_stress(stress_field)
        else:
            base_stress_matrix = (
                voigt_6_to_full_3x3_stress(base_stress)
                if base_stress.shape == (6,)
                else np.array(base_stress, dtype=float)
            )
            total_stress_matrix = base_stress_matrix.copy()
            if self.include_field_stress:
                total_stress_matrix += stress_field
            self.results["stress"] = full_3x3_to_voigt_6_stress(total_stress_matrix)
