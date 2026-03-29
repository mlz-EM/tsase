import unittest

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from tsase import neb


class DummyCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        natoms = len(atoms)
        self.results = {
            "energy": 5.0,
            "forces": np.full((natoms, 3), 0.25, dtype=float),
            "stress": np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6], dtype=float),
        }


class FieldModelTests(unittest.TestCase):
    def test_charge_helpers_and_enthalpy_wrapper_regression(self):
        atoms = Atoms(
            "Cu2",
            positions=[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
            cell=np.diag([5.0, 5.0, 5.0]),
            pbc=[True, True, True],
        )
        reference = atoms.copy()
        charges = neb.attach_field_charges(atoms, [1.0, -1.0])
        self.assertTrue(np.allclose(charges, [1.0, -1.0]))
        self.assertTrue(np.allclose(neb.build_charge_array(atoms), [1.0, -1.0]))

        field = np.array([0.2, 0.0, 0.0])
        wrapped = neb.EnthalpyWrapper(
            DummyCalculator(),
            field=field,
            reference_atoms=reference,
            charges=charges,
        )
        atoms.calc = wrapped
        atoms.positions[0, 0] += 0.1
        atoms.positions[1, 0] += 0.3

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()

        dipole = np.array([-0.2, 0.0, 0.0])
        field_energy = -np.dot(field, dipole)
        self.assertTrue(np.isclose(energy, 5.0 + field_energy))
        self.assertTrue(np.allclose(forces[:, 0], [0.45, 0.05]))
        self.assertEqual(stress.shape, (6,))
        self.assertTrue(np.allclose(wrapped.results["dipole"], dipole))
        self.assertIn("polarization_c_per_m2", wrapped.results)


if __name__ == "__main__":
    unittest.main()
