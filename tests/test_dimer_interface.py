import unittest

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from tsase.dimer import LanczosDimer, SSDimer, run_lanczos, run_ssdimer


class SaddleLikeMaceFieldCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.array(atoms.get_positions(), dtype=float)
        energy = float(np.sum(-(positions[:, 0] ** 2) + positions[:, 1] ** 2 + positions[:, 2] ** 2))
        forces = np.column_stack(
            (
                2.0 * positions[:, 0],
                -2.0 * positions[:, 1],
                -2.0 * positions[:, 2],
            )
        )
        self.results = {
            "energy": energy,
            "forces": forces,
            "stress": np.zeros(6, dtype=float),
            "polarization": np.array([0.1, 0.2, 0.3], dtype=float),
        }


class MissingPolarizationCalculator(SaddleLikeMaceFieldCalculator):
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results.pop("polarization", None)


def make_atoms():
    return Atoms(
        "Cu",
        positions=[[0.2, 0.1, -0.1]],
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=[True, True, True],
    )


class DimerInterfaceTests(unittest.TestCase):
    def test_ssdimer_from_atoms_preserves_original_structure_by_default(self):
        atoms = make_atoms()
        result = run_ssdimer(
            atoms=atoms,
            calculator=SaddleLikeMaceFieldCalculator(),
            calculator_mode="mace_field",
            mode=np.array([[1.0, 0.0, 0.0]]),
            noZeroModes=False,
            search_kwargs={"minForce": 10.0, "maxForceCalls": 2, "quiet": True},
        )

        self.assertIsInstance(result.search, SSDimer)
        self.assertGreater(result.force_calls, 0)
        self.assertTrue(np.allclose(result.atoms.polarization, [0.1, 0.2, 0.3]))
        self.assertAlmostEqual(float(result.atoms.u), float(result.energy))
        self.assertFalse(hasattr(atoms, "polarization"))

    def test_lanczos_entry_point_supports_mace_field_contract(self):
        atoms = make_atoms()
        result = run_lanczos(
            atoms=atoms,
            calculator=SaddleLikeMaceFieldCalculator(),
            calculator_mode="mace_field",
            mode=np.array([[1.0, 0.0, 0.0]]),
            noZeroModes=False,
            search_kwargs={"minForce": 10.0, "maxForceCalls": 2, "quiet": True},
        )

        self.assertIsInstance(result.search, LanczosDimer)
        self.assertGreater(result.force_calls, 0)
        self.assertTrue(np.allclose(result.atoms.polarization_c_per_m2, [1.602176634, 3.204353268, 4.806529902]))

    def test_mace_field_requires_polarization_results(self):
        atoms = make_atoms()
        search = SSDimer.from_atoms(
            atoms,
            calculator=MissingPolarizationCalculator(),
            calculator_mode="mace_field",
            mode=np.array([[1.0, 0.0, 0.0]]),
            noZeroModes=False,
        )

        with self.assertRaisesRegex(ValueError, "results\\['polarization'\\]"):
            search.update_general_forces(search.R0)


if __name__ == "__main__":
    unittest.main()

