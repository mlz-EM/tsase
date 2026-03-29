import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase import io
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from tsase.neb.core.band import ssneb
from tsase.neb.io.restart import load_band_configuration_from_xyz
from tsase.neb.workflows import run_field_ssneb


class WorkspaceWritingCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cwd = Path.cwd()
        cwd.joinpath("marker.txt").write_text(str(cwd), encoding="utf-8")
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3), dtype=float),
        }


class PositionEnergyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        self.results = {
            "energy": float(np.sum(positions[:, 0])),
            "forces": np.zeros((len(atoms), 3), dtype=float),
        }


def make_atoms(shift):
    return Atoms(
        "Cu2",
        positions=[[1.0 + shift, 2.0, 2.5], [4.0 - shift, 2.0, 2.5]],
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=[True, True, True],
    )


class InterfaceCleanupTests(unittest.TestCase):
    def test_per_image_workspace_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = make_atoms(0.0)
            end = make_atoms(0.5)
            calc = WorkspaceWritingCalculator()
            start.calc = calc
            end.calc = calc
            band = ssneb(
                start,
                end,
                numImages=4,
                output_dir=tmpdir,
                ss=False,
            )
            for index in range(band.numImages):
                marker = band.layout.image_workdir(index) / "marker.txt"
                self.assertTrue(marker.exists(), msg=f"missing marker for image {index}")
                self.assertIn(f"image_{index:04d}", marker.read_text(encoding="utf-8"))

    def test_zero_field_workflow_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = np.diag([6.0, 6.0, 6.0])
            common = dict(symbols="Cu2", cell=cell, pbc=[True, True, True])
            structures = [
                Atoms(positions=[[1.6, 3.0, 3.0], [4.4, 3.0, 3.0]], **common),
                Atoms(positions=[[2.1, 2.0, 3.0], [3.9, 4.0, 3.0]], **common),
            ]
            result = run_field_ssneb(
                structures=structures,
                structure_indices=[0, 4],
                num_images=5,
                calculator=EMT(),
                charge_map=np.array([1.0, -1.0], dtype=float),
                run_dir=Path(tmpdir) / "zero_field",
                method="normal",
                band_kwargs={"ss": False},
                optimizer_kwargs={"maxmove": 0.05, "dt": 0.05, "dtmax": 0.05, "output_interval": 1},
                minimize_kwargs={"forceConverged": 10.0, "maxIterations": 1},
            )
            self.assertTrue(np.allclose(result["field_vector"], np.zeros(3)))
            self.assertTrue(Path(result["artifacts"].diagnostics_file).exists())

    def test_restart_reapplies_endpoint_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = make_atoms(0.0)
            end = make_atoms(0.5)
            calc = PositionEnergyCalculator()
            start.calc = calc
            end.calc = calc
            band = ssneb(
                start,
                end,
                numImages=4,
                output_dir=tmpdir,
                ss=False,
            )
            old_u0 = float(band.path[0].u)
            old_un = float(band.path[-1].u)

            restart_images = [image.copy() for image in band.path]
            restart_images[0].positions[:, 0] += 1.0
            restart_images[-1].positions[:, 0] += 2.0
            restart_path = Path(tmpdir) / "restart.xyz"
            io.write(str(restart_path), restart_images, format="extxyz")

            load_band_configuration_from_xyz(band, str(restart_path), remap=False)

            self.assertNotEqual(float(band.path[0].u), old_u0)
            self.assertNotEqual(float(band.path[-1].u), old_un)
            self.assertAlmostEqual(float(band.path[0].u), np.sum(restart_images[0].positions[:, 0]))
            self.assertAlmostEqual(float(band.path[-1].u), np.sum(restart_images[-1].positions[:, 0]))
            self.assertAlmostEqual(float(band.path[0].base_u), float(band.path[0].u))
            self.assertAlmostEqual(float(band.path[-1].base_u), float(band.path[-1].u))


if __name__ == "__main__":
    unittest.main()
