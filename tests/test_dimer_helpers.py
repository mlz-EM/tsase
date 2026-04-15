import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from tsase.dimer import (
    apply_mode_displacement,
    identify_downhill_connections,
    relax_downhill_from_saddle,
    run_dimer_from_yaml,
)


class DoubleWellCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.array(atoms.get_positions(), dtype=float)
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        energy = np.sum((x ** 2 - 1.0) ** 2 + y ** 2 + z ** 2)
        forces = np.column_stack(
            (
                -4.0 * x * (x ** 2 - 1.0),
                -2.0 * y,
                -2.0 * z,
            )
        )
        self.results = {
            "energy": float(energy),
            "forces": forces,
            "stress": np.zeros(6, dtype=float),
        }


class DimerHelperTests(unittest.TestCase):
    def test_apply_mode_displacement_moves_atoms_along_mode(self):
        atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
        atoms.calc = DoubleWellCalculator()

        displaced = apply_mode_displacement(
            atoms,
            np.array([[2.0, 0.0, 0.0]]),
            step_size=0.1,
            ss=False,
        )

        self.assertTrue(np.allclose(displaced.get_positions(), [[0.1, 0.0, 0.0]]))
        self.assertTrue(np.allclose(atoms.get_positions(), [[0.0, 0.0, 0.0]]))

    def test_relax_downhill_from_saddle_reaches_opposite_minima(self):
        saddle = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
        saddle.calc = DoubleWellCalculator()

        result = relax_downhill_from_saddle(
            saddle,
            np.array([[1.0, 0.0, 0.0]]),
            step_size=0.1,
            optimizer="BFGS",
            fmax=1.0e-4,
            max_steps=200,
        )

        self.assertLess(result.positive.atoms.positions[0, 0], 1.1)
        self.assertGreater(result.positive.atoms.positions[0, 0], 0.9)
        self.assertGreater(result.negative.atoms.positions[0, 0], -1.1)
        self.assertLess(result.negative.atoms.positions[0, 0], -0.9)

    def test_identify_downhill_connections_matches_reference_minima(self):
        saddle = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
        saddle.calc = DoubleWellCalculator()
        downhill = relax_downhill_from_saddle(
            saddle,
            np.array([[1.0, 0.0, 0.0]]),
            step_size=0.1,
            optimizer="BFGS",
            fmax=1.0e-4,
            max_steps=200,
        )
        references = {
            "right": Atoms("Cu", positions=[[1.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
            "left": Atoms("Cu", positions=[[-1.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
        }

        labels = identify_downhill_connections(downhill, references)

        self.assertEqual(labels["positive"].label, "right")
        self.assertEqual(labels["negative"].label, "left")

    def test_run_dimer_from_yaml_writes_downhill_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            start = Atoms("Cu", positions=[[0.1, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
            target = Atoms("Cu", positions=[[1.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
            start_path = root / "start.xyz"
            target_path = root / "target.xyz"
            write(start_path, start, format="extxyz")
            write(target_path, target, format="extxyz")
            config_path = root / "dimer.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  root: run",
                        "structure:",
                        "  file: start.xyz",
                        "model:",
                        "  calculator:",
                        "    kind: emt",
                        "search:",
                        "  method: lanczos",
                        "  quiet: true",
                        "  mode:",
                        "    kind: difference",
                        "    file: target.xyz",
                        "  dimer:",
                        "    ss: false",
                        "    noZeroModes: false",
                        "convergence:",
                        "  minForce: 10.0",
                        "  maxForceCalls: 2",
                        "postprocess:",
                        "  downhill:",
                        "    enabled: true",
                        "    step_size: 0.1",
                        "    optimizer: BFGS",
                        "    fmax: 0.0001",
                        "    max_steps: 200",
                        "outputs:",
                        "  stem: true",
                        "  stem_interval: 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            stem_result = {
                "status": "ok",
                "frame_dir": str(root / "run" / "stem" / "stem_iter_0001" / "frames"),
                "gif": str(root / "run" / "stem" / "stem_iter_0001" / "stem.gif"),
            }
            with mock.patch("tsase.dimer.workflows.EMT", return_value=DoubleWellCalculator()), mock.patch(
                "tsase.dimer.workflows.analyze_stem_sequence_from_xyz",
                return_value=stem_result,
            ) as stem_mock:
                result = run_dimer_from_yaml(config_path)

            self.assertIsNotNone(result["downhill_result"])
            self.assertTrue(Path(result["artifacts"]["connections_summary"]).exists())
            self.assertTrue(Path(result["artifacts"]["positive_structure"]).exists())
            self.assertTrue(Path(result["artifacts"]["negative_structure"]).exists())
            self.assertTrue((result["run_dir"] / "iterations" / "iter_0001.cif").exists())
            self.assertTrue((result["run_dir"] / "logs" / "dimer_progress.tsv").exists())
            self.assertTrue((result["run_dir"] / "logs" / "live_progress.png").exists())
            progress_log = (result["run_dir"] / "logs" / "dimer_progress.tsv").read_text(encoding="utf-8")
            self.assertIn("delta_e_mev_per_atom", progress_log)
            self.assertIn("\n1\t", progress_log)
            self.assertGreaterEqual(stem_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
