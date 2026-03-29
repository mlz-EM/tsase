import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from tsase import neb


class ZeroForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3), dtype=float),
        }


def make_atoms(shift):
    atoms = Atoms(
        "Cu2",
        positions=[[1.0 + shift, 2.0, 2.5], [4.0 - shift, 2.0, 2.5]],
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=[True, True, True],
    )
    atoms.calc = ZeroForceCalculator()
    return atoms


class StagedWorkflowTests(unittest.TestCase):
    def test_staged_run_writes_transition_artifacts_and_gates_ci(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "staged_run"
            result = neb.run_staged_ssneb(
                structures=[make_atoms(0.0), make_atoms(0.5)],
                num_images=5,
                remesh_stages=[
                    neb.RemeshStage(
                        target_num_images=7,
                        trigger=neb.StabilizedPerpForce(
                            min_iterations=1,
                            relative_drop=0.0,
                            window=1,
                            plateau_tolerance=0.0,
                        ),
                    )
                ],
                k=1.5,
                method="ci",
                output_dir=output_dir,
                band_kwargs={"ss": False},
                optimizer_kwargs={"dt": 0.05, "dtmax": 0.05, "output_interval": 1},
                minimize_kwargs={"forceConverged": 10.0, "maxIterations": 1},
                ci_activation_force=10.0,
            )

            self.assertEqual(result["workflow_summary"]["outcome"], "completed")
            self.assertEqual(result["stages"][0]["exit_reason"], "remesh_triggered")
            self.assertFalse(result["stages"][0]["ci_active"])
            self.assertEqual(result["stages"][1]["exit_reason"], "final_converged")
            self.assertTrue((output_dir / "workflow_manifest.json").exists())
            self.assertTrue((output_dir / "workflow_summary.json").exists())
            self.assertTrue((output_dir / "stage_00" / "run_manifest.json").exists())
            self.assertTrue((output_dir / "stage_00" / "stage_exit.json").exists())
            self.assertTrue((output_dir / "stage_01" / "run_manifest.json").exists())
            self.assertTrue((output_dir / "stage_01" / "stage_exit.json").exists())
            self.assertTrue((output_dir / "transitions" / "remesh_00_to_01.json").exists())
            self.assertTrue((output_dir / "transitions" / "remesh_00_to_01.xyz").exists())

    def test_final_convergence_is_disabled_before_remesh_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "gated_convergence"
            result = neb.run_staged_ssneb(
                structures=[make_atoms(0.0), make_atoms(0.5)],
                num_images=5,
                remesh_stages=[
                    neb.RemeshStage(
                        target_num_images=7,
                        trigger=lambda _history: False,
                        max_wait_iterations=1,
                        on_miss="force",
                    )
                ],
                k=1.5,
                method="normal",
                output_dir=output_dir,
                band_kwargs={"ss": False},
                optimizer_kwargs={"dt": 0.05, "dtmax": 0.05, "output_interval": 1},
                minimize_kwargs={"forceConverged": 10.0, "maxIterations": 1},
            )

            first_stage = result["stages"][0]
            self.assertEqual(first_stage["exit_reason"], "max_iterations_reached")
            self.assertEqual(first_stage["follow_up_action"], "force_remesh")
            self.assertEqual(result["stages"][1]["exit_reason"], "final_converged")

            with (output_dir / "workflow_summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["outcome"], "completed")


if __name__ == "__main__":
    unittest.main()
