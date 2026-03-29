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


class FailAfterNCallsCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, fail_after, message):
        super().__init__()
        self.fail_after = int(fail_after)
        self.message = str(message)
        self.call_count = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        self.call_count += 1
        if self.call_count > self.fail_after:
            raise RuntimeError(self.message)
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
            self.assertEqual(Path(result["artifacts"].run_dir), Path(result["stages"][-1]["artifacts"].run_dir))
            self.assertTrue((output_dir / "workflow_manifest.json").exists())
            self.assertTrue((output_dir / "workflow_summary.json").exists())
            self.assertTrue((output_dir / "stage_00" / "run_manifest.json").exists())
            self.assertTrue((output_dir / "stage_00" / "stage_exit.json").exists())
            self.assertTrue((output_dir / "stage_01" / "run_manifest.json").exists())
            self.assertTrue((output_dir / "stage_01" / "stage_exit.json").exists())
            self.assertTrue((output_dir / "transitions" / "remesh_00_to_01.json").exists())
            self.assertTrue((output_dir / "transitions" / "remesh_00_to_01.xyz").exists())
            with (output_dir / "workflow_manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["outputs"]["xyz_dir"], str(result["artifacts"].xyz_dir))
            self.assertEqual(
                manifest["workflow_outputs"]["workflow_summary"],
                str((output_dir / "workflow_summary.json").resolve()),
            )

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

    def test_stage_runtime_failure_is_re_raised_with_original_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "failing_stage"
            calculator = FailAfterNCallsCalculator(
                fail_after=5,
                message="stage boom",
            )
            start = make_atoms(0.0)
            end = make_atoms(0.5)
            start.calc = calculator
            end.calc = calculator

            with self.assertRaisesRegex(RuntimeError, "stage boom"):
                neb.run_staged_ssneb(
                    structures=[start, end],
                    num_images=5,
                    k=1.5,
                    method="normal",
                    output_dir=output_dir,
                    band_kwargs={"ss": False},
                    optimizer_kwargs={"dt": 0.05, "dtmax": 0.05, "output_interval": 1},
                    minimize_kwargs={"forceConverged": 10.0, "maxIterations": 1},
                )

            with (output_dir / "workflow_summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            with (output_dir / "stage_00" / "stage_exit.json").open("r", encoding="utf-8") as handle:
                stage_exit = json.load(handle)
            self.assertEqual(summary["outcome"], "failed")
            self.assertEqual(summary["error"], "stage boom")
            self.assertEqual(stage_exit["exit_reason"], "error")

    def test_stage_exit_always_writes_final_xyz_and_plot_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "stage_outputs"
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
                method="normal",
                output_dir=output_dir,
                band_kwargs={"ss": False},
                optimizer_kwargs={"dt": 0.05, "dtmax": 0.05, "output_interval": 50},
                minimize_kwargs={"forceConverged": 10.0, "maxIterations": 1},
            )

            first_stage_artifacts = result["stages"][0]["artifacts"]
            self.assertTrue((Path(first_stage_artifacts.xyz_dir) / "iter_0001.xyz").exists())
            self.assertTrue((Path(first_stage_artifacts.xyz_dir) / "energy_iter_0001.png").exists())


if __name__ == "__main__":
    unittest.main()
