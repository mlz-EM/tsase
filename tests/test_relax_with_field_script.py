import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write


def _load_relax_with_field_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "relax_with_field.py"
    spec = importlib.util.spec_from_file_location("test_relax_with_field_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyMaceFieldCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    last_init_kwargs = None

    def __init__(self, **kwargs):
        super().__init__()
        type(self).last_init_kwargs = dict(kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        natoms = len(atoms)
        self.results = {
            "energy": 1.5,
            "forces": np.zeros((natoms, 3), dtype=float),
            "stress": np.zeros(6, dtype=float),
            "polarization": np.array([0.1, 0.2, 0.3], dtype=float),
        }


class FakeOptimizer:
    trajectory_paths = []
    instance_count = 0
    targets = []

    def __init__(self, _target, logfile=None, trajectory=None):
        self.logfile = logfile
        self.trajectory = trajectory
        self.nsteps = 1
        type(self).instance_count += 1
        type(self).trajectory_paths.append(trajectory)
        type(self).targets.append(_target)
        if trajectory is not None:
            Path(trajectory).write_text("fake trajectory\n", encoding="utf-8")
        if logfile is not None:
            Path(logfile).write_text("fake optimizer log\n", encoding="utf-8")

    def run(self, fmax, steps):
        return True


class FakeBFGS(FakeOptimizer):
    trajectory_paths = []
    instance_count = 0
    targets = []


class FakeFIRE(FakeOptimizer):
    trajectory_paths = []
    instance_count = 0
    targets = []


class RelaxWithFieldScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_relax_with_field_module()

    def setUp(self):
        DummyMaceFieldCalculator.last_init_kwargs = None
        FakeOptimizer.trajectory_paths = []
        FakeOptimizer.instance_count = 0
        FakeOptimizer.targets = []
        FakeBFGS.trajectory_paths = []
        FakeBFGS.instance_count = 0
        FakeBFGS.targets = []
        FakeFIRE.trajectory_paths = []
        FakeFIRE.instance_count = 0
        FakeFIRE.targets = []

    def test_build_calculator_uses_model_paths_contract(self):
        with mock.patch.object(self.module, "load_mace_calculator", return_value=DummyMaceFieldCalculator):
            calc = self.module.build_calculator(
                model_path=Path("dummy.model"),
                field_vector=[0.0, 0.0, 2000.0],
                device="cpu",
                head="test_head",
                default_dtype="float64",
            )

        self.assertIsInstance(calc, DummyMaceFieldCalculator)
        self.assertEqual(DummyMaceFieldCalculator.last_init_kwargs["model_paths"], "dummy.model")
        self.assertNotIn("model_path", DummyMaceFieldCalculator.last_init_kwargs)
        self.assertEqual(DummyMaceFieldCalculator.last_init_kwargs["electric_field"], [0.0, 0.0, 0.02])
        self.assertEqual(DummyMaceFieldCalculator.last_init_kwargs["default_dtype"], "float64")

    def test_normalize_cell_mask_accepts_axis_triplets(self):
        self.assertEqual(self.module.normalize_cell_mask(None), None)
        self.assertEqual(self.module.normalize_cell_mask([1, 0, 1]), [1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.module.normalize_cell_mask([1, 0, 1, 0, 1, 0]), [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "--cell-mask"):
            self.module.normalize_cell_mask([1, 0])

    def test_run_sweep_writes_per_field_trajectory_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.cif"
            model_path = tmpdir / "dummy.model"
            output_dir = tmpdir / "run"
            model_path.write_text("dummy model\n", encoding="utf-8")
            write(
                input_path,
                Atoms(
                    "Cu2",
                    positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                    cell=[5.0, 5.0, 5.0],
                    pbc=True,
                ),
            )

            args = self.module.parse_args(
                [
                    "--input",
                    str(input_path),
                    "--model-path",
                    str(model_path),
                    "--output-dir",
                    str(output_dir),
                    "--field-start",
                    "0.0",
                    "0.0",
                    "0.0",
                    "--field-end",
                    "0.0",
                    "0.0",
                    "2000.0",
                    "--num-fields",
                    "2",
                    "--device",
                    "cpu",
                    "--head",
                    "test_head",
                    "--cell-mask",
                    "1",
                    "0",
                    "1",
                    "--no-stem",
                ]
            )
            self.assertEqual(args.default_dtype, "float64")
            self.assertEqual(args.optimizer, "bfgs")
            self.assertEqual(args.cell_mask, [1.0, 0.0, 1.0])

            with mock.patch.object(self.module, "load_mace_calculator", return_value=DummyMaceFieldCalculator):
                with mock.patch.object(self.module, "BFGS", FakeBFGS):
                    with mock.patch.object(self.module, "FIRE", FakeFIRE):
                        summary = self.module.run_sweep(args)

            self.assertEqual(len(summary["records"]), 2)
            self.assertEqual(
                FakeBFGS.trajectory_paths,
                [
                    str(output_dir / "logs" / "field_0000.traj"),
                    str(output_dir / "logs" / "field_0001.traj"),
                ],
            )
            self.assertEqual(FakeBFGS.instance_count, 2)
            self.assertEqual(FakeFIRE.instance_count, 0)
            self.assertEqual(type(FakeBFGS.targets[0]).__name__, "FrechetCellFilter")
            self.assertEqual(FakeBFGS.targets[0].mask.tolist(), [[1, 0, 0], [0, 0, 0], [0, 0, 1]])
            expected_polarization = [160.2176634, 320.4353268, 480.6529902]
            for record in summary["records"]:
                self.assertTrue(Path(record["trajectory"]).exists())
                self.assertTrue(Path(record["logfile"]).exists())
                self.assertEqual(Path(record["trajectory"]).parent, output_dir / "logs")
                self.assertEqual(Path(record["logfile"]).parent, output_dir / "logs")
                self.assertTrue(Path(record["structure_cif"]).exists())
                self.assertTrue(Path(record["structure_extxyz"]).exists())
                self.assertEqual(record["field_vector"][0], 0.0)
                self.assertEqual(record["field_vector"][1], 0.0)
                self.assertIn(record["field_vector"][2], [0.0, 2000.0])
                self.assertTrue(np.allclose(record["polarization"], expected_polarization))
            self.assertTrue(Path(summary["summary_csv"]).exists())
            self.assertTrue(Path(summary["sweep_cif"]).exists())
            self.assertIsNone(summary["stem_input_xyz"])
            sweep_images = read(summary["sweep_cif"], ":")
            self.assertEqual(len(sweep_images), 2)
            final_structure = read(summary["records"][-1]["structure_extxyz"])
            self.assertAlmostEqual(
                float(final_structure.info["electric_field_z_kv_per_cm"]),
                2000.0,
            )
            self.assertAlmostEqual(
                float(final_structure.info["polarization_magnitude_uc_per_cm2"]),
                np.linalg.norm(expected_polarization),
            )
            self.assertEqual(Path(summary["logs_dir"]), output_dir / "logs")
            self.assertEqual(summary["default_dtype"], "float64")
            self.assertEqual(summary["optimizer"], "bfgs")
            self.assertEqual(summary["cell_mask"], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.assertEqual(summary["field_units"], "kV/cm")
            self.assertEqual(summary["polarization_units"], "uC/cm^2")


if __name__ == "__main__":
    unittest.main()
