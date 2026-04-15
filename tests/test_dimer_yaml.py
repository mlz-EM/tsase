import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from tsase.dimer import DimerConfig, load_dimer_config, run_dimer_from_yaml


class DummyMaceFieldCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    last_init_kwargs = None

    def __init__(self, **kwargs):
        super().__init__()
        type(self).last_init_kwargs = dict(kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.array(atoms.get_positions(), dtype=float)
        self.results = {
            "energy": float(np.sum(-(positions[:, 0] ** 2) + positions[:, 1] ** 2 + positions[:, 2] ** 2)),
            "forces": np.column_stack(
                (
                    2.0 * positions[:, 0],
                    -2.0 * positions[:, 1],
                    -2.0 * positions[:, 2],
                )
            ),
            "stress": np.zeros(6, dtype=float),
            "polarization": np.array([0.1, 0.2, 0.3], dtype=float),
        }


class DimerYamlWorkflowTests(unittest.TestCase):
    def _write_structure_files(self, root):
        im = Atoms("Cu", positions=[[0.1, 0.2, -0.1]], cell=[5.0, 5.0, 5.0], pbc=True)
        fe = Atoms("Cu", positions=[[0.3, 0.2, -0.1]], cell=[5.0, 5.0, 5.0], pbc=True)
        im_path = root / "im.xyz"
        fe_path = root / "fe.xyz"
        write(im_path, im, format="extxyz")
        write(fe_path, fe, format="extxyz")
        return im_path, fe_path

    def test_load_dimer_config_resolves_mace_field_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            im_path, fe_path = self._write_structure_files(root)
            config_path = root / "lanczos.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  root: run",
                        "  name: im_search",
                        "structure:",
                        "  file: im.xyz",
                        "model:",
                        "  calculator:",
                        "    kind: mace_field",
                        "    model_path: dummy.model",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0]",
                        "  field:",
                        "    kind: cartesian",
                        "    value: [0.0, 0.0, 0.02]",
                        "search:",
                        "  method: lanczos",
                        "  quiet: true",
                        "  mode:",
                        "    kind: difference",
                        f"    file: {fe_path.name}",
                        "  dimer:",
                        "    ss: false",
                        "    noZeroModes: false",
                        "convergence:",
                        "  minForce: 10.0",
                        "  maxForceCalls: 2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch("tsase.dimer.workflows.load_mace_calculator", return_value=DummyMaceFieldCalculator):
                config = load_dimer_config(config_path)

            self.assertIsInstance(config, DimerConfig)
            self.assertEqual(config.method, "lanczos")
            self.assertEqual(config.calculator_mode, "mace_field")
            self.assertTrue(np.allclose(config.field_vector, [0.0, 0.0, 0.02]))
            self.assertTrue(np.allclose(config.mode, [[0.2, 0.0, 0.0]]))
            self.assertEqual(DummyMaceFieldCalculator.last_init_kwargs["electric_field"], [0.0, 0.0, 0.02])

    def test_run_dimer_from_yaml_executes_lanczos_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            im_path, fe_path = self._write_structure_files(root)
            config_path = root / "run_lanczos_dimer.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  root: run",
                        "  name: yaml_lanczos",
                        "structure:",
                        f"  file: {im_path.name}",
                        "model:",
                        "  calculator:",
                        "    kind: mace_field",
                        "    model_path: dummy.model",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0]",
                        "  field:",
                        "    kind: cartesian",
                        "    value: [0.0, 0.0, 0.02]",
                        "search:",
                        "  method: lanczos",
                        "  quiet: true",
                        "  mode:",
                        "    kind: difference",
                        f"    file: {fe_path.name}",
                        "  dimer:",
                        "    ss: false",
                        "    noZeroModes: false",
                        "outputs:",
                        "  movie: lanczos_movie.vasp",
                        "  movie_interval: 1",
                        "convergence:",
                        "  minForce: 10.0",
                        "  maxForceCalls: 2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch("tsase.dimer.workflows.load_mace_calculator", return_value=DummyMaceFieldCalculator):
                result = run_dimer_from_yaml(config_path)

            self.assertEqual(result["config"].method, "lanczos")
            self.assertTrue(np.allclose(result["atoms"].polarization, [0.1, 0.2, 0.3]))
            self.assertTrue((result["run_dir"] / "lanczos_movie.vasp").exists())
            self.assertTrue(Path(result["artifacts"]["resolved_config"]).exists())
            self.assertTrue(Path(result["artifacts"]["summary_file"]).exists())
            self.assertTrue(Path(result["artifacts"]["saddle_structure"]).exists())

    def test_resolved_yaml_preserves_calculator_arguments_for_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            im_path, fe_path = self._write_structure_files(root)
            model_path = root / "dummy.model"
            model_path.write_text("placeholder", encoding="utf-8")
            config_path = root / "run_lanczos_dimer.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  root: run",
                        "  name: yaml_lanczos",
                        "structure:",
                        f"  file: {im_path.name}",
                        "model:",
                        "  calculator:",
                        "    kind: mace_field",
                        f"    model_path: {model_path.name}",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0]",
                        "  field:",
                        "    kind: cartesian",
                        "    value: [0.0, 0.0, 0.02]",
                        "search:",
                        "  method: lanczos",
                        "  quiet: true",
                        "  mode:",
                        "    kind: difference",
                        f"    file: {fe_path.name}",
                        "  dimer:",
                        "    ss: false",
                        "    noZeroModes: false",
                        "convergence:",
                        "  minForce: 10.0",
                        "  maxForceCalls: 2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch("tsase.dimer.workflows.load_mace_calculator", return_value=DummyMaceFieldCalculator):
                result = run_dimer_from_yaml(config_path)
                resolved = load_dimer_config(result["artifacts"]["resolved_config"])

            self.assertEqual(resolved.calculator_mode, "mace_field")
            self.assertEqual(
                DummyMaceFieldCalculator.last_init_kwargs["model_paths"],
                str(model_path.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
