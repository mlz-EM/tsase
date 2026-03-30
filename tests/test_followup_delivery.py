import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from tsase.neb.workflows import load_field_ssneb_config, run_field_ssneb


class DummyMaceFieldCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    last_init_kwargs = None

    def __init__(self, **_kwargs):
        super().__init__()
        type(self).last_init_kwargs = dict(_kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        natoms = len(atoms)
        self.results = {
            "energy": 7.5,
            "forces": np.full((natoms, 3), 0.125, dtype=float),
            "stress": np.zeros(6, dtype=float),
            "polarization": np.array([0.1, 0.2, 0.3], dtype=float),
        }


class MissingPolarizationMaceFieldCalculator(DummyMaceFieldCalculator):
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results.pop("polarization", None)


class FollowupDeliveryTests(unittest.TestCase):
    def test_mace_field_mode_uses_direct_contract_and_filtered_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start_path = Path(tmpdir) / "start.xyz"
            end_path = Path(tmpdir) / "end.xyz"
            write(
                start_path,
                Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            write(
                end_path,
                Atoms("Cu2", positions=[[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            config_path = Path(tmpdir) / "mace_field.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  root: run",
                        "path:",
                        "  source:",
                        "    kind: control_points",
                        "    files:",
                        "      - start.xyz",
                        "      - end.xyz",
                        "    indices: [0, 2]",
                        "  num_images: 3",
                        "model:",
                        "  calculator:",
                        "    kind: mace_field",
                        "    model_path: dummy.model",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0, -1.0]",
                        "  field:",
                        "    kind: cartesian",
                        "    value: [0.0, 0.0, 0.02]",
                        "band:",
                        "  ss: false",
                        "optimizer:",
                        "  output_interval: 1",
                        "  dt: 0.01",
                        "  dtmax: 0.01",
                        "  maxmove: 0.01",
                        "  convergence:",
                        "    fmax: 10.0",
                        "    max_steps: 1",
                        "outputs:",
                        "  energy_profile:",
                        "    entries:",
                        "      - enthalpy_adjusted",
                        "      - intrinsic_energy",
                        "      - field_energy",
                        "      - polarization_mag",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch("tsase.neb.workflows.config.load_mace_calculator", return_value=DummyMaceFieldCalculator):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    config = load_field_ssneb_config(config_path)

                self.assertTrue(any("Skipping unsupported" in str(item.message) for item in caught))
                self.assertEqual(config.calculator_mode.kind, "mace_field")
                self.assertIsNone(config.reference_atoms)
                self.assertEqual(
                    DummyMaceFieldCalculator.last_init_kwargs["electric_field"],
                    [0.0, 0.0, 0.02],
                )
                self.assertEqual(
                    config.output_settings["energy_profile_entries"],
                    ["enthalpy_adjusted", "polarization_mag"],
                )

                result = run_field_ssneb(config=config)

            middle_image = result["band"].path[1]
            self.assertIsNone(middle_image.base_u)
            self.assertIsNone(middle_image.field_u)
            self.assertAlmostEqual(float(middle_image.u), 7.5)
            self.assertTrue(np.allclose(middle_image.f, np.full((2, 3), 0.125)))
            self.assertTrue(np.allclose(middle_image.polarization, [0.1, 0.2, 0.3]))

            csv_path = Path(result["artifacts"].energy_dir) / "profile_iter_0001.csv"
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(
                header,
                "image,enthalpy_adjusted_mev_per_atom,polarization_magnitude_c_per_m2",
            )
            diagnostics_path = Path(result["artifacts"].diagnostics_file)
            diagnostics_row = diagnostics_path.read_text(encoding="utf-8").splitlines()[1]
            columns = diagnostics_row.split(",")
            self.assertEqual(columns[3].lower(), "nan")
            self.assertEqual(columns[4].lower(), "nan")

    def test_mace_field_requires_polarization_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start_path = Path(tmpdir) / "start.xyz"
            end_path = Path(tmpdir) / "end.xyz"
            write(
                start_path,
                Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            write(
                end_path,
                Atoms("Cu2", positions=[[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            config_path = Path(tmpdir) / "mace_field_missing_polarization.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  root: run",
                        "path:",
                        "  source:",
                        "    kind: control_points",
                        "    files:",
                        "      - start.xyz",
                        "      - end.xyz",
                        "    indices: [0, 2]",
                        "  num_images: 3",
                        "model:",
                        "  calculator:",
                        "    kind: mace_field",
                        "    model_path: dummy.model",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0, -1.0]",
                        "  field:",
                        "    kind: cartesian",
                        "    value: [0.0, 0.0, 0.02]",
                        "band:",
                        "  ss: false",
                        "optimizer:",
                        "  output_interval: 1",
                        "  dt: 0.01",
                        "  dtmax: 0.01",
                        "  maxmove: 0.01",
                        "  convergence:",
                        "    fmax: 10.0",
                        "    max_steps: 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch(
                "tsase.neb.workflows.config.load_mace_calculator",
                return_value=MissingPolarizationMaceFieldCalculator,
            ):
                config = load_field_ssneb_config(config_path)
                with self.assertRaisesRegex(ValueError, "results\\['polarization'\\]"):
                    run_field_ssneb(config=config)


if __name__ == "__main__":
    unittest.main()
