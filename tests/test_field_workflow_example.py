import tempfile
import unittest
from pathlib import Path

from ase import Atoms, io

from examples.preprocess_field_ssneb_control_points import main as preprocess_main
from examples.run_field_ssneb_interpolated import main
from tsase.neb.workflows.config import load_field_ssneb_config, load_yaml_file


class FieldWorkflowExampleTests(unittest.TestCase):
    def test_maintained_example_declares_staging_and_energy_profile_entries(self):
        config_path = Path(__file__).resolve().parents[1] / "examples" / "configs" / "run_field_ssneb_interpolated.yaml"
        raw = load_yaml_file(config_path)

        self.assertIn("staging", raw)
        self.assertTrue(raw["staging"]["remesh"])
        self.assertEqual(
            raw["outputs"]["energy_profile"]["entries"],
            ["enthalpy_adjusted", "intrinsic_energy", "field_energy", "polarization_mag"],
        )

    def test_smoke_config_routes_energy_profile_entries_through_outputs(self):
        smoke_config = Path(__file__).resolve().parents[1] / "examples" / "configs" / "run_field_ssneb_interpolated_smoke.yaml"
        resolved = load_field_ssneb_config(smoke_config)

        self.assertEqual(
            resolved.optimizer_kwargs["energy_profile_entries"],
            ["enthalpy_adjusted", "polarization_x"],
        )
        self.assertEqual(len(resolved.remesh_stages), 1)

    def test_optimizer_plot_property_is_rejected_in_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start_path = Path(tmpdir) / "start.xyz"
            end_path = Path(tmpdir) / "end.xyz"
            io.write(
                start_path,
                Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            io.write(
                end_path,
                Atoms("Cu2", positions=[[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            config_path = Path(tmpdir) / "bad_config.yaml"
            config_path.write_text(
                "\n".join(
                    [
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
                        "    kind: emt",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0, -1.0]",
                        "optimizer:",
                        "  plot_property: px",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "outputs.energy_profile.entries"):
                load_field_ssneb_config(config_path)

    def test_optimizer_kind_is_resolved_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start_path = Path(tmpdir) / "start.xyz"
            end_path = Path(tmpdir) / "end.xyz"
            io.write(
                start_path,
                Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            io.write(
                end_path,
                Atoms("Cu2", positions=[[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            config_path = Path(tmpdir) / "bfgs.yaml"
            config_path.write_text(
                "\n".join(
                    [
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
                        "    kind: emt",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0, -1.0]",
                        "optimizer:",
                        "  kind: bfgs",
                        "  maxmove: 0.03",
                        "  alpha: 50.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            resolved = load_field_ssneb_config(config_path)
            self.assertEqual(resolved.optimizer_kind, "bfgs")
            self.assertEqual(resolved.optimizer_kwargs["maxmove"], 0.03)
            self.assertEqual(resolved.optimizer_kwargs["alpha"], 50.0)

    def test_band_express_is_resolved_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start_path = Path(tmpdir) / "start.xyz"
            end_path = Path(tmpdir) / "end.xyz"
            io.write(
                start_path,
                Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            io.write(
                end_path,
                Atoms("Cu2", positions=[[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
                format="extxyz",
            )
            config_path = Path(tmpdir) / "express.yaml"
            config_path.write_text(
                "\n".join(
                    [
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
                        "    kind: emt",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0, -1.0]",
                        "band:",
                        "  express:",
                        "    units: GPa",
                        "    tensor:",
                        "      - [1.0, 3.0, 4.0]",
                        "      - [5.0, 2.0, 6.0]",
                        "      - [7.0, 8.0, 9.0]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            resolved = load_field_ssneb_config(config_path)
            self.assertEqual(
                resolved.band_kwargs["express"].tolist(),
                [[1.0, 0.0, 0.0], [5.0, 2.0, 0.0], [7.0, 8.0, 9.0]],
            )

    def test_preprocess_example_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocess_dir = Path(tmpdir) / "preprocessed"
            smoke_config = Path(__file__).resolve().parents[1] / "examples" / "configs" / "run_field_ssneb_interpolated_smoke.yaml"
            result = preprocess_main(
                [
                    "--config",
                    str(smoke_config),
                    "--output-dir",
                    str(preprocess_dir),
                ]
            )

            self.assertTrue(Path(result["processed_config"]).exists())
            self.assertTrue(Path(result["polarization_reference"]).exists())
            self.assertEqual(len(result["processed_control_points"]), 3)
            for path in result["processed_control_points"]:
                self.assertTrue(Path(path).exists())

    def test_example_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "field_example"
            preprocess_dir = Path(tmpdir) / "preprocessed"
            smoke_config = Path(__file__).resolve().parents[1] / "examples" / "configs" / "run_field_ssneb_interpolated_smoke.yaml"
            preprocess_result = preprocess_main(
                [
                    "--config",
                    str(smoke_config),
                    "--output-dir",
                    str(preprocess_dir),
                ]
            )
            result = main(
                [
                    "--config",
                    str(preprocess_result["processed_config"]),
                    "--output-dir",
                    str(output_dir),
                    "--max-steps",
                    "1",
                    "--num-images",
                    "5",
                    "--fmax",
                    "10.0",
                ]
            )

            artifacts = result["artifacts"]
            workflow_output = result["workflow_output"]
            self.assertTrue(Path(artifacts.run_dir).exists())
            self.assertTrue(Path(artifacts.manifest_file).exists())
            self.assertTrue((workflow_output.paths.config_dir / "workflow_summary.json").exists())
            self.assertTrue(Path(artifacts.diagnostics_file).exists())
            self.assertTrue(Path(artifacts.path_dir, "iter_0001.xyz").exists())


if __name__ == "__main__":
    unittest.main()
