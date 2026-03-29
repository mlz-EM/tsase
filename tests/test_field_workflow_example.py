import tempfile
import unittest
from pathlib import Path

from examples.run_field_ssneb_interpolated import main


class FieldWorkflowExampleTests(unittest.TestCase):
    def test_example_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "field_example"
            smoke_config = Path(__file__).resolve().parents[1] / "examples" / "configs" / "run_field_ssneb_interpolated_smoke.yaml"
            result = main(
                [
                    "--config",
                    str(smoke_config),
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
