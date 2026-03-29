import tempfile
import unittest
from pathlib import Path

from example.run_field_ssneb_interpolated import main
from examples.run_field_ssneb_interpolated_refactored import main as refactored_main


class FieldWorkflowExampleTests(unittest.TestCase):
    def test_example_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "field_example"
            result = main(
                [
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
            self.assertTrue(Path(artifacts.run_dir).exists())
            self.assertTrue(Path(artifacts.diagnostics_file).exists())
            self.assertTrue(Path(artifacts.manifest_file).exists())
            self.assertTrue(Path(artifacts.xyz_dir, "iter_0000.xyz").exists())

    def test_refactored_example_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "field_example_refactored"
            result = refactored_main(
                [
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
            self.assertTrue(Path(artifacts.run_dir).exists())
            self.assertTrue(Path(artifacts.diagnostics_file).exists())
            self.assertTrue(Path(artifacts.manifest_file).exists())
            self.assertTrue(Path(artifacts.xyz_dir, "iter_0000.xyz").exists())


if __name__ == "__main__":
    unittest.main()
