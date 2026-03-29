import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
from ase import Atoms

from tsase.neb.io.manager import OutputManager
from tsase.neb.viz import stem


class StemVisualizationTests(unittest.TestCase):
    def test_save_projected_neb_sequence_writes_gif_and_frames(self):
        analyses = [
            SimpleNamespace(
                pb_plot_frac=np.zeros((4, 4, 2), dtype=float),
                horizontal_pair_family="oh",
            ),
            SimpleNamespace(
                pb_plot_frac=np.zeros((4, 4, 2), dtype=float),
                horizontal_pair_family="oh",
            ),
        ]

        fake_imageio = types.ModuleType("imageio.v2")

        def fake_imread(path):
            return str(path)

        def fake_mimsave(path, frames, duration):
            Path(path).write_text(f"{duration}:{len(frames)}", encoding="utf-8")

        fake_imageio.imread = fake_imread
        fake_imageio.mimsave = fake_mimsave
        fake_imageio_pkg = types.ModuleType("imageio")
        fake_imageio_pkg.v2 = fake_imageio

        with TemporaryDirectory() as tmpdir:
            xyz_dir = Path(tmpdir)

            def fake_render(_analysis, output_path):
                Path(output_path).write_text("frame", encoding="utf-8")
                return str(output_path)

            with mock.patch.object(stem, "analyze_projected_neb_image", side_effect=analyses), mock.patch.object(
                stem, "render_projected_frame", side_effect=fake_render
            ), mock.patch.dict(
                sys.modules,
                {"imageio": fake_imageio_pkg, "imageio.v2": fake_imageio},
            ):
                result = stem.save_projected_neb_sequence(
                    [object(), object()],
                    xyz_dir=xyz_dir,
                    iteration=7,
                )

            self.assertEqual(result["status"], "ok")
            self.assertTrue((xyz_dir / "stem_iter_0007" / "frame_0000.png").exists())
            self.assertTrue((xyz_dir / "stem_iter_0007" / "frame_0001.png").exists())
            self.assertTrue((xyz_dir / "stem_iter_0007.gif").exists())

    def test_save_projected_neb_sequence_writes_diagnostics_on_analysis_failure(self):
        fake_imageio = types.ModuleType("imageio.v2")
        fake_imageio.imread = lambda path: str(path)
        fake_imageio.mimsave = lambda path, frames, duration: None
        fake_imageio_pkg = types.ModuleType("imageio")
        fake_imageio_pkg.v2 = fake_imageio

        with TemporaryDirectory() as tmpdir:
            xyz_dir = Path(tmpdir)
            with mock.patch.object(
                stem,
                "analyze_projected_neb_image",
                side_effect=stem.StemAnalysisError("bad frame", diagnostics={"frame": 0}),
            ), mock.patch.dict(
                sys.modules,
                {"imageio": fake_imageio_pkg, "imageio.v2": fake_imageio},
            ):
                result = stem.save_projected_neb_sequence(
                    [object()],
                    xyz_dir=xyz_dir,
                    iteration=3,
                )

            self.assertEqual(result["status"], "failed")
            diagnostics_file = Path(result["diagnostics_file"])
            self.assertTrue(diagnostics_file.exists())
            self.assertIn("bad frame", diagnostics_file.read_text(encoding="utf-8"))

    def test_output_manager_surfaces_stem_skip_diagnostics(self):
        with TemporaryDirectory() as tmpdir:
            output = OutputManager.from_run_dir(tmpdir, settings={"stem": True})
            images = [Atoms("H", positions=[[0.0, 0.0, 0.0]])]

            stream = io.StringIO()
            with redirect_stdout(stream):
                output.write_path_snapshot(
                    images,
                    5,
                    lambda *_args, **_kwargs: {
                        "status": "failed",
                        "diagnostics_file": "/tmp/stem_iter_0005_diagnostics.txt",
                    },
                )

            self.assertIn("Projected STEM visualization skipped", stream.getvalue())
            self.assertIn("iter_0005", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
