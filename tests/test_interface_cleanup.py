import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase import io
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from tsase.neb.constraints.adapters import CellFilterAdapter
from tsase.neb.core.band import ssneb
from tsase.neb.optimize.bfgs import bfgs_ssneb
from tsase.neb.optimize.fire import fire_ssneb
from tsase.neb.workflows import FieldSSNEBConfig, load_field_ssneb_config, run_field_ssneb


class WorkspaceWritingCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cwd = Path.cwd()
        cwd.joinpath("marker.txt").write_text(str(cwd), encoding="utf-8")
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3), dtype=float),
        }


class PositionEnergyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        self.results = {
            "energy": float(np.sum(positions[:, 0])),
            "forces": np.zeros((len(atoms), 3), dtype=float),
        }


class ConstantStressCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, stress):
        super().__init__()
        self.stress = np.array(stress, dtype=float)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3), dtype=float),
            "stress": self.stress.copy(),
        }


def make_atoms(shift):
    return Atoms(
        "Cu2",
        positions=[[1.0 + shift, 2.0, 2.5], [4.0 - shift, 2.0, 2.5]],
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=[True, True, True],
    )


class InterfaceCleanupTests(unittest.TestCase):
    def test_per_image_workspace_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = make_atoms(0.0)
            end = make_atoms(0.5)
            calc = WorkspaceWritingCalculator()
            start.calc = calc
            end.calc = calc
            band = ssneb(
                start,
                end,
                numImages=4,
                output_dir=tmpdir,
                ss=False,
            )
            for index in range(band.numImages):
                marker = band.layout.image_workdir(index) / "marker.txt"
                self.assertTrue(marker.exists(), msg=f"missing marker for image {index}")
                self.assertIn(f"image_{index:04d}", marker.read_text(encoding="utf-8"))

    def test_zero_field_workflow_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = np.diag([6.0, 6.0, 6.0])
            common = dict(symbols="Cu2", cell=cell, pbc=[True, True, True])
            structures = [
                Atoms(positions=[[1.6, 3.0, 3.0], [4.4, 3.0, 3.0]], **common),
                Atoms(positions=[[2.1, 2.0, 3.0], [3.9, 4.0, 3.0]], **common),
            ]
            config = FieldSSNEBConfig.from_inputs(
                structures=structures,
                structure_indices=[0, 4],
                num_images=5,
                calculator=EMT(),
                charge_map=np.array([1.0, -1.0], dtype=float),
                run_dir=Path(tmpdir) / "zero_field",
                method="normal",
                band_kwargs={"ss": False},
                optimizer_kwargs={"maxmove": 0.05, "dt": 0.05, "dtmax": 0.05, "output_interval": 1},
                minimize_kwargs={"forceConverged": 10.0, "maxIterations": 1},
            )
            result = run_field_ssneb(config=config)
            self.assertTrue(np.allclose(result["field_vector"], np.zeros(3)))
            self.assertTrue(Path(result["stages"][-1]["artifacts"].diagnostics_file).exists())

    def test_run_field_ssneb_requires_explicit_config(self):
        with self.assertRaisesRegex(TypeError, "config=FieldSSNEBConfig"):
            run_field_ssneb(structures=[])

    def test_yaml_constraints_filter_reaches_runtime_band(self):
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

            config_path = Path(tmpdir) / "constraints.yaml"
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
                        "    kind: emt",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0, -1.0]",
                        "band:",
                        "  ss: false",
                        "  method: normal",
                        "constraints:",
                        "  filter:",
                        "    mask: [0, 1, 0, 1, 0, 1]",
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

            config = load_field_ssneb_config(config_path)
            result = run_field_ssneb(config=config)

            self.assertTrue(all(isinstance(adapter, CellFilterAdapter) for adapter in result["band"].image_adapters))
            self.assertTrue(
                np.allclose(
                    result["band"].image_adapters[0].mask,
                    np.array(
                        [
                            [0.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [0.0, 1.0, 0.0],
                        ]
                    ),
                )
            )

    def test_yaml_dneb_flags_reach_runtime_band(self):
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

            config_path = Path(tmpdir) / "dneb.yaml"
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
                        "    kind: emt",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0, -1.0]",
                        "band:",
                        "  ss: false",
                        "  method: normal",
                        "  dneb: true",
                        "  dnebOrg: true",
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

            config = load_field_ssneb_config(config_path)
            self.assertTrue(config.band_kwargs["dneb"])
            self.assertTrue(config.band_kwargs["dnebOrg"])

            result = run_field_ssneb(config=config)
            self.assertTrue(result["band"].dneb)
            self.assertTrue(result["band"].dnebOrg)

    def test_full_path_xyz_replaces_restart_xyz_in_maintained_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [make_atoms(shift) for shift in (0.0, 0.2, 0.4, 0.5)]
            restart_path = Path(tmpdir) / "saved_path.xyz"
            io.write(str(restart_path), images, format="extxyz")

            config_path = Path(tmpdir) / "path.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "path:",
                        "  source:",
                        "    kind: full_path_xyz",
                        "    file: saved_path.xyz",
                        "    remap_on_restart: none",
                        "model:",
                        "  calculator:",
                        "    kind: emt",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0, -1.0]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            resolved = load_field_ssneb_config(config_path)
            self.assertEqual(resolved.num_images, 4)
            self.assertTrue(np.allclose(resolved.structures[0].positions, images[0].positions))
            self.assertTrue(np.allclose(resolved.structures[-1].positions, images[-1].positions))

    def test_restart_xyz_is_rejected_in_maintained_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [make_atoms(shift) for shift in (0.0, 0.5)]
            restart_path = Path(tmpdir) / "restart.xyz"
            io.write(str(restart_path), images, format="extxyz")

            config_path = Path(tmpdir) / "restart.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "path:",
                        "  source:",
                        "    kind: restart_xyz",
                        "    file: restart.xyz",
                        "model:",
                        "  calculator:",
                        "    kind: emt",
                        "  charges:",
                        "    kind: array",
                        "    values: [1.0, -1.0]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "full_path_xyz"):
                load_field_ssneb_config(config_path)

    def test_optimizer_outputs_stay_under_the_band_output_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = make_atoms(0.0)
            end = make_atoms(0.5)
            calc = EMT()
            start.calc = calc
            end.calc = calc
            band = ssneb(
                start,
                end,
                numImages=4,
                output_dir=Path(tmpdir) / "band_outputs",
                ss=False,
                method="normal",
            )
            optimizer = fire_ssneb(
                band,
                output_interval=1,
                dt=0.01,
                dtmax=0.01,
            )
            optimizer.minimize(forceConverged=10.0, maxIterations=1)

            self.assertTrue((Path(band.output.path_dir) / "iter_0001.cif").exists())
            self.assertTrue((Path(band.output.energy_dir) / "profile_iter_0001.png").exists())
            self.assertTrue((Path(band.output.energy_dir) / "profile.png").exists())
            self.assertTrue(Path(band.output.log_file).exists())
            self.assertFalse((Path(band.output.path_dir) / "iter_0000.cif").exists())

    def test_live_energy_profile_plot_updates_outside_snapshot_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = make_atoms(0.0)
            end = make_atoms(0.5)
            calc = EMT()
            start.calc = calc
            end.calc = calc
            band = ssneb(
                start,
                end,
                numImages=4,
                output_dir=Path(tmpdir) / "live_profile_outputs",
                ss=False,
                method="normal",
            )
            optimizer = fire_ssneb(
                band,
                output_interval=50,
                dt=0.01,
                dtmax=0.01,
            )

            optimizer._begin_run()
            try:
                optimizer.run_iteration(
                    2,
                    max_iterations=100,
                    force_converged=-1.0,
                    convergence_enabled=False,
                )
            finally:
                optimizer._abort_run()

            self.assertFalse((Path(band.output.path_dir) / "iter_0002.cif").exists())
            self.assertFalse((Path(band.output.energy_dir) / "profile_iter_0002.png").exists())
            self.assertTrue((Path(band.output.energy_dir) / "profile.png").exists())

    def test_direct_plot_property_still_overrides_default_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = make_atoms(0.0)
            end = make_atoms(0.5)
            calc = EMT()
            start.calc = calc
            end.calc = calc
            band = ssneb(
                start,
                end,
                numImages=4,
                output_dir=Path(tmpdir) / "plot_property_outputs",
                ss=False,
                method="normal",
            )
            for image in band.path:
                image.polarization_c_per_m2 = np.array([1.0, 2.0, 3.0], dtype=float)
            optimizer = fire_ssneb(
                band,
                output_interval=1,
                dt=0.01,
                dtmax=0.01,
                plot_property="px",
            )
            optimizer.minimize(forceConverged=10.0, maxIterations=1)

            csv_path = Path(band.output.energy_dir) / "profile_iter_0001.csv"
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header, "image,polarization_x_c_per_m2")

    def test_invalid_direct_energy_profile_entries_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = make_atoms(0.0)
            end = make_atoms(0.5)
            calc = EMT()
            start.calc = calc
            end.calc = calc
            band = ssneb(
                start,
                end,
                numImages=4,
                output_dir=Path(tmpdir) / "bad_entries_outputs",
                ss=False,
                method="normal",
            )

            with self.assertRaisesRegex(ValueError, "energy_profile_entries"):
                fire_ssneb(
                    band,
                    output_interval=1,
                    dt=0.01,
                    dtmax=0.01,
                    energy_profile_entries=["polariztion_x"],
                )

    def test_bfgs_records_actual_applied_step_for_variable_cell_updates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            start = make_atoms(0.0)
            end = make_atoms(0.5)
            calc = ConstantStressCalculator([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            start.calc = calc
            end.calc = calc
            band = ssneb(
                start,
                end,
                numImages=4,
                output_dir=Path(tmpdir) / "bfgs_variable_cell",
                ss=True,
                method="normal",
            )
            optimizer = bfgs_ssneb(
                band,
                maxmove=0.2,
                alpha=1.0,
                output_interval=1,
            )

            previous_positions = band.path[1].get_positions().copy()
            previous_cell = np.array(band.path[1].get_cell(), dtype=float)

            optimizer.step()

            stored_step = optimizer._previous_steps[0].reshape(band.natom + 3, 3)
            actual_position_step = band.path[1].get_positions() - previous_positions
            actual_cell_step = (
                np.linalg.solve(
                    previous_cell,
                    np.array(band.path[1].get_cell(), dtype=float) - previous_cell,
                )
                * band.jacobian
            )

            self.assertGreater(np.linalg.norm(actual_position_step), 0.0)
            self.assertTrue(np.allclose(stored_step[: band.natom], actual_position_step))
            self.assertTrue(np.allclose(stored_step[band.natom :], actual_cell_step))


if __name__ == "__main__":
    unittest.main()
