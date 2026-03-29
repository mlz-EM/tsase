"""Base optimizer utilities for SSNEB."""

from dataclasses import replace
from pathlib import Path

from numpy import dot, sqrt, vdot

from tsase.neb.io.reporting import Reporter
from tsase.neb.models.field import POLARIZATION_E_A2_TO_C_M2
from tsase.neb.util import sPBC, vmag
from tsase.neb.viz.stem import save_projected_neb_sequence


class minimizer_ssneb:
    """NEB minimizer superclass."""

    def __init__(
        self,
        band,
        xyz_dir=None,
        output_interval=1,
        plot_property=None,
        log_file=None,
        image_mobility_rates=None,
    ):
        self.band = band
        self.reporter = getattr(band, "reporter", None)
        band_layout = getattr(band, "layout", None)
        if xyz_dir is None:
            xyz_dir = getattr(band, "xyz_dir", "neb_xyz")
        self.xyz_dir = xyz_dir
        self.log_file = log_file if log_file is not None else getattr(band, "log_file", "fe.out")
        if (
            self.reporter is not None
            and getattr(self.reporter, "is_active", False)
            and band_layout is not None
            and (
                str(band_layout.xyz_dir) != str(Path(self.xyz_dir).expanduser())
                or str(band_layout.log_file) != str(Path(self.log_file).expanduser())
            )
        ):
            self.reporter = Reporter(
                replace(
                    band_layout,
                    xyz_dir=Path(self.xyz_dir).expanduser().resolve(),
                    log_file=Path(self.log_file).expanduser().resolve(),
                )
        )
        self.output_interval = max(1, int(output_interval))
        self.plot_property = self._normalize_plot_property(plot_property)
        self.image_mobility_rates = {}
        self.set_image_mobility_rates(image_mobility_rates)

    def _normalize_plot_property(self, plot_property):
        if plot_property is None:
            return None
        normalized = str(plot_property).strip().lower()
        if normalized in {"", "none", "off", "false"}:
            return None
        aliases = {
            "px": "px",
            "py": "py",
            "pz": "pz",
            "p_mag": "p_mag",
            "pmag": "p_mag",
            "p-magnitude": "p_mag",
            "p_magnitude": "p_mag",
        }
        if normalized not in aliases:
            raise ValueError("plot_property must be one of: none, Px, Py, Pz, P_mag")
        return aliases[normalized]

    def _get_plot_series(self):
        if self.plot_property is None:
            return None, None

        values = []
        for img in self.band.path:
            polarization = getattr(img, "polarization_c_per_m2", None)
            if polarization is None or len(polarization) != 3:
                polarization = getattr(img, "polarization", None)
                if polarization is None or len(polarization) != 3:
                    return None, None
                polarization = POLARIZATION_E_A2_TO_C_M2 * polarization

            if self.plot_property == "px":
                values.append(float(polarization[0]))
            elif self.plot_property == "py":
                values.append(float(polarization[1]))
            elif self.plot_property == "pz":
                values.append(float(polarization[2]))
            elif self.plot_property == "p_mag":
                values.append(float((polarization[0] ** 2 + polarization[1] ** 2 + polarization[2] ** 2) ** 0.5))

        labels = {
            "px": "Polarization Px (C/m^2)",
            "py": "Polarization Py (C/m^2)",
            "pz": "Polarization Pz (C/m^2)",
            "p_mag": "Polarization |P| (C/m^2)",
        }
        return values, labels[self.plot_property]

    def _should_output(self, iteration, max_iterations, converged):
        return (
            iteration == 1
            or iteration % self.output_interval == 0
            or converged
            or iteration == max_iterations
        )

    def set_image_mobility_rates(self, rates):
        normalized = {}
        if rates is None:
            rates = {}
        for index, rate in dict(rates).items():
            image_index = int(index)
            if image_index <= 0 or image_index >= self.band.numImages - 1:
                raise ValueError("only intermediate images can have mobility rates")
            mobility_rate = float(rate)
            if mobility_rate < 0.0 or mobility_rate > 1.0:
                raise ValueError("image mobility rates must be within [0.0, 1.0]")
            normalized[image_index] = mobility_rate
        self.image_mobility_rates = normalized
        self.band.frozen_images = {
            index for index, rate in normalized.items() if rate == 0.0
        }
        if hasattr(self.band, "_refresh_band_state"):
            self.band._refresh_band_state()

    def get_image_mobility_rate(self, image_index):
        return float(self.image_mobility_rates.get(int(image_index), 1.0))

    def _write_iteration_xyz(self, iteration):
        if self.reporter is None or not self.reporter.is_active:
            return
        images = self.reporter.make_snapshot_images(self.band.path)
        self.reporter.write_iteration_xyz(images, iteration, save_projected_neb_sequence)

    def _save_energy_plot(self, iteration):
        if self.reporter is None or not self.reporter.is_active:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        adjusted_energies = []
        raw_energies = []
        for img in self.band.path:
            if hasattr(img, "u"):
                adjusted_energies.append(float(img.u))
            else:
                try:
                    adjusted_energies.append(float(img.get_potential_energy()))
                except Exception:
                    adjusted_energies.append(0.0)
            raw_energies.append(float(getattr(img, "base_u", adjusted_energies[-1])))
        natoms = max(1, len(self.band.path[0])) if self.band.path else 1
        adjusted_reference = adjusted_energies[0] if adjusted_energies else 0.0
        raw_reference = raw_energies[0] if raw_energies else adjusted_reference
        rel_adjusted = [1000.0 * (energy - adjusted_reference) / natoms for energy in adjusted_energies]
        rel_raw = [1000.0 * (energy - raw_reference) / natoms for energy in raw_energies]
        x = list(range(len(rel_adjusted)))
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(x, rel_raw, marker="o", linewidth=1.5, color="tab:blue", label="Raw energy")
        ax1.plot(x, rel_adjusted, marker="^", linewidth=1.5, color="tab:green", label="Field-adjusted enthalpy")
        ax1.set_xlabel("Image Index")
        ax1.set_ylabel("Relative Energy - E0 (meV/atom)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_title(f"Relative NEB Energies (iter {iteration})")
        ax1.grid(True, alpha=0.3)

        plot_values, plot_label = self._get_plot_series()
        lines = list(ax1.get_lines())
        labels = [line.get_label() for line in lines]
        if plot_values is not None:
            ax2 = ax1.twinx()
            line2, = ax2.plot(
                x,
                plot_values,
                marker="s",
                linewidth=1.3,
                linestyle="--",
                color="tab:orange",
                label=plot_label,
            )
            ax2.set_ylabel(plot_label, color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            lines.append(line2)
            labels.append(plot_label)
        ax1.legend(lines, labels, loc="best")
        self.reporter.save_energy_plot(fig, iteration)
        plt.close(fig)

    def ensure_iteration_outputs(self, iteration):
        self._write_iteration_xyz(iteration)
        self._save_energy_plot(iteration)

    def _begin_run(self):
        if self.reporter is not None and self.reporter.is_active:
            self._status_header = "{:>10} {:>16} {:>16} {:>12} {:>8} {:>16} {:>10} {:>8}".format(
                "Iteration",
                "Total Force",
                "Perp Force",
                "MaxU",
                "MaxI",
                "Stress on CI",
                "dt",
                "Mode",
            )
            self._status_separator = "-" * len(self._status_header)
            print(self._status_header)
            print(self._status_separator)
            self.reporter.open_log(self._status_header, self._status_separator)
        else:
            self._status_header = ""
            self._status_separator = ""

    def run_iteration(self, iteration, *, max_iterations, force_converged, convergence_enabled):
        self.step()
        force_max = 0.0
        force_perp_max = 0.0
        for i in range(1, self.band.numImages - 1):
            fi = vmag(self.band.path[i].totalf)
            f_pi = vmag(self.band.path[i].fPerp)
            if fi > force_max:
                force_max = fi
            if f_pi > force_perp_max:
                force_perp_max = f_pi

        maxi = self.band.Umaxi
        cii = getattr(self.band, "CI_index", None)
        if cii is None:
            cii = maxi
        ci_stress = vmag(self.band.path[cii].st)
        converged = convergence_enabled and force_max <= force_converged
        output = "{:10d} {:16.9g} {:16.9g} {:12.9g} {:8d} {:16.9g} {:10.5g} {:>8}".format(
            iteration,
            force_max,
            force_perp_max,
            self.band.Umax - self.band.path[0].u,
            self.band.Umaxi,
            ci_stress,
            self.dt,
            self.band.method,
        )
        should_output = self._should_output(iteration, max_iterations, converged)
        if hasattr(self.band, "write_diagnostics"):
            self.band.write_diagnostics(iteration)
        if self.reporter is not None and self.reporter.is_active:
            self.reporter.write_status_block(
                self._status_header,
                output,
                self._status_separator,
            )
        if should_output:
            self.ensure_iteration_outputs(iteration)
        return {
            "iteration": int(iteration),
            "fmax": float(force_max),
            "fperp_max": float(force_perp_max),
            "ci_stress": float(ci_stress),
            "max_energy_index": int(self.band.Umaxi),
            "max_energy_delta": float(self.band.Umax - self.band.path[0].u),
            "mode": str(self.band.method),
            "converged": bool(converged),
            "output_written": bool(should_output),
        }

    def _finish_run(self):
        if self.reporter is not None and self.reporter.is_active:
            summary_header = "{:>10} {:>16} {:>16} {:>16} {:>10}".format(
                "Image", "ReCoords", "E", "RealForce", "Image"
            )
            summary_separator = "-" * len(summary_header)
            self.reporter.write_summary_header(summary_header, summary_separator)
        for i in range(self.band.numImages):
            if i == 0:
                Rm1 = 0.0
                R20 = 0.0
                realtotalf = 0.0
            else:
                Rm1 = sPBC(self.band.path[i - 1].vdir - self.band.path[i].vdir)
                avgb = 0.5 * (self.band.path[i - 1].get_cell() + self.band.path[i].get_cell())
                Rm1 = dot(Rm1, avgb)
                dh = self.band.path[i - 1].cellt - self.band.path[i].cellt
                Rm1b = dot(self.band.path[i].icell, dh)
                Rm1 = sqrt(vdot(Rm1, Rm1) + vdot(Rm1b, Rm1b))
                if i == self.band.numImages - 1:
                    realtotalf = 0
                else:
                    realtotalf = vdot(self.band.path[i].realtf, self.band.path[i].n)
            R20 += Rm1
            if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0):
                line = "%3i %13.6f %13.6f %13.6f %3i" % (
                    i,
                    float(R20),
                    float(self.band.path[i].u - self.band.path[0].u),
                    float(realtotalf),
                    i,
                )
                self.reporter.write_summary_line(line)

        if self.reporter is not None and self.reporter.is_active:
            self.reporter.close()

    def _abort_run(self):
        if self.reporter is not None and self.reporter.is_active:
            self.reporter.close()

    def minimize(self, forceConverged=0.01, maxIterations=1000):
        self._begin_run()
        iteration = 0
        while iteration < maxIterations:
            iteration += 1
            metrics = self.run_iteration(
                iteration,
                max_iterations=maxIterations,
                force_converged=forceConverged,
                convergence_enabled=True,
            )
            if metrics["converged"]:
                break
        self._finish_run()
