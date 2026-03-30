"""Base optimizer utilities for SSNEB."""

from numpy import dot, sqrt, vdot
import warnings

from tsase.neb.io.energy_profile import (
    ENTRY_METADATA,
    build_energy_profile_rows,
    normalize_energy_profile_entries,
)
from tsase.neb.util import sPBC, vmag
from tsase.neb.viz.stem import analyze_stem_sequence_from_xyz


class minimizer_ssneb:
    """NEB minimizer superclass."""

    def __init__(
        self,
        band,
        output_interval=1,
        energy_profile_entries=None,
        plot_property=None,
        image_mobility_rates=None,
    ):
        self.band = band
        self.output = getattr(band, "output", None)
        self.output_interval = max(1, int(output_interval))
        self.energy_profile_entries = self._normalize_energy_profile_entries(
            energy_profile_entries=energy_profile_entries,
            plot_property=plot_property,
        )
        self.image_mobility_rates = {}
        self.set_image_mobility_rates(image_mobility_rates)

    def _normalize_energy_profile_entries(self, *, energy_profile_entries, plot_property):
        if energy_profile_entries is not None:
            return normalize_energy_profile_entries(energy_profile_entries)
        if plot_property is None:
            if self.output is not None:
                configured = self.output.settings.get("energy_profile_entries")
                if configured is not None:
                    return normalize_energy_profile_entries(configured)
            return []
        warnings.warn(
            "plot_property is deprecated; use energy_profile_entries instead",
            stacklevel=3,
        )
        normalized = str(plot_property).strip().lower()
        aliases = {
            "px": "polarization_x",
            "py": "polarization_y",
            "pz": "polarization_z",
            "p_mag": "polarization_mag",
            "pmag": "polarization_mag",
            "p-magnitude": "polarization_mag",
            "p_magnitude": "polarization_mag",
        }
        if normalized in {"", "none", "off", "false"}:
            return []
        if normalized not in aliases:
            raise ValueError("plot_property must be one of: none, Px, Py, Pz, P_mag")
        return normalize_energy_profile_entries([aliases[normalized]])

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
        if self.output is None:
            return
        self.output.write_path_snapshot(
            self.band.path,
            iteration,
            analyze_stem_sequence_from_xyz,
        )

    def _save_energy_plot(self, iteration):
        if self.output is None:
            return
        entries, rows, skipped_entries = build_energy_profile_rows(self.band.path, self.energy_profile_entries)
        if skipped_entries and self.output.is_active:
            print(
                "Skipping unavailable energy-profile entries: "
                + ", ".join(skipped_entries)
            )
        self.output.write_energy_profile_csv(iteration, entries, rows)
        if not entries or not self.output.settings.get("energy_profile_plot", True):
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        x = [row["image"] for row in rows]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        energy_entries = [
            entry for entry in entries if ENTRY_METADATA[entry]["axis"] == "energy"
        ]
        polarization_entries = [
            entry for entry in entries if ENTRY_METADATA[entry]["axis"] == "polarization"
        ]

        primary_axis = ax1
        secondary_axis = None
        if not energy_entries and polarization_entries:
            primary_axis.set_ylabel("Polarization (C/m^2)")
        elif energy_entries:
            primary_axis.set_ylabel("Energy (meV/atom)")
        primary_axis.set_xlabel("Image Index")
        primary_axis.set_title(f"NEB profile (iter {iteration})")
        primary_axis.grid(True, alpha=0.3)

        lines = []
        labels = []
        for entry in energy_entries or polarization_entries:
            line, = primary_axis.plot(
                x,
                [row[entry] for row in rows],
                marker="o",
                linewidth=1.5,
                label=ENTRY_METADATA[entry]["label"],
            )
            lines.append(line)
            labels.append(ENTRY_METADATA[entry]["label"])

        if energy_entries and polarization_entries:
            secondary_axis = ax1.twinx()
            secondary_axis.set_ylabel("Polarization (C/m^2)")
            for entry in polarization_entries:
                line, = secondary_axis.plot(
                    x,
                    [row[entry] for row in rows],
                    marker="s",
                    linewidth=1.3,
                    linestyle="--",
                    label=ENTRY_METADATA[entry]["label"],
                )
                lines.append(line)
                labels.append(ENTRY_METADATA[entry]["label"])

        primary_axis.legend(lines, labels, loc="best")
        self.output.save_energy_plot(fig, iteration)
        plt.close(fig)

    def ensure_iteration_outputs(self, iteration):
        self._write_iteration_xyz(iteration)
        self._save_energy_plot(iteration)

    def _begin_run(self):
        if self.output is not None and self.output.is_active:
            self.output.initialize_diagnostics()
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
            self.output.open_log(self._status_header, self._status_separator)
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
        if self.output is not None and self.output.is_active:
            self.output.write_status_block(
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
        if self.output is not None and self.output.is_active:
            summary_header = "{:>10} {:>16} {:>16} {:>16} {:>10}".format(
                "Image", "ReCoords", "E", "RealForce", "Image"
            )
            summary_separator = "-" * len(summary_header)
            self.output.write_summary_header(summary_header, summary_separator)
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
                self.output.write_summary_line(line)

        if self.output is not None and self.output.is_active:
            self.output.write_final_path_snapshot(self.band.path)
            self.output.close()

    def _abort_run(self):
        if self.output is not None and self.output.is_active:
            self.output.close()

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
