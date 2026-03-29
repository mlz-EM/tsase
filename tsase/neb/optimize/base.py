"""Base optimizer utilities for SSNEB."""

import os
from copy import deepcopy

from ase import io
from numpy import dot, sqrt, vdot

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
        ci_activation_iteration=None,
        ci_activation_force=None,
    ):
        self.band = band
        if xyz_dir is None:
            xyz_dir = getattr(band, "xyz_dir", "neb_xyz")
        self.xyz_dir = xyz_dir
        self.log_file = log_file if log_file is not None else getattr(band, "log_file", "fe.out")
        self.output_interval = max(1, int(output_interval))
        self.plot_property = self._normalize_plot_property(plot_property)
        self.ci_activation_iteration = ci_activation_iteration
        self.ci_activation_force = ci_activation_force
        self._ci_target_method = getattr(self.band, "method", "normal")
        self._ci_active = self._ci_target_method == "ci"
        if self._ci_active and (
            self.ci_activation_iteration is not None
            or self.ci_activation_force is not None
        ):
            self.band.method = "normal"
            self._ci_active = False

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

    def _maybe_activate_ci(self, iteration, force_max):
        if self._ci_active or self._ci_target_method != "ci":
            return False
        if self.ci_activation_iteration is None and self.ci_activation_force is None:
            return False

        activate = False
        if self.ci_activation_iteration is not None and iteration >= self.ci_activation_iteration:
            activate = True
        if self.ci_activation_force is not None and force_max <= self.ci_activation_force:
            activate = True

        if not activate:
            return False

        self.band.method = self._ci_target_method
        self._ci_active = True
        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0):
            print(f"Climbing image activated at iteration {iteration} (Total Force={force_max:.9g})")
        return True

    def _write_iteration_xyz(self, iteration):
        if self.band.parallel and self.band.rank != 0:
            return
        os.makedirs(self.xyz_dir, exist_ok=True)
        images = []
        for i, img in enumerate(self.band.path):
            snap = img.copy()
            snap.calc = None
            snap.info = deepcopy(getattr(img, "info", {}))
            snap.info.pop("spatial_map_order", None)
            snap.info["neb_image"] = i
            if hasattr(img, "u"):
                try:
                    snap.info["energy"] = float(img.u)
                except Exception:
                    pass
            if hasattr(img, "polarization_c_per_m2"):
                try:
                    snap.info["polarization_c_per_m2"] = [float(value) for value in img.polarization_c_per_m2]
                except Exception:
                    pass
            if hasattr(img, "f"):
                try:
                    snap.arrays["forces"] = img.f.copy()
                except Exception:
                    pass
            images.append(snap)
        outfile = os.path.join(self.xyz_dir, f"iter_{iteration:04d}.xyz")
        io.write(outfile, images, format="extxyz")
        save_projected_neb_sequence(images, xyz_dir=self.xyz_dir, iteration=iteration)

    def _save_energy_plot(self, iteration):
        if self.band.parallel and self.band.rank != 0:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        os.makedirs(self.xyz_dir, exist_ok=True)
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
        outfile = os.path.join(self.xyz_dir, f"energy_iter_{iteration:04d}.png")
        fig.tight_layout()
        fig.savefig(outfile, dpi=150)
        plt.close(fig)

    def minimize(self, forceConverged=0.01, maxIterations=1000):
        fMax = 1e300
        iterations = 0
        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0):
            status_header = "{:>10} {:>16} {:>16} {:>12} {:>8} {:>16} {:>10} {:>8}".format(
                "Iteration",
                "Total Force",
                "Perp Force",
                "MaxU",
                "MaxI",
                "Stress on CI",
                "dt",
                "Mode",
            )
            status_separator = "-" * len(status_header)
            print(status_header)
            print(status_separator)
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            feout = open(self.log_file, "a")
            feout.write(status_header + "\n")
            feout.write(status_separator + "\n")
        while fMax > forceConverged and iterations < maxIterations:
            if hasattr(self.band, "update_image_rates"):
                self.band.update_image_rates(iterations)
            self.step()
            fMax = 0.0
            fPMax = 0.0
            for i in range(1, self.band.numImages - 1):
                fi = vmag(self.band.path[i].totalf)
                fPi = vmag(self.band.path[i].fPerp)
                if fi > fMax:
                    fMax = fi
                if fPi > fPMax:
                    fPMax = fPi

            maxi = self.band.Umaxi
            cii = getattr(self.band, "CI_index", None)
            if cii is None:
                cii = maxi
            fci = vmag(self.band.path[cii].st)
            output = "{:10d} {:16.9g} {:16.9g} {:12.9g} {:8d} {:16.9g} {:10.5g} {:>8}".format(
                iterations + 1,
                fMax,
                fPMax,
                self.band.Umax - self.band.path[0].u,
                self.band.Umaxi,
                fci,
                self.dt,
                self.band.method,
            )

            iteration = iterations + 1
            should_output = self._should_output(iteration, maxIterations, fMax <= forceConverged)
            if hasattr(self.band, "write_diagnostics"):
                self.band.write_diagnostics(iteration)
            if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0):
                print("-------------------------SSNEB------------------------------")
                print(status_header)
                print(output)
                print(status_separator)
                feout.write(status_header + "\n")
                feout.write(output + "\n")
                feout.write(status_separator + "\n")
            if should_output:
                self._write_iteration_xyz(iteration)
                self._save_energy_plot(iteration)

            self._maybe_activate_ci(iteration, fMax)
            iterations += 1

        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0):
            summary_header = "{:>10} {:>16} {:>16} {:>16} {:>10}".format(
                "Image", "ReCoords", "E", "RealForce", "Image"
            )
            summary_separator = "-" * len(summary_header)
            print("-----------------------SSNEB Finished------------------------------")
            print(summary_header)
            print(summary_separator)
            feout.write("-----------------------SSNEB Finished------------------------------\n")
            feout.write(summary_header + "\n")
            feout.write(summary_separator + "\n")
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
                print(line)
                feout.write(line + "\n")

        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0):
            feout.close()

