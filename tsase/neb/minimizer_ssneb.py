'''
ssneb mimizer superclass
'''

import os
from copy import deepcopy
from .util import vmag, sPBC
from ase import io
from numpy import dot, sqrt, vdot

class minimizer_ssneb:
    '''
    Neb minimizer superclass
    '''

    def __init__(
        self,
        band,
        xyz_dir=None,
        output_interval=1,
        ci_activation_iteration=None,
        ci_activation_force=None,
    ):
        self.band = band
        if xyz_dir is None:
            xyz_dir = getattr(band, "xyz_dir", "neb_xyz")
        self.xyz_dir = xyz_dir
        self.output_interval = max(1, int(output_interval))
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
        reasons = []
        if (
            self.ci_activation_iteration is not None
            and iteration >= self.ci_activation_iteration
        ):
            activate = True
            reasons.append(f"iteration >= {self.ci_activation_iteration}")
        if (
            self.ci_activation_force is not None
            and force_max <= self.ci_activation_force
        ):
            activate = True
            reasons.append(f"Total Force <= {self.ci_activation_force}")

        if not activate:
            return False

        self.band.method = self._ci_target_method
        self._ci_active = True
        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0):
            print(
                f"Climbing image activated at iteration {iteration} "
                f"(Total Force={force_max:.9g}; reason: {', '.join(reasons)})"
            )
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
            snap.info["neb_image"] = i
            if hasattr(img, "u"):
                try:
                    snap.info["energy"] = float(img.u)
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
        energies = []
        for img in self.band.path:
            if hasattr(img, "u"):
                energies.append(float(img.u))
            else:
                try:
                    energies.append(float(img.get_potential_energy()))
                except Exception:
                    energies.append(0.0)
        natoms = max(1, len(self.band.path[0])) if self.band.path else 1
        reference_energy = energies[0] if energies else 0.0
        rel_energies = [1000.0 * (energy - reference_energy) / natoms for energy in energies]
        x = list(range(len(rel_energies)))
        plt.figure(figsize=(6, 4))
        plt.plot(x, rel_energies, marker="o", linewidth=1.5)
        plt.xlabel("Image Index")
        plt.ylabel("Relative Energy - E0 (meV/atom)")
        plt.title(f"Relative NEB Energies (iter {iteration})")
        plt.grid(True, alpha=0.3)
        outfile = os.path.join(self.xyz_dir, f"energy_iter_{iteration:04d}.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()

    def minimize(self, forceConverged = 0.01, maxIterations = 1000):
        '''
        Minimize the neb
            forceConverged  - stopping criterion; magnitue of the force vector
            maxForceCalls   - maximum number of force calls allowed
            maxIterations   - maximum number of iterations allowed
        '''
        fMax = 1e300
        iterations = 0
        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0) :
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
            feout = open('fe.out','a')
            feout.write(status_header + '\n')
            feout.write(status_separator + '\n')
        while fMax > forceConverged and iterations < maxIterations:
            self.step()
            fMax = 0.0
            fPMax = 0.0
            for i in range(1, self.band.numImages - 1):
                fi  = vmag(self.band.path[i].totalf)
                fPi = vmag(self.band.path[i].fPerp)
                #fi  = np.max(abs(self.band.path[i].totalf))
                #fPi = np.max(abs(self.band.path[i].fPerp))/self.band.jacobian
                if fi > fMax:
                    fMax = fi
                if fPi > fPMax:
                    fPMax = fPi

            maxi=self.band.Umaxi
            fci =self.band.path[maxi].st 
            fci =vmag(fci)
            #fci =np.max(abs(fci))/self.band.jacobian
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
            should_output = self._should_output(
                iteration, maxIterations, fMax <= forceConverged
            )
            if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0) :
                print("-------------------------SSNEB------------------------------")
                print(status_header)
                print(output)
                print(status_separator)
                feout.write(status_header + '\n')
                feout.write(output+'\n')
                feout.write(status_separator + '\n')
            if should_output:
                self._write_iteration_xyz(iteration)
                self._save_energy_plot(iteration)

            self._maybe_activate_ci(iteration, fMax)
            iterations += 1

        # write data for neb.dat
        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0) :
            summary_header = "{:>10} {:>16} {:>16} {:>16} {:>10}".format(
                "Image",
                "ReCoords",
                "E",
                "RealForce",
                "Image",
            )
            summary_separator = "-" * len(summary_header)
            print("-----------------------SSNEB Finished------------------------------")
            print(summary_header)
            print(summary_separator)
            feout.write("-----------------------SSNEB Finished------------------------------\n")
            feout.write(summary_header + '\n')
            feout.write(summary_separator + '\n')
        for i in range(self.band.numImages):
            if i==0:
                Rm1 = 0.0
                R20 = 0.0
                realtotalf = 0.0
            else:
                Rm1  = sPBC(self.band.path[i - 1].vdir - self.band.path[i].vdir)
                avgb = 0.5*(self.band.path[i - 1].get_cell() + self.band.path[i].get_cell())
                Rm1  = dot(Rm1,avgb) 
                dh   = self.band.path[i - 1].cellt - self.band.path[i].cellt
                Rm1b = dot(self.band.path[i].icell, dh)
                Rm1  = sqrt(vdot(Rm1,Rm1)+vdot(Rm1b,Rm1b))
                if i==self.band.numImages-1:
                    realtotalf = 0
                else:
                    realtotalf = vdot(self.band.path[i].realtf,self.band.path[i].n)
            R20 += Rm1
            if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0) :
                row = "{:10d} {:16.6f} {:16.6f} {:16.6f} {:10d}".format(
                    i,
                    float(R20),
                    float(self.band.path[i].u - self.band.path[0].u),
                    float(realtotalf),
                    i,
                )
                print(row)
                feout.write(row + '\n')

        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0) :
            feout.close()
