"""Solid-state dimer saddle search helpers.

This module keeps the historical ``SSDimer_atoms`` interface used in TSASE
examples while exposing a cleaner maintained API that matches the refactored
NEB package more closely.
"""

from dataclasses import dataclass
from math import atan, cos, pi, sin, tan
from typing import Optional

import numpy as np
from ase import io

from tsase.neb.core.state import stress_to_virial
from tsase.neb.models.field import POLARIZATION_E_A2_TO_C_M2
from tsase.neb.util import vmag, vrand, vunit


@dataclass
class DimerSearchResult:
    """Summary of a completed dimer search."""

    search: "SSDimer"
    atoms: object
    mode: np.ndarray
    curvature: float
    converged: bool
    steps: int
    force_calls: int
    energy: Optional[float]
    max_force: float


def _copy_atoms_with_calculator(atoms):
    snapshot = atoms.copy()
    snapshot.calc = atoms.calc
    snapshot.info = dict(atoms.info)
    return snapshot


def _safe_atan_ratio(numerator, denominator):
    if abs(denominator) < 1.0e-16:
        return 0.0
    return atan(numerator / denominator)


class SSDimer:
    """Atoms-like wrapper for the solid-state dimer saddle search."""

    def __init__(
        self,
        R0=None,
        mode=None,
        maxStep=0.2,
        dT=0.1,
        dR=0.001,
        phi_tol=5,
        rotationMax=4,
        ss=False,
        express=None,
        rotationOpt="cg",
        weight=1,
        noZeroModes=True,
        calculator=None,
        calculator_mode=None,
    ):
        if R0 is None:
            raise ValueError("R0 must be provided")
        self.R0 = R0
        if calculator is not None:
            self.R0.calc = calculator
        if self.R0.calc is None:
            raise ValueError("SSDimer requires an Atoms object with an attached calculator")

        self.steps = 0
        self.dT = float(dT)
        self.dR = float(dR)
        self.phi_tol = float(phi_tol) / 180.0 * pi
        self.maxStep = float(maxStep)
        self.rotationMax = int(rotationMax)
        self.rotationOpt = str(rotationOpt)
        self.noZeroModes = bool(noZeroModes)
        self.ss = bool(ss)
        self.express = np.zeros((3, 3), dtype=float) if express is None else np.array(express, dtype=float)
        self.weight = float(weight)
        self.calculator_mode = str(
            calculator_mode
            if calculator_mode is not None
            else self.R0.info.get("tsase_calculator_mode", "intrinsic")
        )
        self.R0.info["tsase_calculator_mode"] = self.calculator_mode

        self.natom = len(self.R0)
        self.N = self._normalize_mode(mode)

        self.Ftrans = None
        self.F0 = None
        self.forceCalls = 0
        self.curvature = 1.0
        self.converged = False
        self.E = None
        self.initial_energy = None
        self.target_min_force = None
        self.step_diagnostics = {}
        self.last_rotation_substeps = 0
        self.last_rotation_angle_deg = 0.0
        self.last_mode_status = "rotating"
        self.last_alpha = 0.0
        self.last_ftrans_norm = 0.0
        self.last_translation_mode = "drag_up"

        self.R1 = self._copy_runtime_atoms(self.R0)
        self.R1_prime = self._copy_runtime_atoms(self.R0)

        volume = self.R0.get_volume()
        avglen = (volume / self.natom) ** (1.0 / 3.0)
        self.jacobian = avglen * self.natom ** 0.5 * self.weight

        if self.rotationOpt == "bfgs":
            ndim = (self.natom + 3) * 3
            self.Binv0 = np.eye(ndim) / 60.0
            self.Binv = self.Binv0
            self.B0 = np.eye(ndim) * 60.0
            self.B = self.B0

    @classmethod
    def from_atoms(
        cls,
        atoms,
        *,
        calculator=None,
        calculator_mode=None,
        copy_atoms=True,
        **kwargs,
    ):
        """Build a maintained dimer search object from an ASE ``Atoms``."""

        working_atoms = _copy_atoms_with_calculator(atoms) if copy_atoms else atoms
        if calculator is None and working_atoms.calc is None and atoms.calc is not None:
            working_atoms.calc = atoms.calc
        if calculator is not None:
            working_atoms.calc = calculator
        if calculator_mode is not None:
            working_atoms.info["tsase_calculator_mode"] = str(calculator_mode)
        return cls(R0=working_atoms, calculator_mode=calculator_mode, **kwargs)

    def __getattr__(self, attr):
        """Proxy unknown attributes to the wrapped ``Atoms`` object."""

        return getattr(self.R0, attr)

    def __len__(self):
        return self.natom + 3 if self.ss else self.natom

    def _copy_runtime_atoms(self, atoms):
        snapshot = _copy_atoms_with_calculator(atoms)
        snapshot.info["tsase_calculator_mode"] = self.calculator_mode
        return snapshot

    def _normalize_mode(self, mode):
        if mode is None:
            print("randomly initialize the lowest eigenvector")
            normalized = vrand(np.zeros((self.natom + 3, 3)))
            normalized[-3:] *= 0.0
        else:
            normalized = np.array(mode, dtype=float, copy=True)
            if len(normalized) == self.natom:
                normalized = np.vstack((normalized, np.zeros((3, 3))))
        return vunit(normalized)

    def _record_calculator_results(self, atoms, energy):
        results = getattr(atoms.calc, "results", {})
        calculator_mode = str(atoms.info.get("tsase_calculator_mode", self.calculator_mode))

        if calculator_mode == "mace_field":
            if "polarization" not in results:
                raise ValueError(
                    "mace_field calculators must populate results['polarization'] "
                    "for the maintained dimer workflow"
                )
            atoms.base_u = None
            atoms.field_u = None
        else:
            atoms.base_u = float(results.get("base_energy", energy))
            atoms.field_u = float(results.get("field_energy", 0.0))

        dipole = np.array(results.get("dipole", np.zeros(3)), dtype=float)
        polarization = np.array(results.get("polarization", np.zeros(3)), dtype=float)
        polarization_c_per_m2 = results.get("polarization_c_per_m2")
        if polarization_c_per_m2 is None and polarization.shape == (3,):
            polarization_c_per_m2 = POLARIZATION_E_A2_TO_C_M2 * polarization

        atoms.u = float(energy)
        atoms.dipole = dipole
        atoms.polarization = polarization
        atoms.polarization_c_per_m2 = np.array(polarization_c_per_m2, dtype=float)

    def get_positions(self):
        positions = self.R0.get_positions()
        if self.ss:
            return np.vstack((positions * 0.0, self.R0.get_cell() * 0.0))
        return positions

    def set_positions(self, positions):
        if self.ss:
            cell = self.R0.get_cell()
            cell += np.dot(cell, positions[-3:]) / self.jacobian
            self.R0.set_cell(cell, scale_atoms=True)
            atoms = self.R0.get_positions() + positions[:-3]
            self.R0.set_positions(atoms)
            return
        self.R0.set_positions(positions)

    def update_general_forces(self, atoms):
        """Evaluate one dimer image and return generalized forces."""

        self.forceCalls += 1
        energy = float(atoms.get_potential_energy())
        forces = np.array(atoms.get_forces(), dtype=float)
        self._record_calculator_results(atoms, energy)

        virial = np.zeros((3, 3), dtype=float)
        if self.ss:
            stress = atoms.get_stress()
            virial = stress_to_virial(stress, atoms.get_volume())
            virial -= self.express * atoms.get_volume()
        return np.vstack((forces, virial / self.jacobian))

    def get_curvature(self):
        return self.curvature

    def get_mode(self):
        if self.ss:
            return self.N
        return self.N[:-3]

    def get_forces(self):
        F0 = self.min_mode_search()
        self.F0 = F0
        Fparallel = np.vdot(F0, self.N) * self.N
        Fperp = F0 - Fparallel
        f0_magnitude = vmag(F0)
        alpha = 0.0 if f0_magnitude == 0 else vmag(Fperp) / f0_magnitude
        self.last_alpha = float(alpha)

        if self.curvature > 0:
            self.Ftrans = -1.0 * Fparallel
            self.last_translation_mode = "drag_up"
        else:
            self.Ftrans = Fperp - Fparallel
            self.last_translation_mode = "climb"
        self.last_ftrans_norm = float(vmag(self.Ftrans))
        if self.ss:
            return self.Ftrans
        return self.Ftrans[:-3]

    def project_translation_rotation(self, mode, atoms):
        if not self.noZeroModes:
            return mode

        projected = np.array(mode, copy=True)
        for axis_index in range(3):
            translation = np.zeros((self.natom + 3, 3))
            translation[:, axis_index] = 1.0
            translation = vunit(translation)
            projected -= np.vdot(projected, translation) * translation

        for axis in ("x", "y", "z"):
            rotated = atoms.copy()
            rotated.rotate(axis, 0.02, center="COM", rotate_cell=False)
            rotation_vector = rotated.get_positions() - atoms.get_positions()
            rotation_vector = vunit(rotation_vector)
            projected[:-3] -= np.vdot(projected[:-3], rotation_vector) * rotation_vector
        return projected

    def set_endpoint_position(self, mode, reference_atoms, image):
        displacement = self.dR * mode
        cell0 = reference_atoms.get_cell()
        cell1 = cell0 + np.dot(cell0, displacement[-3:]) / self.jacobian
        image.set_cell(cell1, scale_atoms=True)
        scaled_positions = reference_atoms.get_scaled_positions()
        atoms = np.dot(scaled_positions, cell1) + displacement[:-3]
        image.set_positions(atoms)

    def rotation_update(self):
        self.set_endpoint_position(self.N, self.R0, self.R1)
        return self.update_general_forces(self.R1)

    def rotation_plane(self, Fperp, Fperp_old, previous_mode):
        if self.rotationOpt == "sd":
            self.T = vunit(Fperp)
            return

        if self.rotationOpt == "cg":
            overlap = abs(np.vdot(Fperp, Fperp_old))
            previous_norm = np.vdot(Fperp_old, Fperp_old)
            if overlap <= 0.5 * previous_norm and previous_norm != 0:
                gamma = np.vdot(Fperp, Fperp - Fperp_old) / previous_norm
            else:
                gamma = 0.0
            self.Tnorm = getattr(self, "Tnorm", 0.0)
            direction = Fperp + gamma * self.T * self.Tnorm
            direction -= np.vdot(direction, self.N) * self.N
            self.Tnorm = np.linalg.norm(direction)
            self.T = vunit(direction)
            return

        if self.rotationOpt != "bfgs":
            raise ValueError("rotationOpt must be one of: sd, cg, bfgs")

        step = (self.N - previous_mode).flatten()
        gradient = -Fperp.flatten() / self.dR
        previous_gradient = -Fperp_old.flatten() / self.dR
        y = gradient - previous_gradient

        a = np.dot(step, y)
        dg = np.dot(self.B, step)
        b = np.dot(step, dg)
        if abs(a) < 1.0e-16 or abs(b) < 1.0e-16:
            self.B = self.B0
            self.T = vunit(Fperp)
            return
        self.B += np.outer(y, y) / a - np.outer(dg, dg) / b
        omega, eigenvectors = np.linalg.eigh(self.B)
        direction = np.dot(eigenvectors, np.dot(-gradient, eigenvectors) / np.fabs(omega)).reshape((-1, 3))
        if np.vdot(vunit(direction), vunit(Fperp)) < 0.05:
            direction = Fperp
            self.B = self.B0
        direction -= np.vdot(direction, self.N) * self.N
        self.T = vunit(direction)

    def min_mode_search(self):
        if not self.ss:
            self.N = self.project_translation_rotation(self.N, self.R0)

        F0 = self.update_general_forces(self.R0)
        F1 = self.rotation_update()

        phi_min = 1.5
        Fperp = np.zeros_like(F1)
        iteration = 0
        c0 = self.curvature
        rotation_angle_deg = 0.0
        mode_status = "rotating"

        while abs(phi_min) > self.phi_tol and iteration < self.rotationMax:
            if iteration == 0:
                F0perp = F0 - np.vdot(F0, self.N) * self.N
                F1perp = F1 - np.vdot(F1, self.N) * self.N
                Fperp = 2.0 * (F1perp - F0perp)
                self.T = vunit(Fperp)

            if not self.ss:
                self.T = self.project_translation_rotation(self.T, self.R0)

            c0 = np.vdot(F0 - F1, self.N) / self.dR
            c0d = np.vdot(F0 - F1, self.T) / self.dR * 2.0
            phi_1 = -0.5 * _safe_atan_ratio(c0d, 2.0 * abs(c0))
            if abs(phi_1) <= self.phi_tol:
                rotation_angle_deg = abs(phi_1) * 180.0 / pi
                mode_status = "locked"
                break

            rotated_mode = vunit(self.N * cos(phi_1) + self.T * sin(phi_1))
            self.set_endpoint_position(rotated_mode, self.R0, self.R1_prime)
            F1_prime = self.update_general_forces(self.R1_prime)
            c0_prime = np.vdot(F0 - F1_prime, rotated_mode) / self.dR

            b1 = 0.5 * c0d
            a1 = (c0 - c0_prime + b1 * sin(2.0 * phi_1)) / (1.0 - cos(2.0 * phi_1))
            a0 = 2.0 * (c0 - a1)
            phi_min = 0.5 * _safe_atan_ratio(b1, a1)
            c0_min = 0.5 * a0 + a1 * cos(2.0 * phi_min) + b1 * sin(2.0 * phi_min)

            if c0_min > c0:
                phi_min += pi * 0.5
                c0_min = 0.5 * a0 + a1 * cos(2.0 * phi_min) + b1 * sin(2.0 * phi_min)
            if phi_min > pi * 0.5:
                phi_min -= pi

            previous_mode = self.N
            self.N = vunit(self.N * cos(phi_min) + self.T * sin(phi_min))
            if not self.ss:
                self.N = self.project_translation_rotation(self.N, self.R0)
            c0 = c0_min

            F1 = (
                F1 * (sin(phi_1 - phi_min) / sin(phi_1))
                + F1_prime * (sin(phi_min) / sin(phi_1))
                + F0 * (1.0 - cos(phi_min) - sin(phi_min) * tan(phi_1 * 0.5))
            )

            F0perp = F0 - np.vdot(F0, self.N) * self.N
            F1perp = F1 - np.vdot(F1, self.N) * self.N
            Fperp_old = Fperp
            Fperp = 2.0 * (F1perp - F0perp)
            self.rotation_plane(Fperp, Fperp_old, previous_mode)
            iteration += 1
            rotation_angle_deg = abs(phi_min) * 180.0 / pi
            mode_status = "locked" if abs(phi_min) <= self.phi_tol else "rotating"

        self.curvature = c0
        self.last_rotation_substeps = int(iteration)
        self.last_rotation_angle_deg = float(rotation_angle_deg)
        self.last_mode_status = mode_status
        return F0

    def step(self):
        if self.steps == 0:
            shape = (self.natom + 3, 3) if self.ss else (self.natom, 3)
            self.V = np.zeros(shape, dtype=float)

        self.steps += 1
        force_calls_before = self.forceCalls
        Ftrans = self.get_forces()
        if self.initial_energy is None:
            self.initial_energy = float(getattr(self.R0, "u", 0.0))
        dV = Ftrans * self.dT
        if np.vdot(self.V, Ftrans) > 0 and np.vdot(dV, dV) != 0:
            self.V = dV * (1.0 + np.vdot(dV, self.V) / np.vdot(dV, dV))
        else:
            self.V = dV

        step = self.V * self.dT
        if vmag(step) > self.maxStep:
            step = self.maxStep * vunit(step)
        step_norm = float(vmag(step))

        self.set_positions(self.get_positions() + step)
        self.E = float(self.get_potential_energy())
        self._record_calculator_results(self.R0, self.E)
        if self.initial_energy is None:
            self.initial_energy = float(self.E)
        force_calls_step = int(self.forceCalls - force_calls_before)
        max_force = float(self.get_max_atom_force())
        self.step_diagnostics = {
            "step": int(self.steps),
            "region": self._classify_region(max_force),
            "fmax": max_force,
            "curvature": float(self.curvature),
            "delta_e_mev_per_atom": float(1000.0 * (self.E - self.initial_energy) / self.natom),
            "step_norm": step_norm,
            "force_calls_step": force_calls_step,
            "force_calls_total": int(self.forceCalls),
            "alpha": float(self.last_alpha),
            "rotation_angle_deg": float(self.last_rotation_angle_deg),
            "rotation_substeps": int(self.last_rotation_substeps),
            "ftrans_norm": float(self.last_ftrans_norm),
            "mode_status": str(self.last_mode_status),
            "translation_mode": str(self.last_translation_mode),
            "converged": bool(max_force <= (self.target_min_force or 0.0) and self.curvature <= 0.0),
        }

    def get_max_atom_force(self):
        if self.Ftrans is None:
            return 1000.0
        return max(vmag(component) for component in self.Ftrans)

    def _classify_region(self, max_force):
        threshold = 0.0 if self.target_min_force is None else float(self.target_min_force)
        if self.curvature > 0:
            return "stable"
        if max_force <= threshold:
            return "converged"
        if threshold > 0 and max_force <= 5.0 * threshold:
            return "refine"
        return "saddle"

    def _print_search_header(self):
        print("Step  Region     Fmax(eV/A)      Curv   dE(meV/atom)   StepNorm  FC(step/tot)   Alpha   Mode")
        print("------------------------------------------------------------------------------------------------")

    def _format_progress_line(self):
        diagnostics = dict(self.step_diagnostics)
        return (
            f"{diagnostics['step']:4d}  "
            f"{diagnostics['region']:<9}"
            f"{diagnostics['fmax']:12.6f}  "
            f"{diagnostics['curvature']:9.6f}  "
            f"{diagnostics['delta_e_mev_per_atom']:13.6f}  "
            f"{diagnostics['step_norm']:9.6f}  "
            f"{diagnostics['force_calls_step']:3d}/{diagnostics['force_calls_total']:<7d}  "
            f"{diagnostics['alpha']:7.4f}  "
            f"{diagnostics['mode_status']}"
        )

    def build_result(self):
        return DimerSearchResult(
            search=self,
            atoms=self.R0,
            mode=np.array(self.get_mode(), copy=True),
            curvature=float(self.curvature),
            converged=bool(self.converged),
            steps=int(self.steps),
            force_calls=int(self.forceCalls),
            energy=None if self.E is None else float(self.E),
            max_force=float(self.get_max_atom_force()),
        )

    def search(
        self,
        minForce=0.01,
        quiet=False,
        maxForceCalls=100000,
        movie=None,
        interval=50,
        output_callback=None,
    ):
        self.converged = False
        self.initial_energy = None
        self.target_min_force = float(minForce)
        self.step_diagnostics = {}
        if movie:
            io.write(movie, self.R0, format="vasp")

        cc = getattr(self, "curvature", 1.0)
        while (self.get_max_atom_force() > minForce or cc > 0) and self.forceCalls < maxForceCalls:
            self.step()

            if movie and self.steps % interval == 0:
                io.write(movie, self.R0, format="vasp", append=True)
            if output_callback is not None:
                output_callback(self)

            if not quiet:
                if self.steps % 50 == 0 or self.steps == 1:
                    self._print_search_header()
                print(self._format_progress_line())
            cc = self.curvature

        if self.get_max_atom_force() <= minForce and self.curvature <= 0:
            self.converged = True
        return self.build_result()

    # Backward-compatible method aliases for the historical TSASE API.
    project_translt_rott = project_translation_rotation
    iset_endpoint_pos = set_endpoint_position
    minmodesearch = min_mode_search
    getMaxAtomForce = get_max_atom_force


def run_ssdimer(
    *,
    atoms,
    calculator=None,
    calculator_mode=None,
    copy_atoms=True,
    search_kwargs=None,
    **dimer_kwargs,
):
    """Convenience entry point that mirrors the maintained NEB workflow style."""

    search = SSDimer.from_atoms(
        atoms,
        calculator=calculator,
        calculator_mode=calculator_mode,
        copy_atoms=copy_atoms,
        **dimer_kwargs,
    )
    return search.search(**({} if search_kwargs is None else dict(search_kwargs)))


SSDimer_atoms = SSDimer
