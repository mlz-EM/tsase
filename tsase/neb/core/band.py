"""Core SSNEB band implementation."""

from contextlib import contextmanager
import os
import sys
from copy import deepcopy
from math import atan, pi

import numpy
from ase import units

from tsase.neb.constraints.adapters import make_filter_adapter
from tsase.neb.io.paths import RunLayout
from tsase.neb.io.reporting import make_reporter
from tsase.neb.util import sPBC, vmag, vmag2, vproj, vunit
from tsase.neb.viz.stem import save_projected_neb_sequence

from .geometry import compute_jacobian, image_distance_vector, initialize_image_properties
from .interfaces import ExecutionContext, ImageEvalResult, PathSpec
from .mapping import ensure_atom_ids
from .springs import build_spring_constants
from .state import stress_to_virial
from .tangent import energy_weighted_tangent, geometric_tangent


class ssneb:
    """The generalized nudged elastic path (SSNEB) class."""

    def __init__(
        self,
        p1,
        p2,
        numImages=7,
        k=5.0,
        tangent="new",
        dneb=False,
        dnebOrg=False,
        method="normal",
        onlyci=False,
        weight=1,
        parallel=False,
        ss=True,
        express=numpy.zeros((3, 3)),
        fixstrain=numpy.ones((3, 3)),
        xyz_dir=None,
        filter_factory=None,
        output_dir=None,
        log_file=None,
        adaptive_springs=False,
        kmin=None,
        kmax=None,
        adaptive_eps=1e-3,
        diagnostics_file=None,
        image_update_schedule=None,
    ):
        self.numImages = numImages
        self.k = k * numImages
        self.kmin = (k if kmin is None else kmin) * numImages
        self.kmax = (k if kmax is None else kmax) * numImages
        self.tangent = tangent
        self.dneb = dneb
        self.dnebOrg = dnebOrg
        self.method = method
        self.onlyci = onlyci
        self.weight = weight
        self.ss = ss
        self.express = express * units.GPa
        self.filter_factory = filter_factory
        self.adaptive_springs = adaptive_springs
        self.adaptive_eps = adaptive_eps
        self.context = ExecutionContext.from_parallel_flag(parallel)
        self.parallel = self.context.is_parallel
        self.rank = self.context.rank
        self.size = self.context.size
        self.layout = RunLayout.from_inputs(
            output_dir=output_dir,
            xyz_dir=xyz_dir,
            diagnostics_file=diagnostics_file,
            log_file=log_file,
        )
        self.output_dir = str(self.layout.public_dir)
        self.xyz_dir = str(self.layout.xyz_dir)
        self.diagnostics_file = str(self.layout.diagnostics_file)
        self.log_file = str(self.layout.log_file)
        self.image_update_schedule = []
        self.image_update_rates = {}
        self.frozen_images = set()
        self.CI_index = None
        self.reporter = make_reporter(self.context, self.layout)

        if express[0][1] ** 2 + express[0][2] ** 2 + express[1][2] ** 2 > 1e-3:
            express[0][1] = 0
            express[0][2] = 0
            express[1][2] = 0
            if self.context.is_output_owner:
                print("warning: xy, xz, yz components of the external pressure will be set to zero")
        self.fixstrain = fixstrain

        self.path_spec = PathSpec.from_legacy_inputs(p1, p2, self.numImages)
        for p in self.path_spec.endpoints:
            cr = p.get_cell()
            if cr[0][1] ** 2 + cr[0][2] ** 2 + cr[1][2] ** 2 > 1e-3:
                if self.context.is_output_owner:
                    print("check the orientation of the cell, make sure a is along x, b is on the x-y plane")
                sys.exit()

        n = self.numImages - 1
        p1_first = self.path_spec.structures[0]
        p2_last = self.path_spec.structures[-1]
        self.path = self.path_spec.build_path()
        ensure_atom_ids(self.path)

        calc = p1_first.calc
        for i in range(1, n):
            self.path[i].calc = calc

        self.path[0].calc = p1_first.calc
        self.path[n].calc = p2_last.calc
        self.Umaxi = 1
        self.spring_constants = numpy.full(self.numImages - 1, self.k, dtype=float)
        self.image_filters = []
        self.image_adapters = []
        for image in self.path:
            image_filter = None if self.filter_factory is None else self.filter_factory(image)
            self.image_filters.append(image_filter)
            self.image_adapters.append(make_filter_adapter(image, image_filter))

        self.natom = len(self.path[0])
        self.jacobian = compute_jacobian(
            self.path[0].get_volume(),
            self.path[self.numImages - 1].get_volume(),
            self.natom,
            self.weight,
        )

        endpoint_results = {}
        if self.parallel:
            if self.context.is_output_owner:
                for i in [0, n]:
                    endpoint_results[i] = self._evaluate_image(i)
            endpoint_results = self.context.allgather_image_results(endpoint_results)
            for i in [0, n]:
                self._apply_image_result(i, endpoint_results[i])
                initialize_image_properties(self.path[i], self.jacobian)
                self._finalize_image_state(i)
        else:
            for i in [0, n]:
                self._apply_image_result(i, self._evaluate_image(i))
                initialize_image_properties(self.path[i], self.jacobian)
                self._finalize_image_state(i)
        if not self.parallel:
            for i in range(1, n):
                self._apply_image_result(i, self._evaluate_image(i))
                self._finalize_image_state(i)
        if image_update_schedule is not None:
            for entry in image_update_schedule:
                self.schedule_image_updates(**entry)
        self._update_spring_constants()
        self.update_image_rates(0)
        if self.reporter.is_active:
            self.reporter.initialize_diagnostics()
            if all(hasattr(image, "u") for image in self.path):
                self.write_diagnostics(0)
            self.reporter.write_iteration_xyz(
                self.reporter.make_snapshot_images(self.path),
                0,
                save_projected_neb_sequence,
            )

    @contextmanager
    def image_workspace(self, index):
        workdir = self.layout.image_workdir(index)
        workdir.mkdir(parents=True, exist_ok=True)
        previous = os.getcwd()
        os.chdir(str(workdir))
        try:
            yield workdir
        finally:
            os.chdir(previous)

    def _evaluate_image(self, index):
        image = self.path[index]
        with self.image_workspace(index):
            u = image.get_potential_energy()
            f = image.get_forces()
            base_u = float(getattr(image.calc, "results", {}).get("base_energy", u))
            field_u = float(getattr(image.calc, "results", {}).get("field_energy", 0.0))
            dipole = numpy.array(
                getattr(image.calc, "results", {}).get("dipole", numpy.zeros(3)),
                dtype=float,
            )
            polarization = numpy.array(
                getattr(image.calc, "results", {}).get("polarization", numpy.zeros(3)),
                dtype=float,
            )
            polarization_c_per_m2 = numpy.array(
                getattr(image.calc, "results", {}).get("polarization_c_per_m2", numpy.zeros(3)),
                dtype=float,
            )
            st = numpy.zeros((3, 3))
            if self.ss:
                stress = image.get_stress()
                st = stress_to_virial(stress, image.get_volume())
                st -= self.express * image.get_volume()
                st *= self.fixstrain
        return ImageEvalResult(
            u=float(u),
            base_u=base_u,
            field_u=field_u,
            f=self.image_adapters[index].project_atomic_forces(f),
            st=self.image_adapters[index].project_cell_virial(st),
            dipole=dipole,
            polarization=polarization,
            polarization_c_per_m2=polarization_c_per_m2,
        )

    def _apply_image_result(self, index, result):
        image = self.path[index]
        image.u = float(result.u)
        image.base_u = float(result.base_u)
        image.field_u = float(result.field_u)
        image.f = numpy.array(result.f, dtype=float)
        image.st = numpy.array(result.st, dtype=float)
        image.dipole = numpy.array(result.dipole, dtype=float)
        image.polarization = numpy.array(result.polarization, dtype=float)
        image.polarization_c_per_m2 = numpy.array(result.polarization_c_per_m2, dtype=float)

    def _finalize_image_state(self, index):
        initialize_image_properties(self.path[index], self.jacobian)
        dcell = self.path[index].get_cell() - self.path[0].get_cell()
        strain = numpy.dot(self.path[0].icell, dcell)
        pv = numpy.vdot(self.express, strain) * self.path[0].get_volume()
        self.path[index].pv = pv
        self.path[index].u += pv
        self.path[index].enthalpy = self.path[index].u

    def _update_spring_constants(self):
        self.spring_constants = build_spring_constants(
            self.path,
            self.numImages,
            self.k,
            adaptive_springs=self.adaptive_springs,
            kmin=self.kmin,
            kmax=self.kmax,
            adaptive_eps=self.adaptive_eps,
        )
        for i, image in enumerate(self.path):
            image.k_left = self.spring_constants[i - 1] if i > 0 else 0.0
            image.k_right = self.spring_constants[i] if i < self.numImages - 1 else 0.0
        self.CI_index = self._get_ci_index()

    def schedule_image_updates(self, factors, iterations, start_iteration=0):
        if iterations is None or int(iterations) <= 0:
            return
        normalized = {}
        for index, factor in dict(factors).items():
            image_index = int(index)
            if image_index <= 0 or image_index >= self.numImages - 1:
                raise ValueError("only intermediate images can be scheduled for update scaling")
            factor = float(factor)
            if factor < 0.0:
                raise ValueError("image update factor must be non-negative")
            normalized[image_index] = factor
        if not normalized:
            return
        self.image_update_schedule.append(
            {
                "factors": dict(sorted(normalized.items())),
                "start_iteration": int(start_iteration),
                "end_iteration": int(start_iteration) + int(iterations),
            }
        )

    def update_image_rates(self, iteration):
        rates = {index: 1.0 for index in range(1, self.numImages - 1)}
        for entry in self.image_update_schedule:
            if entry["start_iteration"] <= int(iteration) < entry["end_iteration"]:
                for index, factor in entry["factors"].items():
                    rates[int(index)] = float(factor)
        self.image_update_rates = rates
        self.frozen_images = {index for index, factor in rates.items() if factor == 0.0}
        for index, image in enumerate(self.path):
            image.update_rate = rates.get(index, 1.0)
            image.is_frozen = image.update_rate == 0.0
        self.CI_index = self._get_ci_index()

    def _get_ci_index(self):
        if self.method != "ci":
            return None
        candidates = [
            index
            for index in range(1, self.numImages - 1)
            if index not in self.frozen_images
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda index: self.path[index].u)

    def write_diagnostics(self, iteration):
        self.reporter.write_diagnostics(iteration, self.path, self.frozen_images)

    def forces(self):
        if self.parallel:
            imgi = self.rank + 1
            local_results = {}
            if 1 <= imgi < self.numImages - 1:
                local_results[imgi] = self._evaluate_image(imgi)
            image_results = self.context.allgather_image_results(local_results)
            for i in range(1, self.numImages - 1):
                self._apply_image_result(i, image_results[i])
        else:
            for i in range(1, self.numImages - 1):
                self._apply_image_result(i, self._evaluate_image(i))

        for i in range(1, self.numImages - 1):
            self._finalize_image_state(i)
            if i == 1 or self.path[i].u > self.Umax:
                self.Umax = self.path[i].u
                self.Umaxi = i
        self._update_spring_constants()

        for i in range(1, self.numImages - 1):
            self.path[i].totalf = numpy.vstack((self.path[i].f, self.path[i].st / self.jacobian))
            self.path[i].realtf = deepcopy(self.path[i].totalf)

            if self.tangent == "old":
                self.path[i].n = self.path[i + 1].r - self.path[i - 1].r
            else:
                self.path[i].n = energy_weighted_tangent(self.path, i)
                if vmag2(self.path[i].n) <= 1e-30:
                    self.path[i].n = geometric_tangent(self.path, i)

        if self.context.is_output_owner:
            tangent_header = "{:>10} {:>16} {:>16} {:>12}".format(
                "ImageNum", "atom", "cell", "pv"
            )
            print("==========!tangent contribution!==========")
            print("Jacobian:", self.jacobian)
            print(tangent_header)
            print("-" * len(tangent_header))
        for i in range(1, self.numImages - 1):
            self.path[i].n = vunit(self.path[i].n)
            if self.context.is_output_owner:
                print(
                    "{:10d} {:16.8f} {:16.8f} {:12.6f}".format(
                        i,
                        vmag(self.path[i].n[:-3]),
                        vmag(self.path[i].n[-3:]),
                        float(getattr(self.path[i], "pv", 0.0)),
                    )
                )

        for i in range(1, self.numImages - 1):
            if self.method == "ci" and self.CI_index is not None and i == self.CI_index:
                self.path[i].totalf -= 2.0 * vproj(self.path[i].totalf, self.path[i].n)
                self.path[i].fPerp = self.path[i].totalf
            else:
                self.path[i].fPerp = self.path[i].totalf - vproj(self.path[i].totalf, self.path[i].n)
                Rm1 = image_distance_vector(self.path[i - 1], self.path[i])
                Rp1 = image_distance_vector(self.path[i + 1], self.path[i])
                km1 = self.spring_constants[i - 1]
                kp1 = self.spring_constants[i]
                self.path[i].fsN = (kp1 * vmag(Rp1) - km1 * vmag(Rm1)) * self.path[i].n

                if self.dneb:
                    self.path[i].fs = Rp1 * kp1 + Rm1 * km1
                    self.path[i].fsperp = self.path[i].fs - vproj(self.path[i].fs, self.path[i].n)
                    self.path[i].fsdneb = self.path[i].fsperp - vproj(self.path[i].fs, self.path[i].fPerp)
                    if not self.dnebOrg:
                        FperpSQ = vmag2(self.path[i].fPerp)
                        FsperpSQ = vmag2(self.path[i].fsperp)
                        if FsperpSQ > 0:
                            self.path[i].fsdneb *= 2.0 / pi * atan(FperpSQ / FsperpSQ)
                else:
                    self.path[i].fsdneb = 0

                self.path[i].totalf = self.path[i].fsdneb + self.path[i].fsN + self.path[i].fPerp

                if self.method == "ci" and self.onlyci:
                    self.path[i].totalf *= 0.0

            rate = float(getattr(self.path[i], "update_rate", 1.0))
            if rate != 1.0:
                self.path[i].fPerp *= rate
                self.path[i].fsN *= rate
                self.path[i].fsdneb *= rate
                self.path[i].totalf *= rate
