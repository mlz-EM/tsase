"""
The (generalized solid state) nudged elastic path, (ss)neb, module.

When "parallel=True" is set, (ss)neb is parallelized over images through mpi4py.
Each image can only use one processor, because the MPI comminicator cannot be
passed to the calculator. The command to run the neb script should look like:

mpirun -np N python filename.py

where N equals the number of intermedia images, excluding the two end points.

The other parallel version of (ss)neb, pssneb.py, parallelizes over images through
python pool and then each image invokes mpirun when calling the calculator.
pssneb has only been tested for the VASP calculator.
"""


import numpy
import os,sys
from copy import deepcopy
from math import sqrt, atan, pi
from tsase.neb.util import vmag, vmag2, vunit, vproj, vdot, sPBC
from tsase.neb.run_artifacts import resolve_output_paths
from tsase.neb.ssneb_utils import (compute_jacobian, interpolate_path,
                                    initialize_image_properties,
                                    image_distance_vector,
                                    generate_multi_point_path)
from tsase.neb.filtering import make_filter_adapter
from ase import atoms, units, io


def _geometric_tangent(path, index):
    """Return a solid-state geometric tangent for image ``index``.

    Uses the same image-distance metric as the spring term, falling back to the
    nonzero forward/backward segment if the centered difference vanishes.
    """
    forward = image_distance_vector(path[index + 1], path[index])
    backward = image_distance_vector(path[index], path[index - 1])
    tangent = forward + backward
    if vmag2(tangent) > 1e-30:
        return tangent
    if vmag2(forward) > 1e-30:
        return forward
    if vmag2(backward) > 1e-30:
        return backward
    return tangent


class ssneb:
    """
    The generalized nudged elastic path (ssneb) class.
    """

    def __init__(self, p1, p2, numImages = 7, k = 5.0, tangent = "new",       \
                 dneb = False, dnebOrg = False, method = 'normal',            \
                 onlyci = False, weight = 1, parallel = False, ss = True,     \
                 express = numpy.zeros((3,3)), fixstrain = numpy.ones((3,3)), \
                 xyz_dir = "neb_xyz", filter_factory = None,                   \
                 output_dir = None, log_file = None,                           \
                 adaptive_springs = False, kmin = None, kmax = None,          \
                 adaptive_eps = 1e-3, diagnostics_file = "neb_diagnostics.csv", \
                 image_update_schedule = None):
        """
        The neb constructor.
        Parameters:
            p1.......... one endpoint of the path
            p2.......... the other endpoint of the path
            numImages... the total number of images in the path, including the
                         endpoints
            k........... the spring force constant
            tangent..... the tangent method to use, "new" for the new tangent,
                         anything else for the old tangent
            dneb........ set to true to use the double-nudging method
            dnebOrg..... set to true to use the original double-nudging method
            method...... "ci" for the climbing image method, anything else for
                         normal NEB method
            ss.......... boolean, solid-state dimer or regular dimer
            express..... external press, 3*3 lower triangular matrix in the
                         unit of GPa
            fixstrain... 3*3 matrix as express.
                         0 fixes strain at the corresponding direction
        """

        self.numImages = numImages
        self.k         = k * numImages
        self.kmin      = (k if kmin is None else kmin) * numImages
        self.kmax      = (k if kmax is None else kmax) * numImages
        self.tangent   = tangent
        self.dneb      = dneb
        self.dnebOrg   = dnebOrg
        self.method    = method
        self.onlyci    = onlyci
        self.weight    = weight
        self.parallel  = parallel
        self.ss        = ss
        self.express   = express * units.GPa
        self.filter_factory = filter_factory
        self.adaptive_springs = adaptive_springs
        self.adaptive_eps = adaptive_eps
        output_paths = resolve_output_paths(
            output_dir=output_dir,
            xyz_dir=xyz_dir,
            diagnostics_file=diagnostics_file,
            log_file=log_file,
        )
        self.output_dir = output_paths["output_dir"]
        self.diagnostics_file = output_paths["diagnostics_file"]
        self.log_file = output_paths["log_file"]
        self.image_update_schedule = []
        self.image_update_rates = {}
        self.frozen_images = set()
        self.CI_index = None
        self.rank = 0
        self.size = 1
        if express[0][1]**2+express[0][2]**2+express[1][2]**2 > 1e-3:
           express[0][1] = 0
           express[0][2] = 0
           express[1][2] = 0
           if (not self.parallel) or (self.parallel and self.rank == 0):
               print("warning: xy, xz, yz components of the external pressure will be set to zero")
        self.fixstrain = fixstrain

        endpoints_for_check = p1 if isinstance(p1, (list, tuple)) else [p1, p2]
        for p in endpoints_for_check:
            cr = p.get_cell()
            if cr[0][1]**2+cr[0][2]**2+cr[1][2]**2 > 1e-3:
                if (not self.parallel) or (self.parallel and self.rank == 0):
                    print("check the orientation of the cell, make sure a is along x, b is on the x-y plane")
                sys.exit()

        if self.parallel:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.size
            self.rank = self.comm.rank
            self.MPIDB= MPI.DOUBLE

        self.xyz_dir = output_paths["xyz_dir"]

        n = self.numImages - 1
        if isinstance(p1, (list, tuple)):
            endpoints = p1
            indices = p2
            if not isinstance(indices, (list, tuple)):
                raise ValueError("when p1 is a list of endpoints, p2 must be a list of indices")
            p1_first = endpoints[0]
            p2_last = endpoints[-1]
            self.path = generate_multi_point_path(endpoints, indices, self.numImages)
        else:
            p1_first = p1
            p2_last = p2
            self.path = interpolate_path(p1, p2, self.numImages)
        calc = p1_first.calc
        for i in range(1, n):
            self.path[i].calc = calc
        if (not self.parallel) or (self.parallel and self.rank == 0):
            os.makedirs(self.xyz_dir, exist_ok=True)
            io.write(os.path.join(self.xyz_dir, "iter_0000.xyz"), self.path, format="extxyz")
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
            self.path[self.numImages-1].get_volume(),
            self.natom,
            self.weight
        )

        for i in [0,n]:
            self._evaluate_image(i)
            initialize_image_properties(self.path[i], self.jacobian)
            self._finalize_image_state(i)
        if not self.parallel:
            for i in range(1, n):
                self._evaluate_image(i)
                self._finalize_image_state(i)
        if image_update_schedule is not None:
            for entry in image_update_schedule:
                self.schedule_image_updates(**entry)
        self._update_spring_constants()
        self.update_image_rates(0)
        if (not self.parallel) or (self.parallel and self.rank == 0):
            self._initialize_diagnostics()
            if all(hasattr(image, "u") for image in self.path):
                self.write_diagnostics(0)

    def _stress_to_virial(self, stress, volume):
        virial = numpy.zeros((3, 3))
        virial[0][0] = stress[0] * (-volume)
        virial[1][1] = stress[1] * (-volume)
        virial[2][2] = stress[2] * (-volume)
        virial[2][1] = stress[3] * (-volume)
        virial[2][0] = stress[4] * (-volume)
        virial[1][0] = stress[5] * (-volume)
        return virial

    def _evaluate_image(self, index):
        image = self.path[index]
        image.u = image.get_potential_energy()
        image.f = image.get_forces()
        image.base_u = float(getattr(image.calc, "results", {}).get("base_energy", image.u))
        image.field_u = float(getattr(image.calc, "results", {}).get("field_energy", 0.0))
        image.dipole = numpy.array(
            getattr(image.calc, "results", {}).get("dipole", numpy.zeros(3)),
            dtype=float,
        )
        image.polarization = numpy.array(
            getattr(image.calc, "results", {}).get("polarization", numpy.zeros(3)),
            dtype=float,
        )
        image.polarization_c_per_m2 = numpy.array(
            getattr(
                image.calc,
                "results",
                {},
            ).get("polarization_c_per_m2", numpy.zeros(3)),
            dtype=float,
        )
        image.st = numpy.zeros((3, 3))
        if self.ss:
            stress = image.get_stress()
            image.st = self._stress_to_virial(stress, image.get_volume())
            image.st -= self.express * image.get_volume()
            image.st *= self.fixstrain
        image.f = self.image_adapters[index].project_atomic_forces(image.f)
        image.st = self.image_adapters[index].project_cell_virial(image.st)

    def _finalize_image_state(self, index):
        initialize_image_properties(self.path[index], self.jacobian)
        dcell = self.path[index].get_cell() - self.path[0].get_cell()
        strain = numpy.dot(self.path[0].icell, dcell)
        pv = numpy.vdot(self.express, strain) * self.path[0].get_volume()
        self.path[index].pv = pv
        self.path[index].u += pv
        self.path[index].enthalpy = self.path[index].u

    def _update_spring_constants(self):
        if not self.adaptive_springs:
            self.spring_constants = numpy.full(self.numImages - 1, self.k, dtype=float)
        else:
            enthalpies = numpy.array([image.u for image in self.path], dtype=float)
            denominator = max(float(enthalpies.max() - enthalpies.min()), self.adaptive_eps)
            spring_constants = []
            for i in range(self.numImages - 1):
                delta_h = abs(enthalpies[i + 1] - enthalpies[i]) / denominator
                spring_constants.append(
                    self.kmin + (self.kmax - self.kmin) * delta_h
                )
            self.spring_constants = numpy.array(spring_constants, dtype=float)
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
        frozen = {index for index, factor in rates.items() if factor == 0.0}
        self.frozen_images = frozen
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

    def _initialize_diagnostics(self):
        diagnostics_dir = os.path.dirname(self.diagnostics_file)
        if diagnostics_dir:
            os.makedirs(diagnostics_dir, exist_ok=True)
        with open(self.diagnostics_file, "w") as handle:
            handle.write(
                "iteration,image,frozen,update_rate,U,field_term,pv,H,px,py,pz,Px,Py,Pz,k_left,k_right\n"
            )

    def write_diagnostics(self, iteration):
        with open(self.diagnostics_file, "a") as handle:
            for i, image in enumerate(self.path):
                handle.write(
                    "{:d},{:d},{:d},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n".format(
                        int(iteration),
                        i,
                        1 if i in self.frozen_images else 0,
                        float(getattr(image, "update_rate", 1.0)),
                        float(getattr(image, "base_u", image.u - getattr(image, "pv", 0.0))),
                        float(getattr(image, "field_u", 0.0)),
                        float(getattr(image, "pv", 0.0)),
                        float(image.u),
                        float(getattr(image, "dipole", numpy.zeros(3))[0]),
                        float(getattr(image, "dipole", numpy.zeros(3))[1]),
                        float(getattr(image, "dipole", numpy.zeros(3))[2]),
                        float(getattr(image, "polarization", numpy.zeros(3))[0]),
                        float(getattr(image, "polarization", numpy.zeros(3))[1]),
                        float(getattr(image, "polarization", numpy.zeros(3))[2]),
                        float(getattr(image, "k_left", 0.0)),
                        float(getattr(image, "k_right", 0.0)),
                    )
                )

    def forces(self):
        """
        Calculate the forces for each image on the path.  Applies the force due
        to the potential and the spring forces.
        Parameters:
            force - the potential energy force.
        """

        if self.parallel:
            imgi  = self.rank+1
            self._evaluate_image(imgi)

            ui    = self.path[imgi].u
            fi    = self.path[imgi].f
            sti   = self.path[imgi].st
            msg_s = numpy.vstack((fi, sti, [ui,0.0,0.0]))
            msg_r = numpy.zeros((self.size, self.natom+4,3))

            self.comm.Allgather([msg_s, self.MPIDB], [msg_r, self.MPIDB])

            for i in range(1, self.numImages - 1):
                self.path[i].f = msg_r[i-1][:-4]
                self.path[i].st = msg_r[i-1][-4:-1]
                self.path[i].u = msg_r[i-1][-1][0]
        else:
            for i in range(1, self.numImages - 1):
                self._evaluate_image(i)

        for i in range(1, self.numImages - 1):
            self._finalize_image_state(i)

            if i == 1 or self.path[i].u > self.Umax:
                self.Umax  = self.path[i].u
                self.Umaxi = i
        self._update_spring_constants()

        for i in range(1, self.numImages - 1):
            self.path[i].totalf = numpy.vstack((self.path[i].f, self.path[i].st / self.jacobian))
            self.path[i].realtf = deepcopy(self.path[i].totalf)

            if self.tangent == 'old':
                self.path[i].n = (self.path[i + 1].r - self.path[i - 1].r)
            else:
                UPm1 = self.path[i - 1].u > self.path[i].u
                UPp1 = self.path[i + 1].u > self.path[i].u

                if(UPm1 != UPp1):
                    if(UPm1):
                        dr_dir  = sPBC(self.path[i].vdir - self.path[i - 1].vdir)
                        avgbox  = 0.5*(self.path[i].get_cell() + self.path[i - 1].get_cell())
                        sn  = numpy.dot(dr_dir,avgbox)
                        dh  = self.path[i].cellt - self.path[i - 1].cellt
                        snb = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i - 1].icell, dh)*0.5
                        self.path[i].n = numpy.vstack((sn,snb))
                    else:
                        dr_dir  = sPBC(self.path[i + 1].vdir - self.path[i].vdir)
                        avgbox  = 0.5*(self.path[i+1].get_cell() + self.path[i].get_cell())
                        sn  = numpy.dot(dr_dir,avgbox)
                        dh  = self.path[i + 1].cellt - self.path[i].cellt
                        snb = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i + 1].icell, dh)*0.5
                        self.path[i].n = numpy.vstack((sn,snb))
                else:
                    Um1 = self.path[i - 1].u - self.path[i].u
                    Up1 = self.path[i + 1].u - self.path[i].u
                    Umin = min(abs(Up1), abs(Um1))
                    Umax = max(abs(Up1), abs(Um1))
                    if Umax == 0:
                        self.path[i].n = _geometric_tangent(self.path, i)
                    elif(Um1 > Up1):
                        dr_dir  = sPBC(self.path[i + 1].vdir - self.path[i].vdir)
                        avgbox  = 0.5*(self.path[i + 1].get_cell() + self.path[i].get_cell())
                        sn      = numpy.dot(dr_dir,avgbox) * Umin
                        dr_dir  = sPBC(self.path[i].vdir - self.path[i - 1].vdir)
                        avgbox  = 0.5*(self.path[i].get_cell() + self.path[i - 1].get_cell())
                        sn     += numpy.dot(dr_dir,avgbox) * Umax

                        dh   = self.path[i + 1].cellt - self.path[i].cellt
                        snb1 = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i + 1].icell, dh)*0.5
                        dh   = self.path[i].cellt - self.path[i - 1].cellt
                        snb2 = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i - 1].icell, dh)*0.5
                        snb  = snb1 * Umin + snb2 * Umax
                        self.path[i].n = numpy.vstack((sn,snb))
                    else:
                        dr_dir  = sPBC(self.path[i + 1].vdir - self.path[i].vdir)
                        avgbox  = 0.5*(self.path[i + 1].get_cell() + self.path[i].get_cell())
                        sn      = numpy.dot(dr_dir,avgbox) * Umax
                        dr_dir  = sPBC(self.path[i].vdir - self.path[i - 1].vdir)
                        avgbox  = 0.5*(self.path[i].get_cell() + self.path[i - 1].get_cell())
                        sn     += numpy.dot(dr_dir,avgbox) * Umin

                        dh   = self.path[i + 1].cellt - self.path[i].cellt
                        snb1 = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i + 1].icell, dh)*0.5
                        dh   = self.path[i].cellt - self.path[i - 1].cellt
                        snb2 = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i - 1].icell, dh)*0.5
                        snb  = snb1 * Umax + snb2 * Umin
                        self.path[i].n = numpy.vstack((sn,snb))
                    if vmag2(self.path[i].n) <= 1e-30:
                        self.path[i].n = _geometric_tangent(self.path, i)

        if (not self.parallel) or (self.parallel and self.rank == 0):
            tangent_header = "{:>10} {:>16} {:>16} {:>12}".format(
                "ImageNum", "atom", "cell", "pv"
            )
            print("==========!tangent contribution!==========")
            print("Jacobian:", self.jacobian)
            print(tangent_header)
            print("-" * len(tangent_header))
        for i in range(1,self.numImages-1):
            self.path[i].n = vunit(self.path[i].n)
            if (not self.parallel) or (self.parallel and self.rank == 0):
                print(
                    "{:10d} {:16.8f} {:16.8f} {:12.6f}".format(
                        i,
                        vmag(self.path[i].n[:-3]),
                        vmag(self.path[i].n[-3:]),
                        float(getattr(self.path[i], "pv", 0.0)),
                    )
                )

        for i in range(1, self.numImages - 1):
            if self.method == 'ci' and self.CI_index is not None and i == self.CI_index:
                self.path[i].totalf -= 2.0 * vproj(self.path[i].totalf, self.path[i].n)
                self.path[i].fPerp   = self.path[i].totalf
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
                            self.path[i].fsdneb *= 2.0/pi*atan(FperpSQ/FsperpSQ)
                else:
                    self.path[i].fsdneb = 0

                self.path[i].totalf = self.path[i].fsdneb + self.path[i].fsN + self.path[i].fPerp

                if(self.method == 'ci' and self.onlyci):
                    self.path[i].totalf *= 0.0

            rate = float(getattr(self.path[i], "update_rate", 1.0))
            if rate != 1.0:
                self.path[i].fPerp *= rate
                self.path[i].fsN *= rate
                self.path[i].fsdneb *= rate
                self.path[i].totalf *= rate
