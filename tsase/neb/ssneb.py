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
from tsase.neb.ssneb_utils import (compute_jacobian, interpolate_path,
                                    initialize_image_properties,
                                    image_distance_vector)
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
                 xyz_dir = "neb_xyz"):
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
        self.tangent   = tangent
        self.dneb      = dneb
        self.dnebOrg   = dnebOrg
        self.method    = method
        self.onlyci    = onlyci
        self.weight    = weight 
        self.parallel  = parallel 
        self.ss        = ss
        self.express   = express * units.GPa
        self.rank = 0
        self.size = 1
        if express[0][1]**2+express[0][2]**2+express[1][2]**2 > 1e-3:
           express[0][1] = 0
           express[0][2] = 0
           express[1][2] = 0
           if (not self.parallel) or (self.parallel and self.rank == 0):
               print("warning: xy, xz, yz components of the external pressure will be set to zero")
        self.fixstrain = fixstrain

        # check the orientation of the cell, make sure a is along x, b is on xoy plane
        endpoints_for_check = p1 if isinstance(p1, (list, tuple)) else [p1, p2]
        for p in endpoints_for_check:
            cr = p.get_cell()
            if cr[0][1]**2+cr[0][2]**2+cr[1][2]**2 > 1e-3: 
                if (not self.parallel) or (self.parallel and self.rank == 0):
                    print("check the orientation of the cell, make sure a is along x, b is on the x-y plane")
                sys.exit()

        # parallel over images through mpi4py
        if self.parallel:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.size
            self.rank = self.comm.rank
            self.MPIDB= MPI.DOUBLE

        self.xyz_dir = xyz_dir

        # set the path by linear interpolation between end points
        n = self.numImages - 1
        if isinstance(p1, (list, tuple)):
            endpoints = p1
            indices = p2
            if not isinstance(indices, (list, tuple)):
                raise ValueError("when p1 is a list of endpoints, p2 must be a list of indices")
            p1_first = endpoints[0]
            p2_last = endpoints[-1]
            self.path = interpolate_path(endpoints, indices, self.numImages)
        else:
            p1_first = p1
            p2_last = p2
            self.path = interpolate_path(p1, p2, self.numImages)
        calc = p1_first.calc
        for i in range(1, n):
            fdname = '0'+str(i)
            if (not self.parallel) or (self.parallel and self.rank == 0):
                if not os.path.exists(fdname): os.mkdir(fdname)
            self.path[i].calc = calc
        if (not self.parallel) or (self.parallel and self.rank == 0):
            os.makedirs(self.xyz_dir, exist_ok=True)
            io.write(os.path.join(self.xyz_dir, "iter_0000.xyz"), self.path, format="extxyz")
        self.path[0].calc = p1_first.calc
        self.path[n].calc = p2_last.calc
        self.Umaxi = 1

        # calculate the Jacobian so that a cell move have the same units and weight as an atomic move
        self.natom = len(self.path[0])
        self.jacobian = compute_jacobian(
            self.path[0].get_volume(),
            self.path[self.numImages-1].get_volume(),
            self.natom,
            self.weight
        )

        # add some new properties
        for i in [0,n]:
            fdname = '0'+str(i)
            backfd = '../'
            if self.parallel: 
                fdname += '/'+str(self.rank)+str(i)
                backfd  = '../../'
            if not os.path.exists(fdname): os.makedirs(fdname)
            os.chdir(fdname)
            self.path[i].u = self.path[i].get_potential_energy()
            self.path[i].f = self.path[i].get_forces()
            if self.ss: stt = self.path[i].get_stress()
            os.chdir(backfd)
            initialize_image_properties(self.path[i], self.jacobian)
            self.path[i].st = numpy.zeros((3,3))
            # solid-state or not
            if self.ss:
                vol = self.path[i].get_volume()*(-1)
                self.path[i].st[0][0] = stt[0] * vol
                self.path[i].st[1][1] = stt[1] * vol
                self.path[i].st[2][2] = stt[2] * vol
                self.path[i].st[2][1] = stt[3] * vol
                self.path[i].st[2][0] = stt[4] * vol
                self.path[i].st[1][0] = stt[5] * vol
                self.path[i].st      -= self.express * (-1)*vol
                self.path[i].st      *= self.fixstrain 

            # calculate the PV term in the enthalpy E+PV, setting image 0 as reference
            dcell  = self.path[i].get_cell() - self.path[0].get_cell()
            strain = numpy.dot(self.path[0].icell, dcell)
            pv     = numpy.vdot(self.express, strain) * self.path[0].get_volume()
            self.path[i].pv = pv
            self.path[i].u += pv

    def forces(self):
        """
        Calculate the forces for each image on the path.  Applies the force due
        to the potential and the spring forces.
        Parameters:
            force - the potential energy force.
        """

        # Calculate the force due to the potential on the intermediate points
        
        #=========================== Begin potential energy evaluation ==============================
        #--------------------------- MPI version -------------------------
        if self.parallel:
            imgi  = self.rank+1
            fdname = '0'+str(imgi)
            os.chdir(fdname)
            self.path[imgi].u    = self.path[imgi].get_potential_energy()
            self.path[imgi].f    = self.path[imgi].get_forces()
            if self.ss: stt      = self.path[imgi].get_stress()
            os.chdir('../')

            try:
                self.path[imgi].st
            except:
                self.path[imgi].st = numpy.zeros((3,3))
            # solid-state or not
            if self.ss:
                vol = self.path[imgi].get_volume()*(-1)
                self.path[imgi].st[0][0] = stt[0] * vol
                self.path[imgi].st[1][1] = stt[1] * vol
                self.path[imgi].st[2][2] = stt[2] * vol
                self.path[imgi].st[2][1] = stt[3] * vol
                self.path[imgi].st[2][0] = stt[4] * vol
                self.path[imgi].st[1][0] = stt[5] * vol
                self.path[imgi].st[0][1] = 0.0
                self.path[imgi].st[0][2] = 0.0
                self.path[imgi].st[1][2] = 0.0
                self.path[imgi].st      -= self.express * vol*(-1)
                self.path[imgi].st      *= self.fixstrain 

            ui    = self.path[imgi].u 
            fi    = self.path[imgi].f 
            sti   = self.path[imgi].st 
            msg_s = numpy.vstack((fi, sti, [ui,0.0,0.0]))
            msg_r = numpy.zeros((self.size, self.natom+4,3))

            #The following pypar send and receive are equivalent to Allgather()
            #msg_r=pypar.gather(msg_s,0,buffer=msg_r)
            #msg_r=pypar.broadcast(msg_r,0)
            self.comm.Allgather([msg_s, self.MPIDB], [msg_r, self.MPIDB])

            for i in range(1, self.numImages - 1):
                self.path[i].f = msg_r[i-1][:-4]
                self.path[i].st = msg_r[i-1][-4:-1]
                self.path[i].u = msg_r[i-1][-1][0]
        #--------------------------- Serial version -------------------------
        else: 
            for i in range(1, self.numImages - 1):
                # writing input and do the calculation in images' directories respectively
                fdname = '0'+str(i)
                os.chdir(fdname)
                self.path[i].u     = self.path[i].get_potential_energy()
                self.path[i].f     = self.path[i].get_forces()
                if self.ss: stt    = self.path[i].get_stress()
                os.chdir('../')
                try:
                    self.path[i].st
                except:
                    self.path[i].st = numpy.zeros((3,3))
                # solid-state or not
                if self.ss:
                    vol = self.path[i].get_volume()*(-1)
                    self.path[i].st[0][0] = stt[0] * vol
                    self.path[i].st[1][1] = stt[1] * vol
                    self.path[i].st[2][2] = stt[2] * vol
                    self.path[i].st[2][1] = stt[3] * vol
                    self.path[i].st[2][0] = stt[4] * vol
                    self.path[i].st[1][0] = stt[5] * vol
                    self.path[i].st[0][1] = 0.0
                    self.path[i].st[0][2] = 0.0
                    self.path[i].st[1][2] = 0.0
                    self.path[i].st      -= self.express * vol*(-1)
                    self.path[i].st      *= self.fixstrain 
        #=========================== End potential energy evaluation ==============================

        for i in range(1, self.numImages - 1):
            initialize_image_properties(self.path[i], self.jacobian)

            # calculate the PV term in the enthalpy E+PV, setting image 0 as reference
            dcell  = self.path[i].get_cell() - self.path[0].get_cell()
            strain = numpy.dot(self.path[0].icell, dcell)
            pv     = numpy.vdot(self.express, strain) * self.path[0].get_volume()
            self.path[i].pv = pv
            self.path[i].u += pv

            if i == 1 or self.path[i].u > self.Umax:
                self.Umax  = self.path[i].u
                self.Umaxi = i
            
        # Loop over each intermediate point and calculate the tangent.
        for i in range(1, self.numImages - 1):

            # Here st should be the Cauchy stress tensor times cell volume. 
            # Timing box volume should have been done.
            self.path[i].totalf = numpy.vstack((self.path[i].f, self.path[i].st / self.jacobian))
            # realtf that needed by nebspline.pl is saved for output
            self.path[i].realtf = deepcopy(self.path[i].totalf)
            
            # If we're using the 'old' tangent, the tangent is defined as the
            # vector from the point behind the current image to the point in
            # front of the current image.
            # Haven't implemented for ssneb
            if self.tangent == 'old':
                self.path[i].n = (self.path[i + 1].r - self.path[i - 1].r)
            
            # Otherwise, we're using the 'new' tangent.
            # Ref:
            # G. Henkelman and H. Jonsson,  Improved tangent estimate in the 
            # nudged elastic path method for finding minimum energy paths and 
            # saddle points, J. Chem. Phys. 113, 9978-9985 (2000)
            else:
                # UPm1: is the previous image higher in energy
                # UPp1: is the next image higher in energy
                UPm1 = self.path[i - 1].u > self.path[i].u
                UPp1 = self.path[i + 1].u > self.path[i].u
                
                # if V(i+1)>V(i)>V(i-1)
                # or V(i+1)<V(i)<V(i-1)
                # (this is the usual case along the MEP)
                '''
                tangent
                '''
                if(UPm1 != UPp1):
                    if(UPm1):
                        # use direct coordinates to avoid double counting cell motion
                        dr_dir  = sPBC(self.path[i].vdir - self.path[i - 1].vdir)
                        avgbox  = 0.5*(self.path[i].get_cell() + self.path[i - 1].get_cell())
                        sn  = numpy.dot(dr_dir,avgbox)
                        dh  = self.path[i].cellt - self.path[i - 1].cellt
                        snb = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i - 1].icell, dh)*0.5
                        #---------------another way to average strain----------------------
                        #iavgbox = numpy.linalg.inv(avgbox)
                        #snb = numpy.dot(iavgbox, snb)
                        #------------------------------------------------------------------
                        self.path[i].n = numpy.vstack((sn,snb))
                    else:
                        dr_dir  = sPBC(self.path[i + 1].vdir - self.path[i].vdir)
                        avgbox  = 0.5*(self.path[i+1].get_cell() + self.path[i].get_cell())
                        sn  = numpy.dot(dr_dir,avgbox)
                        dh  = self.path[i + 1].cellt - self.path[i].cellt
                        snb = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i + 1].icell, dh)*0.5
                        #---------------another way to average strain----------------------
                        #iavgbox = numpy.linalg.inv(avgbox)
                        #snb = numpy.dot(iavgbox, snb)
                        #------------------------------------------------------------------
                        self.path[i].n = numpy.vstack((sn,snb))
                # otherwise, we are near some extremum
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

        # Normalize each tangent
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

        # Loop over each intermediate image and adjust the potential energy,
        # force, and apply the spring force.
        for i in range(1, self.numImages - 1):

            # push the climbing image uphill
            if self.method == 'ci' and i == self.Umaxi:
                self.path[i].totalf -= 2.0 * vproj(self.path[i].totalf, self.path[i].n) 
                self.path[i].fPerp   = self.path[i].totalf

            # and for the non-climbing images...
            else:

                # Calculate the force perpendicular to the tangent. 
                self.path[i].fPerp = self.path[i].totalf - vproj(self.path[i].totalf,   \
                                                            self.path[i].n)
                # Calculate the spring force.
                Rm1 = image_distance_vector(self.path[i - 1], self.path[i])
                Rp1 = image_distance_vector(self.path[i + 1], self.path[i])

                self.path[i].fsN = (vmag(Rp1) - vmag(Rm1)) * self.k * self.path[i].n
                #print i, vmag(Rp1),vmag(Rm1)

                # For dneb use total spring force -spring force in the grad direction.
                if self.dneb:
                    self.path[i].fs = (Rp1 + Rm1) * self.k
                    self.path[i].fsperp = self.path[i].fs -                   \
                                          vproj(self.path[i].fs, self.path[i].n)
                    self.path[i].fsdneb = self.path[i].fsperp -               \
                                          vproj(self.path[i].fs, self.path[i].fPerp)

                    # dneb modification so that it will converge
                    if not self.dnebOrg:
                        FperpSQ = vmag2(self.path[i].fPerp)
                        FsperpSQ = vmag2(self.path[i].fsperp)
                        if FsperpSQ > 0:
                            self.path[i].fsdneb *= 2.0/pi*atan(FperpSQ/FsperpSQ)

                # Not using double-nudging, so set the double-nudging spring
                # force to zero.
                else:
                    self.path[i].fsdneb = 0

                # The final force is the sum of these forces.    
                self.path[i].totalf = self.path[i].fsdneb + self.path[i].fsN +     \
                                 self.path[i].fPerp

                # only move the climing image
                if(self.method == 'ci' and self.onlyci): 
                    self.path[i].totalf *= 0.0
