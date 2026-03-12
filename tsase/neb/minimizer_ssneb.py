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

    def __init__(self, band, xyz_dir=None):
        self.band = band
        if xyz_dir is None:
            xyz_dir = getattr(band, "xyz_dir", "neb_xyz")
        self.xyz_dir = xyz_dir

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
        x = list(range(len(energies)))
        plt.figure(figsize=(6, 4))
        plt.plot(x, energies, marker="o", linewidth=1.5)
        plt.xlabel("Image Index")
        plt.ylabel("Energy")
        plt.title(f"NEB Energies (iter {iteration})")
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
            print("Iteration       Total Force       Perp Force        MaxU       MaxI    Stress on CI    ")
            print("---------------------------------------------------------------------------------------")
            feout = open('fe.out','a')
            feout.write('Iteration       Total Force       Perp Force        MaxU       MaxI    Stress on CI     \n')
            feout.write('------------------------------------------------------------  \n')
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
            output = str(iterations+1)+'     '+str(fMax)+'     '+str(fPMax)+'     ' \
                 +str(self.band.Umax-self.band.path[0].u)+'     '+str(self.band.Umaxi) \
                 +'     '+str(fci) + '    '+str(self.dt)

            if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0) :
                print("-------------------------SSNEB------------------------------")
                print(output)
                feout.write(output+'\n')
            self._write_iteration_xyz(iterations + 1)
            self._save_energy_plot(iterations + 1)

            iterations += 1

        # write data for neb.dat
        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0) :
            print("-----------------------SSNEB Finished------------------------------")
            print("Image    ReCoords      E      RealForce      Image")
            feout.write("-----------------------SSNEB Finished------------------------------\n")
            feout.write("Image    ReCoords      E      RealForce      Image \n")
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
                print("%3i %13.6f %13.6f %13.6f %3i" % (i,float(R20),float(self.band.path[i].u-self.band.path[0].u),float(realtotalf),i))
                feout.write( "%3i %13.6f %13.6f %13.6f %3i %s" % (i,float(R20),float(self.band.path[i].u-self.band.path[0].u),float(realtotalf),i,'\n'))

        if (not self.band.parallel) or (self.band.parallel and self.band.rank == 0) :
            feout.close()
