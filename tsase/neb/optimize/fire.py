"""FIRE optimizer for SSNEB."""

import numpy as np

from tsase.neb.util import vdot, vmag, vunit

from .base import minimizer_ssneb


class fire_ssneb(minimizer_ssneb):
    def __init__(
        self,
        path,
        maxmove=0.2,
        dt=0.1,
        dtmax=1.0,
        Nmin=5,
        finc=1.1,
        fdec=0.5,
        astart=0.1,
        fa=0.99,
        output_interval=1,
        plot_property=None,
        image_mobility_rates=None,
    ):
        minimizer_ssneb.__init__(
            self,
            path,
            output_interval=output_interval,
            plot_property=plot_property,
            image_mobility_rates=image_mobility_rates,
        )
        self.maxmove = maxmove
        self.dt = dt
        self.dtmax = dtmax
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.a = astart
        self.fa = fa
        self.Nsteps = 0

        i = self.band.numImages - 2
        j = self.band.natom + 3
        self.v = np.zeros((i, j, 3))

    def step(self):
        self.band.forces()
        totalf = self.v.copy()
        for i in range(1, self.band.numImages - 1):
            totalf[i - 1] = self.get_image_mobility_rate(i) * self.band.path[i].totalf
        Power = vdot(totalf, self.v)

        if Power > 0.0:
            self.v = (1.0 - self.a) * self.v + self.a * vmag(self.v) * vunit(totalf)
            if self.Nsteps > self.Nmin:
                self.dt = min(self.dt * self.finc, self.dtmax)
                self.a *= self.fa
            self.Nsteps += 1
        else:
            self.v *= 0.0
            self.a = self.astart
            self.dt *= self.fdec
            self.Nsteps = 0

        self.v += self.dt * totalf

        for i in range(1, self.band.numImages - 1):
            dR = self.dt * self.v[i - 1]
            if vmag(dR) > self.maxmove:
                dR = self.maxmove * vunit(dR)
            self.band.image_adapters[i].apply_step(dR, self.band.jacobian)
