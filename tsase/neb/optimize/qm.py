"""Quick-min optimizer for SSNEB."""

import numpy as np

from tsase.neb.util import vdot, vmag, vproj, vunit

from .base import minimizer_ssneb


class qm_ssneb(minimizer_ssneb):
    """Quick-min optimizer for SSNEB bands."""

    def __init__(
        self,
        path,
        maxmove=0.2,
        dt=0.05,
        output_interval=1,
        energy_profile_entries=None,
        plot_property=None,
        image_mobility_rates=None,
    ):
        super().__init__(
            path,
            output_interval=output_interval,
            energy_profile_entries=energy_profile_entries,
            plot_property=plot_property,
            image_mobility_rates=image_mobility_rates,
        )
        self.maxmove = float(maxmove)
        self.dt = float(dt)
        self.v = self._zero_generalized_array()

    def step(self):
        totalf = self._collect_generalized_forces()
        steps = self._zero_generalized_array()
        velocity_limit = np.inf if self.dt == 0.0 else self.maxmove / self.dt

        for index in range(self.band.numImages - 2):
            image_force = totalf[index]
            power = vdot(image_force, self.v[index])
            if power > 0.0:
                self.v[index] = vproj(self.v[index], image_force)
            else:
                self.v[index].fill(0.0)

            self.v[index] += self.dt * image_force
            speed = vmag(self.v[index])
            if speed > velocity_limit:
                self.v[index] = velocity_limit * vunit(self.v[index])
            steps[index] = self.dt * self.v[index]

        self._apply_generalized_steps(steps)
