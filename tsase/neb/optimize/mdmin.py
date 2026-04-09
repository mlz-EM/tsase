"""MDMin optimizer for SSNEB."""

from tsase.neb.util import vdot, vmag, vunit

from .base import minimizer_ssneb


class mdmin_ssneb(minimizer_ssneb):
    """MDMin optimizer adapted to the SSNEB generalized coordinates."""

    def __init__(
        self,
        path,
        dt=0.2,
        maxmove=0.1,
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
        self.dt = float(dt)
        self.maxmove = float(maxmove)
        self.v = None

    def step(self):
        totalf = self._collect_generalized_forces()
        if self.v is None:
            self.v = self._zero_generalized_array()
        else:
            self.v += 0.5 * self.dt * totalf
            for index in range(self.band.numImages - 2):
                image_force = totalf[index]
                force_norm_sq = vdot(image_force, image_force)
                velocity_force = vdot(self.v[index], image_force)
                if velocity_force <= 0.0 or force_norm_sq <= 0.0:
                    self.v[index].fill(0.0)
                else:
                    self.v[index] = image_force * (velocity_force / force_norm_sq)

        self.v += 0.5 * self.dt * totalf
        steps = self.dt * self.v
        for index in range(self.band.numImages - 2):
            if vmag(steps[index]) > self.maxmove:
                steps[index] = self.maxmove * vunit(steps[index])
        self._apply_generalized_steps(steps)
