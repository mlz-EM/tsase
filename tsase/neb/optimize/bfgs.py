"""BFGS optimizer for SSNEB."""

import numpy as np

from .base import minimizer_ssneb


class bfgs_ssneb(minimizer_ssneb):
    """Full-memory BFGS optimizer over the SSNEB generalized coordinates."""

    def __init__(
        self,
        path,
        maxmove=0.04,
        alpha=70.0,
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
        self.alpha = float(alpha)
        self.dt = 0.0
        self._inverse_hessians = None
        self._previous_forces = None
        self._previous_steps = None

    def _capture_intermediate_state(self):
        state = []
        for index in range(1, self.band.numImages - 1):
            image = self.band.path[index]
            state.append(
                (
                    np.array(image.get_positions(), dtype=float),
                    np.array(image.get_cell(), dtype=float),
                )
            )
        return state

    def _measure_applied_steps(self, previous_state):
        applied = self._zero_generalized_array()
        for index, (previous_positions, previous_cell) in enumerate(previous_state, start=1):
            image = self.band.path[index]
            applied[index - 1, : self.band.natom, :] = (
                np.array(image.get_positions(), dtype=float) - previous_positions
            )
            cell_delta = np.array(image.get_cell(), dtype=float) - previous_cell
            applied[index - 1, self.band.natom :, :] = (
                np.linalg.solve(previous_cell, cell_delta) * self.band.jacobian
            )
        return applied.reshape(self.band.numImages - 2, -1)

    def _ensure_state(self, vector_size):
        if self._inverse_hessians is not None:
            return
        base = np.eye(vector_size, dtype=float) / self.alpha
        self._inverse_hessians = np.repeat(
            base[np.newaxis, :, :],
            self.band.numImages - 2,
            axis=0,
        )

    def _update_inverse_hessian(self, index, flat_forces):
        if self._previous_forces is None or self._previous_steps is None:
            return

        s = self._previous_steps[index]
        if not np.any(s):
            return

        y = -(flat_forces[index] - self._previous_forces[index])
        ys = float(np.dot(y, s))
        if ys <= 1.0e-12:
            return

        hessian_inv = self._inverse_hessians[index]
        identity = np.eye(hessian_inv.shape[0], dtype=float)
        rho = 1.0 / ys
        sy = np.outer(s, y)
        ys_outer = np.outer(y, s)
        updated = (
            (identity - rho * sy)
            @ hessian_inv
            @ (identity - rho * ys_outer)
            + rho * np.outer(s, s)
        )
        self._inverse_hessians[index] = 0.5 * (updated + updated.T)

    def step(self):
        totalf = self._collect_generalized_forces()
        flat_forces = totalf.reshape(self.band.numImages - 2, -1)
        self._ensure_state(flat_forces.shape[1])

        for index in range(self.band.numImages - 2):
            self._update_inverse_hessian(index, flat_forces)

        flat_steps = np.zeros_like(flat_forces)
        for index in range(self.band.numImages - 2):
            flat_steps[index] = self._inverse_hessians[index] @ flat_forces[index]
            step_norm = float(np.linalg.norm(flat_steps[index]))
            if step_norm > self.maxmove:
                flat_steps[index] *= self.maxmove / step_norm
            if not np.all(np.isfinite(flat_steps[index])):
                flat_steps[index].fill(0.0)

        steps = flat_steps.reshape(self.generalized_shape)
        previous_state = self._capture_intermediate_state()
        self._apply_generalized_steps(steps)
        self._previous_forces = flat_forces.copy()
        self._previous_steps = self._measure_applied_steps(previous_state)
