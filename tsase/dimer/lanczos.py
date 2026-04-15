"""Lanczos min-mode variant of the solid-state dimer search."""

import numpy as np

from tsase.neb.util import vmag, vunit

from tsase.dimer.ssdimer import SSDimer


class LanczosDimer(SSDimer):
    """SSDimer variant that uses a Lanczos Krylov search for the min mode."""

    def min_mode_search(self):
        if not self.ss:
            self.N = self.project_translation_rotation(self.N, self.R0)

        F0 = self.update_general_forces(self.R0)
        F1 = self.rotation_update()
        delta_phi = 100.0

        vector = self.N.flatten()
        beta = vmag(vector)
        size = (self.natom + 3) * 3
        tridiagonal = np.zeros((size, size), dtype=float)
        basis = np.zeros((size, size), dtype=float)

        c0 = self.curvature
        previous_mode = None
        for iteration in range(size):
            basis[:, iteration] = vector / beta
            hessian_vector = -(F1 - F0) / self.dR
            vector = hessian_vector.flatten()

            if iteration > 0:
                vector = vector - beta * basis[:, iteration - 1]
            alpha = np.vdot(basis[:, iteration], vector)
            vector = vector - alpha * basis[:, iteration]
            vector = vector - np.dot(basis, np.dot(basis.T, vector))

            tridiagonal[iteration, iteration] = alpha
            if iteration > 0:
                tridiagonal[iteration - 1, iteration] = beta
                tridiagonal[iteration, iteration - 1] = beta

            beta = vmag(vector)
            if beta == 0:
                break

            self.N = np.reshape(vector / beta, (-1, 3))
            if not self.ss:
                self.N = self.project_translation_rotation(self.N, self.R0)
            F1 = self.rotation_update()

            if iteration == 0:
                mode = np.array(self.N, copy=True)
                c0 = np.vdot(hessian_vector, mode)
            else:
                eigenvalues, eigenvectors = np.linalg.eig(tridiagonal[: iteration + 1, : iteration + 1])
                lowest = eigenvalues.argsort()[0]
                c0 = float(eigenvalues[lowest])
                estimated_mode = basis[:, : iteration + 1].dot(eigenvectors[:, lowest])
                estimated_mode = vunit(estimated_mode)
                mode = np.reshape(estimated_mode, (-1, 3))
                if not self.ss:
                    mode = self.project_translation_rotation(mode, self.R0)
                dot_product = 0.0 if previous_mode is None else np.vdot(mode, previous_mode)
                dot_product = min(1.0, max(-1.0, dot_product))
                delta_phi = np.arccos(dot_product)
                if delta_phi > np.pi / 2.0:
                    delta_phi = np.pi - delta_phi

            previous_mode = np.array(mode, copy=True)
            if delta_phi < self.phi_tol or iteration == size - 1 or iteration >= self.rotationMax:
                self.N = mode
                break

        self.curvature = c0
        return F0

    minmodesearch = min_mode_search


def run_lanczos(
    *,
    atoms,
    calculator=None,
    calculator_mode=None,
    copy_atoms=True,
    search_kwargs=None,
    **dimer_kwargs,
):
    """Convenience wrapper for the maintained Lanczos-dimer entry point."""

    search = LanczosDimer.from_atoms(
        atoms,
        calculator=calculator,
        calculator_mode=calculator_mode,
        copy_atoms=copy_atoms,
        **dimer_kwargs,
    )
    return search.search(**({} if search_kwargs is None else dict(search_kwargs)))


lanczos_atoms = LanczosDimer
