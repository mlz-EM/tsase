"""Protocols for force/cell projection adapters used by SSNEB."""

from typing import Protocol


class ProjectionAdapter(Protocol):
    def project_atomic_forces(self, forces):
        ...

    def project_cell_virial(self, virial):
        ...

    def project_cell_step(self, cell_step):
        ...

    def apply_step(self, delta, jacobian):
        ...

