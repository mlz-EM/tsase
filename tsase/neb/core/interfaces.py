"""Normalized runtime interfaces for the refactored NEB codebase."""

from dataclasses import dataclass
from typing import Optional

from .path import generate_multi_point_path, interpolate_path


@dataclass
class ImageEvalResult:
    """One complete per-image evaluation payload."""

    u: float
    base_u: Optional[float]
    field_u: Optional[float]
    f: object
    st: object
    dipole: object
    polarization: object
    polarization_c_per_m2: object


class ExecutionContext:
    """Centralized execution-mode state for serial and MPI runs."""

    def __init__(self, parallel=False, communicator=None):
        self.is_parallel = bool(parallel)
        self.communicator = communicator
        if self.is_parallel and self.communicator is not None:
            self.rank = self.communicator.rank
            self.size = self.communicator.size
        else:
            self.rank = 0
            self.size = 1
        self.is_output_owner = self.rank == 0

    @classmethod
    def from_parallel_flag(cls, parallel):
        if parallel:
            from mpi4py import MPI

            return cls(parallel=True, communicator=MPI.COMM_WORLD)
        return cls(parallel=False, communicator=None)

    def allgather_image_results(self, result_map):
        """Gather a full ``index -> ImageEvalResult`` mapping onto every rank."""
        if not self.is_parallel:
            return dict(result_map)
        gathered = self.communicator.allgather(dict(result_map))
        merged = {}
        for payload in gathered:
            merged.update(payload)
        return merged


@dataclass
class PathSpec:
    """Normalized path-construction specification for SSNEB."""

    structures: list
    indices: list
    num_images: int

    @classmethod
    def from_legacy_inputs(cls, p1, p2, num_images):
        if isinstance(p1, (list, tuple)):
            if not isinstance(p2, (list, tuple)):
                raise ValueError("when p1 is a list of endpoints, p2 must be a list of indices")
            return cls(list(p1), list(p2), int(num_images))
        return cls([p1, p2], [0, int(num_images) - 1], int(num_images))

    @property
    def endpoints(self):
        return self.structures

    def build_path(self):
        if len(self.structures) == self.num_images and self.indices == list(range(self.num_images)):
            return [image.copy() for image in self.structures]
        if len(self.structures) == 2 and self.indices == [0, self.num_images - 1]:
            return interpolate_path(self.structures[0], self.structures[1], self.num_images)
        return generate_multi_point_path(self.structures, self.indices, self.num_images)
