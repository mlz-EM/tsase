"""ASE-filter compatibility helpers for SSNEB image updates."""

import numpy as np
from ase.filters import ExpCellFilter, Filter, StrainFilter, UnitCellFilter
from ase.stress import voigt_6_to_full_3x3_stress


def _atom_mask_from_filter(filter_object, natoms):
    index = np.asarray(filter_object.index)
    if index.dtype == bool:
        if len(index) != natoms:
            raise ValueError("boolean filter mask length must match the number of atoms")
        return index.copy()
    mask = np.zeros(natoms, dtype=bool)
    mask[index.astype(int)] = True
    return mask


def _mask_matrix(mask):
    mask_array = np.asarray(mask, dtype=float)
    if mask_array.shape == (6,):
        return voigt_6_to_full_3x3_stress(mask_array)
    if mask_array.shape == (3, 3):
        return mask_array.copy()
    raise ValueError("filter mask must have shape (6,) or (3, 3)")


class ImageFilterAdapter:
    """Translate ASE filters into SSNEB force and step projections."""

    def __init__(self, atoms):
        self.atoms = atoms
        self.natoms = len(atoms)

    def project_atomic_forces(self, forces):
        return np.array(forces, dtype=float)

    def project_cell_virial(self, virial):
        return np.array(virial, dtype=float)

    def project_cell_step(self, cell_step):
        return np.array(cell_step, dtype=float)

    def apply_step(self, delta, jacobian):
        delta = np.array(delta, dtype=float)
        positions = self.atoms.get_positions()
        positions += delta[: self.natoms]
        self.atoms.set_positions(positions)

        cell_step = self.project_cell_step(delta[self.natoms :])
        if np.linalg.norm(cell_step) > 0.0:
            cell = np.array(self.atoms.get_cell(), dtype=float)
            cell += np.dot(cell, cell_step) / jacobian
            self.atoms.set_cell(cell, scale_atoms=True)


class SubsetFilterAdapter(ImageFilterAdapter):
    def __init__(self, atoms, filter_object):
        super().__init__(atoms)
        self.atom_mask = _atom_mask_from_filter(filter_object, self.natoms)

    def project_atomic_forces(self, forces):
        projected = np.zeros_like(forces, dtype=float)
        projected[self.atom_mask] = np.array(forces, dtype=float)[self.atom_mask]
        return projected

    def project_cell_virial(self, virial):
        return np.zeros((3, 3), dtype=float)

    def apply_step(self, delta, jacobian):
        del jacobian
        delta = np.array(delta, dtype=float)
        positions = self.atoms.get_positions()
        positions[self.atom_mask] += delta[: self.natoms][self.atom_mask]
        self.atoms.set_positions(positions)


class CellFilterAdapter(ImageFilterAdapter):
    def __init__(
        self,
        atoms,
        mask,
        hydrostatic_strain=False,
        constant_volume=False,
        move_atoms=True,
    ):
        super().__init__(atoms)
        self.mask = _mask_matrix(mask)
        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.move_atoms = move_atoms

    def project_atomic_forces(self, forces):
        if self.move_atoms:
            return np.array(forces, dtype=float)
        return np.zeros_like(forces, dtype=float)

    def _project_cell_tensor(self, tensor):
        projected = np.array(tensor, dtype=float)
        if self.hydrostatic_strain:
            projected = np.eye(3) * np.trace(projected) / 3.0
        projected *= self.mask
        if self.constant_volume:
            trace = np.trace(projected)
            projected -= np.eye(3) * trace / 3.0
        return projected

    def project_cell_virial(self, virial):
        return self._project_cell_tensor(virial)

    def project_cell_step(self, cell_step):
        return self._project_cell_tensor(cell_step)


def make_filter_adapter(atoms, filter_object=None):
    if filter_object is None:
        return ImageFilterAdapter(atoms)
    if isinstance(filter_object, StrainFilter):
        return CellFilterAdapter(
            atoms,
            mask=filter_object.mask,
            hydrostatic_strain=False,
            constant_volume=False,
            move_atoms=False,
        )
    if isinstance(filter_object, (ExpCellFilter, UnitCellFilter)):
        return CellFilterAdapter(
            atoms,
            mask=filter_object.mask,
            hydrostatic_strain=getattr(filter_object, "hydrostatic_strain", False),
            constant_volume=getattr(filter_object, "constant_volume", False),
            move_atoms=True,
        )
    if isinstance(filter_object, Filter):
        return SubsetFilterAdapter(atoms, filter_object)
    raise TypeError(f"unsupported ASE filter type: {type(filter_object).__name__}")

