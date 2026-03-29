"""Atom-ID and spatial mapping helpers for SSNEB paths."""

import numpy
from scipy.optimize import linear_sum_assignment

from tsase.neb.util import sPBC

NEB_ATOM_ID_ARRAY = "neb_atom_id"


def _normalized_atom_ids(image, array_name=NEB_ATOM_ID_ARRAY):
    if array_name not in image.arrays:
        return None
    atom_ids = numpy.array(image.arrays[array_name], dtype=int)
    if atom_ids.shape != (len(image),):
        raise ValueError(
            f"{array_name!r} must have shape ({len(image)},), got {atom_ids.shape}"
        )
    if len(numpy.unique(atom_ids)) != len(atom_ids):
        raise ValueError(f"{array_name!r} must contain unique per-atom identifiers")
    return atom_ids


def ensure_atom_ids(images, reference=None, array_name=NEB_ATOM_ID_ARRAY):
    """Attach stable per-atom IDs to one image or a list of images."""
    if hasattr(images, "get_positions"):
        images = [images]
    if not images:
        return numpy.array([], dtype=int)

    reference_image = images[0] if reference is None else reference
    reference_ids = _normalized_atom_ids(reference_image, array_name=array_name)
    if reference_ids is None:
        reference_ids = numpy.arange(len(reference_image), dtype=int)
        reference_image.arrays[array_name] = reference_ids.copy()

    for image in images:
        if len(image) != len(reference_ids):
            raise ValueError("all images must contain the same number of atoms")
        image.arrays[array_name] = reference_ids.copy()

    return reference_ids.copy()


def reorder_by_atom_ids(reference, candidate, array_name=NEB_ATOM_ID_ARRAY):
    """Reorder ``candidate`` to match ``reference`` using persistent atom IDs."""
    reference_ids = _normalized_atom_ids(reference, array_name=array_name)
    candidate_ids = _normalized_atom_ids(candidate, array_name=array_name)
    if reference_ids is None or candidate_ids is None:
        return None
    if len(reference_ids) != len(candidate_ids):
        raise ValueError("reference and candidate must contain the same number of atom IDs")
    if set(reference_ids.tolist()) != set(candidate_ids.tolist()):
        raise ValueError("restart image atom IDs do not match the current band atom IDs")

    candidate_lookup = {
        int(atom_id): index for index, atom_id in enumerate(candidate_ids.tolist())
    }
    order = numpy.array(
        [candidate_lookup[int(atom_id)] for atom_id in reference_ids.tolist()],
        dtype=int,
    )
    reordered = candidate[order].copy()
    reordered.info = dict(getattr(candidate, "info", {}))
    reordered.arrays[array_name] = reference_ids.copy()
    return reordered


def spatial_map(reference, candidate):
    """Reorder a structure to match the reference atom indexing."""
    if len(reference) != len(candidate):
        raise ValueError("all structures must contain the same number of atoms")

    ref_symbols = numpy.array(reference.get_chemical_symbols())
    cand_symbols = numpy.array(candidate.get_chemical_symbols())
    if sorted(ref_symbols.tolist()) != sorted(cand_symbols.tolist()):
        raise ValueError("all structures must contain the same species counts")

    ref_frac = reference.get_scaled_positions()
    cand_frac = candidate.get_scaled_positions()
    avg_cell = 0.5 * (numpy.array(reference.get_cell()) + numpy.array(candidate.get_cell()))
    order = numpy.empty(len(reference), dtype=int)

    for symbol in sorted(set(ref_symbols.tolist())):
        ref_idx = numpy.where(ref_symbols == symbol)[0]
        cand_idx = numpy.where(cand_symbols == symbol)[0]
        cost = numpy.zeros((len(ref_idx), len(cand_idx)))
        for i, ref_i in enumerate(ref_idx):
            deltas = sPBC(cand_frac[cand_idx] - ref_frac[ref_i])
            cart = numpy.dot(deltas, avg_cell)
            cost[i] = numpy.linalg.norm(cart, axis=1)
        row_ind, col_ind = linear_sum_assignment(cost)
        order[ref_idx[row_ind]] = cand_idx[col_ind]

    reordered = candidate[order].copy()
    reordered.info = dict(getattr(candidate, "info", {}))
    reordered.info["spatial_map_order"] = order.tolist()
    reference_ids = _normalized_atom_ids(reference, array_name=NEB_ATOM_ID_ARRAY)
    if reference_ids is not None:
        reordered.arrays[NEB_ATOM_ID_ARRAY] = reference_ids.copy()
    return reordered

