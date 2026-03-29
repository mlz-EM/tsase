"""Path interpolation helpers for SSNEB."""

import numpy

from tsase.neb.util import sPBC

from .mapping import ensure_atom_ids, reorder_by_atom_ids


def _interpolate_segment(p1, p2, num_images):
    n = num_images - 1
    cell1 = p1.get_cell()
    cell2 = p2.get_cell()
    dRB = (cell2 - cell1) / n

    icell1 = numpy.linalg.inv(cell1)
    vdir1 = numpy.dot(p1.get_positions(), icell1)
    icell2 = numpy.linalg.inv(cell2)
    vdir2 = numpy.dot(p2.get_positions(), icell2)
    dR = sPBC(vdir2 - vdir1) / n

    path = [p1.copy()]
    for i in range(1, n):
        img = p1.copy()
        cellt = cell1 + dRB * i
        vdirt = vdir1 + dR * i
        rt = numpy.dot(vdirt, cellt)
        img.set_cell(cellt)
        img.set_positions(rt)
        path.append(img)
    path.append(p2.copy())
    return path


def generate_multi_point_path(structures, indices, total_images):
    """Generate a segmented path passing through user-specified images."""
    if structures is None or indices is None:
        raise ValueError("structures and indices must be provided")
    if len(structures) < 2:
        raise ValueError("structures must contain at least two images")
    if len(structures) != len(indices):
        raise ValueError("structures and indices must be the same length")
    if sorted(indices) != list(indices):
        raise ValueError("indices must be sorted in ascending order")
    if indices[0] != 0:
        raise ValueError("indices must start at 0")
    if indices[-1] != total_images - 1:
        raise ValueError("last index must be total_images - 1")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must be unique")
    for index in indices:
        if index < 0 or index >= total_images:
            raise ValueError("indices must be within [0, total_images - 1]")
    for i in range(len(indices) - 1):
        if indices[i + 1] <= indices[i]:
            raise ValueError("indices must be strictly increasing")

    reference = structures[0]
    ensure_atom_ids(reference)
    mapped_structures = [reference.copy()]
    for structure in structures[1:]:
        reordered = reorder_by_atom_ids(reference, structure)
        mapped_structures.append(structure.copy() if reordered is None else reordered)

    path = []
    for i in range(len(mapped_structures) - 1):
        seg_images = indices[i + 1] - indices[i] + 1
        if seg_images < 2:
            raise ValueError("each segment must span at least 2 images")
        segment = _interpolate_segment(mapped_structures[i], mapped_structures[i + 1], seg_images)
        if i > 0:
            segment = segment[1:]
        path.extend(segment)

    if len(path) != total_images:
        raise ValueError("interpolation produced an unexpected number of images")
    ensure_atom_ids(path, reference=reference)
    return path


def interpolate_path(endpoints, indices, num_images):
    """Interpolate a path between two endpoints or across multiple fixed images."""
    if not isinstance(endpoints, (list, tuple)):
        return _interpolate_segment(endpoints, indices, num_images)
    return generate_multi_point_path(endpoints, indices, num_images)
