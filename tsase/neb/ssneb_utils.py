"""
Standalone utility functions extracted from the ssneb class.

These functions implement the core geometry operations used in solid-state NEB:
linear interpolation, Jacobian scaling, image property initialization, and the
inter-image distance metric. They operate on ASE Atoms objects and numpy arrays
without requiring an ssneb instance.
"""

import numpy
from ase import io
from scipy.optimize import linear_sum_assignment
from tsase.neb.util import sPBC


def compute_jacobian(vol1, vol2, natom, weight=1.0):
    """Compute the Jacobian scaling factor for solid-state NEB.

    The Jacobian balances the units and weight of cell deformations against
    atomic displacements, so that the NEB spring forces treat both on equal
    footing.

    Args:
        vol1: Volume of the first endpoint cell.
        vol2: Volume of the second endpoint cell.
        natom: Number of atoms per image.
        weight: Relative weight of cell vs atomic degrees of freedom.

    Returns:
        The Jacobian scaling factor (float).
    """
    vol = (vol1 + vol2) * 0.5
    avglen = (vol / natom) ** (1.0 / 3.0)
    return avglen * natom ** 0.5 * weight


def _interpolate_segment(p1, p2, num_images):
    """Linearly interpolate a path between two endpoint structures.

    Interpolates both the cell vectors and fractional atomic coordinates.
    Fractional coordinate differences are wrapped via sPBC so that atoms
    take the shortest periodic path.

    Args:
        p1: ASE Atoms object for the first endpoint.
        p2: ASE Atoms object for the second endpoint.
        num_images: Total number of images including both endpoints.

    Returns:
        List of num_images ASE Atoms objects. The first and last are copies
        of p1 and p2 respectively; intermediates are interpolated.
    """
    n = num_images - 1
    cell1 = p1.get_cell()
    cell2 = p2.get_cell()
    dRB = (cell2 - cell1) / n

    # Use Cartesian -> fractional via inverse cell (not get_scaled_positions)
    # so that atoms moving more than half a lattice vector are handled correctly.
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


def interpolate_path(endpoints, indices, num_images):
    """Interpolate a path through endpoints.

    Supports two usage patterns:
      1) interpolate_path(p1, p2, num_images) for a single segment.
      2) interpolate_path(endpoints, indices, num_images) for multiple segments.

    Args:
        endpoints: List of ASE Atoms objects [p0, p1, ...].
        indices: List of integer indices (same length as endpoints) marking
                 where each endpoint should appear in the final path.
        num_images: Total number of images in the final path.

    Returns:
        List of num_images ASE Atoms objects with linear interpolation between
        successive endpoints, including all endpoints exactly at the given
        indices.
    """
    # Backwards-compatible single-segment path: (p1, p2, num_images)
    if not isinstance(endpoints, (list, tuple)):
        return _interpolate_segment(endpoints, indices, num_images)

    return generate_multi_point_path(endpoints, indices, num_images)


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
    return reordered


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
    mapped_structures = [reference.copy()]
    for structure in structures[1:]:
        mapped_structures.append(spatial_map(reference, structure))

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
    return path


def load_band_configuration_from_xyz(band, xyz_path):
    """Load image positions and cells from a saved iter_*.xyz file into a band."""
    images = io.read(xyz_path, ":")
    if len(images) != band.numImages:
        raise ValueError(
            f"restart file contains {len(images)} images, expected {band.numImages}"
        )
    for index, image in enumerate(images):
        band.path[index].set_cell(image.get_cell(), scale_atoms=False)
        band.path[index].set_positions(image.get_positions())


def initialize_image_properties(image, jacobian):
    """Attach derived NEB properties to an ASE Atoms image.

    Sets three attributes used throughout the ssneb force calculation:
      - image.cellt: cell matrix scaled by the Jacobian
      - image.icell: inverse of the (unscaled) cell matrix
      - image.vdir:  fractional (scaled) positions

    Args:
        image: An ASE Atoms object.
        jacobian: The Jacobian scaling factor from compute_jacobian().
    """
    image.cellt = numpy.array(image.get_cell()) * jacobian
    image.icell = numpy.linalg.inv(image.get_cell())
    image.vdir = image.get_scaled_positions()


def image_distance_vector(image_a, image_b):
    """Compute the combined atom + cell distance vector between two images.

    This is the metric used by ssneb for spring forces. It combines:
      - Atomic part: sPBC-wrapped fractional coordinate differences converted
        to Cartesian via the average cell matrix. Shape (natom, 3).
      - Cell part: Jacobian-scaled cell differences averaged through both
        images' inverse cells. Shape (3, 3).

    The two parts are vertically stacked into a single array whose Euclidean
    norm (via vmag) gives the scalar inter-image distance.

    Both images must have .cellt, .icell, and .vdir attributes set (see
    initialize_image_properties).

    Args:
        image_a: ASE Atoms with NEB properties (the "from" image).
        image_b: ASE Atoms with NEB properties (the "to" image).

    Returns:
        numpy array of shape (natom + 3, 3) — the stacked distance vector.
    """
    # Atomic part: fractional diff -> Cartesian via average cell
    frac_diff = sPBC(image_a.vdir - image_b.vdir)
    avgbox = 0.5 * (numpy.array(image_a.get_cell()) + numpy.array(image_b.get_cell()))
    atom_part = numpy.dot(frac_diff, avgbox)

    # Cell part: Jacobian-scaled cell diff averaged through inverse cells
    dh = image_a.cellt - image_b.cellt
    cell_part = numpy.dot(image_b.icell, dh) * 0.5 + numpy.dot(image_a.icell, dh) * 0.5

    return numpy.vstack((atom_part, cell_part))
