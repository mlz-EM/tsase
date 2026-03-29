"""Core geometry helpers used by SSNEB."""

import numpy

from tsase.neb.util import sPBC


def compute_jacobian(vol1, vol2, natom, weight=1.0):
    """Compute the SSNEB Jacobian scaling factor."""
    vol = (vol1 + vol2) * 0.5
    avglen = (vol / natom) ** (1.0 / 3.0)
    return avglen * natom ** 0.5 * weight


def initialize_image_properties(image, jacobian):
    """Attach cached geometry state used by SSNEB force calculations."""
    image.cellt = numpy.array(image.get_cell()) * jacobian
    image.icell = numpy.linalg.inv(image.get_cell())
    image.vdir = image.get_scaled_positions()


def image_distance_vector(image_a, image_b):
    """Return the combined atom-plus-cell displacement vector."""
    frac_diff = sPBC(image_a.vdir - image_b.vdir)
    avgbox = 0.5 * (numpy.array(image_a.get_cell()) + numpy.array(image_b.get_cell()))
    atom_part = numpy.dot(frac_diff, avgbox)

    dh = image_a.cellt - image_b.cellt
    cell_part = numpy.dot(image_b.icell, dh) * 0.5 + numpy.dot(image_a.icell, dh) * 0.5
    return numpy.vstack((atom_part, cell_part))

