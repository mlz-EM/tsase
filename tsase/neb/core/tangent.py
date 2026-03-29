"""Tangent helpers for SSNEB."""

import numpy

from tsase.neb.util import sPBC, vmag2

from .geometry import image_distance_vector


def geometric_tangent(path, index):
    """Return a geometric tangent with the zero-vector fallback used in the current code."""
    forward = image_distance_vector(path[index + 1], path[index])
    backward = image_distance_vector(path[index], path[index - 1])
    tangent = forward + backward
    if vmag2(tangent) > 1e-30:
        return tangent
    if vmag2(forward) > 1e-30:
        return forward
    if vmag2(backward) > 1e-30:
        return backward
    return tangent


def energy_weighted_tangent(path, index):
    """Return the legacy Henkelman-Jonsson tangent used by TSASE."""
    UPm1 = path[index - 1].u > path[index].u
    UPp1 = path[index + 1].u > path[index].u

    if UPm1 != UPp1:
        if UPm1:
            dr_dir = sPBC(path[index].vdir - path[index - 1].vdir)
            avgbox = 0.5 * (path[index].get_cell() + path[index - 1].get_cell())
            sn = numpy.dot(dr_dir, avgbox)
            dh = path[index].cellt - path[index - 1].cellt
            snb = numpy.dot(path[index].icell, dh) * 0.5 + numpy.dot(path[index - 1].icell, dh) * 0.5
            return numpy.vstack((sn, snb))

        dr_dir = sPBC(path[index + 1].vdir - path[index].vdir)
        avgbox = 0.5 * (path[index + 1].get_cell() + path[index].get_cell())
        sn = numpy.dot(dr_dir, avgbox)
        dh = path[index + 1].cellt - path[index].cellt
        snb = numpy.dot(path[index].icell, dh) * 0.5 + numpy.dot(path[index + 1].icell, dh) * 0.5
        return numpy.vstack((sn, snb))

    Um1 = path[index - 1].u - path[index].u
    Up1 = path[index + 1].u - path[index].u
    Umin = min(abs(Up1), abs(Um1))
    Umax = max(abs(Up1), abs(Um1))
    if Umax == 0:
        return geometric_tangent(path, index)

    if Um1 > Up1:
        dr_dir = sPBC(path[index + 1].vdir - path[index].vdir)
        avgbox = 0.5 * (path[index + 1].get_cell() + path[index].get_cell())
        sn = numpy.dot(dr_dir, avgbox) * Umin
        dr_dir = sPBC(path[index].vdir - path[index - 1].vdir)
        avgbox = 0.5 * (path[index].get_cell() + path[index - 1].get_cell())
        sn += numpy.dot(dr_dir, avgbox) * Umax

        dh = path[index + 1].cellt - path[index].cellt
        snb1 = numpy.dot(path[index].icell, dh) * 0.5 + numpy.dot(path[index + 1].icell, dh) * 0.5
        dh = path[index].cellt - path[index - 1].cellt
        snb2 = numpy.dot(path[index].icell, dh) * 0.5 + numpy.dot(path[index - 1].icell, dh) * 0.5
        snb = snb1 * Umin + snb2 * Umax
        tangent = numpy.vstack((sn, snb))
    else:
        dr_dir = sPBC(path[index + 1].vdir - path[index].vdir)
        avgbox = 0.5 * (path[index + 1].get_cell() + path[index].get_cell())
        sn = numpy.dot(dr_dir, avgbox) * Umax
        dr_dir = sPBC(path[index].vdir - path[index - 1].vdir)
        avgbox = 0.5 * (path[index].get_cell() + path[index - 1].get_cell())
        sn += numpy.dot(dr_dir, avgbox) * Umin

        dh = path[index + 1].cellt - path[index].cellt
        snb1 = numpy.dot(path[index].icell, dh) * 0.5 + numpy.dot(path[index + 1].icell, dh) * 0.5
        dh = path[index].cellt - path[index - 1].cellt
        snb2 = numpy.dot(path[index].icell, dh) * 0.5 + numpy.dot(path[index - 1].icell, dh) * 0.5
        snb = snb1 * Umax + snb2 * Umin
        tangent = numpy.vstack((sn, snb))

    if vmag2(tangent) <= 1e-30:
        tangent = geometric_tangent(path, index)
    return tangent

