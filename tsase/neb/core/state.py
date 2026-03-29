"""Image-state helpers for SSNEB."""

import numpy


def stress_to_virial(stress, volume):
    """Convert ASE stress ordering to the lower-triangular virial tensor used by SSNEB."""
    virial = numpy.zeros((3, 3))
    virial[0][0] = stress[0] * (-volume)
    virial[1][1] = stress[1] * (-volume)
    virial[2][2] = stress[2] * (-volume)
    virial[2][1] = stress[3] * (-volume)
    virial[2][0] = stress[4] * (-volume)
    virial[1][0] = stress[5] * (-volume)
    return virial

