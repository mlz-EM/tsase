"""Spring-constant helpers for SSNEB."""

import numpy


def build_spring_constants(path, num_images, k, adaptive_springs=False, kmin=None, kmax=None, adaptive_eps=1e-3):
    """Return the spring-constant array for the current path."""
    if not adaptive_springs:
        return numpy.full(num_images - 1, k, dtype=float)

    if kmin is None:
        kmin = k
    if kmax is None:
        kmax = k
    enthalpies = numpy.array([image.u for image in path], dtype=float)
    denominator = max(float(enthalpies.max() - enthalpies.min()), adaptive_eps)
    spring_constants = []
    for i in range(num_images - 1):
        delta_h = abs(enthalpies[i + 1] - enthalpies[i]) / denominator
        spring_constants.append(kmin + (kmax - kmin) * delta_h)
    return numpy.array(spring_constants, dtype=float)

