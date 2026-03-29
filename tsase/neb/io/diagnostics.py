"""Diagnostics formatting helpers for SSNEB."""

import os


DIAGNOSTICS_HEADER = (
    "iteration,image,frozen,update_rate,U,field_term,pv,H,px,py,pz,Px,Py,Pz,k_left,k_right\n"
)


def initialize_diagnostics_file(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(DIAGNOSTICS_HEADER)


def append_diagnostics_rows(path, iteration, images, frozen_images):
    with open(path, "a") as handle:
        for i, image in enumerate(images):
            handle.write(
                "{:d},{:d},{:d},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n".format(
                    int(iteration),
                    i,
                    1 if i in frozen_images else 0,
                    float(getattr(image, "update_rate", 1.0)),
                    float(getattr(image, "base_u", image.u - getattr(image, "pv", 0.0))),
                    float(getattr(image, "field_u", 0.0)),
                    float(getattr(image, "pv", 0.0)),
                    float(image.u),
                    float(getattr(image, "dipole", [0.0, 0.0, 0.0])[0]),
                    float(getattr(image, "dipole", [0.0, 0.0, 0.0])[1]),
                    float(getattr(image, "dipole", [0.0, 0.0, 0.0])[2]),
                    float(getattr(image, "polarization", [0.0, 0.0, 0.0])[0]),
                    float(getattr(image, "polarization", [0.0, 0.0, 0.0])[1]),
                    float(getattr(image, "polarization", [0.0, 0.0, 0.0])[2]),
                    float(getattr(image, "k_left", 0.0)),
                    float(getattr(image, "k_right", 0.0)),
                )
            )

