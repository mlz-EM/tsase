"""Diagnostics formatting helpers for SSNEB."""

import os


DIAGNOSTICS_HEADER = (
    "iteration,image,U,field_term,pv,H,dipole_x,dipole_y,dipole_z,polarization_x,polarization_y,polarization_z\n"
)


def initialize_diagnostics_file(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(DIAGNOSTICS_HEADER)


def append_diagnostics_rows(path, iteration, images):
    with open(path, "a", encoding="utf-8") as handle:
        for i, image in enumerate(images):
            base_u = getattr(image, "base_u", None)
            field_u = getattr(image, "field_u", 0.0)
            base_u = float("nan") if base_u is None else float(base_u)
            field_u = float("nan") if field_u is None else float(field_u)
            handle.write(
                "{:d},{:d},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n".format(
                    int(iteration),
                    i,
                    base_u,
                    field_u,
                    float(getattr(image, "pv", 0.0)),
                    float(image.u),
                    float(getattr(image, "dipole", [0.0, 0.0, 0.0])[0]),
                    float(getattr(image, "dipole", [0.0, 0.0, 0.0])[1]),
                    float(getattr(image, "dipole", [0.0, 0.0, 0.0])[2]),
                    float(getattr(image, "polarization", [0.0, 0.0, 0.0])[0]),
                    float(getattr(image, "polarization", [0.0, 0.0, 0.0])[1]),
                    float(getattr(image, "polarization", [0.0, 0.0, 0.0])[2]),
                )
            )
