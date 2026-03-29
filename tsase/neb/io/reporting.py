"""Owner-aware reporting helpers for SSNEB user-facing outputs."""

import os
from copy import deepcopy

from ase import io

from .diagnostics import append_diagnostics_rows, initialize_diagnostics_file


class NullReporter:
    """No-op reporter for non-owner ranks."""

    is_active = False

    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop


class Reporter:
    """Reporter that owns shared user-facing outputs."""

    is_active = True

    def __init__(self, layout):
        self.layout = layout
        self._log_handle = None

    def initialize_diagnostics(self):
        initialize_diagnostics_file(str(self.layout.diagnostics_file))

    def write_diagnostics(self, iteration, images, frozen_images):
        append_diagnostics_rows(str(self.layout.diagnostics_file), iteration, images, frozen_images)

    def write_iteration_xyz(self, images, iteration, save_projected_neb_sequence):
        os.makedirs(self.layout.xyz_dir, exist_ok=True)
        outfile = self.layout.xyz_dir / f"iter_{iteration:04d}.xyz"
        io.write(str(outfile), images, format="extxyz")
        save_projected_neb_sequence(images, xyz_dir=str(self.layout.xyz_dir), iteration=iteration)

    def save_energy_plot(self, fig, iteration):
        os.makedirs(self.layout.xyz_dir, exist_ok=True)
        outfile = self.layout.xyz_dir / f"energy_iter_{iteration:04d}.png"
        fig.tight_layout()
        fig.savefig(str(outfile), dpi=150)

    def open_log(self, header, separator):
        log_dir = os.path.dirname(str(self.layout.log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._log_handle = open(self.layout.log_file, "a")
        self._log_handle.write(header + "\n")
        self._log_handle.write(separator + "\n")

    def write_status_block(self, header, output, separator):
        print("-------------------------SSNEB------------------------------")
        print(header)
        print(output)
        print(separator)
        if self._log_handle is not None:
            self._log_handle.write(header + "\n")
            self._log_handle.write(output + "\n")
            self._log_handle.write(separator + "\n")

    def write_summary_header(self, header, separator):
        print("-----------------------SSNEB Finished------------------------------")
        print(header)
        print(separator)
        if self._log_handle is not None:
            self._log_handle.write("-----------------------SSNEB Finished------------------------------\n")
            self._log_handle.write(header + "\n")
            self._log_handle.write(separator + "\n")

    def write_summary_line(self, line):
        print(line)
        if self._log_handle is not None:
            self._log_handle.write(line + "\n")

    def close(self):
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def make_snapshot_images(self, path):
        images = []
        for i, img in enumerate(path):
            snap = img.copy()
            snap.calc = None
            snap.info = deepcopy(getattr(img, "info", {}))
            snap.info.pop("spatial_map_order", None)
            snap.info["neb_image"] = i
            if hasattr(img, "u"):
                try:
                    snap.info["energy"] = float(img.u)
                except Exception:
                    pass
            if hasattr(img, "polarization_c_per_m2"):
                try:
                    snap.info["polarization_c_per_m2"] = [float(value) for value in img.polarization_c_per_m2]
                except Exception:
                    pass
            if hasattr(img, "f"):
                try:
                    snap.arrays["forces"] = img.f.copy()
                except Exception:
                    pass
            images.append(snap)
        return images


def make_reporter(context, layout):
    if context.is_output_owner:
        return Reporter(layout)
    return NullReporter()
