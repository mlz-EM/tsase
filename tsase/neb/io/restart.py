"""Restart helpers for SSNEB bands."""

from ase import io

from tsase.neb.core.geometry import initialize_image_properties
from tsase.neb.core.mapping import ensure_atom_ids, reorder_by_atom_ids, spatial_map


def load_band_configuration_from_xyz(band, xyz_path, remap=True):
    """Load image positions and cells from a saved iter_*.xyz file into a band."""
    images = io.read(xyz_path, ":")
    if len(images) != band.numImages:
        raise ValueError(
            f"restart file contains {len(images)} images, expected {band.numImages}"
        )

    reference_image = band.path[0].copy()
    reference_ids = ensure_atom_ids(reference_image)
    for index, image in enumerate(images):
        loaded = image
        if remap:
            loaded = reorder_by_atom_ids(reference_image, image)
            if loaded is None:
                loaded = spatial_map(reference_image, image)
        band.path[index].set_cell(loaded.get_cell(), scale_atoms=False)
        band.path[index].set_positions(loaded.get_positions())
        band.path[index].arrays["neb_atom_id"] = reference_ids.copy()

    ensure_atom_ids(band.path, reference=reference_image)
    for image in band.path:
        initialize_image_properties(image, band.jacobian)

    if hasattr(band, "_evaluate_image") and hasattr(band, "_finalize_image_state"):
        for index in (0, band.numImages - 1):
            result = band._evaluate_image(index)
            if hasattr(band, "_apply_image_result"):
                band._apply_image_result(index, result)
            band._finalize_image_state(index)
        if hasattr(band, "_update_spring_constants"):
            band._update_spring_constants()
        if hasattr(band, "update_image_rates"):
            band.update_image_rates(0)
