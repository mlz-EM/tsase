"""Backward-compatible re-exports for refactored SSNEB helpers."""

from tsase.neb.core.geometry import (
    compute_jacobian,
    image_distance_vector,
    initialize_image_properties,
)
from tsase.neb.core.mapping import (
    NEB_ATOM_ID_ARRAY,
    ensure_atom_ids,
    reorder_by_atom_ids,
    spatial_map,
)
from tsase.neb.core.path import generate_multi_point_path, interpolate_path
from tsase.neb.io.restart import load_band_configuration_from_xyz
