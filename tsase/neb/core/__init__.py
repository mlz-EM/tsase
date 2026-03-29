"""Core SSNEB math helpers and band implementation.

Allowed dependencies:
- core may depend only on other core modules and legacy math utilities.
"""

from .band import ssneb
from .geometry import compute_jacobian, image_distance_vector, initialize_image_properties
from .interfaces import ExecutionContext, ImageEvalResult, PathSpec
from .mapping import (
    NEB_ATOM_ID_ARRAY,
    ensure_atom_ids,
    reorder_by_atom_ids,
    spatial_map,
)
from .path import generate_multi_point_path, interpolate_path
