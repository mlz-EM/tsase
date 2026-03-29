"""Maintained public surface for the refactored NEB package."""

from tsase.neb.core.band import ssneb
from tsase.neb.core.geometry import compute_jacobian, image_distance_vector, initialize_image_properties
from tsase.neb.core.mapping import NEB_ATOM_ID_ARRAY, ensure_atom_ids, reorder_by_atom_ids, spatial_map
from tsase.neb.core.path import generate_multi_point_path, interpolate_path
from tsase.neb.core.remesh import uniform_remesh
from tsase.neb.io import OutputManager, RunPaths, load_band_configuration_from_xyz, resolve_output_paths
from tsase.neb.models.charges import attach_field_charges, build_charge_array
from tsase.neb.models.field import EnthalpyWrapper, crystal_field_to_cartesian
from tsase.neb.optimize.fire import fire_ssneb
from tsase.neb.qm_ssneb import qm_ssneb
from tsase.neb.viz.stem import (
    ProjectedFrameAnalysis,
    StemAnalysisError,
    analyze_projected_neb_image,
    render_projected_frame,
    save_projected_neb_sequence,
)
from tsase.neb.workflows import (
    FieldSSNEBConfig,
    RemeshStage,
    StabilizedPerpForce,
    load_field_ssneb_config,
    run_field_ssneb,
    run_field_ssneb_from_yaml,
    run_staged_ssneb,
)
