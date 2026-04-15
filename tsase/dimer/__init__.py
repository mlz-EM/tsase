"""Maintained public surface for TSASE dimer-based saddle searches."""

from tsase.dimer.lanczos import LanczosDimer, lanczos_atoms, run_lanczos
from tsase.dimer.ssdimer import DimerSearchResult, SSDimer, SSDimer_atoms, run_ssdimer
from tsase.dimer.workflows import (
    DimerConfig,
    DownhillConnectionResult,
    RelaxationResult,
    StructureMatch,
    apply_mode_displacement,
    compare_structures,
    identify_downhill_connections,
    load_dimer_config,
    match_structure_to_references,
    relax_downhill_from_saddle,
    relax_structure,
    run_dimer,
    run_dimer_from_yaml,
)

__all__ = [
    "DimerSearchResult",
    "DimerConfig",
    "DownhillConnectionResult",
    "LanczosDimer",
    "RelaxationResult",
    "SSDimer",
    "SSDimer_atoms",
    "StructureMatch",
    "apply_mode_displacement",
    "compare_structures",
    "identify_downhill_connections",
    "lanczos_atoms",
    "load_dimer_config",
    "match_structure_to_references",
    "relax_downhill_from_saddle",
    "relax_structure",
    "run_dimer",
    "run_dimer_from_yaml",
    "run_lanczos",
    "run_ssdimer",
]
