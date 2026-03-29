"""Normalized workflow configuration for field-coupled SSNEB."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tsase.neb.models.field import resolve_field_vector


@dataclass
class FieldSSNEBConfig:
    """Normalized field-SSNEB workflow options."""

    structures: list
    structure_indices: list
    num_images: int
    calculator: object
    charge_map: object
    field_vector: object
    reference_atoms: object
    run_dir: Path
    restart_xyz: Optional[Path] = None
    spring: float = 5.0
    method: str = "ci"
    filter_factory: object = None
    remesh_stages: object = None
    image_mobility_rates: object = None
    ci_activation_iteration: Optional[int] = None
    ci_activation_force: Optional[float] = None
    band_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    minimize_kwargs: dict = field(default_factory=dict)
    script_path: Optional[str] = None
    manifest_config: Optional[dict] = None

    @classmethod
    def from_inputs(
        cls,
        *,
        structures,
        structure_indices,
        num_images,
        calculator,
        charge_map,
        field=None,
        field_crystal=None,
        reference_atoms=None,
        run_dir=None,
        restart_xyz=None,
        spring=5.0,
        method="ci",
        filter_factory=None,
        remesh_stages=None,
        image_mobility_rates=None,
        ci_activation_iteration=None,
        ci_activation_force=None,
        band_kwargs=None,
        optimizer_kwargs=None,
        minimize_kwargs=None,
        script_path=None,
        manifest_config=None,
    ):
        structures = list(structures)
        reference_atoms = structures[0] if reference_atoms is None else reference_atoms
        run_dir = Path("field_ssneb_runs") if run_dir is None else Path(run_dir)
        return cls(
            structures=structures,
            structure_indices=list(structure_indices),
            num_images=int(num_images),
            calculator=calculator,
            charge_map=charge_map,
            field_vector=resolve_field_vector(structures[0].get_cell(), field=field, field_crystal=field_crystal),
            reference_atoms=reference_atoms,
            run_dir=run_dir,
            restart_xyz=None if restart_xyz is None else Path(restart_xyz),
            spring=spring,
            method=method,
            filter_factory=filter_factory,
            remesh_stages=remesh_stages,
            image_mobility_rates=image_mobility_rates,
            ci_activation_iteration=ci_activation_iteration,
            ci_activation_force=ci_activation_force,
            band_kwargs={} if band_kwargs is None else dict(band_kwargs),
            optimizer_kwargs={} if optimizer_kwargs is None else dict(optimizer_kwargs),
            minimize_kwargs={} if minimize_kwargs is None else dict(minimize_kwargs),
            script_path=script_path,
            manifest_config=manifest_config,
        )
