"""Reusable field-coupled SSNEB workflow helpers."""

from tsase.neb.core.mapping import spatial_map
from tsase.neb.models.charges import attach_field_charges, build_charge_array
from tsase.neb.models.field import EnthalpyWrapper

from .config import FieldSSNEBConfig
from .staged import run_staged_ssneb


def prepare_field_images(structures, charge_map, calculator):
    """Attach a calculator and fixed charges to each structure."""
    prepared = []
    for atoms in structures:
        image = atoms.copy()
        image.calc = calculator
        attach_field_charges(image, charge_map)
        prepared.append(image)
    return prepared


def run_field_ssneb(
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
    """Prepare and run a field-coupled SSNEB optimization."""
    config = FieldSSNEBConfig.from_inputs(
        structures=structures,
        structure_indices=structure_indices,
        num_images=num_images,
        calculator=calculator,
        charge_map=charge_map,
        field=field,
        field_crystal=field_crystal,
        reference_atoms=reference_atoms,
        run_dir=run_dir,
        restart_xyz=restart_xyz,
        spring=spring,
        method=method,
        filter_factory=filter_factory,
        remesh_stages=remesh_stages,
        image_mobility_rates=image_mobility_rates,
        ci_activation_iteration=ci_activation_iteration,
        ci_activation_force=ci_activation_force,
        band_kwargs=band_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        minimize_kwargs=minimize_kwargs,
        script_path=script_path,
        manifest_config=manifest_config,
    )
    prepared = prepare_field_images(config.structures, config.charge_map, config.calculator)
    polarization_reference = spatial_map(prepared[0], config.reference_atoms)
    wrapped_calc = EnthalpyWrapper(
        config.calculator,
        field=config.field_vector,
        reference_atoms=polarization_reference,
        charges=build_charge_array(prepared[0], charge_map=config.charge_map),
    )
    for atoms in prepared:
        atoms.calc = wrapped_calc

    result = run_staged_ssneb(
        structures=prepared,
        structure_indices=config.structure_indices,
        num_images=config.num_images,
        remesh_stages=config.remesh_stages,
        restart_xyz=config.restart_xyz,
        k=config.spring,
        method=config.method,
        filter_factory=config.filter_factory,
        output_dir=config.run_dir,
        band_kwargs=config.band_kwargs,
        optimizer_kwargs=config.optimizer_kwargs,
        minimize_kwargs=config.minimize_kwargs,
        image_mobility_rates=config.image_mobility_rates,
        ci_activation_iteration=config.ci_activation_iteration,
        ci_activation_force=config.ci_activation_force,
        script_path=config.script_path,
        manifest_config=config.manifest_config,
    )
    result.update(
        {
        "field_vector": config.field_vector,
        "prepared_structures": prepared,
        "config": config,
        }
    )
    return result
