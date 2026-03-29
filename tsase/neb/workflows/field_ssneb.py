"""Reusable field-coupled SSNEB workflow helpers."""

from tsase.neb.core.mapping import spatial_map
from tsase.neb.models.charges import attach_field_charges, build_charge_array
from tsase.neb.models.field import EnthalpyWrapper

from .config import FieldSSNEBConfig, load_field_ssneb_config
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


def run_field_ssneb(*, config=None, **unexpected_kwargs):
    """Prepare and run a field-coupled SSNEB optimization from a resolved config."""

    if unexpected_kwargs:
        unexpected = ", ".join(sorted(unexpected_kwargs))
        raise TypeError(
            "run_field_ssneb no longer accepts workflow keyword arguments "
            f"({unexpected}). Pass config=FieldSSNEBConfig(...) instead, or "
            "use run_field_ssneb_from_yaml(...) for YAML-driven runs."
        )

    if not isinstance(config, FieldSSNEBConfig):
        raise TypeError(
            "run_field_ssneb expects config=FieldSSNEBConfig(...). "
            "Use load_field_ssneb_config(...) or run_field_ssneb_from_yaml(...) "
            "for YAML-driven runs."
        )

    resolved = config
    prepared = prepare_field_images(resolved.structures, resolved.charge_map, resolved.calculator)
    polarization_reference = spatial_map(prepared[0], resolved.reference_atoms)
    wrapped_calc = EnthalpyWrapper(
        resolved.calculator,
        field=resolved.field_vector,
        reference_atoms=polarization_reference,
        charges=build_charge_array(prepared[0], charge_map=resolved.charge_map),
    )
    for atoms in prepared:
        atoms.calc = wrapped_calc

    result = run_staged_ssneb(
        structures=prepared,
        structure_indices=resolved.structure_indices,
        num_images=resolved.num_images,
        remesh_stages=resolved.remesh_stages,
        k=resolved.spring,
        method=resolved.method,
        filter_factory=resolved.filter_factory,
        output_dir=resolved.run_dir,
        output_settings=resolved.output_settings,
        band_kwargs=resolved.band_kwargs,
        optimizer_kwargs=resolved.optimizer_kwargs,
        minimize_kwargs=resolved.minimize_kwargs,
        image_mobility_rates=resolved.image_mobility_rates,
        ci_activation_iteration=resolved.ci_activation_iteration,
        ci_activation_force=resolved.ci_activation_force,
        script_path=resolved.script_path,
        manifest_config=resolved.manifest_config,
        resolved_config=resolved.resolved_config,
        input_config_path=resolved.config_path,
    )
    result.update(
        {
            "field_vector": resolved.field_vector,
            "config": resolved,
        }
    )
    return result


def run_field_ssneb_from_yaml(path, *, overrides=None):
    """Load, resolve, and run the maintained YAML-driven field SSNEB workflow."""

    return run_field_ssneb(config=load_field_ssneb_config(path, overrides=overrides))
