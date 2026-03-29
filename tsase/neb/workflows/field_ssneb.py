"""Reusable field-coupled SSNEB workflow helpers."""

from pathlib import Path

from tsase.neb.core.mapping import spatial_map
from tsase.neb.io.artifacts import RunArtifacts
from tsase.neb.io.restart import load_band_configuration_from_xyz
from tsase.neb.models.charges import attach_field_charges, build_charge_array
from tsase.neb.models.field import EnthalpyWrapper
from tsase.neb.optimize.fire import fire_ssneb

from .config import FieldSSNEBConfig


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
    adaptive_springs=False,
    kmin=None,
    kmax=None,
    image_update_schedule=None,
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
        adaptive_springs=adaptive_springs,
        kmin=kmin,
        kmax=kmax,
        image_update_schedule=image_update_schedule,
        band_kwargs=band_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        minimize_kwargs=minimize_kwargs,
        script_path=script_path,
        manifest_config=manifest_config,
    )
    artifacts = RunArtifacts.create(base_dir=config.run_dir.parent, run_name=config.run_dir.name, timestamp=False)
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

    from tsase.neb.core.band import ssneb

    band = ssneb(
        prepared,
        config.structure_indices,
        numImages=config.num_images,
        k=config.spring,
        method=config.method,
        output_dir=str(artifacts.output_dir),
        filter_factory=config.filter_factory,
        adaptive_springs=config.adaptive_springs,
        kmin=config.kmin,
        kmax=config.kmax,
        image_update_schedule=config.image_update_schedule,
        **config.band_kwargs,
    )
    if config.restart_xyz is not None:
        load_band_configuration_from_xyz(band, str(config.restart_xyz))

    optimizer = fire_ssneb(band, **config.optimizer_kwargs)
    optimizer.minimize(**config.minimize_kwargs)

    prepared_paths = artifacts.write_structures(prepared, indices=config.structure_indices)
    if config.script_path is not None:
        artifacts.snapshot_script(config.script_path)
    artifacts.write_manifest(
        script_path=config.script_path,
        config=config.manifest_config,
        prepared_structures=prepared_paths,
    )
    return {
        "artifacts": artifacts,
        "band": band,
        "optimizer": optimizer,
        "field_vector": config.field_vector,
        "prepared_structures": prepared,
        "config": config,
    }
