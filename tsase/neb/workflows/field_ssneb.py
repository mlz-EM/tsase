"""Reusable field-coupled SSNEB workflow helpers."""

from pathlib import Path

from ase import io

from tsase.neb.core.mapping import spatial_map
from tsase.neb.io.artifacts import RunArtifacts
from tsase.neb.io.restart import load_band_configuration_from_xyz
from tsase.neb.models.charges import attach_field_charges, build_charge_array
from tsase.neb.models.field import EnthalpyWrapper, crystal_field_to_cartesian
from tsase.neb.optimize.fire import fire_ssneb


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
    run_dir = Path("field_ssneb_runs") if run_dir is None else Path(run_dir)
    artifacts = RunArtifacts.create(base_dir=run_dir.parent, run_name=run_dir.name, timestamp=False)
    prepared = prepare_field_images(structures, charge_map, calculator)
    polarization_reference = prepared[0] if reference_atoms is None else spatial_map(prepared[0], reference_atoms)
    field_vector = (
        crystal_field_to_cartesian(prepared[0].get_cell(), field_crystal)
        if field_crystal is not None
        else field
    )
    wrapped_calc = EnthalpyWrapper(
        calculator,
        field=field_vector,
        reference_atoms=polarization_reference,
        charges=build_charge_array(prepared[0], charge_map=charge_map),
    )
    for atoms in prepared:
        atoms.calc = wrapped_calc

    from tsase.neb.core.band import ssneb

    band_options = {} if band_kwargs is None else dict(band_kwargs)
    band = ssneb(
        prepared,
        structure_indices,
        numImages=num_images,
        k=spring,
        method=method,
        output_dir=str(artifacts.output_dir),
        filter_factory=filter_factory,
        adaptive_springs=adaptive_springs,
        kmin=kmin,
        kmax=kmax,
        image_update_schedule=image_update_schedule,
        **band_options,
    )
    if restart_xyz is not None:
        load_band_configuration_from_xyz(band, str(restart_xyz))

    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    minimize_kwargs = {} if minimize_kwargs is None else dict(minimize_kwargs)
    optimizer = fire_ssneb(band, **optimizer_kwargs)
    optimizer.minimize(**minimize_kwargs)

    prepared_paths = artifacts.write_structures(prepared, indices=structure_indices)
    if script_path is not None:
        artifacts.snapshot_script(script_path)
    artifacts.write_manifest(
        script_path=script_path,
        config=manifest_config,
        prepared_structures=prepared_paths,
    )
    return {
        "artifacts": artifacts,
        "band": band,
        "optimizer": optimizer,
        "field_vector": field_vector,
        "prepared_structures": prepared,
    }

