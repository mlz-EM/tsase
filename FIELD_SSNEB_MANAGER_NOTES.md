# Field-Coupled SSNEB Corrections

## Applied corrections

1. The cell-force term from the electric enthalpy was implemented with ASE's stress convention:
   `virial_field = sym(E ⊗ p)` and `stress_field = -virial_field / V`.
   This keeps `H = U - E·p` consistent with the forces seen by SSNEB and ASE-style cell optimizers.

2. `StrainFilter` was not used as the recommended path for constant-volume coupled atom+cell relaxation.
   In ASE, `StrainFilter` only exposes strain DOFs and keeps scaled positions fixed, so it is not the right tool for a fully coupled NEB image relaxation.
   The production path supports `ExpCellFilter` and `UnitCellFilter` for cell relaxation, with `constant_volume=True` or `hydrostatic_strain=True` when needed.

3. The field coupling is implemented as an ionic dipole model based on fixed user-supplied charges and MIC-tracked displacements relative to Image 0.
   This is physically consistent with the requested `q_i E` force term, but it is not a Berry-phase electronic polarization treatment.
   The workflow now also supports an explicit polarization-reference structure, so a centrosymmetric phase can be used instead of implicitly setting Image 0 to zero polarization.

4. The example files referenced in the manager brief (`PZO_AFE.cif`, `PZO_Intermediate.cif`, `PZO_FE.cif`) are not present in the repository.
   The new driver therefore accepts arbitrary CIF lists and example runs can be performed with the structures that are actually available locally.
