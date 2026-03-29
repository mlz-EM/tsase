# Follow-Up Delivery Phase Plan: Remaining Gaps, Post-Hoc STEM API, and MACE-Field Integration

## Objective

This phase implements the pieces that were requested but are not yet fully delivered in the maintained code path.

The scope of this phase is intentionally narrow and aggressive:
- remove the remaining legacy and duplicate code paths that still survive in the repository,
- finish the transition to one maintained runtime path,
- expose a standalone post-hoc STEM analysis API that operates on saved XYZ files,
- integrate MACE-field calculators safely without corrupting energies, forces, or plotting,
- replace the current single `plot_property` style with explicit YAML-configured plot/series entries.

The hard requirement remains the same:

> be aggressive about deleting redundancy, but do not corrupt the SSNEB math, field-coupling math, or optimizer behavior.

## What is still not fully delivered

The current maintained path already improved significantly, but the following requested changes are still incomplete:

1. Remaining legacy modules and duplicate implementations still exist in the repo.
2. Restart cleanup is not complete because both:
   - the new YAML path-source model exists, and
   - the older overwrite-style band restart helper still exists.
3. The output system is improved, but the STEM functionality is still tied mainly to the iteration-output path rather than exposed as a clear post-hoc API.
4. MACE-field calculators are not yet treated as a first-class calculator mode with explicit energy semantics.
5. Plotting/output control is still too narrow and should be driven by explicit YAML entries rather than one legacy-like `plot_property` channel.
6. Some core physics/control parameters are still not fully surfaced through YAML.

This phase closes those gaps.

---

## Deliverable 1 — Remove the remaining undelivered legacy and duplicate paths

### Goal

Delete the remaining code paths that keep the repo looking like two generations of runtime are still being supported simultaneously.

### Remove aggressively

#### A. Legacy import shims that are no longer needed internally

Delete compatibility modules once internal imports and maintained examples no longer depend on them.

Candidates include:
- `tsase/neb/ssneb.py`
- `tsase/neb/fire_ssneb.py`
- `tsase/neb/minimizer_ssneb.py`
- `tsase/neb/ssneb_utils.py`
- `tsase/neb/run_artifacts.py`
- `tsase/neb/stem_visualization.py`

If a public compatibility promise absolutely requires one temporary shim, keep exactly one minimal façade and remove the rest.

#### B. Obsolete duplicate implementation paths

Delete old parallel/alternate implementations that duplicate the maintained SSNEB path.

Primary candidate:
- `tsase/neb/pssneb.py`

This file duplicates core band-force, tangent, spring, and image-workspace logic and still carries old `fixstrain` semantics and hostfile-specific execution behavior. If the maintained `core.band + ExecutionContext` path is the official path, this duplicate implementation should be removed.

#### C. Old overwrite-style restart path

The maintained runtime must use one path-input model only.

That means removing the residual stage-0 overwrite restart route where a band is first constructed and then overwritten from XYZ.

Specifically:
- remove `restart_xyz` from the staged runtime as a separate overwrite step,
- remove or retire `load_band_configuration_from_xyz(band, xyz_path, ...)` as a maintained runtime dependency,
- keep a migration/helper function only if it is moved under an explicitly legacy or developer-only namespace.

#### D. Remaining path and naming inconsistencies

Standardize the maintained example/output naming convention.

The maintained code should not mix:
- `example/`
- `examples/`

for the same active workflow surface.

---

## Deliverable 2 — Complete YAML parameter coverage

### Goal

All maintained user-facing runtime settings must come from YAML.

### Missing coverage to add

#### A. External pressure / stress (`express`)

`express` is still part of the maintained band physics but is not yet a first-class YAML field.

Add explicit YAML support, for example:

```yaml
band:
  spring: 5.0
  method: ci
  ss: true
  weight: 1.0
  express:
    units: GPa
    tensor:
      - [0.0, 0.0, 0.0]
      - [0.0, 0.0, 0.0]
      - [0.0, 0.0, 0.0]
```

or an equivalent lower-triangular form if preferred.

Design rules:
- the maintained YAML must make the units explicit,
- the lower-triangular TSASE convention must be documented and validated,
- unsupported upper-triangular entries must still be normalized/warned consistently.

#### B. Tangent mode and optional band-level settings

Any maintained band setting still exposed programmatically should either:
- be surfaced in YAML, or
- be removed from the maintained path.

#### C. Output plotting entries

Replace the narrow single-series plotting interface with explicit YAML-configured plot entries.

---

## Deliverable 3 — Standalone post-hoc STEM analysis API

### Goal

Expose STEM analysis as a reusable public API that can be called after the SSNEB run has finished.

This API must operate on saved XYZ files and save requested artifacts on demand.

### User need

The user wants to take an already written XYZ path and then request:
- PNG frames,
- GIF sequence,
- NPY analysis output,
- and optionally diagnostics,

without rerunning SSNEB.

### Required public API

Add a maintained public function, for example:

```python
from tsase.neb.viz import analyze_stem_sequence_from_xyz
```

or equivalent under a maintained workflow/API module.

### Minimum interface

```python
analyze_stem_sequence_from_xyz(
    xyz_file,
    output_dir=None,
    emit_png=True,
    emit_gif=True,
    emit_npy=True,
    emit_diagnostics=True,
    cutoff_angstrom=...,
)
```

### Required behavior

Given a saved path XYZ file, the function must:
1. read all images,
2. run the same projected STEM-style analysis used by the maintained visualization logic,
3. save outputs only for the requested artifact types,
4. return a structured summary of generated outputs.

### NPY requirement

The `.npy` output is important and must be treated as a first-class artifact, not an afterthought.

The saved numeric analysis should be structured enough for later quantitative automation.

At minimum it should preserve per-frame arrays/fields such as:
- Pb projected coordinates,
- Zr projected coordinates,
- oxygen-site coordinates,
- tilt values,
- Pb displacement vectors,
- mean-subtracted displacement vectors,
- frame diagnostics,
- metadata needed to interpret shapes and units.

If one `.npy` file becomes awkward, use:
- either a single `.npz`, or
- one `.npy`/`.json` pair.

The important requirement is a stable post-hoc machine-readable artifact.

### Required CLI/example

Add a thin maintained script, for example:
- `examples/analyze_stem_from_xyz.py`

It should accept:
- `--xyz`
- `--output-dir`
- `--png`
- `--gif`
- `--npy`
- `--diagnostics`

This script should just call the maintained API.

### Integration rule

The online iteration-output path may still call the same STEM API internally, but the post-hoc API must be the authoritative implementation.

That means:
- no duplicated STEM analysis code in the optimizer/output manager,
- the iteration output path should become a wrapper around the post-hoc-capable STEM API.

---

## Deliverable 4 — Safe MACE-field calculator integration

### Goal

Support MACE-field models as a first-class calculator mode for polarization-aware SSNEB runs.

### User-provided constraint

For a MACE-field ASE calculator:
- `atoms.get_potential_energy()` is already enthalpy-adjusted,
- polarization is available directly via:
  - `atoms.calc.results["polarization"]`

The integration must not corrupt:
- the path energy definitions,
- the atomic forces,
- the field-coupled physics,
- the plotting logic.

### Critical design rule

A MACE-field calculator that already includes polar enthalpy must **not** be wrapped again in `EnthalpyWrapper`.

Otherwise the code risks:
- double-counting field enthalpy in the energy,
- adding an extra field-force contribution on top of forces that are already field-consistent,
- producing incorrect plots and diagnostics.

### Required calculator modes

Introduce explicit calculator semantics in YAML, for example:

```yaml
model:
  calculator:
    kind: mace_field
    model_path: ...
    device: cuda
    energy_semantics: enthalpy_adjusted
    polarization_source: results.polarization
```

Supported maintained calculator modes should explicitly distinguish at least:

1. `emt` / generic intrinsic calculator
2. `mace` intrinsic calculator
3. `mace_field` field-aware calculator with built-in enthalpy/polarization
4. optional future hybrid modes

### Runtime contract for `mace_field`

For `mace_field`, the maintained runtime must assume:
- calculator energy is already the enthalpy-like quantity to use as the path energy,
- atomic forces returned by the calculator are already the correct forces for that enthalpy model,
- polarization is obtained from the calculator results,
- no extra field-force correction is added externally,
- no second enthalpy wrapper is applied.

### Energy bookkeeping requirement

The maintained code must normalize per-image energy fields into explicit semantic slots, for example:
- `intrinsic_energy`
- `field_energy`
- `enthalpy_adjusted`

However, the values available depend on calculator mode.

#### For wrapped intrinsic calculators (`EnthalpyWrapper` path)

The code can populate:
- intrinsic/base energy,
- field contribution,
- enthalpy-adjusted energy.

#### For `mace_field`

The code may only know enthalpy-adjusted energy directly unless the calculator exposes additional decomposition.

Therefore the implementation must not invent a fake intrinsic energy decomposition.

Design rule:
- only populate components that are genuinely available,
- do not reconstruct `intrinsic_energy = enthalpy - E·P` unless the model contract explicitly guarantees that interpretation and needed quantities are available in a trustworthy way.

### Optional enriched support

If a future MACE-field calculator can return additional result keys such as:
- `intrinsic_energy`
- `field_energy`
- `polarization`

then the normalized result object should use them.

But this must be opt-in and based on explicit result availability, not guessing.

### Required internal abstraction

Introduce a normalized per-image model result contract, for example:

```python
ImagePhysicsResult(
    energy_total,
    energy_intrinsic=None,
    energy_field=None,
    forces=...,
    stress=...,
    polarization=...,
    polarization_c_per_m2=...,
    dipole=None,
)
```

The band/runtime should consume this normalized object rather than making calculator-specific assumptions directly in `band.py`.

### Required implementation outcome

The maintained runtime should support two clean paths:

#### Path A — intrinsic calculator + external field wrapper
- use `EnthalpyWrapper`
- base energy and field energy are known separately
- forces include the wrapper-added field term

#### Path B — native `mace_field`
- do not wrap with `EnthalpyWrapper`
- trust the calculator’s returned total energy/forces/polarization semantics
- only use energy decomposition fields that the calculator explicitly exposes

---

## Deliverable 5 — Replace `plot_property` with explicit YAML plot entries

### Goal

The current plotting/output selection is too narrow.

The maintained YAML should accept explicit entries to plot/export, including:
- intrinsic energy,
- enthalpy-adjusted energy,
- field contribution when available,
- polarization magnitude,
- polarization X,
- polarization Y,
- polarization Z.

### Proposed YAML structure

```yaml
outputs:
  energy_profile:
    enabled: true
    emit_csv: true
    emit_png: true
    entries:
      - enthalpy_adjusted
      - intrinsic_energy
      - field_energy
      - polarization_mag
      - polarization_x
      - polarization_y
      - polarization_z
```

### Design rule

The plotting/export code must only plot entries that are both:
- requested, and
- available under the chosen calculator mode.

If a requested quantity is unavailable, the code should:
- either skip it with a clear warning, or
- error early if strict mode is requested.

### Required internal cleanup

This phase removes the single-value `plot_property` interface from the maintained path.

Keep backward compatibility only if absolutely needed, and if kept, route it through the new entries model as a deprecated alias.

### CSV/export requirement

The CSV output should also reflect the same explicit entry model rather than being hard-coded to one fixed set only.

---

## Deliverable 6 — Final energy/force correctness audit

### Goal

Because MACE-field integration touches the most sensitive part of the code, this phase must include a small explicit correctness audit.

### Must verify

#### A. Wrapped intrinsic calculator path
- energy decomposition remains correct,
- forces remain equal to intrinsic forces plus the intended field contribution,
- polarization/dipole reporting remains consistent.

#### B. Native `mace_field` path
- no second field wrapper is applied,
- energy used along the path equals the calculator-returned enthalpy-adjusted energy,
- forces are taken directly from the calculator and not modified by an extra field term,
- polarization for plots/diagnostics is pulled from calculator results,
- stress handling is consistent with the calculator’s semantics.

#### C. Plot outputs
- enthalpy-adjusted series is correct for both calculator modes,
- intrinsic/base-energy series only appears when genuinely available,
- polarization magnitude and components match the normalized stored result.

---

## Implementation sequence

### Step 0 — Freeze current maintained behavior

Capture a short regression baseline for:
- current wrapped-calculator path,
- current YAML workflow path,
- current output artifacts,
- current STEM iteration rendering behavior.

### Step 1 — Finish the delete pass

Remove:
- unused legacy shims,
- obsolete duplicate implementations,
- residual overwrite-style restart flow from the maintained runtime,
- leftover naming inconsistencies where feasible.

### Step 2 — Surface remaining physics/control parameters in YAML

Add explicit YAML support for:
- `express`
- any remaining maintained band/runtime settings still exposed only programmatically.

### Step 3 — Introduce normalized calculator semantics

Add calculator-mode resolution for:
- intrinsic calculators,
- wrapped field calculators,
- `mace_field` calculators.

Normalize per-image energy/polarization/force/stress outputs through a common result contract.

### Step 4 — Integrate `mace_field`

Implement the safe native-field path:
- no external enthalpy wrapper,
- direct polarization retrieval from calculator results,
- correct normalized energy semantics.

### Step 5 — Replace `plot_property` with explicit output entries

Refactor the plotting/export layer to use YAML-configured entry lists and availability-aware rendering.

### Step 6 — Expose standalone post-hoc STEM API

Implement the public API that takes XYZ input and emits requested PNG/GIF/NPY/diagnostics outputs.

### Step 7 — Route online STEM emission through that API

Refactor the iteration-output STEM hook to call the same maintained post-hoc API so there is only one STEM implementation path.

### Step 8 — Add maintained example scripts

Provide:
- a maintained YAML-driven field SSNEB example that exercises the updated output model,
- a maintained post-hoc STEM analysis example/CLI for saved XYZ files,
- optionally a minimal MACE-field example config if dependencies are available.

### Step 9 — Final correctness and cleanup pass

Remove any leftover deprecated code or options that became unnecessary during implementation.

---

## Tests required for this phase

### A. Remaining-delivery tests

1. old overwrite-style restart path is no longer part of the maintained runtime
2. maintained internal modules no longer import through deleted legacy shims
3. obsolete duplicate implementation files are removed or explicitly quarantined

### B. YAML-coverage tests

4. YAML `band.express` is parsed and applied correctly
5. YAML strain/filter settings still produce the intended projected cell updates
6. YAML plot entry requests resolve correctly

### C. STEM API tests

7. post-hoc API can read a saved XYZ path and emit PNG frames on request
8. post-hoc API can emit GIF on request
9. post-hoc API can emit machine-readable NPY/NPZ analysis output on request
10. iteration-time STEM output uses the same API path

### D. MACE-field tests

11. `mace_field` mode does not apply `EnthalpyWrapper`
12. `mace_field` energies are consumed as enthalpy-adjusted totals without double counting
13. `mace_field` forces are taken directly from the calculator without extra field-force corruption
14. `mace_field` polarization is retrieved from `results["polarization"]`
15. plots/CSV correctly handle availability of intrinsic/base energy versus total enthalpy

### E. Regression-preservation tests

16. wrapped intrinsic-calculator field path remains numerically consistent with the preserved baseline
17. FIRE and SSNEB force/tangent behavior remain intact for unchanged modes
18. final maintained example scripts run successfully

---

## Recommended git plan

### Working branch

Use a dedicated follow-up branch, for example:
- `refactor/neb-followup-delivery`

### Recommended commit sequence

1. `test: capture follow-up baselines for remaining delivery gaps`
2. `refactor: delete remaining legacy shims and duplicate runtime paths`
3. `feat: expose remaining maintained band parameters through yaml`
4. `refactor: add normalized calculator semantics for field-aware models`
5. `feat: support native mace-field calculator mode without double counting`
6. `refactor: replace plot_property with explicit yaml plot entries`
7. `feat: add post-hoc stem analysis api for saved xyz paths`
8. `refactor: route online stem rendering through the post-hoc api`
9. `feat: add maintained examples for updated yaml workflow and stem api`
10. `test: verify mace-field correctness and final maintained runtime behavior`
11. `docs: summarize follow-up delivery phase`

---

## Exit criteria

This follow-up phase is complete when all of the following are true:

- [ ] remaining legacy and duplicate runtime paths requested for deletion are removed aggressively
- [ ] the maintained runtime no longer relies on overwrite-style restart behavior
- [ ] all maintained user-facing runtime parameters, including `express`, are represented through YAML
- [ ] a public post-hoc STEM analysis API exists for saved XYZ files
- [ ] that API can emit PNG, GIF, and NPY/NPZ analysis artifacts on request
- [ ] online STEM output uses the same authoritative API
- [ ] native `mace_field` calculator mode is supported safely
- [ ] `mace_field` mode does not double count enthalpy or corrupt forces
- [ ] polarization can be taken directly from `results["polarization"]` for `mace_field`
- [ ] energy/profile plotting is driven by explicit YAML entries
- [ ] requested plot entries include intrinsic energy, enthalpy-adjusted energy, polarization magnitude, and polarization components when available
- [ ] maintained example scripts reflect the updated codebase
- [ ] regression tests confirm that preserved SSNEB math/physics remain intact

---

## Phase result

At the end of this phase, the codebase should no longer merely have a cleaner maintained path sitting beside old infrastructure.

It should instead behave like a finished maintained system with:
- one authoritative runtime,
- one authoritative YAML interface,
- one authoritative STEM analysis API,
- safe support for both wrapped intrinsic-field calculators and native MACE-field calculators,
- explicit output plotting/export semantics,
- and no remaining large redundant code paths that obscure the intended design.
