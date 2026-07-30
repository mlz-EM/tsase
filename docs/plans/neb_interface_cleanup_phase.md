# Interface Cleanup Phase Plan for the Refactored NEB Codebase

## Scope of this phase

This phase starts from the **already refactored package layout** and cleans up the **code interface only**.

### Primary objective

Replace repeated scenario-specific branching with a small set of normalized interface objects so that:
- execution ownership is defined once,
- run paths and workspaces are defined once,
- image evaluation returns one normalized result object,
- workflow options are normalized before the run starts, and
- user-facing outputs are emitted through one reporting interface.

### Hard constraint

This phase must **not intentionally change any mathematics, physics, or optimizer formulas**.

That means no intentional changes to:
- SSNEB tangent formulas,
- spring-force formulas,
- Jacobian scaling,
- FIRE update equations,
- field-coupling formulas,
- charge handling formulas,
- polarization formulas,
- restart mapping logic,
- output numerical definitions.

Execution semantics may be cleaned up where needed to preserve the intended behavior of the existing codebase, but the underlying equations and physical model must remain unchanged.

## Why this phase is needed

The refactored codebase improved package layout, but several interfaces still encode behavior through repeated local branching such as:
- serial vs MPI,
- output-owner vs non-owner,
- two-endpoint vs multi-point path construction,
- field vs crystal-field vs no-field,
- public outputs vs per-image scratch work,
- plotting enabled vs disabled,
- diagnostics enabled vs disabled.

This makes the code harder to maintain and contributed directly to regressions in:
- per-image calculator isolation,
- MPI field metadata synchronization,
- diagnostics ownership,
- optional zero-field workflow behavior.

The goal of this phase is to remove that branching pressure by introducing a few normalized interface objects and clear ownership boundaries.

## Design rules for this phase

1. **One execution owner for shared outputs**
   - Shared user-facing outputs are owned by a single process.
   - In serial, the current process is the owner.
   - In MPI, rank 0 is the owner.

2. **One workspace per image**
   - Every image evaluates in its own stable workspace directory.
   - This is true for both serial and MPI runs.
   - In-memory ASE calculators simply ignore this.
   - File-backed calculators continue to work safely.

3. **One normalized evaluation payload per image**
   - Image evaluation returns one complete object containing all per-image computed values.
   - MPI synchronization exchanges that full object, not selected fields.

4. **One normalized run layout**
   - Public outputs and image-local workspaces are defined in one place.
   - Path logic is not split across multiple modules.

5. **One normalized workflow config**
   - Optional workflow inputs are resolved once before creating the band or optimizer.
   - Core runtime code should not need to repeatedly inspect mutually exclusive options.

6. **One reporting interface**
   - Optimizers should report events to a reporting/output object rather than manually branching on rank, plot flags, or output ownership.

## Deliverables

This phase must deliver all of the following:

1. A cleaned-up interface layer on top of the already refactored codebase.
2. Preserved mathematics and physics behavior.
3. Restored safe per-image workspaces for file-based calculators.
4. Owner-only shared outputs for MPI runs.
5. Full MPI synchronization of field-related image metadata.
6. Normalized zero-field handling.
7. A replacement example script that runs under the new codebase:
   - `examples/run_field_ssneb_interpolated_refactored.py`

## Interface objects to introduce

### 1. `ExecutionContext`

**Purpose**
Centralize execution-mode state so the rest of the code no longer branches directly on `parallel`, `rank`, or `size`.

**Responsibilities**
- store `parallel`, `rank`, `size`, communicator,
- expose `is_output_owner`,
- expose helpers for synchronization,
- expose helpers for owner-only actions.

**Minimum interface**
- `is_parallel`
- `rank`
- `size`
- `communicator`
- `is_output_owner`
- `allgather_image_results(...)`

**Used by**
- `tsase.neb.core.band`
- `tsase.neb.optimize.base`
- any owner-only IO/reporting logic

---

### 2. `RunLayout`

**Purpose**
Define the run directory structure exactly once.

**Responsibilities**
- define public output locations,
- define per-image workspaces,
- define manifest/input/prepared-structure locations,
- eliminate split path logic across `paths.py`, `artifacts.py`, and `band.py`.

**Minimum interface**
- `run_dir`
- `public_dir`
- `xyz_dir`
- `diagnostics_file`
- `log_file`
- `manifest_file`
- `image_workdir(index)`
- `prepared_structures_dir`
- `inputs_dir`

**Required layout principle**
The run root must distinguish between:
- **public shared outputs**, and
- **image-local working directories**.

Example structure:

```text
<run_dir>/
  xyz/
  diagnostics.csv
  fe.out
  run_manifest.json
  inputs/
  prepared_structures/
  images/
    image_0000/
    image_0001/
    ...
```

---

### 3. `ImageEvalResult`

**Purpose**
Represent one complete image evaluation result so serial and MPI follow the same logical path.

**Responsibilities**
Store all per-image values produced by evaluation.

**Fields**
- `u`
- `base_u`
- `field_u`
- `f`
- `st`
- `dipole`
- `polarization`
- `polarization_c_per_m2`

**Used by**
- `band._evaluate_image(...)`
- MPI gather/scatter logic
- diagnostics and plotting

---

### 4. `PathSpec`

**Purpose**
Remove constructor-level ambiguity from the band.

**Responsibilities**
Normalize whether the band is built from:
- two endpoints, or
- multiple control points plus indices.

**Result**
The band should receive one normalized path specification rather than branching on whether `p1` is a list.

---

### 5. `FieldSSNEBConfig`

**Purpose**
Normalize workflow inputs before any band, calculator wrapper, or optimizer is created.

**Responsibilities**
Resolve:
- `field` vs `field_crystal` vs zero-field,
- `reference_atoms` default,
- run directory selection,
- optional restart path,
- band/optimizer/minimizer kwargs defaults.

**Required rule**
If both `field` and `field_crystal` are absent, the resolved field vector must be an explicit zero vector, not `None`.

---

### 6. `Reporter`

**Purpose**
Own all shared user-facing output behavior so the optimizer is no longer full of rank/output-case branching.

**Responsibilities**
- diagnostics writes,
- XYZ snapshots,
- energy plots,
- projected STEM rendering,
- text log file writes,
- end-of-run summary output.

**Key behavior**
- output-owner reporter writes real outputs,
- non-owner ranks get a null/no-op reporter.

## Module-level implementation plan

### A. `tsase.neb.core.band`

#### Current structural issues
- direct branching on `parallel`/`rank` in multiple places,
- path construction mode branching inside the constructor,
- no stable workspace abstraction,
- MPI only synchronizes partial image state,
- diagnostics ownership is not enforced through one common mechanism.

#### Required changes
1. Add `ExecutionContext` to the band.
2. Add `RunLayout` to the band.
3. Create stable per-image workspaces under the run layout.
4. Wrap image evaluation in a workspace context manager.
5. Make `_evaluate_image(...)` return `ImageEvalResult`.
6. Make serial and MPI both apply evaluation results through the same helper.
7. Synchronize full `ImageEvalResult` objects in MPI.
8. Replace direct diagnostics ownership logic with `context.is_output_owner`.
9. Replace constructor input branching with a normalized `PathSpec` or equivalent factory-layer normalization.

#### Explicit non-goals
- do not change the tangent formulas,
- do not change spring-force formulas,
- do not change how PV/enthalpy is computed,
- do not change CI selection logic.

### B. `tsase.neb.optimize.base`

#### Current structural issues
- repeated owner checks,
- repeated output-condition checks,
- direct responsibility for multiple output types,
- diagnostics and reporting mixed into optimization control flow.

#### Required changes
1. Introduce `Reporter`.
2. Replace direct writes with reporter calls.
3. Use a null reporter for non-owner ranks.
4. Preserve log format, CSV format, plot naming, and XYZ naming exactly.

#### Explicit non-goals
- do not change FIRE control flow,
- do not change convergence logic,
- do not change CI activation logic.

### C. `tsase.neb.io.paths` and `tsase.neb.io.artifacts`

#### Current structural issues
- path decisions are split,
- public outputs and image-local work paths are not unified,
- different modules compute related paths independently.

#### Required changes
1. Replace ad hoc path assembly with `RunLayout`.
2. Make `RunArtifacts` consume `RunLayout` rather than re-deriving paths.
3. Keep manifest schema and copied-input metadata unchanged.

### D. `tsase.neb.models.field`

#### Current structural issues
- field normalization is not centralized,
- zero-field behavior is not guaranteed at the interface level.

#### Required changes
1. Add a normalized `resolve_field_vector(...)` helper.
2. Guarantee explicit zero-vector field behavior.
3. Keep all field formulas unchanged.

### E. `tsase.neb.workflows.field_ssneb`

#### Current structural issues
- too many optional arguments are resolved inline,
- workflow logic mixes normalization with orchestration,
- example compatibility is fragile.

#### Required changes
1. Introduce `FieldSSNEBConfig`.
2. Normalize all workflow options before creating runtime objects.
3. Keep runtime body close to straight-line orchestration.
4. Provide a replacement example script that uses the normalized interface.

## Required new example

Provide:
- `examples/run_field_ssneb_interpolated_refactored.py`

### Example goals
This example must:
- run against the cleaned-up interface layer,
- demonstrate the normalized workflow API,
- not require the user to know internal package structure,
- clearly show where run directory, field choice, reference structure, optimizer settings, and optional restart are specified.

### Example structure requirements
The example should:
1. construct a `FieldSSNEBConfig` or use a single high-level workflow entry point,
2. explicitly show zero-field and crystal-field usage patterns,
3. create outputs in the new run layout,
4. still support the refactored codebase without relying on old legacy imports.

## Implementation sequence

### Step 0 — Capture baseline behavior from the current refactored branch
Record golden references for:
- helper outputs,
- band force evaluation,
- field wrapper outputs,
- diagnostics format,
- plot file names,
- XYZ file names,
- manifest schema,
- current example output behavior.

This phase preserves the **current formulas** exactly.

### Step 1 — Add interface objects with no behavior change yet
Introduce:
- `ExecutionContext`
- `RunLayout`
- `ImageEvalResult`
- `PathSpec`
- `FieldSSNEBConfig`
- `Reporter`

At this step they may wrap current behavior, but must not change runtime outputs yet.

### Step 2 — Route `band.py` through `ExecutionContext` and `RunLayout`
- replace local rank/parallel branching with context queries,
- allocate image workspaces through layout,
- add workspace context manager,
- keep formulas unchanged.

### Step 3 — Convert image evaluation to `ImageEvalResult`
- `_evaluate_image(...)` produces normalized result objects,
- serial path applies local results,
- MPI path gathers full result objects and applies them to every rank,
- diagnostics and plotting consume synchronized image state.

### Step 4 — Route owner-only outputs through `Reporter`
- move diagnostics, XYZ, plots, and text log handling behind the reporter,
- owner gets a real reporter,
- non-owner ranks get a null reporter.

### Step 5 — Normalize workflow configuration
- centralize field resolution,
- centralize restart normalization,
- centralize default artifact/layout construction,
- centralize band/optimizer/minimizer option normalization.

### Step 6 — Add the replacement example
- implement `examples/run_field_ssneb_interpolated_refactored.py`,
- verify it runs under the cleaned-up interface,
- keep its numerical behavior aligned with the current refactored implementation.

### Step 7 — Final compatibility and cleanup pass
- keep legacy imports working where already promised,
- remove duplicate branching made unnecessary by the new interfaces,
- confirm no new circular dependencies were introduced.

## Tests required for this phase

These tests are regression-preservation tests plus interface-cleanup integration tests.

### A. Regression-preservation tests
1. `compute_jacobian` unchanged
2. `image_distance_vector` unchanged
3. path interpolation unchanged
4. atom mapping / restart mapping unchanged
5. tangent outputs unchanged
6. spring constant outputs unchanged
7. `ssneb.forces()` outputs unchanged for deterministic mock cases
8. FIRE one-step and short-run outputs unchanged
9. `EnthalpyWrapper` energy/force/stress/dipole/polarization outputs unchanged
10. diagnostics CSV header and row ordering unchanged
11. log formatting unchanged
12. manifest schema unchanged

### B. Interface-cleanup integration tests
13. **Per-image workspace isolation test**
   - use a mock file-writing calculator,
   - verify each image writes only inside its own workspace.

14. **MPI full-state synchronization test**
   - verify rank 0 sees synchronized `base_u`, `field_u`, `dipole`, `polarization`, and `polarization_c_per_m2` for all images.

15. **Owner-only diagnostics test**
   - verify exactly one diagnostics append per iteration in MPI runs.

16. **Zero-field normalization test**
   - verify the workflow runs when both `field` and `field_crystal` are absent.

17. **Replacement example smoke test**
   - run `examples/run_field_ssneb_interpolated_refactored.py` with a short deterministic configuration and verify outputs are created under the new run layout.

## Git plan

### Working branch
Create a new working branch from the already refactored branch:
- `refactor/neb-interface-cleanup`

### Recommended commit sequence
1. `test: capture interface-cleanup regression baselines`
2. `refactor: add execution context and run layout interfaces`
3. `refactor: add normalized image evaluation result interface`
4. `refactor: route band evaluation through workspaces and unified image results`
5. `refactor: add reporter interface for owner-only outputs`
6. `refactor: normalize field ssneb workflow configuration`
7. `feat: add refactored field ssneb example script`
8. `test: verify preserved math and cleaned-up interfaces`
9. `docs: summarize interface cleanup phase and preserved equations`

## Exit criteria

Before merging this phase, confirm:
- [ ] all numerical regression tests pass,
- [ ] no intended formula changes were introduced,
- [ ] per-image workspaces are restored and managed under the run layout,
- [ ] shared outputs are owner-only,
- [ ] MPI field metadata is fully synchronized,
- [ ] zero-field workflow initialization works,
- [ ] the replacement example runs under the cleaned-up interface,
- [ ] repeated scenario branching is materially reduced in `band.py`, `optimize/base.py`, and `workflows/field_ssneb.py`.

## Phase result

At the end of this phase, the refactored NEB codebase should keep the same mathematics and physics but expose a cleaner, more maintainable interface built on a few normalized runtime objects instead of many local scenario-specific branches.
