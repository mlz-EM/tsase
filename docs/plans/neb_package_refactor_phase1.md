# Phase 1 Implementation Plan: NEB Package Restructuring (No Math Changes)

## Objective

Restructure the `tsase.neb` codebase into a layered package layout that separates:
- core SSNEB mathematics,
- optimizers,
- constraints/projection logic,
- physical model wrappers,
- IO/diagnostics,
- visualization,
- runtime helpers, and
- end-user workflows.

This phase **must not intentionally change any mathematics, physics model behavior, or optimizer behavior**. The only goal is to improve code organization, ownership boundaries, and maintainability while preserving existing functionality.

## Non-goals

This phase does **not**:
- modify SSNEB equations,
- modify tangent math,
- modify spring-force math,
- modify field-coupling formulas,
- fix or redesign filter physics,
- redesign FIRE,
- change public behavior on purpose,
- remove old public imports.

Any bug fixes or physics corrections are deferred to later phases unless they are strictly required to keep behavior unchanged after moving code.

## Success criteria

Phase 1 is complete when all of the following are true:
1. Existing external imports from `tsase.neb` still work.
2. Existing example scripts still run without required user-facing API changes.
3. Core mathematical outputs match pre-refactor behavior within numerical tolerance.
4. Field-coupled workflows produce the same energies, forces, stress, dipole, and polarization as before.
5. Restart / path / mapping behavior is unchanged.
6. The package layout clearly separates core math from workflow glue.

## Target package structure

```text
 tsase/
 └── neb/
     ├── __init__.py
     ├── core/
     │   ├── band.py
     │   ├── geometry.py
     │   ├── path.py
     │   ├── mapping.py
     │   ├── tangent.py
     │   ├── springs.py
     │   └── state.py
     ├── optimize/
     │   ├── base.py
     │   └── fire.py
     ├── constraints/
     │   ├── adapters.py
     │   └── protocols.py
     ├── models/
     │   ├── field.py
     │   └── charges.py
     ├── io/
     │   ├── artifacts.py
     │   ├── diagnostics.py
     │   ├── restart.py
     │   └── paths.py
     ├── viz/
     │   └── stem.py
     ├── runtime/
     │   └── calculators.py
     ├── workflows/
     │   └── field_ssneb.py
     └── legacy/
         └── api.py
```

## Dependency rule

Enforce this import direction during the refactor:
- `core/` may not import from `optimize/`, `constraints/`, `models/`, `io/`, `viz/`, `runtime/`, or `workflows/`.
- `optimize/` may import from `core/` and `constraints/` only.
- `models/` may import from `core/` utilities if needed, but not from `optimize/`, `io/`, `viz/`, or `workflows/`.
- `io/`, `viz/`, `runtime/`, and `workflows/` may depend on lower layers.
- `tsase.neb.__init__` remains the backward-compatible public facade.

## File migration map

### Core math
- `tsase/neb/ssneb.py`
  - keep `ssneb` class logic in `tsase/neb/core/band.py`
  - move tangent helper logic into `tsase/neb/core/tangent.py`
  - move adaptive spring logic into `tsase/neb/core/springs.py`
  - move image-state helper logic into `tsase/neb/core/state.py`
  - move Jacobian / distance helper calls to `tsase/neb/core/geometry.py`

- `tsase/neb/ssneb_utils.py`
  - `compute_jacobian`, `image_distance_vector` -> `core/geometry.py`
  - `interpolate_path`, `generate_multi_point_path` -> `core/path.py`
  - `ensure_atom_ids`, `reorder_by_atom_ids`, `spatial_map` -> `core/mapping.py`
  - `initialize_image_properties` -> `core/state.py`
  - `load_band_configuration_from_xyz` -> `io/restart.py`

### Optimizer
- `tsase/neb/minimizer_ssneb.py` -> `tsase/neb/optimize/base.py`
- `tsase/neb/fire_ssneb.py` -> `tsase/neb/optimize/fire.py`

### Constraints
- `tsase/neb/filtering.py` -> `tsase/neb/constraints/adapters.py`
- add `tsase/neb/constraints/protocols.py` for the projection interface used by optimizers / band objects

### Physical models
- `tsase/neb/field.py`
  - `EnthalpyWrapper`, `crystal_field_to_cartesian`, polarization constant -> `models/field.py`
  - `build_charge_array`, `attach_field_charges` -> `models/charges.py`

### IO / diagnostics / restart
- `tsase/neb/run_artifacts.py` -> `tsase/neb/io/artifacts.py`
- output-path helpers -> `tsase/neb/io/paths.py`
- csv/log writing code currently embedded in the minimizer -> `tsase/neb/io/diagnostics.py`
- restart loading helpers -> `tsase/neb/io/restart.py`

### Visualization
- `tsase/neb/stem_visualization.py` -> `tsase/neb/viz/stem.py`

### Runtime helpers
- `tsase/neb/runtime.py` -> `tsase/neb/runtime/calculators.py`

### Workflow glue
- create `tsase/neb/workflows/field_ssneb.py`
- move reusable orchestration from example drivers here without changing logic
- keep example scripts thin and declarative

### Compatibility layer
- `tsase/neb/__init__.py` re-exports all legacy public names
- optional temporary compatibility imports in old module paths if needed during transition

## Implementation steps

### Step 1: Create new package directories and facades
- Add the new subpackages and `__init__.py` files.
- Do not move logic yet.
- Add import comments documenting allowed dependency directions.

### Step 2: Move pure helper functions first
- Move geometry/path/mapping/state helper functions out of `ssneb_utils.py` into their new modules.
- Re-export old names from `ssneb_utils.py` temporarily.
- Update internal imports only after helper tests pass.

### Step 3: Split `ssneb.py`
- Move non-class helper logic (`_geometric_tangent`, spring logic helpers, state helpers) out into `core/*` modules.
- Keep the public `ssneb` class behavior unchanged.
- After extraction, `core/band.py` should orchestrate math, not own every helper.

### Step 4: Move optimizer code
- Move `minimizer_ssneb` and `fire_ssneb` into `optimize/`.
- Keep names and constructor signatures unchanged.
- Preserve old import paths through re-export shims.

### Step 5: Move constraint adapters
- Move filter adapter code into `constraints/adapters.py`.
- Add a small protocol / interface module to define expected adapter methods:
  - `project_atomic_forces`
  - `project_cell_virial`
  - `project_cell_step`
  - `apply_step`
- No math changes.

### Step 6: Move field model code
- Move field-coupling code into `models/field.py` and `models/charges.py`.
- Keep formulas and names unchanged.
- Preserve old imports from `tsase.neb`.

### Step 7: Move IO / artifact code
- Move run-artifact and output-path helpers into `io/`.
- Move restart helpers into `io/restart.py`.
- Extract diagnostics-writing logic from the minimizer into `io/diagnostics.py` without changing output format.

### Step 8: Move visualization and runtime helpers
- Move STEM visualization to `viz/stem.py`.
- Move MACE loading helper to `runtime/calculators.py`.
- Keep all runtime behavior unchanged.

### Step 9: Add workflow helper module
- Create `workflows/field_ssneb.py`.
- Move reusable orchestration out of example scripts into workflow-level helper functions.
- Keep current example behavior unchanged.

### Step 10: Final facade cleanup
- Update `tsase/neb/__init__.py` to be the stable public surface.
- Verify external imports still work.
- Leave temporary legacy shims if needed.

## Tests required for this phase

All tests below are regression tests. Their purpose is to verify that Phase 1 preserved behavior.

### A. Core math regression tests
1. **Jacobian regression**
   - Check `compute_jacobian` returns the same value as before for representative `(vol1, vol2, natom, weight)` inputs.

2. **Image distance regression**
   - For a fixed pair of images, verify `image_distance_vector` matches pre-refactor output exactly or within floating-point tolerance.

3. **Interpolation regression**
   - For a two-endpoint path, verify all interpolated cells and positions match pre-refactor results.
   - For a multi-endpoint path, verify image count, endpoint placement, and intermediate images match previous behavior.

4. **Atom mapping regression**
   - Verify `ensure_atom_ids`, `reorder_by_atom_ids`, and `spatial_map` preserve prior behavior for permuted-but-equivalent structures.

5. **Tangent regression**
   - For a representative non-degenerate path, confirm tangent vectors match pre-refactor results.
   - For the zero-tangent fallback case, confirm the fallback still gives the same post-branch behavior as the current branch.

6. **Spring regression**
   - With adaptive springs off, verify all spring constants match prior behavior.
   - With adaptive springs on, verify the full spring-constant array matches prior behavior.

### B. Band / force regression tests
7. **`ssneb.forces()` regression without field coupling**
   - On a small deterministic path with a mock calculator, compare:
     - `u`
     - `f`
     - `st`
     - `totalf`
     - climbing image index
   - Results should match pre-refactor outputs.

8. **Climbing-image selection regression**
   - Verify `CI_index` / climbing-image behavior is unchanged for normal cases and frozen-image cases.

9. **Image update schedule regression**
   - Verify scheduled image-rate scaling and frozen-image tracking remain unchanged.

### C. Optimizer regression tests
10. **FIRE single-step regression**
    - For a deterministic mock band, verify one optimizer step produces the same:
      - velocity update,
      - `dt`, `a`, `Nsteps`,
      - applied atomic positions,
      - applied cell update.

11. **FIRE multi-step regression**
    - Run a short fixed-number iteration test and compare the full sequence of energies / max forces / positions to the pre-refactor baseline.

### D. Constraint regression tests
12. **No-filter behavior regression**
    - Confirm that using no filter reproduces the old path update behavior.

13. **Subset filter regression**
    - Confirm allowed atoms move and disallowed atoms remain unchanged exactly as before.

14. **Cell-mask regression**
    - Confirm projected cell virial and projected cell step match pre-refactor behavior for representative masks.

### E. Field-model regression tests
15. **Charge helper regression**
    - Verify `build_charge_array` and `attach_field_charges` return exactly the same charge arrays as before.

16. **EnthalpyWrapper regression**
    - For a controlled structure and field, compare:
      - energy,
      - free energy,
      - forces,
      - stress,
      - dipole,
      - polarization,
      - polarization in C/m^2,
      - field energy.

17. **Crystal-field conversion regression**
    - Verify `crystal_field_to_cartesian` returns the same vector as before for representative cells / inputs.

### F. IO / workflow regression tests
18. **Restart regression**
    - Load a saved XYZ restart and verify the resulting band geometry matches pre-refactor behavior.

19. **Manifest / artifact regression**
    - Verify artifact paths, manifest keys, and copied input metadata are unchanged.

20. **Diagnostics/log format regression**
    - Confirm diagnostics CSV header and row ordering are unchanged.
    - Confirm `fe.out` formatting remains unchanged.

21. **Example smoke test**
    - Run the current field SSNEB example and confirm it completes with the same observable outputs as before.

## Test strategy

### Baseline capture
Before moving code, record golden-reference outputs from the current branch for:
- helper function outputs,
- one deterministic band-force evaluation,
- one deterministic FIRE step,
- one short example run,
- diagnostics CSV header / row schema,
- manifest schema.

### Test tolerance
- Use exact equality for schema, headers, keys, shapes, atom IDs, and discrete selections.
- Use tight floating-point tolerances for energies, forces, stresses, dipoles, polarization, tangents, and path coordinates.
- Any mismatch must be treated as a regression unless explicitly justified.

## Git workflow for this phase

### Branches
- Main working branch for the refactor:
  - `refactor/neb-package-structure-phase1`
- Optional sub-branches if the work becomes too large:
  - `refactor/neb-core-layout`
  - `refactor/neb-optimizer-layout`
  - `refactor/neb-io-viz-layout`

### Commit plan
Make small, reviewable commits with one concern per commit.

Recommended commit sequence:
1. `test: capture regression baselines for phase-1 refactor`
2. `refactor: add layered neb package skeleton`
3. `refactor: move geometry path and mapping helpers into core modules`
4. `refactor: move ssneb state and tangent helpers into core modules`
5. `refactor: move ssneb class into core band module with compatibility imports`
6. `refactor: move minimizer and FIRE into optimize package`
7. `refactor: move filter adapters into constraints package`
8. `refactor: move field model and charge helpers into models package`
9. `refactor: move artifacts restart and diagnostics into io package`
10. `refactor: move visualization and runtime helpers into dedicated packages`
11. `refactor: add workflow helper module and thin example entrypoints`
12. `refactor: finalize neb public facade and legacy compatibility layer`
13. `test: verify regression suite passes after phase-1 restructure`
14. `docs: summarize phase-1 package restructure and preserved behavior`

### Pull request scope
The PR description must explicitly state:
- no intended math changes,
- no intended physics changes,
- no intended optimizer changes,
- all changes are structural / packaging,
- regression tests were used to verify preserved behavior.

## Exit checklist

Before merging, verify:
- [ ] all regression tests pass,
- [ ] examples still run,
- [ ] `tsase.neb` legacy imports still work,
- [ ] no new circular imports were introduced,
- [ ] `core/` does not import workflow / IO / viz code,
- [ ] outputs match baseline within tolerance,
- [ ] PR diff is dominated by file moves and import rewiring rather than formula edits.

## Deliverable of Phase 1

A reorganized `tsase.neb` package with a clear layered structure and preserved behavior, ready for later phases that will address the real higher-risk issues:
- constraint / filter correctness,
- field-reference handling,
- stronger physical model separation,
- workflow cleanup.
