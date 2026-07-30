# Redundancy Removal and YAML-Driven Workflow Phase Plan

## Objective

This phase aggressively removes redundant interfaces, stale legacy layers, duplicated output paths, and duplicated configuration paths from the refactored `tsase.neb` codebase while preserving the underlying:
- SSNEB mathematics,
- tangent and spring-force formulas,
- FIRE update equations,
- field-coupled enthalpy model,
- restart remapping behavior where still intentionally supported, and
- physically meaningful strain/cell restriction semantics.

The goal is to leave one clear runtime path:

1. one YAML input file,
2. one normalized resolved config object,
3. one path-construction entry path,
4. one output/reporting stack,
5. one cell-restriction mechanism,
6. one maintained example entrypoint.

## Hard constraint

This phase may be aggressive about deleting redundancy, but it must **not intentionally change the math or physics** of:
- SSNEB force construction,
- image tangent construction,
- spring-force formulas,
- climbing-image inversion,
- Jacobian scaling,
- FIRE integration behavior,
- field-coupling formulas,
- dipole / polarization definitions.

The code structure may change substantially. Public interfaces may also change substantially. However, the preserved runtime path must compute the same intended physical quantities.

## Main decisions for this phase

### 1. Redundancy is removed aggressively

This phase will not preserve redundant legacy code just for convenience. If two code paths currently serve the same purpose, the unused or inferior one should be removed.

### 2. YAML becomes the primary user interface

Runtime parameters must be supplied through a YAML file. This includes:
- path definition,
- restart source,
- field definition,
- charge definition,
- optimizer settings,
- remeshing settings,
- output settings,
- strain / cell restriction settings,
- example workflow settings.

Programmatic helpers may still exist internally, but the maintained end-user workflow must be YAML-driven.

### 3. Cell restriction is normalized through one mechanism

The current code has overlapping cell-restriction mechanisms:
- `fixstrain` in the band,
- filter-mask projection in the filter adapter.

This phase will standardize on one mechanism only for the maintained workflow path. The maintained path will use **filter-based cell restriction** expressed in YAML. The old `fixstrain` path will be retired from the maintained workflow and removed if no required user-facing compatibility obligation remains.

### 4. Restart from XYZ becomes self-contained

Restarting from a saved XYZ path must not require users to also supply initial endpoint structures that are then overwritten. Restart becomes a first-class path source.

### 5. Outputs become explicit, structured, and hook-based

User-facing outputs must be emitted through one output manager using explicit output configuration. STEM visualization becomes an output hook, not a hidden side effect of XYZ writing.

## Redundancies to remove in this phase

### A. Iteration output duplication

Remove the initialization-time path snapshot and diagnostics write from band construction.

Only one iteration-output path should remain, owned by the optimizer/reporting stack.

Result:
- no duplicate `iter_0000` and `iter_0001` caused by separate initialization and iteration writers,
- iteration numbering corresponds to actual optimizer iterations,
- optional initial-state snapshots, if kept at all, must be explicit named events rather than fake iteration files.

### B. Constant-spring cached fields

Remove cached per-image spring bookkeeping that no longer carries independent meaning under constant-spring behavior.

Specifically remove:
- `k_left`
- `k_right`

If spring constants are globally constant, they should not be stored redundantly per image just for diagnostics.

Diagnostics must be updated accordingly.

### C. Dead or stale image-update metadata

Remove stale image state fields that are no longer maintained as live state.

Specifically review and remove:
- `update_rate` as a per-image diagnostic field if it is no longer written onto images,
- dead helper hooks such as `update_image_rates` if they no longer exist on the maintained runtime objects,
- dead restart-time compatibility hooks that probe for methods that the maintained band no longer defines.

If mobility scaling is still needed, keep it in one authoritative runtime location only.

### D. Duplicate runtime ownership fields

Remove duplicated runtime state where one object already owns the information.

Examples include:
- `context` versus duplicated `parallel`, `rank`, `size`,
- `layout` versus duplicated flattened path strings like `output_dir`, `xyz_dir`, `diagnostics_file`, `log_file`,
- `public_dir` if it remains identical to `run_dir` and provides no independent meaning.

There should be one authoritative execution object and one authoritative run/output layout object.

### E. Overlapping IO/output layers

Collapse the current overlapping responsibilities across:
- `RunLayout`,
- `RunArtifacts`,
- `Reporter`.

Replace them with one authoritative output-management layer plus optional emitters/hooks.

This layer must own:
- run directory structure,
- manifests,
- copied YAML input,
- diagnostics,
- logs,
- path snapshots,
- energy-profile exports,
- STEM outputs,
- transitions/remesh outputs.

### F. `prepared_structures` outputs

Remove the `prepared_structures` output category and related functions.

It is redundant with:
- copied inputs,
- stage inputs,
- path snapshots,
- remesh transition outputs.

This phase removes:
- dedicated `prepared_structures` directories,
- manifest keys for `prepared_structures`,
- helper defaults centered on `prepared_structures`.

### G. Legacy import shims and duplicate public facades

Remove unused legacy import layers and duplicated compatibility facades as aggressively as possible.

Candidates include:
- `tsase/neb/ssneb.py`
- `tsase/neb/fire_ssneb.py`
- `tsase/neb/minimizer_ssneb.py`
- `tsase/neb/ssneb_utils.py`
- `tsase/neb/run_artifacts.py`
- `tsase/neb/stem_visualization.py`
- duplicated facade routing through `tsase.neb.__init__` and `legacy/api.py`

The maintained codebase should expose one clear public surface only.

### H. Duplicate workflow entry styles

Remove the doubled public configuration path where users can both:
- manually assemble workflow kwargs, and
- separately build a config object that is then exploded back into kwargs.

The maintained path should be:
- YAML -> resolved config object -> workflow execution.

### I. Duplicate example entrypoints and naming conventions

Remove duplicate or stale example entrypoints under mixed conventions such as `example/` versus `examples/` when they represent the same maintained workflow.

This phase must leave one maintained example path and one maintained naming convention.

### J. Obsolete alternate SSNEB implementations

Aggressively evaluate whether old alternate implementations such as `pssneb.py` are still required.

If the maintained MPI-aware or serial path already covers the intended functionality, the duplicated legacy implementation should be removed.

This is especially important when an old file duplicates:
- tangent construction,
- spring logic,
- force assembly,
- parallel image evaluation,
- working-directory logic.

## Standardized YAML interface

The maintained user interface for this phase is a YAML file.

### Proposed top-level structure

```yaml
run:
  name: pzo_field_ssneb
  root: runs/pzo_field_ssneb

path:
  source:
    kind: control_points      # control_points | full_path_xyz | restart_xyz
    files:
      - example/Pbam_SC_NC.cif
      - example/Cluster#2_SC.cif
    indices: [0, 26]
  num_images: 27
  remap_on_restart: atom_ids  # atom_ids | spatial | none

model:
  calculator:
    kind: mace
    model_path: example/MACE_model.model
    device: cuda
  charges:
    kind: species_map
    values:
      Pb: 2.0
      Zr: 4.0
      O: -2.0
  field:
    kind: crystal             # crystal | cartesian | none
    value: [-0.001, 0.0, 0.0]
  reference:
    kind: file
    file: example/Pm-3m.cif

band:
  spring: 5.0
  method: ci
  ss: true
  weight: 1.0

constraints:
  filter:
    kind: ExpCellFilter       # none | ExpCellFilter | UnitCellFilter | StrainFilter
    mask: [0, 1, 0, 1, 0, 1] # ASE Voigt order [xx, yy, zz, yz, xz, xy]
    hydrostatic_strain: false
    constant_volume: false
    move_atoms: true

optimizer:
  kind: fire
  dt: 0.10
  dtmax: 0.10
  maxmove: 0.10
  output_interval: 200
  convergence:
    fmax: 0.01
    max_steps: 5000
  ci_activation:
    force: 0.5

staging:
  remesh:
    - target_num_images: 27
      trigger:
        kind: stabilized_perp_force
        min_iterations: 20
        relative_drop: 0.3
        window: 5
        plateau_tolerance: 0.05
      max_wait_iterations: 200
      on_miss: force

outputs:
  path_snapshot:
    enabled: true
    format: extxyz
    schedule:
      every: 200
      include_initial: false
      include_final: true
  diagnostics:
    enabled: true
    format: csv
  energy_profile:
    enabled: true
    formats: [csv, png]
    fields: [enthalpy, energy_base, energy_field, polarization_magnitude]
  stem:
    enabled: true
    schedule:
      every: 200
      include_final: true
    emit:
      analysis_npy: true
      frame_png: true
      sequence_gif: true
      diagnostics_txt: true
```

## Path input redesign

Introduce one normalized path input model.

### Supported path source kinds

1. `control_points`
   - structures are given by files plus path indices
   - interpolation or multi-point construction is performed

2. `full_path_xyz`
   - the full path is read from a saved XYZ
   - no endpoint reconstruction is required

3. `restart_xyz`
   - restart is treated as a first-class path source
   - the workflow constructs the band directly from the restart path
   - optional remapping/validation is applied as configured

### Design rule

The band constructor should receive already-normalized images and settings.

It should no longer receive legacy inputs that later get overwritten by restart data.

## Normalized cell restriction / clamping model

### Current issue

Cell restriction is currently split between:
- `fixstrain` in the band force path,
- filter adapter projection for cell virial and cell step.

This duplicates responsibility for restricting strain components.

### New rule

The maintained workflow path will express cell restriction only through the YAML `constraints.filter` block and the filter adapter path.

### Supported semantics

The YAML must support:
- no cell restriction,
- masked strain components,
- hydrostatic strain,
- constant-volume projection,
- atom motion enabled/disabled where relevant.

### Required implementation outcome

- band-level `fixstrain` is removed from the maintained workflow path,
- user-facing strain control is defined only in YAML,
- the mapping from YAML to the chosen filter adapter is explicit and documented,
- the underlying projected update behavior remains physically equivalent to the intended current filter-based restriction semantics.

## Output system redesign

### One authoritative output manager

Replace the split output stack with one output manager that owns:
- `config/`
- `outputs/`
- `stages/`
- `transitions/`
- `work/`

### Proposed run layout

```text
<run_root>/
  config/
    input.yaml
    resolved.yaml
    manifest.json
  outputs/
    diagnostics.csv
    fe.out
    path/
      iter_0200.xyz
      final.xyz
    energy/
      profile_iter_0200.csv
      profile_iter_0200.png
    stem/
      iter_0200/
        analysis.npy
        diagnostics.txt
        frames/
          frame_0000.png
          ...
        sequence.gif
  stages/
    stage_00/
      stage_manifest.json
      exit.json
    stage_01/
      stage_manifest.json
      exit.json
  transitions/
    remesh_00_to_01.json
    remesh_00_to_01.xyz
  work/
    image_0000/
    image_0001/
    ...
```

### Output rules

- public user-facing artifacts live only under `outputs/`, `stages/`, `transitions/`, and `config/`
- scratch/workspace artifacts live only under `work/`
- STEM rendering is an explicit configured emitter
- structured `.npy` STEM analysis output must be supported for later quantitative AutoResearch analysis

## Example requirement

Provide a replacement maintained example:

- `examples/run_field_ssneb_interpolated.py`

### Requirement for this example

The maintained example must work with the updated codebase and demonstrate the YAML-driven workflow.

### Required behavior

The example should:
- consume a YAML config file,
- run through the maintained workflow path,
- show how path source, field, charges, optimizer, remeshing, strain restrictions, and outputs are configured in YAML,
- not rely on deprecated legacy imports,
- not require users to manually unpack config objects into long function argument lists.

### Companion example input

Also provide a companion YAML file, for example:
- `examples/configs/run_field_ssneb_interpolated.yaml`

The example script should be thin and primarily responsible for:
- reading YAML,
- resolving the config,
- invoking the maintained workflow,
- printing final artifact locations.

## Implementation sequence

### Step 0 — Capture regression baselines

Before deleting redundant code, record golden baselines for:
- band force outputs,
- tangent outputs,
- spring-force outputs,
- FIRE one-step behavior,
- field-wrapper outputs,
- restart remap behavior,
- current diagnostics schemas needed for preserved quantities.

### Step 1 — Introduce resolved YAML config

Add a normalized YAML loader and resolved config model that covers:
- path,
- model,
- field,
- charges,
- constraints/filter settings,
- optimizer,
- remeshing,
- outputs.

At this step, it may wrap existing internals temporarily.

### Step 2 — Normalize path input sources

Implement one path-input normalization path for:
- control points,
- full-path XYZ,
- restart XYZ.

Then remove the old restart model that requires initial structures and later overwrites them.

### Step 3 — Remove duplicated iteration output path

Delete initialization-time XYZ/diagnostics output from band construction.

Make the optimizer/output manager the only owner of iteration artifact emission.

### Step 4 — Collapse output management

Unify layout, artifacts, and reporting into one output-management layer.

Remove:
- `prepared_structures` outputs,
- duplicated path bookkeeping,
- optimizer-local path overrides that compete with the band layout.

### Step 5 — Remove stale state and duplicated fields

Remove:
- `k_left`
- `k_right`
- dead `update_rate` image field usage if not maintained
- duplicated runtime ownership/path fields
- dead restart compatibility probes

Update diagnostics accordingly.

### Step 6 — Standardize cell restriction

Route maintained strain restriction entirely through the YAML `constraints.filter` path and filter adapters.

Remove or retire band-level `fixstrain` from the maintained workflow path.

### Step 7 — Remove legacy facades and unused alternative implementations

Aggressively delete unused legacy shims and duplicated implementations, including any obsolete parallel SSNEB implementation that is not part of the maintained workflow.

### Step 8 — Add the replacement YAML-driven example

Provide the maintained `examples/run_field_ssneb_interpolated.py` plus a YAML config file that works with the updated codebase.

### Step 9 — Final cleanup pass

Remove any leftover imports, unused functions, stale helper modules, and duplicated naming conventions discovered during integration.

## Tests required for this phase

### Physics-preservation regression tests

1. Jacobian computation unchanged
2. image-distance vector unchanged
3. path interpolation unchanged
4. tangent outputs unchanged
5. spring-force outputs unchanged
6. `ssneb.forces()` unchanged for deterministic mock cases
7. FIRE one-step and short-run behavior unchanged
8. `EnthalpyWrapper` energy/force/stress/dipole/polarization outputs unchanged
9. restart remapping behavior preserved for supported restart modes

### Redundancy-removal integration tests

10. exactly one iteration snapshot stream is written
11. no `prepared_structures` outputs are created
12. YAML-only workflow runs without manual kwargs duplication
13. restart-from-XYZ runs without requiring endpoint overwrite inputs
14. strain restriction from YAML correctly maps to filter adapter behavior
15. STEM outputs are emitted only when requested
16. STEM `.npy` analysis output is produced when enabled
17. maintained example script runs successfully with the new YAML file
18. no maintained internal module imports through removed legacy shims

## Recommended git plan

### Working branch

Create a dedicated branch for this phase, for example:
- `refactor/neb-redundancy-yaml-cleanup`

### Recommended commit sequence

1. `test: capture baselines before redundancy cleanup phase`
2. `feat: add yaml-driven resolved config for neb workflows`
3. `refactor: normalize path sources including standalone restart xyz`
4. `refactor: unify output manager and remove duplicate iteration writes`
5. `refactor: remove prepared structure outputs and stale state fields`
6. `refactor: standardize strain restriction through yaml filter settings`
7. `refactor: delete unused legacy facades and duplicate implementations`
8. `feat: add maintained yaml-driven run_field_ssneb_interpolated example`
9. `test: verify preserved math and reduced redundant surface`
10. `docs: summarize redundancy cleanup and yaml workflow phase`

## Exit criteria

This phase is complete when all of the following are true:

- [ ] the maintained workflow is YAML-driven
- [ ] restart-from-XYZ is self-contained
- [ ] initialization-time duplicate iteration output is removed
- [ ] `prepared_structures` outputs are removed
- [ ] `k_left` / `k_right` are removed
- [ ] dead `update_rate` image-state usage is removed or intentionally restored in one authoritative place
- [ ] maintained strain restriction is configured only through YAML and the filter adapter path
- [ ] duplicated output-management layers are collapsed
- [ ] unused legacy shims and duplicate public facades are removed aggressively
- [ ] any obsolete duplicate SSNEB implementation that is no longer needed is removed
- [ ] the maintained `examples/run_field_ssneb_interpolated.py` works with the updated codebase
- [ ] regression tests confirm preserved math/physics behavior

## Phase result

At the end of this phase, the codebase should stop looking like a partially refactored system with multiple surviving interface generations.

Instead, it should present one coherent maintained runtime:
- YAML in,
- one resolved config,
- one path source model,
- one restriction/clamping model,
- one output manager,
- one maintained example,
- no redundant legacy surface that the core code still depends on.
