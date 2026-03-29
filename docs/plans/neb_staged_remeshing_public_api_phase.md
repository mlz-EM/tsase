# Staged remeshing phase plan for TSASE SSNEB

## Objective

This phase adds staged remeshing to the refactored TSASE SSNEB workflow while keeping the core SSNEB force model simple.

The main goal is not just to support remeshing internally. The main goal is to expose a clean, intentional **public API** for:
- direct user-side uniform remeshing before a run,
- staged internal remeshing during a run,
- CI gating after remeshing is complete,
- simple per-image mobility-rate control,
- clean staged I/O.

This phase also removes adaptive spring constants entirely.

---

## Hard decisions already made

1. **Adaptive spring constants are removed completely.**
   - Keep one constant spring constant `k` in the core band force.
   - Remove all adaptive-`k` parameters, helper functions, file/module support, docs, tests, and example usage.
   - Do not leave legacy compatibility shims for adaptive springs.

2. **Remeshing is staged execution, not in-place band mutation.**
   - Internally, staged remeshing is treated as multiple SSNEB stages executed in sequence.
   - When a remesh trigger fires, the current stage ends, the band is remeshed, and a new stage starts.

3. **Restart after remesh should be maximally conservative.**
   - Remeshing should reset the optimizer exactly as if a new calculation were being started.
   - Do not try to preserve FIRE velocities or carry over stage-local optimizer momentum.

4. **If a remesh plan is defined, CI and final `fmax < threshold` termination are both disabled until the last remesh stage has completed.**

5. **Image mobility uses only a scalar rate.**
   - `rate = 0.0` means frozen.
   - `0 < rate < 1` means slowed.
   - `rate = 1.0` means normal.
   - No separate public freeze declaration should exist.

6. **The first remeshing mode is uniform remeshing in the SSNEB path metric.**
   - This same remesher should be callable directly from user scripts.

---

## Public API target at this stage

This is the most important section of this plan.

At the end of this phase, the public API should support the following workflow without users needing to assemble low-level internals by hand.

### A. Keep the low-level public API

These should remain usable and straightforward:
- `neb.ssneb`
- `neb.fire_ssneb`
- `neb.EnthalpyWrapper`
- `neb.attach_field_charges`
- `neb.build_charge_array`
- current path helpers that still make sense to keep public

These remain the low-level building blocks.

### B. Add a direct public remesh utility

Add a public function:

```python
neb.uniform_remesh(images, num_images=None, upsample_ratio=None)
```

#### Required semantics
- `num_images` is the primary control.
- `upsample_ratio` is convenience sugar.
- if both are passed, raise an error.
- if neither is passed, raise an error.
- preserve endpoints exactly.
- preserve atom IDs and per-atom field-charge arrays.
- define uniformity in the SSNEB path metric, not plain Cartesian distance.

This function must be usable directly in user scripts to initialize a finer band from a coarse path.

### C. Add a public staged workflow entry point

Add a public high-level workflow entry point. Either of the following shapes is acceptable:

```python
neb.run_staged_ssneb(...)
```

or

```python
neb.StagedSSNEBRunner(...).run()
```

The implementation should choose one and keep it minimal. Do not build a large generic framework unless it is necessary.

### D. Add a minimal public stage specification object

The public staged workflow should accept a remesh plan built from one or more stage specifications.

Recommended public shape:

```python
neb.RemeshStage(
    target_num_images=27,
    trigger=...,          # see below
    max_wait_iterations=..., 
    on_miss="force",     # "force", "skip", or "error"
)
```

A simple `RemeshPlan([...])` wrapper is acceptable, but it is not required if a plain ordered list is sufficient.

### E. Keep mobility public and simple

The public workflow should accept image mobility only as rates.

Recommended public shape:

```python
image_mobility_rates={
    10: 0.5,
    11: 0.0,
}
```

Do not require a separate public freeze object.

### F. Require a working staged example under the current interface

This phase must include a **working example script** for the field-coupled interpolated workflow we discussed.

Required target:
- `examples/run_field_ssneb_interpolated.py`

Required behavior for that example at this stage:
- use the public interface available after this phase
- demonstrate **one-stage upsampling**
- initial image count: `15`
- remesh target: `27`
- CI only active in the final stage
- final `fmax < threshold` only active in the final stage
- produce clean staged output directories / manifests

This example is not optional. It is part of the deliverable because it validates whether the public API is actually usable.

---

## What must change to achieve that public API target

## 1. Core band must become strictly constant-`k`

### Required changes
In `tsase/neb/core/band.py`:
- remove `adaptive_springs`
- remove `kmin`
- remove `kmax`
- remove `adaptive_eps`
- remove any `spring_constants` array machinery
- remove adaptive spring bookkeeping on images
- compute spring force directly with one scalar `self.k`

### Required deletions
Delete any adaptive-spring-specific helper module, especially:
- `tsase/neb/core/springs.py`

Also remove all adaptive-spring references from:
- examples
- tests
- workflow configs
- manifests
- docs
- compatibility facades

### Design rule
The core band must not know:
- whether remeshing is planned
- whether the current stage is coarse or final
- whether CI is temporarily gated by workflow logic
- whether the workflow should terminate or remesh next

It only computes the current band force.

---

## 2. Optimizer must stop owning CI activation and final stopping

### Required changes
The optimizer should own only:
- FIRE state
- step generation
- step application
- mobility-rate application
- reset behavior between stages

Move out of the optimizer:
- CI activation by force threshold
- CI activation by iteration threshold
- final workflow stopping ownership

Those become workflow-runner decisions.

### Mobility handling
Apply image mobility in the optimizer, not in the band force engine.

That means the band still returns the raw NEB force and the optimizer uses the configured rate when converting force into motion.

---

## 3. Add a dedicated public remesh utility

### Required functionality
Implement `neb.uniform_remesh(...)` as a real public utility, not just an internal helper.

### Required algorithm
Define remeshing as redistribution onto a new uniform path grid:
1. compute cumulative arc length of the current path using the SSNEB path metric
2. choose new uniformly spaced target arc-length values
3. interpolate images piecewise between bracketing old images

### Important rule
Do **not** define remeshing as “insert images between each pair.”

That insertion model is fine as an internal special case intuition, but the actual implementation should be one unified redistribution algorithm so non-integer ratio-derived targets work naturally.

### Ratio conversion
If `upsample_ratio` is used, convert it internally to a new target image count using:

```python
new_num_images = int(round((old_num_images - 1) * upsample_ratio)) + 1
```

That should be the only ratio logic needed.

---

## 4. Add a staged workflow runner

### Required responsibility
Add one public staged workflow layer that owns:
- stage creation
- stage execution
- stage exit reasons
- remesh transitions
- CI gating
- final convergence gating
- staged output layout

### Conceptual behavior
A run with remeshing should behave as:
1. build stage 0 band
2. run stage 0
3. if the stage remesh trigger fires, terminate the stage
4. remesh the current band
5. create a fresh stage 1 band and a fresh optimizer
6. continue until the last remesh stage has completed
7. only then allow CI and final force convergence

### Key point
This is effectively multiple SSNEB calls internally, by design.

That is acceptable and preferred because it matches the conservative restart semantics we want.

---

## 5. Introduce explicit stage termination semantics

At this phase, the workflow can no longer think in terms of one global termination condition only.

### Required distinction
A stage may end because:
- a remesh trigger fired
- final convergence fired
- max iterations were reached
- an error occurred

### Recommended stage exit reasons
- `remesh_triggered`
- `final_converged`
- `max_iterations_reached`
- `error`

### Recommended workflow outcomes
- `completed`
- `incomplete_max_iterations`
- `failed`

The plan, the runner, and the manifests should all reflect this explicitly.

---

## 6. CI and final convergence must be gated by remesh-plan completion

### Required behavior
If a remesh plan is defined:
- CI must be ineligible until the last remesh stage has been applied.
- final `fmax < threshold` termination must be ineligible until the last remesh stage has been applied.

### Final-stage semantics
After the last remesh stage is complete:
- CI becomes eligible if requested by the run config
- final force convergence becomes eligible

### Why this is necessary
A coarse band is intentionally temporary. It should not:
- activate CI too early, or
- terminate as “converged” before the final discretization is in place.

---

## 7. Remesh triggers should be based primarily on `fPerp_max`

### Guiding idea
For staged refinement, the relevant question is not “is the run converged?”
It is “is the current coarse path stable enough to justify refinement?”

### Best current indicator
Use `fPerp_max` as the primary signal.

That is better than total force because it tracks path-shape relaxation more directly.

### But do not use `fPerp_max` alone
The first-stage trigger should combine:
- a minimum number of stage iterations
- a sufficient drop in `fPerp_max`
- a plateau or small recent trend in `fPerp_max`

### Recommended first trigger shape
Conceptually:

```python
StabilizedPerpForce(
    min_iterations=20,
    relative_drop=0.3,
    window=5,
    plateau_tolerance=...,
)
```

The exact implementation can be simple, but this should be the target behavior.

### Fallback policy
Each remesh stage should have an explicit fallback if the trigger never fires:
- `force`
- `skip`
- `error`

Do not leave this implicit.

---

## 8. Staged I/O must be clean and explicit

### Required top-level structure
One workflow run should produce one top-level run directory.
Inside that directory, each internal stage should have its own subdirectory.

### Recommended layout

```text
<run_dir>/
  workflow_manifest.json
  workflow_summary.json
  initial_inputs/
  stage_00/
    run_manifest.json
    xyz/
    diagnostics.csv
    fe.out
    prepared_structures/
    images/
    stage_exit.json
  stage_01/
    run_manifest.json
    xyz/
    diagnostics.csv
    fe.out
    prepared_structures/
    images/
    stage_exit.json
  transitions/
    remesh_00_to_01.xyz
    remesh_00_to_01.json
```

### Required transition artifacts
For each remesh transition, record:
- source stage id
- target stage id
- source image count
- target image count
- trigger metrics at transition
- remeshed path snapshot

This is needed for debugging, reproducibility, and restart clarity.

---

## 9. Minimal public objects recommended for this phase

To achieve the public API target without overbuilding, the recommended new public surface is:

- `neb.uniform_remesh`
- `neb.RemeshStage`
- either `neb.run_staged_ssneb` or `neb.StagedSSNEBRunner`

Optionally:
- `neb.StabilizedPerpForce` as a named public trigger helper

That is enough for this phase.

Do not introduce a large policy/plugin framework unless it is clearly needed to support the example and tests.

---

## 10. Required working example

This phase must ship a working example:

- `examples/run_field_ssneb_interpolated.py`

### Example requirements
It must:
- use the public API available after this phase
- demonstrate a staged field-coupled run
- start with `15` images
- use one remesh stage to `27` images
- keep CI disabled in the first stage
- allow CI only in the final stage
- use final `fmax < threshold` only in the final stage
- produce clean staged output directories and manifests

### Why this is mandatory
The example is the best test of whether the public API is actually usable.
If the example still has to reach deep into internal objects, then the public API target was not achieved.

---

## Implementation sequence

### Step 1
Remove adaptive spring support completely from code, tests, examples, and docs.

### Step 2
Implement `neb.uniform_remesh(...)` and its tests.

### Step 3
Move mobility-rate application fully into the optimizer layer.

### Step 4
Implement staged workflow runner with stage exit reasons and conservative stage restarts.

### Step 5
Move CI activation and final convergence gating into the staged runner.

### Step 6
Implement clean staged I/O layout and workflow/stage manifests.

### Step 7
Provide the working `examples/run_field_ssneb_interpolated.py` staged example.

### Step 8
Add tests covering:
- remesh utility behavior
- staged workflow transitions
- CI gating
- final convergence gating
- staged output layout
- example smoke run

---

## Required tests for this phase

### Core regression-preservation tests
1. constant-`k` spring behavior unchanged relative to previous constant-`k` runs
2. tangent behavior unchanged
3. CI force reversal unchanged once activated
4. FIRE behavior unchanged when no staged workflow actions are applied

### New remesh tests
5. uniform remesh preserves endpoints
6. uniform remesh preserves atom IDs and field-charge arrays
7. uniform remesh works for target image count changes derived from integer and non-integer ratios
8. remeshed bands are uniformly spaced in the SSNEB metric within tolerance

### Workflow tests
9. one-stage no-remesh workflow matches a normal run
10. one-remesh workflow exits stage 0 with `remesh_triggered`
11. CI never activates before remesh-plan completion
12. final `fmax < threshold` cannot terminate before remesh-plan completion
13. stage restart after remesh resets optimizer state
14. staged output directory layout is correct
15. transition artifacts are written correctly
16. `examples/run_field_ssneb_interpolated.py` runs as a smoke test

---

## Exit criteria

This phase is complete when all of the following are true:
- [ ] adaptive spring support is fully removed, with no legacy hooks left behind
- [ ] core SSNEB force code uses only constant `k`
- [ ] `neb.uniform_remesh(...)` exists and is publicly usable
- [ ] staged internal remeshing is implemented through a public staged workflow entry point
- [ ] image mobility is represented publicly only through rates
- [ ] CI is gated until remesh-plan completion
- [ ] final `fmax < threshold` convergence is gated until remesh-plan completion
- [ ] staged I/O is clean and reproducible
- [ ] `examples/run_field_ssneb_interpolated.py` is a working staged 15→27 example under the public interface
- [ ] the test suite covers the new public API and the staged workflow behavior

---

## Bottom line

The key standard for this phase is not just whether staged remeshing works internally.

The key standard is whether the repository exposes a clean and believable **public API target at this stage**:
- a public uniform remesher,
- a public staged SSNEB entry point,
- simple public mobility-rate control,
- clean CI/final-convergence gating semantics,
- and a working staged field-SSNEB example that proves the interface is real.
