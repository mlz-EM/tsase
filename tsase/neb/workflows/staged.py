"""Public staged SSNEB workflow helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ase.io import write

from tsase.neb.core.remesh import uniform_remesh
from tsase.neb.io.manager import OutputManager
from tsase.neb.optimize.fire import fire_ssneb


@dataclass(frozen=True)
class RemeshStage:
    """One requested remeshing transition in a staged SSNEB run."""

    target_num_images: int
    trigger: object
    max_wait_iterations: Optional[int] = None
    on_miss: str = "force"

    def __post_init__(self):
        if int(self.target_num_images) < 2:
            raise ValueError("target_num_images must be at least 2")
        if self.on_miss not in {"force", "skip", "error"}:
            raise ValueError("on_miss must be one of: force, skip, error")
        if self.max_wait_iterations is not None and int(self.max_wait_iterations) <= 0:
            raise ValueError("max_wait_iterations must be positive when provided")


@dataclass(frozen=True)
class StabilizedPerpForce:
    """Trigger remeshing once the perpendicular force has dropped and plateaued."""

    min_iterations: int = 20
    relative_drop: float = 0.3
    window: int = 5
    plateau_tolerance: float = 0.05

    def __post_init__(self):
        if int(self.min_iterations) <= 0:
            raise ValueError("min_iterations must be positive")
        if not 0.0 <= float(self.relative_drop) <= 1.0:
            raise ValueError("relative_drop must be within [0.0, 1.0]")
        if int(self.window) <= 0:
            raise ValueError("window must be positive")
        if float(self.plateau_tolerance) < 0.0:
            raise ValueError("plateau_tolerance must be non-negative")

    def is_triggered(self, history):
        if len(history) < max(self.min_iterations, self.window):
            return False
        baseline = max(entry["fperp_max"] for entry in history[: self.min_iterations])
        current = history[-1]["fperp_max"]
        if baseline <= 0.0:
            return True
        if current > baseline * (1.0 - self.relative_drop):
            return False
        recent = [entry["fperp_max"] for entry in history[-self.window :]]
        plateau_span = max(recent) - min(recent)
        return plateau_span <= self.plateau_tolerance * max(current, 1.0e-12)


def _copy_images_with_calculators(images):
    copied = []
    for image in images:
        snapshot = image.copy()
        snapshot.calc = image.calc
        copied.append(snapshot)
    return copied


def _normalize_structure_indices(structures, structure_indices, num_images):
    if structure_indices is not None:
        return list(structure_indices)
    if len(structures) == 2:
        return [0, int(num_images) - 1]
    if len(structures) == int(num_images):
        return list(range(int(num_images)))
    raise ValueError("structure_indices are required when structures do not already span the full path")


def _normalize_remesh_stages(remesh_stages):
    if remesh_stages is None:
        return []
    return [stage if isinstance(stage, RemeshStage) else RemeshStage(**stage) for stage in remesh_stages]


def _should_activate_ci(iteration, metrics, ci_activation_iteration, ci_activation_force):
    if ci_activation_iteration is None and ci_activation_force is None:
        return True
    if ci_activation_iteration is not None and int(iteration) >= int(ci_activation_iteration):
        return True
    if ci_activation_force is not None and float(metrics["fmax"]) <= float(ci_activation_force):
        return True
    return False


def _trigger_fired(trigger, history):
    if trigger is None:
        return False
    if hasattr(trigger, "is_triggered"):
        return bool(trigger.is_triggered(history))
    return bool(trigger(history))


def _activate_ci(band, iteration, force_max):
    band.method = "ci"
    if hasattr(band, "_refresh_band_state"):
        band._refresh_band_state()
    if band.context.is_output_owner:
        print(f"Climbing image activated at iteration {iteration} (Total Force={force_max:.9g})")


def _build_stage_record(*, stage_index, stage_output, band, is_final_stage, stage_result):
    last_metrics = None if not stage_result["metrics_history"] else stage_result["metrics_history"][-1]
    return {
        "stage_index": int(stage_index),
        "stage_dir": str(stage_output.run_dir),
        "num_images": int(band.numImages),
        "is_final_stage": bool(is_final_stage),
        "exit_reason": stage_result["reason"],
        "iterations": int(stage_result["iterations"]),
        "ci_active": bool(stage_result["ci_active"]),
        "metrics": last_metrics,
    }


def _write_stage_outputs(
    *,
    stage_output,
    stage_index,
    band,
    is_final_stage,
    stage_result,
    remesh_stage,
    method,
    force_converged,
    stage_max_iterations,
    image_mobility_rates,
    script_path,
    git_cwd,
    follow_up_action=None,
):
    stage_record = _build_stage_record(
        stage_index=stage_index,
        stage_output=stage_output,
        band=band,
        is_final_stage=is_final_stage,
        stage_result=stage_result,
    )
    stage_record["follow_up_action"] = follow_up_action
    stage_output.write_json(stage_output.run_dir / "stage_exit.json", stage_record)
    stage_output.write_manifest(
        script_path=script_path,
        git_cwd=git_cwd,
        config={
            "stage_index": stage_index,
            "num_images": band.numImages,
            "requested_method": method if is_final_stage else "normal",
            "force_converged": force_converged,
            "max_iterations": stage_max_iterations,
            "image_mobility_rates": None if not image_mobility_rates else dict(image_mobility_rates),
            "remesh_stage": None
            if remesh_stage is None
            else {
                "target_num_images": remesh_stage.target_num_images,
                "max_wait_iterations": remesh_stage.max_wait_iterations,
                "on_miss": remesh_stage.on_miss,
                "trigger": type(remesh_stage.trigger).__name__,
            },
        },
        extra_metadata={
            "outputs": stage_output.as_public_paths(),
            "stage_exit": stage_record,
        },
    )
    return stage_record


def _write_workflow_outputs(
    *,
    workflow_output,
    num_images,
    method,
    force_converged,
    max_iterations,
    image_mobility_rates,
    remesh_plan,
    manifest_config,
    script_path,
    git_cwd,
    executed_stages,
    transition_records,
    outcome,
    error_message,
):
    summary = {
        "outcome": outcome,
        "run_directory": str(workflow_output.run_dir),
        "stages": [
            {key: value for key, value in stage.items() if key != "artifacts"}
            for stage in executed_stages
        ],
        "transitions": transition_records,
        "error": error_message,
    }
    summary_file = workflow_output.paths.config_dir / "workflow_summary.json"
    workflow_output.write_json(summary_file, summary)
    final_output = workflow_output if not executed_stages else executed_stages[-1]["artifacts"]
    workflow_output.write_manifest(
        script_path=script_path,
        git_cwd=git_cwd,
        config={
            "num_images": int(num_images),
            "requested_method": method,
            "force_converged": force_converged,
            "max_iterations": max_iterations,
            "image_mobility_rates": None if not image_mobility_rates else dict(image_mobility_rates),
            "remesh_stages": [
                {
                    "target_num_images": stage.target_num_images,
                    "max_wait_iterations": stage.max_wait_iterations,
                    "on_miss": stage.on_miss,
                    "trigger": type(stage.trigger).__name__,
                }
                for stage in remesh_plan
            ],
            "user_config": manifest_config,
        },
        extra_metadata={
            "summary_file": str(summary_file),
            "transitions_dir": str(workflow_output.paths.transitions_dir),
            "stage_directories": [str(stage["artifacts"].run_dir) for stage in executed_stages],
            "final_outputs": final_output.as_public_paths(),
            "outcome": outcome,
        },
    )
    return summary, summary_file


def _run_stage(
    *,
    band,
    optimizer,
    max_iterations,
    force_converged,
    requested_method,
    ci_activation_iteration,
    ci_activation_force,
    trigger,
    final_convergence_enabled,
):
    def finalize_stage_iteration(metrics):
        if metrics is not None and not metrics.get("output_written", False):
            optimizer.ensure_iteration_outputs(metrics["iteration"])

    metrics_history = []
    completed_iteration = False
    ci_requested = requested_method == "ci"
    ci_active = ci_requested and ci_activation_iteration is None and ci_activation_force is None
    band.method = "ci" if ci_active else "normal" if ci_requested else requested_method
    if hasattr(band, "_refresh_band_state"):
        band._refresh_band_state()

    optimizer._begin_run()
    try:
        for iteration in range(1, int(max_iterations) + 1):
            metrics = optimizer.run_iteration(
                iteration,
                max_iterations=int(max_iterations),
                force_converged=float(force_converged),
                convergence_enabled=bool(final_convergence_enabled),
            )
            completed_iteration = True
            metrics_history.append(metrics)
            if ci_requested and not ci_active and _should_activate_ci(
                iteration,
                metrics,
                ci_activation_iteration,
                ci_activation_force,
            ):
                _activate_ci(band, iteration, metrics["fmax"])
                ci_active = True
            if trigger is not None and _trigger_fired(trigger, metrics_history):
                finalize_stage_iteration(metrics)
                return {
                    "reason": "remesh_triggered",
                    "iterations": int(iteration),
                    "metrics_history": metrics_history,
                    "ci_active": ci_active,
                }
            if final_convergence_enabled and metrics["converged"]:
                finalize_stage_iteration(metrics)
                return {
                    "reason": "final_converged",
                    "iterations": int(iteration),
                    "metrics_history": metrics_history,
                    "ci_active": ci_active,
                }
        finalize_stage_iteration(metrics_history[-1] if metrics_history else None)
        return {
            "reason": "max_iterations_reached",
            "iterations": int(max_iterations),
            "metrics_history": metrics_history,
            "ci_active": ci_active,
        }
    finally:
        if completed_iteration:
            optimizer._finish_run()
        else:
            optimizer._abort_run()


def run_staged_ssneb(
    *,
    structures,
    structure_indices=None,
    num_images,
    remesh_stages=None,
    restart_xyz=None,
    k=5.0,
    method="ci",
    filter_factory=None,
    output_dir=None,
    output_settings=None,
    band_kwargs=None,
    optimizer_kwargs=None,
    minimize_kwargs=None,
    image_mobility_rates=None,
    ci_activation_iteration=None,
    ci_activation_force=None,
    script_path=None,
    manifest_config=None,
    resolved_config=None,
    input_config_path=None,
    git_cwd=None,
):
    """Run SSNEB as one or more conservative stages with optional remeshing."""

    from tsase.neb.core.band import ssneb
    from tsase.neb.io.restart import load_band_configuration_from_xyz

    structures = list(structures)
    remesh_plan = _normalize_remesh_stages(remesh_stages)
    indices = _normalize_structure_indices(structures, structure_indices, num_images)
    band_options = {} if band_kwargs is None else dict(band_kwargs)
    optimizer_options = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    minimize_options = {} if minimize_kwargs is None else dict(minimize_kwargs)
    force_converged = float(minimize_options.get("forceConverged", 0.01))
    max_iterations = int(minimize_options.get("maxIterations", 1000))

    workflow_output = (
        OutputManager.create(
            base_dir="neb_runs",
            run_name="staged_ssneb",
            timestamp=True,
            settings=output_settings,
        )
        if output_dir is None
        else OutputManager.from_run_dir(output_dir, settings=output_settings)
    )
    workflow_output.paths.config_dir.mkdir(parents=True, exist_ok=True)
    workflow_output.paths.transitions_dir.mkdir(parents=True, exist_ok=True)
    workflow_output.paths.stages_dir.mkdir(parents=True, exist_ok=True)
    if input_config_path is not None:
        workflow_output.copy_input_config(input_config_path)
    if resolved_config is not None:
        from .config import dump_yaml

        workflow_output.write_text(
            workflow_output.paths.resolved_config_file,
            dump_yaml(resolved_config) + "\n",
        )
    if script_path is not None:
        workflow_output.snapshot_script(script_path)

    current_structures = _copy_images_with_calculators(structures)
    current_indices = list(indices)
    current_num_images = int(num_images)
    executed_stages = []
    transition_records = []
    final_band = None
    final_optimizer = None
    outcome = "completed"
    error_message = None

    total_stage_count = len(remesh_plan) + 1
    for stage_index in range(total_stage_count):
        is_final_stage = stage_index == total_stage_count - 1
        remesh_stage = None if is_final_stage else remesh_plan[stage_index]
        stage_dir = workflow_output.paths.stages_dir / f"stage_{stage_index:02d}"
        stage_output = workflow_output.child(
            stage_dir,
            manifest_file=stage_dir / "stage_manifest.json",
        )

        stage_method = method if is_final_stage else "normal"
        band = ssneb(
            current_structures,
            current_indices,
            numImages=current_num_images,
            k=k,
            method=stage_method,
            filter_factory=filter_factory,
            output_manager=stage_output,
            **band_options,
        )
        if stage_index == 0 and restart_xyz is not None:
            load_band_configuration_from_xyz(band, str(restart_xyz))

        optimizer = fire_ssneb(
            band,
            image_mobility_rates=image_mobility_rates,
            **optimizer_options,
        )

        stage_max_iterations = (
            max_iterations
            if remesh_stage is None or remesh_stage.max_wait_iterations is None
            else remesh_stage.max_wait_iterations
        )
        try:
            stage_result = _run_stage(
                band=band,
                optimizer=optimizer,
                max_iterations=stage_max_iterations,
                force_converged=force_converged,
                requested_method=method if is_final_stage else "normal",
                ci_activation_iteration=ci_activation_iteration if is_final_stage else None,
                ci_activation_force=ci_activation_force if is_final_stage else None,
                trigger=None if is_final_stage else remesh_stage.trigger,
                final_convergence_enabled=is_final_stage,
            )
        except Exception as exc:
            stage_result = {
                "reason": "error",
                "iterations": 0,
                "metrics_history": [],
                "ci_active": False,
            }
            stage_record = _write_stage_outputs(
                stage_output=stage_output,
                stage_index=stage_index,
                band=band,
                is_final_stage=is_final_stage,
                stage_result=stage_result,
                remesh_stage=remesh_stage,
                method=method,
                force_converged=force_converged,
                stage_max_iterations=stage_max_iterations,
                image_mobility_rates=image_mobility_rates,
                script_path=script_path,
                git_cwd=git_cwd,
            )
            executed_stages.append({**stage_record, "artifacts": stage_output})
            final_band = band
            final_optimizer = optimizer
            outcome = "failed"
            error_message = str(exc)
            _write_workflow_outputs(
                workflow_output=workflow_output,
                num_images=num_images,
                method=method,
                force_converged=force_converged,
                max_iterations=max_iterations,
                image_mobility_rates=image_mobility_rates,
                remesh_plan=remesh_plan,
                manifest_config=manifest_config,
                script_path=script_path,
                git_cwd=git_cwd,
                executed_stages=executed_stages,
                transition_records=transition_records,
                outcome=outcome,
                error_message=error_message,
            )
            raise

        follow_up_action = None
        if not is_final_stage and stage_result["reason"] == "max_iterations_reached":
            if remesh_stage.on_miss == "force":
                follow_up_action = "force_remesh"
            elif remesh_stage.on_miss == "skip":
                follow_up_action = "skip_remesh"
            else:
                stage_result = {
                    **stage_result,
                    "reason": "remesh_trigger_missed",
                }

        stage_record = _write_stage_outputs(
            stage_output=stage_output,
            stage_index=stage_index,
            band=band,
            is_final_stage=is_final_stage,
            stage_result=stage_result,
            remesh_stage=remesh_stage,
            method=method,
            force_converged=force_converged,
            stage_max_iterations=stage_max_iterations,
            image_mobility_rates=image_mobility_rates,
            script_path=script_path,
            git_cwd=git_cwd,
            follow_up_action=follow_up_action,
        )
        executed_stages.append({**stage_record, "artifacts": stage_output})
        final_band = band
        final_optimizer = optimizer

        if is_final_stage:
            break

        if stage_result["reason"] == "remesh_trigger_missed" and remesh_stage.on_miss == "error":
            outcome = "failed"
            error_message = "remesh trigger did not fire before the configured maximum wait"
            break

        if stage_result["reason"] == "max_iterations_reached" and remesh_stage.on_miss == "skip":
            current_structures = _copy_images_with_calculators(band.path)
            current_indices = list(range(band.numImages))
            current_num_images = band.numImages
            continue

        remeshed = uniform_remesh(band.path, num_images=remesh_stage.target_num_images)
        transition_index = len(transition_records)
        transition_xyz = workflow_output.paths.transitions_dir / f"remesh_{transition_index:02d}_to_{transition_index + 1:02d}.xyz"
        transition_json = workflow_output.paths.transitions_dir / f"remesh_{transition_index:02d}_to_{transition_index + 1:02d}.json"
        if workflow_output.is_active:
            write(str(transition_xyz), [image.copy() for image in remeshed], format="extxyz")
        transition_record = {
            "from_stage": stage_index,
            "to_stage": stage_index + 1,
            "from_num_images": band.numImages,
            "to_num_images": remesh_stage.target_num_images,
            "reason": stage_result["reason"],
            "follow_up_action": follow_up_action,
            "xyz_file": str(transition_xyz),
            "json_file": str(transition_json),
        }
        workflow_output.write_json(transition_json, transition_record)
        transition_records.append(transition_record)

        current_structures = _copy_images_with_calculators(remeshed)
        current_indices = list(range(remesh_stage.target_num_images))
        current_num_images = remesh_stage.target_num_images

    workflow_summary, summary_file = _write_workflow_outputs(
        workflow_output=workflow_output,
        num_images=num_images,
        method=method,
        force_converged=force_converged,
        max_iterations=max_iterations,
        image_mobility_rates=image_mobility_rates,
        remesh_plan=remesh_plan,
        manifest_config=manifest_config,
        script_path=script_path,
        git_cwd=git_cwd,
        executed_stages=executed_stages,
        transition_records=transition_records,
        outcome=outcome,
        error_message=error_message,
    )

    return {
        "band": final_band,
        "optimizer": final_optimizer,
        "artifacts": executed_stages[-1]["artifacts"] if executed_stages else workflow_output,
        "workflow_output": workflow_output,
        "workflow_artifacts": workflow_output,
        "stages": executed_stages,
        "transitions": transition_records,
        "workflow_summary": workflow_summary,
        "summary_file": str(summary_file),
    }
