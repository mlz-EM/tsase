#!/usr/bin/env python

"""Run the maintained YAML-driven field-coupled SSNEB example.

For the PbZrO3 example config, first run
``examples/preprocess_field_ssneb_control_points.py`` to produce NEB-ready CIFs
and a derived YAML input for the actual NEB run.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsase.neb.workflows import (
    load_field_ssneb_config,
    load_yaml_file,
    preprocess_field_ssneb_control_points,
    run_field_ssneb,
)


DEFAULT_CONFIG = "/home/gridsan/mzhu/Tools/STEM_TOOL/tsase/example/preprocessed/run_field_ssneb_interpolated/run_field_ssneb_interpolated_preprocessed.yaml" #ROOT / "examples" / "configs" / "run_field_ssneb_interpolated.yaml"


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    return parser.parse_args(argv)


def _build_overrides(args):
    overrides = {}
    if args.output_dir is not None:
        overrides.setdefault("run", {})["root"] = args.output_dir
    if args.max_steps is not None:
        overrides.setdefault("optimizer", {}).setdefault("convergence", {})["max_steps"] = args.max_steps
    if args.fmax is not None:
        overrides.setdefault("optimizer", {}).setdefault("convergence", {})["fmax"] = args.fmax
    if args.num_images is not None:
        overrides.setdefault("path", {})["num_images"] = args.num_images
    return overrides or None


def _expected_preprocessed_config_path(config_path, raw_config):
    preprocess = dict(raw_config.get("preprocess", {}))
    output_dir = preprocess.get("output_dir")
    if output_dir is None:
        return None
    processed_name = str(
        dict(preprocess.get("outputs", {})).get(
            "config_name",
            "run_field_ssneb_interpolated_preprocessed.yaml",
        )
    )
    return (config_path.parent / output_dir / processed_name).resolve()


def _resolve_runtime_config_path(config_path):
    config_path = Path(config_path).expanduser().resolve()
    raw_config = load_yaml_file(config_path)
    if "preprocess" not in raw_config or config_path.stem.endswith("_preprocessed"):
        return config_path

    expected = _expected_preprocessed_config_path(config_path, raw_config)
    if expected is not None and expected.exists():
        print(f"Using preprocessed config: {expected}")
        return expected

    result = preprocess_field_ssneb_control_points(config_path)
    resolved = Path(result["processed_config"]).resolve()
    print(f"Using preprocessed config: {resolved}")
    return resolved


def main(argv=None):
    args = parse_args(argv)
    runtime_config_path = _resolve_runtime_config_path(args.config)
    config = load_field_ssneb_config(runtime_config_path, overrides=_build_overrides(args))
    result = run_field_ssneb(config=config)

    workflow_output = result["workflow_output"]
    artifacts = result["artifacts"]
    print(f"Run directory: {workflow_output.run_dir}")
    print(f"Diagnostics: {artifacts.diagnostics_file}")
    print(f"Path snapshots: {artifacts.path_dir}")
    return result


if __name__ == "__main__":
    main()
