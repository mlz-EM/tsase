#!/usr/bin/env python

"""Run the bundled preprocessed PbZrO3 field-SSNEB example."""

from argparse import ArgumentParser
from pathlib import Path
import sys


EXAMPLE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tsase.neb.runtime import detect_world_size
from tsase.neb.workflows import load_field_ssneb_config, run_field_ssneb


DEFAULT_CONFIG = EXAMPLE_DIR / "preprocessed" / "run.yaml"


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--parallel", choices=("true", "false", "auto"), default="auto")
    return parser.parse_args(argv)


def _build_overrides(args):
    overrides = {}
    if args.output_dir is not None:
        overrides.setdefault("run", {})["root"] = str(args.output_dir.expanduser().resolve())
    if args.max_steps is not None:
        overrides.setdefault("optimizer", {}).setdefault("convergence", {})["max_steps"] = args.max_steps
    if args.fmax is not None:
        overrides.setdefault("optimizer", {}).setdefault("convergence", {})["fmax"] = args.fmax
    if args.num_images is not None:
        overrides.setdefault("path", {})["num_images"] = args.num_images
    if args.device is not None:
        overrides.setdefault("model", {}).setdefault("calculator", {})["device"] = args.device
    if args.parallel is not None:
        parallel = detect_world_size() > 1 if args.parallel == "auto" else args.parallel == "true"
        overrides.setdefault("band", {})["parallel"] = parallel
    return overrides or None


def main(argv=None):
    args = parse_args(argv)
    config = load_field_ssneb_config(args.config, overrides=_build_overrides(args))
    result = run_field_ssneb(config=config)
    artifacts = result["artifacts"]
    print(f"Run directory: {artifacts.run_dir}")
    print(f"Diagnostics: {artifacts.diagnostics_file}")
    print(f"Path snapshots: {artifacts.path_dir}")
    return result


if __name__ == "__main__":
    main()
