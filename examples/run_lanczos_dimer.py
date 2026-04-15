#!/usr/bin/env python

"""Run the maintained YAML-driven Lanczos-dimer example workflow."""

from argparse import ArgumentParser
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsase.dimer import load_dimer_config, run_dimer


DEFAULT_CONFIG = ROOT / "examples" / "configs" / "run_lanczos_dimer.yaml"


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-force-calls", type=int, default=None)
    parser.add_argument("--min-force", type=float, default=None)
    parser.add_argument("--downhill-fmax", type=float, default=None)
    parser.add_argument("--downhill-max-steps", type=int, default=None)
    return parser.parse_args(argv)


def _build_overrides(args):
    overrides = {}
    if args.output_dir is not None:
        overrides.setdefault("run", {})["root"] = args.output_dir
    if args.max_force_calls is not None:
        overrides.setdefault("convergence", {})["maxForceCalls"] = args.max_force_calls
    if args.min_force is not None:
        overrides.setdefault("convergence", {})["minForce"] = args.min_force
    if args.downhill_fmax is not None:
        overrides.setdefault("postprocess", {}).setdefault("downhill", {})["fmax"] = args.downhill_fmax
    if args.downhill_max_steps is not None:
        overrides.setdefault("postprocess", {}).setdefault("downhill", {})["max_steps"] = args.downhill_max_steps
    return overrides or None


def main(argv=None):
    args = parse_args(argv)
    config = load_dimer_config(args.config, overrides=_build_overrides(args))
    result = run_dimer(config=config)

    print(f"Run directory: {result['run_dir']}")
    print(f"Saddle summary: {result['artifacts']['summary_file']}")
    print(f"Saddle structure: {result['artifacts']['saddle_structure']}")
    if result["downhill_result"] is not None:
        print(f"Connection summary: {result['artifacts']['connections_summary']}")
        print(f"Positive branch: {result['artifacts']['positive_structure']}")
        print(f"Negative branch: {result['artifacts']['negative_structure']}")
    return result


if __name__ == "__main__":
    main()

