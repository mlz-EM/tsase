#!/usr/bin/env python

"""Prepare the bundled PbZrO3 control points for the field-SSNEB run."""

from argparse import ArgumentParser
from pathlib import Path
import sys


EXAMPLE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tsase.neb.workflows import preprocess_field_ssneb_control_points


DEFAULT_CONFIG = EXAMPLE_DIR / "input.yaml"


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = preprocess_field_ssneb_control_points(
        args.config,
        output_dir=args.output_dir,
    )
    print(f"Preprocessed inputs: {result['output_dir']}")
    print(f"Run configuration: {result['processed_config']}")
    return result


if __name__ == "__main__":
    main()
