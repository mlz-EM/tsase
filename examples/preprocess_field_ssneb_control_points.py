#!/usr/bin/env python

"""Preprocess raw field-SSNEB control points into NEB-ready CIF inputs."""

from argparse import ArgumentParser
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsase.neb.workflows import preprocess_field_ssneb_control_points


DEFAULT_CONFIG = ROOT / "examples" / "configs" / "run_field_ssneb_interpolated.yaml"


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = preprocess_field_ssneb_control_points(
        args.config,
        output_dir=args.output_dir,
    )
    if result["space_group"] is not None:
        print(f"Reference space group: {result['space_group']}")
    print(f"Preprocessed inputs: {result['output_dir']}")
    print(f"Derived config: {result['processed_config']}")
    endpoint_stem = result.get("endpoint_stem")
    if endpoint_stem is not None:
        if endpoint_stem.get("status") == "ok":
            print(f"Endpoint STEM frames: {endpoint_stem['frame_dir']}")
            if endpoint_stem.get("gif") is not None:
                print(f"Endpoint STEM gif: {endpoint_stem['gif']}")
        else:
            print(f"Endpoint STEM diagnostics: {endpoint_stem.get('diagnostics_file')}")
    return result


if __name__ == "__main__":
    main()
