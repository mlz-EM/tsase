#!/usr/bin/env python

"""Analyze a saved SSNEB XYZ path with the maintained post-hoc STEM API."""

from argparse import ArgumentParser
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsase.neb.viz import analyze_stem_sequence_from_xyz


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--npy", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = analyze_stem_sequence_from_xyz(
        args.xyz,
        output_dir=args.output_dir,
        emit_png=args.png or not any((args.png, args.gif, args.npy)),
        emit_gif=args.gif or not any((args.png, args.gif, args.npy)),
        emit_npy=args.npy or not any((args.png, args.gif, args.npy)),
        emit_diagnostics=args.diagnostics,
    )
    print(f"Status: {result['status']}")
    print(f"Output directory: {result['output_dir']}")
    if "gif" in result:
        print(f"GIF: {result['gif']}")
    if "npz" in result:
        print(f"NPZ: {result['npz']}")
    return result


if __name__ == "__main__":
    main()
