#!/usr/bin/env python

"""Translate one image in a saved SSNEB extxyz band and write a new restart file."""

import argparse
from pathlib import Path

import numpy as np
from ase import io
from tsase.neb import NEB_ATOM_ID_ARRAY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_xyz")
    parser.add_argument("output_xyz")
    parser.add_argument("--image-index", type=int, required=True)
    parser.add_argument(
        "--shift",
        nargs=3,
        type=float,
        required=True,
        metavar=("DX", "DY", "DZ"),
        help="Shift vector in Cartesian Angstrom unless --fractional is set.",
    )
    parser.add_argument(
        "--fractional",
        action="store_true",
        help="Interpret --shift in fractional lattice coordinates.",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap translated atoms back into the periodic cell before saving.",
    )
    parser.add_argument(
        "--atom-index",
        type=int,
        help="Translate only this atom index within the selected image.",
    )
    parser.add_argument(
        "--atom-id",
        type=int,
        help=f"Translate only the atom with this persistent `{NEB_ATOM_ID_ARRAY}`.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images = io.read(args.input_xyz, ":")
    if not images:
        raise ValueError(f"no images found in {args.input_xyz!r}")
    if args.image_index < 0 or args.image_index >= len(images):
        raise IndexError(
            f"image index {args.image_index} is out of range for {len(images)} images"
        )
    if args.atom_index is not None and args.atom_id is not None:
        raise ValueError("use at most one of --atom-index or --atom-id")

    target = images[args.image_index]
    shift = np.array(args.shift, dtype=float)
    if args.fractional:
        shift = np.dot(shift, np.array(target.get_cell(), dtype=float))

    positions = target.get_positions().copy()
    moved_label = f"image {args.image_index}"
    if args.atom_id is not None:
        if NEB_ATOM_ID_ARRAY not in target.arrays:
            raise ValueError(
                f"selected image does not contain `{NEB_ATOM_ID_ARRAY}`; "
                "it cannot be edited by persistent atom ID"
            )
        matches = np.where(
            np.array(target.arrays[NEB_ATOM_ID_ARRAY], dtype=int) == int(args.atom_id)
        )[0]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one atom with {NEB_ATOM_ID_ARRAY}={args.atom_id}, "
                f"found {len(matches)}"
            )
        atom_index = int(matches[0])
        positions[atom_index] += shift
        moved_label = (
            f"atom id {args.atom_id} (index {atom_index}) in image {args.image_index}"
        )
    elif args.atom_index is not None:
        if args.atom_index < 0 or args.atom_index >= len(target):
            raise IndexError(
                f"atom index {args.atom_index} is out of range for {len(target)} atoms"
            )
        positions[int(args.atom_index)] += shift
        moved_label = f"atom index {args.atom_index} in image {args.image_index}"
    else:
        positions += shift
    target.set_positions(positions)
    if args.wrap:
        target.wrap()

    for image in images:
        image.info.pop("spatial_map_order", None)

    output_path = Path(args.output_xyz).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.write(output_path, images, format="extxyz")

    print(f"Wrote {len(images)} images to {output_path}")
    print(f"Translated {moved_label} by {shift.tolist()} Angstrom")


if __name__ == "__main__":
    main()
