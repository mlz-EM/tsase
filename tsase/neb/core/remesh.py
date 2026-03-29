"""Public path remeshing helpers for SSNEB."""

import numpy as np

from tsase.neb.util import sPBC

from .geometry import compute_jacobian, image_distance_vector, initialize_image_properties
from .mapping import ensure_atom_ids


def _copy_image_with_calc(image):
    copied = image.copy()
    copied.calc = image.calc
    return copied


def _validate_target_image_count(num_images):
    target = int(num_images)
    if target < 2:
        raise ValueError("num_images must be at least 2")
    return target


def _resolve_target_image_count(images, num_images=None, upsample_ratio=None):
    if (num_images is None) == (upsample_ratio is None):
        raise ValueError("pass exactly one of num_images or upsample_ratio")
    if upsample_ratio is not None:
        ratio = float(upsample_ratio)
        if ratio <= 0.0:
            raise ValueError("upsample_ratio must be positive")
        num_images = int(round((len(images) - 1) * ratio)) + 1
    return _validate_target_image_count(num_images)


def _path_metric_cumulative_lengths(images, weight):
    if len(images) < 2:
        raise ValueError("images must contain at least two path images")
    prepared = [_copy_image_with_calc(image) for image in images]
    ensure_atom_ids(prepared, reference=prepared[0])
    jacobian = compute_jacobian(
        prepared[0].get_volume(),
        prepared[-1].get_volume(),
        len(prepared[0]),
        weight=weight,
    )
    for image in prepared:
        initialize_image_properties(image, jacobian)

    lengths = [0.0]
    for left, right in zip(prepared, prepared[1:]):
        delta = image_distance_vector(right, left)
        lengths.append(lengths[-1] + float(np.linalg.norm(delta)))
    return np.asarray(lengths, dtype=float)


def _interpolate_image(left, right, fraction):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0.0, 1.0]")
    if fraction == 0.0:
        return _copy_image_with_calc(left)
    if fraction == 1.0:
        return _copy_image_with_calc(right)

    cell_left = np.asarray(left.get_cell(), dtype=float)
    cell_right = np.asarray(right.get_cell(), dtype=float)
    cell = cell_left + fraction * (cell_right - cell_left)

    inv_left = np.linalg.inv(cell_left)
    frac_left = np.dot(left.get_positions(), inv_left)
    frac_right = right.get_scaled_positions()
    frac = frac_left + fraction * sPBC(frac_right - frac_left)

    image = _copy_image_with_calc(left)
    image.set_cell(cell, scale_atoms=False)
    image.set_positions(np.dot(frac, cell))
    image.calc = left.calc if left.calc is not None else right.calc
    return image


def uniform_remesh(images, num_images=None, upsample_ratio=None, weight=1.0):
    """Redistribute a band onto a uniform SSNEB path grid."""
    source_images = list(images)
    target_count = _resolve_target_image_count(
        source_images,
        num_images=num_images,
        upsample_ratio=upsample_ratio,
    )
    if len(source_images) < 2:
        raise ValueError("images must contain at least two path images")
    if target_count == len(source_images):
        remeshed = [_copy_image_with_calc(image) for image in source_images]
        ensure_atom_ids(remeshed, reference=remeshed[0])
        return remeshed

    cumulative_lengths = _path_metric_cumulative_lengths(source_images, weight=weight)
    total_length = float(cumulative_lengths[-1])
    targets = np.linspace(0.0, total_length, target_count)
    remeshed = []
    for index, target in enumerate(targets):
        if index == 0:
            remeshed.append(_copy_image_with_calc(source_images[0]))
            continue
        if index == target_count - 1:
            remeshed.append(_copy_image_with_calc(source_images[-1]))
            continue
        segment_index = int(np.searchsorted(cumulative_lengths, target, side="right") - 1)
        segment_index = max(0, min(segment_index, len(source_images) - 2))
        left_length = cumulative_lengths[segment_index]
        right_length = cumulative_lengths[segment_index + 1]
        if right_length <= left_length:
            fraction = 0.0
        else:
            fraction = float((target - left_length) / (right_length - left_length))
        remeshed.append(
            _interpolate_image(
                source_images[segment_index],
                source_images[segment_index + 1],
                fraction,
            )
        )
    ensure_atom_ids(remeshed, reference=source_images[0])
    return remeshed
