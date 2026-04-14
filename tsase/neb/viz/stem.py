"""Projected STEM-style NEB analysis and visualization helpers.

The helpers in this module analyze one NEB image at a time in the projected
``ab`` plane, then render a 3-panel visualization across a saved NEB sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from ase.io import read
from scipy.optimize import linear_sum_assignment

_GRID_SIZE = 4
_COLUMN_CUTOFF_ANG = 0.31  # slightly relaxed from 30 pm to avoid obvious over-splitting
_IMAGE_SHIFTS = np.array(
    [(i, j) for i in (-1.0, 0.0, 1.0) for j in (-1.0, 0.0, 1.0)],
    dtype=float,
)
_PM_TO_ANGSTROM = 0.01


class StemAnalysisError(RuntimeError):
    """Raised when one projected frame cannot be analyzed safely."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass
class ColumnComponent:
    """Projected connected component used to build atomic columns."""

    atom_indices: tuple[int, ...]
    species_counts: dict[str, int]
    classification: str
    projected_fractional: np.ndarray


@dataclass
class ProjectedFrameAnalysis:
    """Structured projected result for one ASE ``Atoms`` image."""

    frame_index: int
    grid_size: int
    cell_xy: np.ndarray
    pb_frac: np.ndarray
    zr_frac: np.ndarray
    oh_frac: np.ndarray
    ov_frac: np.ndarray
    pb_plot_frac: np.ndarray
    zr_plot_frac: np.ndarray
    pb_xy: np.ndarray
    zr_xy: np.ndarray
    oh_xy: np.ndarray
    ov_xy: np.ndarray
    pb_plot_xy: np.ndarray
    zr_plot_xy: np.ndarray
    pb_tiles_xy: np.ndarray
    zr_tiles_xy: np.ndarray
    zr_polygons_xy: np.ndarray
    zr_tilt_deg: np.ndarray
    pb_displacement_xy: np.ndarray
    pb_displacement_minus_mean_xy: np.ndarray
    horizontal_pair_family: str
    diagnostics: dict[str, object]
    components: list[ColumnComponent]


def _wrap01(values):
    return np.mod(np.asarray(values, dtype=float), 1.0)


def _cell_xy_from_atoms(atoms):
    cell = np.asarray(atoms.get_cell(), dtype=float)
    return np.asarray(
        [
            [cell[0, 0], cell[0, 1]],
            [cell[1, 0], cell[1, 1]],
        ],
        dtype=float,
    )


def _projected_fractional_positions(atoms):
    positions = np.asarray(atoms.get_positions(), dtype=float)
    inv_cell = np.linalg.inv(np.asarray(atoms.get_cell(), dtype=float))
    fractional = positions @ inv_cell
    return _wrap01(fractional[:, :2])


def _frac_to_xy(projected_fractional, cell_xy):
    return np.asarray(projected_fractional, dtype=float) @ cell_xy


def _minimum_image_delta(projected_target, projected_reference, cell_xy):
    deltas = np.asarray(projected_target, dtype=float) - np.asarray(projected_reference, dtype=float)
    candidates = deltas + _IMAGE_SHIFTS
    cartesian = candidates @ cell_xy
    norms = np.linalg.norm(cartesian, axis=1)
    best = int(np.argmin(norms))
    return candidates[best], cartesian[best], float(norms[best])


def _torus_distance(projected_a, projected_b, cell_xy):
    return _minimum_image_delta(projected_a, projected_b, cell_xy)[2]


def _unwrap_around_reference(projected_points, projected_reference, cell_xy):
    unwrapped = []
    for point in np.asarray(projected_points, dtype=float):
        delta_frac, _, _ = _minimum_image_delta(point, projected_reference, cell_xy)
        unwrapped.append(np.asarray(projected_reference, dtype=float) + delta_frac)
    return np.asarray(unwrapped, dtype=float)


def _torus_average(projected_points, cell_xy, reference=None):
    projected_points = np.asarray(projected_points, dtype=float)
    if len(projected_points) == 0:
        raise ValueError("cannot average zero projected points")
    anchor = projected_points[0] if reference is None else np.asarray(reference, dtype=float)
    unwrapped = _unwrap_around_reference(projected_points, anchor, cell_xy)
    return _wrap01(np.mean(unwrapped, axis=0))


def _torus_weighted_average(projected_points, weights, cell_xy, reference=None):
    projected_points = np.asarray(projected_points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(projected_points) == 0:
        raise ValueError("cannot average zero projected points")
    if len(projected_points) != len(weights):
        raise ValueError("projected points and weights must have the same length")
    anchor = projected_points[0] if reference is None else np.asarray(reference, dtype=float)
    unwrapped = _unwrap_around_reference(projected_points, anchor, cell_xy)
    return _wrap01(np.average(unwrapped, axis=0, weights=weights))


def _connected_components(projected_fractional, cell_xy, cutoff_angstrom):
    count = len(projected_fractional)
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a, index_b):
        root_a = find(index_a)
        root_b = find(index_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for index_a in range(count):
        for index_b in range(index_a + 1, count):
            if _torus_distance(projected_fractional[index_a], projected_fractional[index_b], cell_xy) < cutoff_angstrom:
                union(index_a, index_b)

    components = {}
    for index in range(count):
        components.setdefault(find(index), []).append(index)
    return [tuple(sorted(indices)) for indices in components.values()]


def _format_species_counts(species):
    species = dict(sorted(species.items()))
    return ", ".join(f"{key}:{value}" for key, value in species.items())


def _nearest_component_distances(column_components, cell_xy, limit=12):
    pairs = []
    for i, left in enumerate(column_components):
        for j in range(i + 1, len(column_components)):
            right = column_components[j]
            pairs.append(
                {
                    "distance_angstrom": _torus_distance(
                        left.projected_fractional,
                        right.projected_fractional,
                        cell_xy,
                    ),
                    "left": left.classification,
                    "right": right.classification,
                    "left_species": left.species_counts,
                    "right_species": right.species_counts,
                }
            )
    pairs.sort(key=lambda entry: entry["distance_angstrom"])
    return pairs[:limit]


def _expected_pb_columns(grid_size):
    return int(grid_size * grid_size)


def _expected_zr_columns(grid_size):
    return int(grid_size * grid_size)


def _expected_o_columns(grid_size):
    return int(2 * grid_size * grid_size)


def _repair_oversegmented_columns(columns, weights, cell_xy, target_count, label, grid_size):
    if len(columns) <= target_count:
        return np.asarray(columns, dtype=float), []

    columns = [np.asarray(column, dtype=float) for column in columns]
    weights = [float(weight) for weight in weights]
    base_spacing = min(np.linalg.norm(cell_xy[0]), np.linalg.norm(cell_xy[1])) / float(grid_size)
    merge_threshold = 0.35 * base_spacing
    merges = []

    while len(columns) > target_count:
        best_pair = None
        best_distance = None
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                distance = _torus_distance(columns[i], columns[j], cell_xy)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        if best_pair is None or best_distance is None or best_distance > merge_threshold:
            raise StemAnalysisError(
                f"{label} column clustering produced too many components that could not be merged safely",
                diagnostics={
                    "label": label,
                    "raw_count": len(columns),
                    "target_count": target_count,
                    "best_merge_distance_angstrom": best_distance,
                    "merge_threshold_angstrom": merge_threshold,
                },
            )

        i, j = best_pair
        merged_column = _torus_weighted_average(
            [columns[i], columns[j]],
            [weights[i], weights[j]],
            cell_xy,
        )
        merges.append(
            {
                "distance_angstrom": float(best_distance),
                "merged_weights": [weights[i], weights[j]],
            }
        )
        columns[i] = merged_column
        weights[i] = weights[i] + weights[j]
        del columns[j]
        del weights[j]

    return np.asarray(columns, dtype=float), merges


def _build_column_sets(atoms, cutoff_angstrom, *, grid_size):
    symbols = np.asarray(atoms.get_chemical_symbols())
    cell_xy = _cell_xy_from_atoms(atoms)
    projected_fractional = _projected_fractional_positions(atoms)
    raw_components = _connected_components(projected_fractional, cell_xy, cutoff_angstrom)

    pb_columns = []
    pb_weights = []
    zr_columns = []
    zr_weights = []
    oxygen_columns = []
    column_components = []

    for atom_indices in raw_components:
        component_symbols = symbols[list(atom_indices)]
        species_counts = {}
        for symbol in component_symbols:
            species_counts[symbol] = species_counts.get(symbol, 0) + 1

        species_set = set(species_counts)
        if species_set == {"Pb"}:
            classification = "pb"
            centroid = _torus_average(projected_fractional[list(atom_indices)], cell_xy)
            pb_columns.append(centroid)
            pb_weights.append(len(atom_indices))
        elif species_set == {"O"}:
            classification = "o_only"
            centroid = _torus_average(projected_fractional[list(atom_indices)], cell_xy)
            oxygen_columns.append(centroid)
        elif species_set <= {"Zr", "O"} and "Zr" in species_set:
            zr_indices = [index for index in atom_indices if symbols[index] == "Zr"]
            if not zr_indices:
                raise StemAnalysisError(
                    "encountered a Zr-containing component with no Zr atoms after filtering",
                    diagnostics={"component": atom_indices, "species_counts": species_counts},
                )
            centroid = _torus_average(projected_fractional[zr_indices], cell_xy)
            classification = "mixed_zr_o" if "O" in species_set else "zr_only"
            zr_columns.append(centroid)
            zr_weights.append(len(zr_indices))
        else:
            centroid = _torus_average(projected_fractional[list(atom_indices)], cell_xy)
            classification = "unsupported"
            column_components.append(
                ColumnComponent(
                    atom_indices=tuple(atom_indices),
                    species_counts=species_counts,
                    classification=classification,
                    projected_fractional=centroid,
                )
            )
            raise StemAnalysisError(
                "encountered a projected component with unsupported species content",
                diagnostics={
                    "species_counts": species_counts,
                    "atom_indices": list(atom_indices),
                },
            )

        column_components.append(
            ColumnComponent(
                atom_indices=tuple(atom_indices),
                species_counts=species_counts,
                classification=classification,
                projected_fractional=np.asarray(centroid, dtype=float),
            )
        )

    pb_columns, pb_merges = _repair_oversegmented_columns(
        pb_columns,
        pb_weights,
        cell_xy,
        target_count=_expected_pb_columns(grid_size),
        label="Pb",
        grid_size=grid_size,
    )
    zr_columns, zr_merges = _repair_oversegmented_columns(
        zr_columns,
        zr_weights,
        cell_xy,
        target_count=_expected_zr_columns(grid_size),
        label="Zr",
        grid_size=grid_size,
    )

    diagnostics = {
        "component_count": len(column_components),
        "pb_columns": len(pb_columns),
        "zr_columns": len(zr_columns),
        "o_columns_total": len(oxygen_columns),
        "pb_merges": pb_merges,
        "zr_merges": zr_merges,
        "components": [
            {
                "size": len(component.atom_indices),
                "classification": component.classification,
                "species": dict(component.species_counts),
            }
            for component in sorted(
                column_components,
                key=lambda component: (
                    component.classification,
                    len(component.atom_indices),
                    _format_species_counts(component.species_counts),
                ),
            )
        ],
        "nearest_component_distances": _nearest_component_distances(column_components, cell_xy),
    }

    if (
        len(pb_columns) != _expected_pb_columns(grid_size)
        or len(zr_columns) != _expected_zr_columns(grid_size)
        or len(oxygen_columns) < _expected_o_columns(grid_size)
    ):
        raise StemAnalysisError(
            (
                f"projected column counts do not match the expected {grid_size}x{grid_size} pseudocubic lattice "
                f"(Pb={len(pb_columns)}, Zr={len(zr_columns)}, O-only={len(oxygen_columns)})"
            ),
            diagnostics=diagnostics,
        )

    return (
        np.asarray(pb_columns, dtype=float),
        np.asarray(zr_columns, dtype=float),
        np.asarray(oxygen_columns, dtype=float),
        column_components,
        diagnostics,
        cell_xy,
    )


def _circular_group_centers(values, group_count):
    values = _wrap01(values)
    if len(values) != group_count * group_count:
        raise ValueError(
            f"expected {group_count * group_count} values for circular grouping, got {len(values)}"
        )

    order = np.argsort(values)
    sorted_values = values[order]
    wrapped = np.concatenate([sorted_values, [sorted_values[0] + 1.0]])
    cut_after = int(np.argmax(np.diff(wrapped)))

    unwrapped_values = np.concatenate(
        [sorted_values[cut_after + 1 :], sorted_values[: cut_after + 1] + 1.0]
    )
    if len(unwrapped_values) != len(values):
        raise ValueError("unexpected unwrap length while grouping circular coordinates")

    points_per_group = len(values) // group_count
    centers = []
    for group_index in range(group_count):
        start = group_index * points_per_group
        stop = start + points_per_group
        centers.append(float(np.mean(unwrapped_values[start:stop])))
    return _wrap01(np.asarray(centers, dtype=float))


def _assignment_cost(actual_positions, ideal_positions, cell_xy):
    cost = np.zeros((len(actual_positions), len(ideal_positions)), dtype=float)
    for actual_index, actual in enumerate(actual_positions):
        for ideal_index, ideal in enumerate(ideal_positions):
            cost[actual_index, ideal_index] = _torus_distance(actual, ideal, cell_xy)
    return cost


def _axis_order_candidates(levels):
    levels = np.asarray(levels, dtype=float)
    candidates = []
    for reverse in (False, True):
        base = levels[::-1] if reverse else levels.copy()
        for shift in range(len(base)):
            candidates.append(np.roll(base, -shift))
    return candidates


def _build_pb_grid(pb_columns, cell_xy, *, grid_size, previous_pb_plot_grid=None):
    base_u_levels = _circular_group_centers(pb_columns[:, 0], group_count=grid_size)
    base_v_levels = _circular_group_centers(pb_columns[:, 1], group_count=grid_size)

    best = None
    u_candidates = _axis_order_candidates(base_u_levels) if previous_pb_plot_grid is not None else [base_u_levels]
    v_candidates = _axis_order_candidates(base_v_levels) if previous_pb_plot_grid is not None else [base_v_levels]

    for u_levels in u_candidates:
        for v_levels in v_candidates:
            ideal_nodes = []
            node_indices = []
            for i in range(grid_size):
                for j in range(grid_size):
                    ideal_nodes.append([u_levels[i], v_levels[j]])
                    node_indices.append((i, j))
            ideal_nodes = np.asarray(ideal_nodes, dtype=float)
            ideal_grid = ideal_nodes.reshape(grid_size, grid_size, 2)
            cost = _assignment_cost(pb_columns, ideal_nodes, cell_xy)
            rows, cols = linear_sum_assignment(cost)

            assigned = np.zeros((grid_size, grid_size, 2), dtype=float)
            assignment_cost = 0.0
            for row, col in zip(rows, cols):
                i, j = node_indices[col]
                assigned[i, j] = pb_columns[row]
                assignment_cost += float(cost[row, col])

            temporal_cost = 0.0
            if previous_pb_plot_grid is not None:
                for i in range(grid_size):
                    for j in range(grid_size):
                        temporal_cost += _torus_distance(
                            ideal_grid[i, j],
                            previous_pb_plot_grid[i, j],
                            cell_xy,
                        )

            candidate = {
                "u_levels": np.asarray(u_levels, dtype=float),
                "v_levels": np.asarray(v_levels, dtype=float),
                "assigned": assigned,
                "assignment_cost": assignment_cost,
                "temporal_cost": temporal_cost,
                "score": (1000.0 * temporal_cost + assignment_cost)
                if previous_pb_plot_grid is not None
                else assignment_cost,
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

    if best is None:
        raise StemAnalysisError(f"failed to build a consistent Pb {grid_size}x{grid_size} grid")

    u_spacing = [
        _torus_distance(
            [best["u_levels"][i], best["v_levels"][0]],
            [best["u_levels"][(i + 1) % grid_size], best["v_levels"][0]],
            cell_xy,
        )
        for i in range(grid_size)
    ]
    v_spacing = [
        _torus_distance(
            [best["u_levels"][0], best["v_levels"][j]],
            [best["u_levels"][0], best["v_levels"][(j + 1) % grid_size]],
            cell_xy,
        )
        for j in range(grid_size)
    ]
    max_reasonable_distance = 0.49 * min(u_spacing + v_spacing)
    ideal_nodes = np.zeros((grid_size, grid_size, 2), dtype=float)
    for i in range(grid_size):
        for j in range(grid_size):
            ideal_nodes[i, j] = [best["u_levels"][i], best["v_levels"][j]]
            if _torus_distance(best["assigned"][i, j], ideal_nodes[i, j], cell_xy) > max_reasonable_distance:
                raise StemAnalysisError(
                    "a Pb column was assigned too far from its ideal lattice node",
                    diagnostics={
                        "max_reasonable_distance_angstrom": max_reasonable_distance,
                        "assigned_pb_grid": best["assigned"].tolist(),
                        "ideal_pb_grid": ideal_nodes.tolist(),
                    },
                )

    return best["assigned"], best["u_levels"], best["v_levels"], max_reasonable_distance


def _torus_midpoint(projected_a, projected_b, cell_xy):
    unwrapped = _unwrap_around_reference([projected_a, projected_b], projected_a, cell_xy)
    return _wrap01(np.mean(unwrapped, axis=0))


def _circular_midpoints(levels):
    levels = np.asarray(levels, dtype=float)
    midpoints = []
    for index in range(len(levels)):
        previous = levels[(index - 1) % len(levels)]
        current = levels[index]
        delta = ((current - previous + 0.5) % 1.0) - 0.5
        midpoints.append((previous + 0.5 * delta) % 1.0)
    return np.asarray(midpoints, dtype=float)


def _ideal_pb_sites(u_levels, v_levels):
    grid_size = len(u_levels)
    pb_sites = np.zeros((grid_size, grid_size, 2), dtype=float)
    for i in range(grid_size):
        for j in range(grid_size):
            pb_sites[i, j] = [u_levels[i], v_levels[j]]
    return pb_sites


def _pb_cell_tiles(pb_plot_frac, u_levels, v_levels, cell_xy):
    grid_size = pb_plot_frac.shape[0]
    u_bounds = _circular_midpoints(u_levels)
    v_bounds = _circular_midpoints(v_levels)
    tiles = np.zeros((grid_size, grid_size, 4, 2), dtype=float)
    for i in range(grid_size):
        for j in range(grid_size):
            left = u_bounds[i]
            right = u_bounds[(i + 1) % grid_size]
            bottom = v_bounds[j]
            top = v_bounds[(j + 1) % grid_size]
            tile_frac = np.asarray(
                [
                    [left, bottom],
                    [right, bottom],
                    [right, top],
                    [left, top],
                ],
                dtype=float,
            )
            tile_frac = _unwrap_around_reference(tile_frac, pb_plot_frac[i, j], cell_xy)
            tiles[i, j] = _frac_to_xy(tile_frac, cell_xy)
    return tiles


def _site_grid_tiles(site_plot_frac, cell_xy):
    grid_size = site_plot_frac.shape[0]
    u_levels = site_plot_frac[:, 0, 0]
    v_levels = site_plot_frac[0, :, 1]
    u_bounds = _circular_midpoints(u_levels)
    v_bounds = _circular_midpoints(v_levels)
    tiles = np.zeros((grid_size, grid_size, 4, 2), dtype=float)
    for i in range(grid_size):
        for j in range(grid_size):
            left = u_bounds[i]
            right = u_bounds[(i + 1) % grid_size]
            bottom = v_bounds[j]
            top = v_bounds[(j + 1) % grid_size]
            tile_frac = np.asarray(
                [
                    [left, bottom],
                    [right, bottom],
                    [right, top],
                    [left, top],
                ],
                dtype=float,
            )
            tile_frac = _unwrap_around_reference(tile_frac, site_plot_frac[i, j], cell_xy)
            tiles[i, j] = _frac_to_xy(tile_frac, cell_xy)
    return tiles


def _ideal_zr_sites(pb_grid, cell_xy):
    grid_size = pb_grid.shape[0]
    sites = np.zeros((grid_size, grid_size, 2), dtype=float)
    for i in range(grid_size):
        for j in range(grid_size):
            corners = np.asarray(
                [
                    pb_grid[i, j],
                    pb_grid[(i + 1) % grid_size, j],
                    pb_grid[i, (j + 1) % grid_size],
                    pb_grid[(i + 1) % grid_size, (j + 1) % grid_size],
                ],
                dtype=float,
            )
            sites[i, j] = _torus_average(corners, cell_xy, reference=pb_grid[i, j])
    return sites


def _ideal_o_sites(pb_grid, cell_xy):
    grid_size = pb_grid.shape[0]
    oh = np.zeros((grid_size, grid_size, 2), dtype=float)
    ov = np.zeros((grid_size, grid_size, 2), dtype=float)
    for i in range(grid_size):
        for j in range(grid_size):
            oh[i, j] = _torus_midpoint(
                pb_grid[i, j],
                pb_grid[(i + 1) % grid_size, j],
                cell_xy,
            )
            ov[i, j] = _torus_midpoint(
                pb_grid[i, j],
                pb_grid[i, (j + 1) % grid_size],
                cell_xy,
            )
    return oh, ov


def _assign_to_grid(actual_positions, ideal_grid, cell_xy, label, max_reasonable_distance):
    flat_ideal = ideal_grid.reshape(-1, 2)
    cost = _assignment_cost(actual_positions, flat_ideal, cell_xy)
    rows, cols = linear_sum_assignment(cost)
    assigned = np.zeros_like(ideal_grid)
    for row, col in zip(rows, cols):
        assigned.reshape(-1, 2)[col] = actual_positions[row]
        if cost[row, col] > max_reasonable_distance:
            raise StemAnalysisError(
                f"a {label} column was assigned too far from its ideal topology site",
                diagnostics={
                    "label": label,
                    "max_reasonable_distance_angstrom": max_reasonable_distance,
                    "assigned_distance_angstrom": float(cost[row, col]),
                    "ideal_flat_index": int(col),
                },
            )
    return assigned


def _assign_edge_oxygen_columns(actual_positions, ideal_grid, cell_xy, max_reasonable_distance):
    flat_ideal = ideal_grid.reshape(-1, 2)
    if len(actual_positions) < len(flat_ideal):
        raise StemAnalysisError(
            "not enough oxygen-only columns were found to populate the Pb-edge topology",
            diagnostics={
                "oxygen_columns_found": len(actual_positions),
                "oxygen_columns_required": len(flat_ideal),
            },
        )

    cost = _assignment_cost(actual_positions, flat_ideal, cell_xy)
    rows, cols = linear_sum_assignment(cost)
    assigned = np.zeros_like(ideal_grid)
    selected_rows = []
    selected_costs = []
    for row, col in zip(rows, cols):
        assigned.reshape(-1, 2)[col] = actual_positions[row]
        selected_rows.append(int(row))
        selected_costs.append(float(cost[row, col]))
        if cost[row, col] > max_reasonable_distance:
            raise StemAnalysisError(
                "an oxygen-only column was assigned too far from its ideal Pb-edge site",
                diagnostics={
                    "label": "O-only",
                    "max_reasonable_distance_angstrom": max_reasonable_distance,
                    "assigned_distance_angstrom": float(cost[row, col]),
                    "ideal_flat_index": int(col),
                },
            )

    selected_rows = sorted(selected_rows)
    remaining_rows = [index for index in range(len(actual_positions)) if index not in set(selected_rows)]
    remaining_best_costs = [float(np.min(cost[index])) for index in remaining_rows]
    return assigned, {
        "selected_rows": selected_rows,
        "selected_costs_angstrom": selected_costs,
        "remaining_rows": remaining_rows,
        "remaining_best_edge_costs_angstrom": remaining_best_costs,
    }


def _localize_positions(projected_center, projected_neighbors, cell_xy):
    center_xy = _frac_to_xy(projected_center, cell_xy)
    local_positions = []
    local_vectors = []
    for neighbor in projected_neighbors:
        _, delta_xy, _ = _minimum_image_delta(neighbor, projected_center, cell_xy)
        local_vectors.append(delta_xy)
        local_positions.append(center_xy + delta_xy)
    return np.asarray(local_positions, dtype=float), np.asarray(local_vectors, dtype=float), center_xy


def _signed_angle_degrees(vector_xy, reference_xy):
    ref = np.asarray(reference_xy, dtype=float)
    vec = np.asarray(vector_xy, dtype=float)
    ref_norm = np.linalg.norm(ref)
    vec_norm = np.linalg.norm(vec)
    if ref_norm <= 1e-15 or vec_norm <= 1e-15:
        return 0.0
    ref = ref / ref_norm
    vec = vec / vec_norm
    cross = ref[0] * vec[1] - ref[1] * vec[0]
    dot = float(np.clip(np.dot(ref, vec), -1.0, 1.0))
    return float(np.degrees(np.arctan2(cross, dot)))


def _build_zr_polygons_and_tilts(
    zr_grid,
    oh_grid,
    ov_grid,
    cell_xy,
    zr_plot_grid=None,
    horizontal_pair_family=None,
):
    grid_size = zr_grid.shape[0]
    polygons = np.zeros((grid_size, grid_size, 4, 2), dtype=float)
    tilts = np.zeros((grid_size, grid_size), dtype=float)
    a_axis = np.asarray(cell_xy[0], dtype=float)
    pair_alignment = {"oh": 0.0, "ov": 0.0}
    pair_vectors = {}
    if zr_plot_grid is None:
        zr_plot_grid = zr_grid

    if horizontal_pair_family is None:
        for i in range(grid_size):
            for j in range(grid_size):
                oh_neighbors = np.asarray(
                    [oh_grid[i, j], oh_grid[i, (j + 1) % grid_size]],
                    dtype=float,
                )
                ov_neighbors = np.asarray(
                    [ov_grid[i, j], ov_grid[(i + 1) % grid_size, j]],
                    dtype=float,
                )
                oh_local, _, _ = _localize_positions(zr_grid[i, j], oh_neighbors, cell_xy)
                ov_local, _, _ = _localize_positions(zr_grid[i, j], ov_neighbors, cell_xy)
                pair_alignment["oh"] += abs(np.dot(oh_local[1] - oh_local[0], a_axis))
                pair_alignment["ov"] += abs(np.dot(ov_local[1] - ov_local[0], a_axis))
        horizontal_pair_family = "oh" if pair_alignment["oh"] >= pair_alignment["ov"] else "ov"

    for i in range(grid_size):
        for j in range(grid_size):
            neighbors = np.asarray(
                [
                    oh_grid[i, j],
                    oh_grid[i, (j + 1) % grid_size],
                    ov_grid[i, j],
                    ov_grid[(i + 1) % grid_size, j],
                ],
                dtype=float,
            )
            local_positions, local_vectors, center_xy = _localize_positions(zr_grid[i, j], neighbors, cell_xy)
            plot_center_xy = _frac_to_xy(zr_plot_grid[i, j], cell_xy)
            plot_positions = plot_center_xy + local_vectors
            angles = np.arctan2(
                plot_positions[:, 1] - plot_center_xy[1],
                plot_positions[:, 0] - plot_center_xy[0],
            )
            polygons[i, j] = plot_positions[np.argsort(angles)]

            pair_vectors["oh"] = local_positions[1] - local_positions[0]
            pair_vectors["ov"] = local_positions[3] - local_positions[2]
            pair_vector = np.asarray(pair_vectors[horizontal_pair_family], dtype=float)
            if np.dot(pair_vector, a_axis) < 0.0:
                pair_vector *= -1.0
            angle = _signed_angle_degrees(pair_vector, a_axis)
            tilts[i, j] = ((angle + 180.0) % 360.0) - 180.0

    return polygons, tilts, horizontal_pair_family


def _build_pb_displacements(pb_grid, oh_grid, ov_grid, cell_xy):
    grid_size = pb_grid.shape[0]
    displacements = np.zeros((grid_size, grid_size, 2), dtype=float)
    for i in range(grid_size):
        for j in range(grid_size):
            neighbors = np.asarray(
                [
                    oh_grid[(i - 1) % grid_size, j],
                    oh_grid[i, j],
                    ov_grid[i, (j - 1) % grid_size],
                    ov_grid[i, j],
                ],
                dtype=float,
            )
            local_positions, _, center_xy = _localize_positions(pb_grid[i, j], neighbors, cell_xy)
            cage_center = np.mean(local_positions, axis=0)
            displacements[i, j] = center_xy - cage_center
    mean_displacement = np.mean(displacements.reshape(-1, 2), axis=0)
    return displacements, displacements - mean_displacement


def analyze_projected_neb_image(
    atoms,
    *,
    frame_index=0,
    previous_pb_plot_grid=None,
    horizontal_pair_family=None,
    cutoff_angstrom=_COLUMN_CUTOFF_ANG,
    grid_size=_GRID_SIZE,
):
    """Analyze one ASE ``Atoms`` image in the projected ``ab`` plane."""

    (
        pb_columns,
        zr_columns,
        oxygen_columns,
        column_components,
        diagnostics,
        cell_xy,
    ) = _build_column_sets(atoms, cutoff_angstrom=cutoff_angstrom, grid_size=grid_size)

    pb_grid, u_levels, v_levels, max_reasonable_distance = _build_pb_grid(
        pb_columns,
        cell_xy,
        grid_size=grid_size,
        previous_pb_plot_grid=previous_pb_plot_grid,
    )
    current_pb_plot_grid = _ideal_pb_sites(u_levels, v_levels)
    pb_plot_grid = (
        np.asarray(previous_pb_plot_grid, dtype=float).copy()
        if previous_pb_plot_grid is not None
        else current_pb_plot_grid
    )
    zr_ideal = _ideal_zr_sites(pb_grid, cell_xy)
    zr_plot_grid = _ideal_zr_sites(pb_plot_grid, cell_xy)
    oh_ideal, ov_ideal = _ideal_o_sites(pb_grid, cell_xy)
    zr_grid = _assign_to_grid(
        zr_columns,
        zr_ideal,
        cell_xy,
        label="Zr",
        max_reasonable_distance=max_reasonable_distance,
    )
    oxygen_ideal = np.concatenate([oh_ideal.reshape(-1, 2), ov_ideal.reshape(-1, 2)], axis=0)
    oxygen_assigned, oxygen_selection = _assign_edge_oxygen_columns(
        oxygen_columns,
        oxygen_ideal.reshape(2 * grid_size, grid_size, 2),
        cell_xy,
        max_reasonable_distance=max_reasonable_distance,
    )
    oxygen_assigned = oxygen_assigned.reshape(2 * grid_size * grid_size, 2)
    oh_grid = oxygen_assigned[: grid_size * grid_size].reshape(grid_size, grid_size, 2)
    ov_grid = oxygen_assigned[grid_size * grid_size :].reshape(grid_size, grid_size, 2)

    zr_polygons_xy, zr_tilt_deg, horizontal_pair_family = _build_zr_polygons_and_tilts(
        zr_grid,
        oh_grid,
        ov_grid,
        cell_xy,
        zr_plot_grid=zr_plot_grid,
        horizontal_pair_family=horizontal_pair_family,
    )
    pb_displacement_xy, pb_displacement_minus_mean_xy = _build_pb_displacements(
        pb_grid,
        oh_grid,
        ov_grid,
        cell_xy,
    )
    plot_u_levels = pb_plot_grid[:, 0, 0]
    plot_v_levels = pb_plot_grid[0, :, 1]
    pb_tiles_xy = _pb_cell_tiles(pb_plot_grid, plot_u_levels, plot_v_levels, cell_xy)
    zr_tiles_xy = _site_grid_tiles(zr_plot_grid, cell_xy)

    diagnostics = dict(diagnostics)
    diagnostics["pb_u_levels"] = u_levels.tolist()
    diagnostics["pb_v_levels"] = v_levels.tolist()
    diagnostics["grid_size"] = int(grid_size)
    diagnostics["horizontal_pair_family"] = horizontal_pair_family
    diagnostics["max_assignment_distance_angstrom"] = max_reasonable_distance
    diagnostics["edge_oxygen_selection"] = {
        "selected_count": len(oxygen_selection["selected_rows"]),
        "ignored_count": len(oxygen_selection["remaining_rows"]),
        "selected_cost_min_angstrom": min(oxygen_selection["selected_costs_angstrom"])
        if oxygen_selection["selected_costs_angstrom"]
        else None,
        "selected_cost_median_angstrom": float(np.median(oxygen_selection["selected_costs_angstrom"]))
        if oxygen_selection["selected_costs_angstrom"]
        else None,
        "selected_cost_max_angstrom": max(oxygen_selection["selected_costs_angstrom"])
        if oxygen_selection["selected_costs_angstrom"]
        else None,
        "ignored_best_edge_cost_min_angstrom": min(oxygen_selection["remaining_best_edge_costs_angstrom"])
        if oxygen_selection["remaining_best_edge_costs_angstrom"]
        else None,
        "ignored_best_edge_cost_median_angstrom": float(np.median(oxygen_selection["remaining_best_edge_costs_angstrom"]))
        if oxygen_selection["remaining_best_edge_costs_angstrom"]
        else None,
        "ignored_best_edge_cost_max_angstrom": max(oxygen_selection["remaining_best_edge_costs_angstrom"])
        if oxygen_selection["remaining_best_edge_costs_angstrom"]
        else None,
    }

    return ProjectedFrameAnalysis(
        frame_index=int(frame_index),
        grid_size=int(grid_size),
        cell_xy=np.asarray(cell_xy, dtype=float),
        pb_frac=np.asarray(pb_grid, dtype=float),
        zr_frac=np.asarray(zr_grid, dtype=float),
        oh_frac=np.asarray(oh_grid, dtype=float),
        ov_frac=np.asarray(ov_grid, dtype=float),
        pb_plot_frac=np.asarray(pb_plot_grid, dtype=float),
        zr_plot_frac=np.asarray(zr_plot_grid, dtype=float),
        pb_xy=_frac_to_xy(pb_grid.reshape(-1, 2), cell_xy).reshape(grid_size, grid_size, 2),
        zr_xy=_frac_to_xy(zr_grid.reshape(-1, 2), cell_xy).reshape(grid_size, grid_size, 2),
        oh_xy=_frac_to_xy(oh_grid.reshape(-1, 2), cell_xy).reshape(grid_size, grid_size, 2),
        ov_xy=_frac_to_xy(ov_grid.reshape(-1, 2), cell_xy).reshape(grid_size, grid_size, 2),
        pb_plot_xy=_frac_to_xy(pb_plot_grid.reshape(-1, 2), cell_xy).reshape(grid_size, grid_size, 2),
        zr_plot_xy=_frac_to_xy(zr_plot_grid.reshape(-1, 2), cell_xy).reshape(grid_size, grid_size, 2),
        pb_tiles_xy=pb_tiles_xy,
        zr_tiles_xy=zr_tiles_xy,
        zr_polygons_xy=zr_polygons_xy,
        zr_tilt_deg=zr_tilt_deg,
        pb_displacement_xy=pb_displacement_xy,
        pb_displacement_minus_mean_xy=pb_displacement_minus_mean_xy,
        horizontal_pair_family=horizontal_pair_family,
        diagnostics=diagnostics,
        components=column_components,
    )


def _quiver_colors(vectors_xy, clip_magnitude_angstrom, reference_xy):
    from matplotlib.colors import hsv_to_rgb

    flat = vectors_xy.reshape(-1, 2)
    reference = np.asarray(reference_xy, dtype=float)
    colors = np.zeros((len(flat), 3), dtype=float)
    for index, vector in enumerate(flat):
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= 1e-15:
            colors[index] = 0.0
            continue
        angle = _signed_angle_degrees(vector, reference)
        hue = ((angle + 180.0) % 360.0) / 360.0
        normalized_magnitude = min(magnitude / clip_magnitude_angstrom, 1.0)
        colors[index] = hsv_to_rgb([hue, normalized_magnitude, normalized_magnitude])
    return colors.reshape(vectors_xy.shape[:-1] + (3,))


def _vector_angles_degrees(vectors_xy, reference_xy):
    flat = vectors_xy.reshape(-1, 2)
    angles = []
    for vector in flat:
        if np.linalg.norm(vector) <= 1e-15:
            angles.append(0.0)
        else:
            angles.append(_signed_angle_degrees(vector, reference_xy))
    return np.asarray(angles, dtype=float).reshape(vectors_xy.shape[:-1])


def _arrowhead_vertices(center_xy, angle_deg, length_xy, width_xy):
    angle = np.deg2rad(float(angle_deg))
    direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    center_xy = np.asarray(center_xy, dtype=float)
    tip = center_xy + 0.65 * length_xy * direction
    base_center = center_xy - 0.35 * length_xy * direction
    left = base_center + 0.5 * width_xy * normal
    right = base_center - 0.5 * width_xy * normal
    return np.asarray([tip, left, right], dtype=float)


def _cell_outline(cell_xy):
    return np.asarray(
        [
            [0.0, 0.0],
            cell_xy[0],
            cell_xy[0] + cell_xy[1],
            cell_xy[1],
            [0.0, 0.0],
        ],
        dtype=float,
    )


def render_projected_frame(analysis, output_path):
    """Render one analyzed frame to ``output_path``."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    grid_size = int(getattr(analysis, "grid_size", analysis.pb_plot_xy.shape[0]))
    outline = _cell_outline(analysis.cell_xy)
    a_axis = np.asarray(analysis.cell_xy[0], dtype=float)
    x_values = outline[:, 0].tolist()
    y_values = outline[:, 1].tolist()
    for polygon in analysis.zr_polygons_xy.reshape(-1, 4, 2):
        x_values.extend(polygon[:, 0].tolist())
        y_values.extend(polygon[:, 1].tolist())
    margin_x = 0.08 * max(x_values) if x_values else 0.5
    margin_y = 0.08 * max(y_values) if y_values else 0.5

    for axis in axes:
        axis.plot(outline[:, 0], outline[:, 1], color="0.2", linewidth=1.2)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(min(x_values) - margin_x, max(x_values) + margin_x)
        axis.set_ylim(min(y_values) - margin_y, max(y_values) + margin_y)
        axis.set_xticks([])
        axis.set_yticks([])

    axes[0].set_title("Zr-O Tilt")
    zr_tiles = analysis.zr_tiles_xy.reshape(-1, 4, 2)
    patches = [Polygon(points, closed=True) for points in zr_tiles]
    patch_values = np.clip(analysis.zr_tilt_deg.reshape(-1), -1.0, 1.0)
    collection = PatchCollection(
        patches,
        cmap="RdBu",
        edgecolor="0.25",
        linewidth=0.8,
    )
    collection.set_array(patch_values)
    collection.set_clim(-1.0, 1.0)
    axes[0].add_collection(collection)
    axes[0].scatter(
        analysis.zr_plot_xy[..., 0].reshape(-1),
        analysis.zr_plot_xy[..., 1].reshape(-1),
        s=12,
        color="black",
        zorder=3,
    )
    colorbar = fig.colorbar(collection, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("deg")

    pb_plot_xy = analysis.pb_plot_xy.reshape(-1, 2)
    pb_tiles = analysis.pb_tiles_xy.reshape(-1, 4, 2)
    raw_colors = _quiver_colors(
        analysis.pb_displacement_xy,
        clip_magnitude_angstrom=35.0 * _PM_TO_ANGSTROM,
        reference_xy=a_axis,
    ).reshape(-1, 3)
    raw_angles = _vector_angles_degrees(analysis.pb_displacement_xy, a_axis).reshape(-1)
    axes[1].set_title("Pb Displacement")
    raw_tile_collection = PatchCollection(
        [Polygon(tile, closed=True) for tile in pb_tiles],
        facecolor=raw_colors,
        edgecolor="none",
        linewidth=0.0,
        zorder=1,
    )
    axes[1].add_collection(raw_tile_collection)
    tile_scale = min(np.linalg.norm(analysis.cell_xy[0]), np.linalg.norm(analysis.cell_xy[1])) / float(grid_size)
    arrow_length = 0.22 * tile_scale
    arrow_width = 0.12 * tile_scale
    for (x_coord, y_coord), angle_deg in zip(pb_plot_xy, raw_angles):
        axes[1].add_patch(
            Polygon(
                _arrowhead_vertices((x_coord, y_coord), angle_deg, arrow_length, arrow_width),
                closed=True,
                facecolor="white",
                edgecolor="white",
                linewidth=0.0,
                zorder=3,
            )
        )

    mean_sub_colors = _quiver_colors(
        analysis.pb_displacement_minus_mean_xy,
        clip_magnitude_angstrom=3.0 * _PM_TO_ANGSTROM,
        reference_xy=a_axis,
    ).reshape(-1, 3)
    mean_sub_angles = _vector_angles_degrees(
        analysis.pb_displacement_minus_mean_xy,
        a_axis,
    ).reshape(-1)
    axes[2].set_title("Pb Displacement - Mean")
    mean_tile_collection = PatchCollection(
        [Polygon(tile, closed=True) for tile in pb_tiles],
        facecolor=mean_sub_colors,
        edgecolor="none",
        linewidth=0.0,
        zorder=1,
    )
    axes[2].add_collection(mean_tile_collection)
    for (x_coord, y_coord), angle_deg in zip(pb_plot_xy, mean_sub_angles):
        axes[2].add_patch(
            Polygon(
                _arrowhead_vertices((x_coord, y_coord), angle_deg, arrow_length, arrow_width),
                closed=True,
                facecolor="white",
                edgecolor="white",
                linewidth=0.0,
                zorder=3,
            )
        )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _format_diagnostics_message(error, frame_index):
    lines = [
        f"Projected STEM analysis failed for frame {frame_index}: {error}",
    ]
    diagnostics = getattr(error, "diagnostics", {}) or {}
    if diagnostics:
        lines.append("")
        lines.append("Diagnostics:")
        for key, value in diagnostics.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _failed_stem_sequence_result(
    *,
    xyz_path,
    output_dir,
    frame_dir,
    diagnostics_path,
    emit_diagnostics,
    frame_index,
    frames_rendered=0,
    error,
):
    if emit_diagnostics:
        diagnostics_path.write_text(
            _format_diagnostics_message(error, frame_index),
            encoding="utf-8",
        )
    return {
        "status": "failed",
        "xyz_file": str(xyz_path),
        "output_dir": str(output_dir),
        "frame_dir": str(frame_dir),
        "frames_rendered": int(frames_rendered),
        "diagnostics_file": str(diagnostics_path),
    }


def _failed_direct_stem_sequence_result(
    *,
    frame_dir,
    diagnostics_path,
    emit_diagnostics,
    frame_index,
    frames_rendered=0,
    error,
):
    if emit_diagnostics:
        diagnostics_path.write_text(
            _format_diagnostics_message(error, frame_index),
            encoding="utf-8",
        )
    return {
        "status": "failed",
        "frames_rendered": int(frames_rendered),
        "frame_dir": str(frame_dir),
        "diagnostics_file": str(diagnostics_path),
    }


def _analyze_projected_sequence(images, *, cutoff_angstrom, grid_size):
    analyses = []
    anchor_pb_plot_grid = None
    horizontal_pair_family = None
    for frame_index, atoms in enumerate(images):
        analysis = analyze_projected_neb_image(
            atoms,
            frame_index=frame_index,
            previous_pb_plot_grid=anchor_pb_plot_grid,
            horizontal_pair_family=horizontal_pair_family,
            cutoff_angstrom=cutoff_angstrom,
            grid_size=grid_size,
        )
        if anchor_pb_plot_grid is None:
            anchor_pb_plot_grid = analysis.pb_plot_frac.copy()
        horizontal_pair_family = analysis.horizontal_pair_family
        analyses.append(analysis)
    return analyses


def _write_npz_analysis(analyses, output_dir):
    output_dir = Path(output_dir)
    npz_path = output_dir / "stem_analysis.npz"
    metadata_path = output_dir / "stem_analysis_metadata.json"
    np.savez(
        npz_path,
        frame_indices=np.asarray([analysis.frame_index for analysis in analyses], dtype=int),
        cell_xy=np.asarray([analysis.cell_xy for analysis in analyses], dtype=float),
        pb_frac=np.asarray([analysis.pb_frac for analysis in analyses], dtype=float),
        zr_frac=np.asarray([analysis.zr_frac for analysis in analyses], dtype=float),
        oh_frac=np.asarray([analysis.oh_frac for analysis in analyses], dtype=float),
        ov_frac=np.asarray([analysis.ov_frac for analysis in analyses], dtype=float),
        pb_xy=np.asarray([analysis.pb_xy for analysis in analyses], dtype=float),
        zr_xy=np.asarray([analysis.zr_xy for analysis in analyses], dtype=float),
        oh_xy=np.asarray([analysis.oh_xy for analysis in analyses], dtype=float),
        ov_xy=np.asarray([analysis.ov_xy for analysis in analyses], dtype=float),
        zr_tilt_deg=np.asarray([analysis.zr_tilt_deg for analysis in analyses], dtype=float),
        pb_displacement_xy=np.asarray([analysis.pb_displacement_xy for analysis in analyses], dtype=float),
        pb_displacement_minus_mean_xy=np.asarray(
            [analysis.pb_displacement_minus_mean_xy for analysis in analyses],
            dtype=float,
        ),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "units": {
                    "cartesian_coordinates": "angstrom",
                    "projected_fractional": "fractional_ab",
                    "zr_tilt_deg": "degrees",
                    "polar_displacements": "angstrom",
                },
                "grid_size": [analysis.grid_size for analysis in analyses],
                "frame_diagnostics": [analysis.diagnostics for analysis in analyses],
                "horizontal_pair_family": [analysis.horizontal_pair_family for analysis in analyses],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(npz_path), str(metadata_path)


def analyze_stem_sequence_from_xyz(
    xyz_file,
    *,
    output_dir=None,
    iteration=None,
    emit_png=True,
    emit_gif=True,
    emit_npy=True,
    emit_diagnostics=True,
    gif_duration_seconds=0.4,
    cutoff_angstrom=_COLUMN_CUTOFF_ANG,
    grid_size=_GRID_SIZE,
):
    """Analyze a saved XYZ path and emit requested STEM artifacts."""

    xyz_path = Path(xyz_file).expanduser().resolve()
    destination = (
        xyz_path.parent / f"{xyz_path.stem}_stem"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    if iteration is not None:
        destination = destination / f"stem_iter_{int(iteration):04d}"
    destination.mkdir(parents=True, exist_ok=True)
    frame_dir = destination / "frames"
    gif_path = destination / "stem.gif"
    diagnostics_path = destination / "diagnostics.txt"

    try:
        images = read(str(xyz_path), ":")
        analyses = _analyze_projected_sequence(images, cutoff_angstrom=cutoff_angstrom, grid_size=grid_size)
    except StemAnalysisError as error:
        return _failed_stem_sequence_result(
            xyz_path=xyz_path,
            output_dir=destination,
            frame_dir=frame_dir,
            diagnostics_path=diagnostics_path,
            emit_diagnostics=emit_diagnostics,
            frame_index=getattr(error, "frame_index", 0),
            error=error,
        )
    except Exception as error:
        return _failed_stem_sequence_result(
            xyz_path=xyz_path,
            output_dir=destination,
            frame_dir=frame_dir,
            diagnostics_path=diagnostics_path,
            emit_diagnostics=emit_diagnostics,
            frame_index=0,
            error=StemAnalysisError(
                f"{type(error).__name__}: {error}",
                diagnostics={"exception_type": type(error).__name__},
            ),
        )

    result = {
        "status": "ok",
        "xyz_file": str(xyz_path),
        "output_dir": str(destination),
        "frame_dir": str(frame_dir),
        "frames_rendered": len(analyses),
        "grid_size": int(grid_size),
    }
    if emit_png:
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        try:
            for index, analysis in enumerate(analyses):
                frame_index = int(getattr(analysis, "frame_index", index))
                frame_path = frame_dir / f"frame_{frame_index:04d}.png"
                render_projected_frame(analysis, frame_path)
                frame_paths.append(str(frame_path))
        except Exception as error:
            return _failed_stem_sequence_result(
                xyz_path=xyz_path,
                output_dir=destination,
                frame_dir=frame_dir,
                diagnostics_path=diagnostics_path,
                emit_diagnostics=emit_diagnostics,
                frame_index=frame_index,
                frames_rendered=len(frame_paths),
                error=StemAnalysisError(
                    f"{type(error).__name__}: {error}",
                    diagnostics={"exception_type": type(error).__name__},
                ),
            )
        result["frames"] = frame_paths
    if emit_gif:
        try:
            import imageio.v2 as imageio
        except Exception as error:
            return _failed_stem_sequence_result(
                xyz_path=xyz_path,
                output_dir=destination,
                frame_dir=frame_dir,
                diagnostics_path=diagnostics_path,
                emit_diagnostics=emit_diagnostics,
                frame_index=0,
                frames_rendered=len(result.get("frames", [])),
                error=StemAnalysisError(
                    f"{type(error).__name__}: {error}",
                    diagnostics={"exception_type": type(error).__name__},
                ),
            )
        if "frames" not in result:
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_paths = []
            try:
                for index, analysis in enumerate(analyses):
                    frame_index = int(getattr(analysis, "frame_index", index))
                    frame_paths.append(
                        render_projected_frame(
                            analysis,
                            frame_dir / f"frame_{frame_index:04d}.png",
                        )
                    )
            except Exception as error:
                return _failed_stem_sequence_result(
                    xyz_path=xyz_path,
                    output_dir=destination,
                    frame_dir=frame_dir,
                    diagnostics_path=diagnostics_path,
                    emit_diagnostics=emit_diagnostics,
                    frame_index=frame_index,
                    frames_rendered=len(frame_paths),
                    error=StemAnalysisError(
                        f"{type(error).__name__}: {error}",
                        diagnostics={"exception_type": type(error).__name__},
                    ),
                )
            result["frames"] = frame_paths
        try:
            gif_frames = [imageio.imread(frame_path) for frame_path in result["frames"]]
            imageio.mimsave(gif_path, gif_frames, duration=gif_duration_seconds)
        except Exception as error:
            return _failed_stem_sequence_result(
                xyz_path=xyz_path,
                output_dir=destination,
                frame_dir=frame_dir,
                diagnostics_path=diagnostics_path,
                emit_diagnostics=emit_diagnostics,
                frame_index=0,
                frames_rendered=len(result.get("frames", [])),
                error=StemAnalysisError(
                    f"{type(error).__name__}: {error}",
                    diagnostics={"exception_type": type(error).__name__},
                ),
            )
        result["gif"] = str(gif_path)
    if emit_npy:
        npz_path, metadata_path = _write_npz_analysis(analyses, destination)
        result["npz"] = npz_path
        result["metadata"] = metadata_path
    if diagnostics_path.exists():
        diagnostics_path.unlink()
    return result


def save_projected_neb_sequence(
    images,
    *,
    xyz_dir,
    iteration,
    emit_png=True,
    emit_gif=True,
    emit_npy=False,
    emit_diagnostics=True,
    gif_duration_seconds=0.4,
    cutoff_angstrom=_COLUMN_CUTOFF_ANG,
    grid_size=_GRID_SIZE,
):
    """Analyze and render one saved NEB sequence.

    The output layout is:

    - ``<xyz_dir>/stem_iter_XXXX/frame_0000.png``
    - ``<xyz_dir>/stem_iter_XXXX.gif``

    If any frame fails validation, the helper writes a diagnostics text file and
    stops rendering that sequence instead of generating misleading plots.
    """

    xyz_dir = Path(xyz_dir)
    xyz_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = xyz_dir / f"stem_iter_{int(iteration):04d}"
    gif_path = xyz_dir / f"stem_iter_{int(iteration):04d}.gif"
    diagnostics_path = xyz_dir / f"stem_iter_{int(iteration):04d}_diagnostics.txt"

    try:
        analyses = _analyze_projected_sequence(images, cutoff_angstrom=cutoff_angstrom, grid_size=grid_size)
    except StemAnalysisError as error:
        if emit_diagnostics:
            diagnostics_path.write_text(
                _format_diagnostics_message(error, 0),
                encoding="utf-8",
            )
        return {
            "status": "failed",
            "frames_rendered": 0,
            "frame_dir": str(frame_dir),
            "diagnostics_file": str(diagnostics_path),
        }
    except Exception as error:
        if emit_diagnostics:
            diagnostics_path.write_text(
                _format_diagnostics_message(
                    StemAnalysisError(
                        f"{type(error).__name__}: {error}",
                        diagnostics={"exception_type": type(error).__name__},
                    ),
                    0,
                ),
                encoding="utf-8",
            )
        return {
            "status": "failed",
            "frames_rendered": 0,
            "frame_dir": str(frame_dir),
            "diagnostics_file": str(diagnostics_path),
        }

    result = {
        "status": "ok",
        "frames_rendered": len(analyses),
        "frame_dir": str(frame_dir),
        "grid_size": int(grid_size),
    }
    if emit_png:
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        try:
            for index, analysis in enumerate(analyses):
                frame_index = int(getattr(analysis, "frame_index", index))
                frame_path = frame_dir / f"frame_{frame_index:04d}.png"
                render_projected_frame(analysis, frame_path)
                frame_paths.append(str(frame_path))
        except Exception as error:
            return _failed_direct_stem_sequence_result(
                frame_dir=frame_dir,
                diagnostics_path=diagnostics_path,
                emit_diagnostics=emit_diagnostics,
                frame_index=frame_index,
                frames_rendered=len(frame_paths),
                error=StemAnalysisError(
                    f"{type(error).__name__}: {error}",
                    diagnostics={"exception_type": type(error).__name__},
                ),
            )
        result["frames"] = frame_paths
    if emit_gif:
        try:
            import imageio.v2 as imageio
        except Exception as error:
            return _failed_direct_stem_sequence_result(
                frame_dir=frame_dir,
                diagnostics_path=diagnostics_path,
                emit_diagnostics=emit_diagnostics,
                frame_index=0,
                frames_rendered=len(result.get("frames", [])),
                error=StemAnalysisError(
                    f"{type(error).__name__}: {error}",
                    diagnostics={"exception_type": type(error).__name__},
                ),
            )
        if "frames" not in result:
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_paths = []
            try:
                for index, analysis in enumerate(analyses):
                    frame_index = int(getattr(analysis, "frame_index", index))
                    frame_paths.append(
                        render_projected_frame(
                            analysis,
                            frame_dir / f"frame_{frame_index:04d}.png",
                        )
                    )
            except Exception as error:
                return _failed_direct_stem_sequence_result(
                    frame_dir=frame_dir,
                    diagnostics_path=diagnostics_path,
                    emit_diagnostics=emit_diagnostics,
                    frame_index=frame_index,
                    frames_rendered=len(frame_paths),
                    error=StemAnalysisError(
                        f"{type(error).__name__}: {error}",
                        diagnostics={"exception_type": type(error).__name__},
                    ),
                )
            result["frames"] = frame_paths
        try:
            gif_frames = [imageio.imread(frame_path) for frame_path in result["frames"]]
            imageio.mimsave(gif_path, gif_frames, duration=gif_duration_seconds)
        except Exception as error:
            return _failed_direct_stem_sequence_result(
                frame_dir=frame_dir,
                diagnostics_path=diagnostics_path,
                emit_diagnostics=emit_diagnostics,
                frame_index=0,
                frames_rendered=len(result.get("frames", [])),
                error=StemAnalysisError(
                    f"{type(error).__name__}: {error}",
                    diagnostics={"exception_type": type(error).__name__},
                ),
            )
        result["gif"] = str(gif_path)
    if emit_npy:
        frame_dir.mkdir(parents=True, exist_ok=True)
        npz_path, metadata_path = _write_npz_analysis(analyses, frame_dir)
        result["npz"] = npz_path
        result["metadata"] = metadata_path
    if diagnostics_path.exists():
        diagnostics_path.unlink()
    return result
