"""Climbing-image selection helpers for SSNEB."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ClimbingImageSelection:
    """Climbing-image selector.

    Exact indices climb directly. Ranges are resolved at runtime by choosing the
    highest-energy unfrozen intermediate image in each requested range. Auto
    mode picks the strongest separated local peaks and then tracks them locally.
    """

    indices: tuple[int, ...] = ()
    ranges: tuple[tuple[int, int], ...] = ()
    auto_count: int = 0
    auto_min_separation: int = 1
    auto_search_radius: int = 2
    auto_prominence: float = 0.0
    auto_update_interval: int = 1
    _tracked_indices: tuple[int, ...] = ()
    _last_update: int | None = None

    @property
    def enabled(self):
        return bool(self.indices or self.ranges or self.auto_count > 0)

    @classmethod
    def from_config(cls, config, *, num_images, enabled_default=True):
        normalized = normalize_climbing_images_config(
            config,
            enabled_default=enabled_default,
        )
        if normalized is None:
            return cls()

        if "indices" in normalized:
            return cls(indices=_normalize_indices(normalized["indices"], num_images))
        if "ranges" in normalized:
            return cls(ranges=_normalize_ranges(normalized["ranges"], num_images))
        if normalized.get("mode") == "auto":
            count = int(normalized["count"])
            min_separation = normalized.get("min_separation")
            if min_separation is None:
                min_separation = max(1, int(num_images) // max(1, count * 4))
            return cls(
                auto_count=count,
                auto_min_separation=max(1, int(min_separation)),
                auto_search_radius=max(1, int(normalized.get("search_radius", 2))),
                auto_prominence=max(0.0, float(normalized.get("prominence", 0.0))),
                auto_update_interval=max(1, int(normalized.get("update_interval", 1))),
            )

        indices, ranges = _normalize_selection(normalized["selection"], num_images)
        return cls(indices=indices, ranges=ranges)

    def to_config(self):
        if not self.enabled:
            return {"enabled": False}
        if self.indices:
            return {"enabled": True, "indices": list(self.indices)}
        if self.auto_count > 0:
            return {
                "enabled": True,
                "mode": "auto",
                "count": int(self.auto_count),
                "min_separation": int(self.auto_min_separation),
                "search_radius": int(self.auto_search_radius),
                "prominence": float(self.auto_prominence),
                "update_interval": int(self.auto_update_interval),
            }
        return {"enabled": True, "ranges": [[start, end] for start, end in self.ranges]}

    def resolve(self, path, frozen_images, auto_select, *, iteration=None):
        if self.indices:
            return self.indices

        if self.auto_count > 0:
            if (
                iteration is not None
                and self._tracked_indices
                and self._last_update is not None
                and int(iteration) - self._last_update < self.auto_update_interval
            ):
                return self._tracked_indices
            tracked = self._resolve_auto_peaks(path, frozen_images)
            object.__setattr__(self, "_tracked_indices", tracked)
            object.__setattr__(
                self,
                "_last_update",
                None if iteration is None else int(iteration),
            )
            return tracked

        indices = []
        for start, end in self.ranges:
            ci_index = auto_select(range(start, end + 1), frozen_images=frozen_images)
            if ci_index is not None and ci_index not in indices:
                indices.append(ci_index)
        return tuple(indices)

    def _resolve_auto_peaks(self, path, frozen_images):
        if self._tracked_indices:
            tracked = []
            for index in self._tracked_indices:
                start = max(1, int(index) - self.auto_search_radius)
                end = min(len(path) - 2, int(index) + self.auto_search_radius)
                peaks = _ranked_local_peaks(
                    path,
                    range(start, end + 1),
                    frozen_images=frozen_images,
                    prominence=self.auto_prominence,
                )
                if peaks:
                    candidate = peaks[0]
                else:
                    candidate = _highest_energy_image(path, range(start, end + 1), frozen_images)
                if candidate is not None:
                    tracked.append(candidate)
            selected = _select_separated_peaks(
                tracked,
                path,
                count=self.auto_count,
                min_separation=self.auto_min_separation,
                preserve_order=True,
            )
            if len(selected) == self.auto_count:
                return selected
            global_candidates = _ranked_local_peaks(
                path,
                range(1, len(path) - 1),
                frozen_images=frozen_images,
                prominence=self.auto_prominence,
            )
            if len(global_candidates) < self.auto_count:
                for index in _ranked_images(path, range(1, len(path) - 1), frozen_images=frozen_images):
                    if index not in global_candidates:
                        global_candidates.append(index)
            return _fill_separated_peaks(
                selected,
                global_candidates,
                path,
                count=self.auto_count,
                min_separation=self.auto_min_separation,
            )
        return _select_auto_peaks(
            path,
            frozen_images=frozen_images,
            count=self.auto_count,
            min_separation=self.auto_min_separation,
            prominence=self.auto_prominence,
            prefer_indices=self._tracked_indices,
            search_radius=self.auto_search_radius,
        )


def normalize_climbing_images_config(config, *, enabled_default):
    """Normalize user-facing climbing-image config without image-count checks."""

    if config is None or config is False:
        return None
    if config is True:
        raise ValueError("climbing_images=true requires indices, ranges, or selection")
    if not isinstance(config, dict):
        config = {"selection": config}

    config = dict(config)
    if not bool(config.get("enabled", enabled_default)):
        return None

    mode = str(config.get("mode", "")).strip().lower()
    provided = [key for key in ("indices", "ranges", "selection") if key in config]
    if mode in {"auto", "auto_peaks", "peaks"} or "count" in config:
        if provided:
            raise ValueError("auto climbing_images cannot also provide indices, ranges, or selection")
        count = int(config.get("count", 1))
        if count <= 0:
            raise ValueError("climbing_images.count must be positive for auto mode")
        normalized = {"enabled": True, "mode": "auto", "count": count}
        for key in ("min_separation", "search_radius", "prominence", "update_interval"):
            if key in config:
                normalized[key] = config[key]
        return normalized
    if len(provided) != 1:
        raise ValueError(
            "climbing_images must provide exactly one of: indices, ranges, selection, "
            "or mode: auto with count"
        )
    key = provided[0]
    values = list(config[key] or [])
    if not values:
        raise ValueError(f"climbing_images.{key} must not be empty when enabled")
    return {"enabled": True, key: values}


def _select_auto_peaks(
    path,
    *,
    frozen_images,
    count,
    min_separation,
    prominence,
    prefer_indices=(),
    search_radius=2,
):
    peaks = _ranked_local_peaks(
        path,
        range(1, len(path) - 1),
        frozen_images=frozen_images,
        prominence=prominence,
    )
    if len(peaks) < count:
        ranked = _ranked_images(path, range(1, len(path) - 1), frozen_images=frozen_images)
        for index in ranked:
            if index not in peaks:
                peaks.append(index)
    if prefer_indices:
        local = [
            index
            for index in peaks
            if min(abs(index - previous) for previous in prefer_indices) <= int(search_radius)
        ]
        remote = [index for index in peaks if index not in local]
        peaks = local + remote
    return _select_separated_peaks(
        peaks,
        path,
        count=count,
        min_separation=min_separation,
        preserve_order=bool(prefer_indices),
    )


def _ranked_local_peaks(path, candidate_indices, *, frozen_images, prominence):
    peaks = []
    for index in candidate_indices:
        energy = _image_energy(path, index)
        if index in frozen_images or energy is None:
            continue
        left = _image_energy(path, index - 1)
        right = _image_energy(path, index + 1)
        left = energy if left is None else left
        right = energy if right is None else right
        if energy < left or energy < right:
            continue
        if prominence > 0.0 and energy - max(left, right) < float(prominence):
            continue
        peaks.append(index)
    return sorted(peaks, key=lambda index: _image_energy(path, index), reverse=True)


def _ranked_images(path, candidate_indices, *, frozen_images):
    return sorted(
        [
            index
            for index in candidate_indices
            if index not in frozen_images and _image_energy(path, index) is not None
        ],
        key=lambda index: _image_energy(path, index),
        reverse=True,
    )


def _highest_energy_image(path, candidate_indices, frozen_images):
    ranked = _ranked_images(path, candidate_indices, frozen_images=frozen_images)
    return None if not ranked else ranked[0]


def _fill_separated_peaks(seed_indices, candidates, path, *, count, min_separation):
    selected = list(seed_indices)
    for index in sorted(
        [candidate for candidate in dict.fromkeys(candidates) if candidate not in selected],
        key=lambda candidate: _image_energy(path, candidate),
        reverse=True,
    ):
        if all(abs(index - previous) >= int(min_separation) for previous in selected):
            selected.append(index)
        if len(selected) == int(count):
            break
    return tuple(sorted(selected))


def _select_separated_peaks(candidates, path, *, count, min_separation, preserve_order=False):
    selected = []
    ranked_candidates = list(dict.fromkeys(candidates))
    if not preserve_order:
        ranked_candidates = sorted(
            ranked_candidates,
            key=lambda candidate: _image_energy(path, candidate),
            reverse=True,
        )
    for index in ranked_candidates:
        if all(abs(index - previous) >= int(min_separation) for previous in selected):
            selected.append(index)
        if len(selected) == int(count):
            break
    return tuple(sorted(selected))


def _image_energy(path, index):
    value = getattr(path[index], "u", None)
    if value is None:
        return None
    return float(value)


def _normalize_selection(selection, num_images):
    values = list(selection or [])
    if not values:
        raise ValueError("climbing_images.selection must not be empty when enabled")
    if all(isinstance(value, Integral) for value in values):
        return _normalize_indices(values, num_images), ()
    return (), _normalize_ranges(values, num_images)


def _normalize_indices(indices, num_images):
    normalized = []
    for value in list(indices or []):
        index = int(value)
        if index <= 0 or index >= int(num_images) - 1:
            raise ValueError("climbing image indices must refer to intermediate images")
        if index not in normalized:
            normalized.append(index)
    if not normalized:
        raise ValueError("climbing image indices must not be empty when enabled")
    return tuple(normalized)


def _normalize_ranges(ranges, num_images):
    normalized = []
    for entry in list(ranges or []):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("climbing image ranges must be [start, end] index pairs")
        start = int(entry[0])
        end = int(entry[1])
        if start > end:
            raise ValueError("climbing image range start must be <= end")
        if end < 1 or start > int(num_images) - 2:
            raise ValueError("climbing image ranges must include at least one intermediate image")
        normalized.append((max(1, start), min(int(num_images) - 2, end)))
    if not normalized:
        raise ValueError("climbing image ranges must not be empty when enabled")
    return tuple(normalized)
