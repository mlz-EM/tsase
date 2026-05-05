"""Climbing-image selection helpers for SSNEB."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class ClimbingImageSelection:
    """Manual climbing-image selector.

    Exact indices climb directly. Ranges are resolved at runtime by choosing the
    highest-energy unfrozen intermediate image in each requested range.
    """

    indices: tuple[int, ...] = ()
    ranges: tuple[tuple[int, int], ...] = ()

    @property
    def enabled(self):
        return bool(self.indices or self.ranges)

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

        indices, ranges = _normalize_selection(normalized["selection"], num_images)
        return cls(indices=indices, ranges=ranges)

    def to_config(self):
        if not self.enabled:
            return {"enabled": False}
        if self.indices:
            return {"enabled": True, "indices": list(self.indices)}
        return {"enabled": True, "ranges": [[start, end] for start, end in self.ranges]}

    def resolve(self, frozen_images, auto_select):
        if self.indices:
            return self.indices

        indices = []
        for start, end in self.ranges:
            ci_index = auto_select(range(start, end + 1), frozen_images=frozen_images)
            if ci_index is not None and ci_index not in indices:
                indices.append(ci_index)
        return tuple(indices)


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

    provided = [key for key in ("indices", "ranges", "selection") if key in config]
    if len(provided) != 1:
        raise ValueError("climbing_images must provide exactly one of: indices, ranges, selection")
    key = provided[0]
    values = list(config[key] or [])
    if not values:
        raise ValueError(f"climbing_images.{key} must not be empty when enabled")
    return {"enabled": True, key: values}


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
