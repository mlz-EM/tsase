"""Optimizers for SSNEB bands."""

from inspect import signature

from .base import minimizer_ssneb
from .bfgs import bfgs_ssneb
from .fire import fire_ssneb
from .mdmin import mdmin_ssneb
from .qm import qm_ssneb

_OPTIMIZER_TYPES = {
    "fire": fire_ssneb,
    "qm": qm_ssneb,
    "mdmin": mdmin_ssneb,
    "bfgs": bfgs_ssneb,
}

_OPTIMIZER_ALIASES = {
    "fire": "fire",
    "qm": "qm",
    "quickmin": "qm",
    "quick-min": "qm",
    "quick_min": "qm",
    "mdmin": "mdmin",
    "md-min": "mdmin",
    "md_min": "mdmin",
    "bfgs": "bfgs",
    "bgfs": "bfgs",
}

_OPTIMIZER_DEFAULTS = {
    "fire": {"maxmove": 0.1, "dt": 0.1, "dtmax": 0.1},
    "qm": {"maxmove": 0.2, "dt": 0.05},
    "mdmin": {"maxmove": 0.1, "dt": 0.2},
    "bfgs": {"maxmove": 0.04, "alpha": 70.0},
}


def normalize_optimizer_kind(kind):
    normalized = str(kind or "fire").strip().lower()
    canonical = _OPTIMIZER_ALIASES.get(normalized)
    if canonical is None:
        choices = ", ".join(sorted(_OPTIMIZER_TYPES))
        raise ValueError(f"optimizer.kind must be one of: {choices}")
    return canonical


def get_optimizer_class(kind):
    return _OPTIMIZER_TYPES[normalize_optimizer_kind(kind)]


def get_optimizer_parameters(kind):
    cls = get_optimizer_class(kind)
    parameters = []
    for name in signature(cls.__init__).parameters:
        if name in {"self", "path", "band"}:
            continue
        parameters.append(name)
    return tuple(parameters)


def build_optimizer_kwargs(kind, optimizer_config, *, energy_profile_entries):
    canonical = normalize_optimizer_kind(kind)
    runtime = {
        key: value
        for key, value in dict(optimizer_config or {}).items()
        if key
        not in {
            "kind",
            "convergence",
            "ci_activation",
        }
    }
    accepted = set(get_optimizer_parameters(canonical))
    unknown = sorted(set(runtime) - accepted)
    if unknown:
        raise ValueError(
            f"unsupported settings for optimizer {canonical!r}: {', '.join(unknown)}"
        )

    kwargs = dict(_OPTIMIZER_DEFAULTS.get(canonical, {}))
    kwargs.update(runtime)
    if canonical == "fire" and "dtmax" not in runtime:
        kwargs["dtmax"] = kwargs.get("dt", _OPTIMIZER_DEFAULTS["fire"]["dtmax"])
    kwargs["energy_profile_entries"] = list(energy_profile_entries or [])
    return kwargs


def create_optimizer(kind, band, **optimizer_kwargs):
    optimizer_class = get_optimizer_class(kind)
    return optimizer_class(band, **optimizer_kwargs)
