"""Runtime helpers for optional heavy dependencies used by the examples."""

import os
from pathlib import Path
import site
import sys


def _user_site_paths():
    try:
        paths = site.getusersitepackages()
    except Exception:
        return []
    if isinstance(paths, str):
        paths = [paths]
    normalized = []
    for entry in paths:
        try:
            normalized.append(str(Path(entry).expanduser().resolve()))
        except Exception:
            normalized.append(str(entry))
    return normalized


def _clear_import_state():
    for name in list(sys.modules):
        if name == "mace" or name.startswith("mace."):
            sys.modules.pop(name, None)
        if name == "torch" or name.startswith("torch."):
            sys.modules.pop(name, None)


def _remove_user_site_from_sys_path():
    user_sites = _user_site_paths()
    if not user_sites:
        return []

    removed = []
    kept = []
    for entry in sys.path:
        try:
            resolved = str(Path(entry).expanduser().resolve())
        except Exception:
            resolved = str(entry)
        if resolved in user_sites:
            removed.append(entry)
        else:
            kept.append(entry)
    if removed:
        sys.path[:] = kept
        _clear_import_state()
    return removed


def _read_int_env(*names):
    for name in names:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return int(str(raw).strip())
        except ValueError:
            continue
    return None


def detect_world_size(default=1):
    """Return the distributed world size from common MPI/Slurm env vars."""

    value = _read_int_env(
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS",
        "WORLD_SIZE",
    )
    return int(default if value is None else max(1, value))


def detect_local_rank(default=0):
    """Return the rank index within the current node from common env vars."""

    value = _read_int_env(
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "LOCAL_RANK",
    )
    return int(default if value is None else max(0, value))


def _visible_cuda_devices():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    devices = [token.strip() for token in str(raw).split(",") if token.strip()]
    return devices or []


def _torch_cuda_device_count():
    try:
        import torch
    except Exception:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def resolve_runtime_device(requested_device):
    """Resolve calculator devices for single- and multi-rank CUDA runs.

    `cuda` is treated as "pick a node-local GPU for this rank". When Slurm or
    MPI already constrains `CUDA_VISIBLE_DEVICES`, the selected device is a
    rank-local `cuda:<index>` within that visible subset.
    """

    if requested_device is None:
        return None

    device = str(requested_device).strip()
    lowered = device.lower()
    if lowered in {"gpu", "cuda:auto"}:
        lowered = "cuda"
    if lowered not in {"auto", "cuda"}:
        return device

    visible_devices = _visible_cuda_devices()
    local_rank = detect_local_rank()
    if visible_devices is not None:
        if not visible_devices:
            return "cpu" if lowered == "auto" else device
        return f"cuda:{local_rank % len(visible_devices)}"

    cuda_count = _torch_cuda_device_count()
    if cuda_count > 0:
        return f"cuda:{local_rank % cuda_count}"

    return "cpu" if lowered == "auto" else device


def load_mace_calculator():
    """Import and return ``MACECalculator`` with a clearer env-mismatch error."""
    try:
        from mace.calculators import MACECalculator

        return MACECalculator
    except Exception as exc:
        removed = _remove_user_site_from_sys_path()
        if removed:
            try:
                from mace.calculators import MACECalculator

                return MACECalculator
            except ModuleNotFoundError as retry_exc:
                if retry_exc.name == "torch":
                    raise RuntimeError(
                        "MACE is installed in the active environment, but PyTorch is "
                        "being picked up from the user site or is missing from the env. "
                        "The loader removed user-site packages from sys.path and retried, "
                        "but `torch` is still unavailable. Install PyTorch inside the "
                        "current conda env, or run with an env that already provides it."
                    ) from exc
                raise
            except Exception as retry_exc:
                raise RuntimeError(
                    "Failed to import MACE after removing user-site packages from "
                    "sys.path. A conflicting `~/.local` PyTorch install is likely "
                    "shadowing the conda env packages."
                ) from retry_exc

        raise RuntimeError(
            "Failed to import MACECalculator. If the traceback mentions "
            "`libcudart.so`, `libcublas.so`, or a user-site `torch` under "
            "`~/.local`, your conda env is mixing with user-site packages. "
            "Try `PYTHONNOUSERSITE=1` and make sure PyTorch is installed inside "
            "the active environment."
        ) from exc
