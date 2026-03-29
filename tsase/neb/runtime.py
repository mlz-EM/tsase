"""Runtime helpers for optional heavy dependencies used by the examples."""

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
