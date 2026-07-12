"""Runtime configuration helpers for JAX-backed entrypoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import json


def apply_runtime_config(
    *,
    platform: str,
    gpu_preallocate: bool | None = None,
    gpu_mem_fraction: float | None = None,
    precision: str = "float32",
) -> dict[str, object]:
    """Apply JAX/XLA runtime environment variables before importing JAX.

    ``precision="float64"`` enables x64. Contact needs it: the IPC barrier resolves a
    gap ``d`` that is many orders of magnitude smaller than the mesh coordinates, and in
    float32 a positive gap can round to zero or below, at which point ``log(d/d_hat)``
    is NaN and the intersection-free guarantee is lost to rounding alone.
    """
    normalized_platform = platform.lower()
    if normalized_platform not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported platform {platform!r}; expected 'cpu' or 'gpu'")

    normalized_precision = precision.lower()
    if normalized_precision not in {"float32", "float64"}:
        raise ValueError(
            f"Unsupported precision {precision!r}; expected 'float32' or 'float64'"
        )
    enable_x64 = normalized_precision == "float64"
    os.environ["JAX_ENABLE_X64"] = "1" if enable_x64 else "0"

    # The environment variable alone is not enough, and quietly so. JAX reads it exactly once,
    # when it is imported -- and by the time an entrypoint calls this, importing that
    # entrypoint has usually already pulled in `diff_vbd`, which imports the solver, which
    # imports JAX. The variable is then set far too late, `jax_enable_x64` stays False, and
    # every float64 array the caller asks for is silently truncated to float32. There is no
    # error; the request simply does not happen. So say it again, directly, for the case where
    # JAX is already loaded.
    jax_module = sys.modules.get("jax")
    if jax_module is not None:
        jax_module.config.update("jax_enable_x64", enable_x64)

    backend_platform = "cuda" if normalized_platform == "gpu" else normalized_platform
    os.environ["JAX_PLATFORMS"] = backend_platform
    if normalized_platform == "gpu":
        effective_preallocate = False if gpu_preallocate is None else gpu_preallocate
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
            "true" if effective_preallocate else "false"
        )
        if gpu_mem_fraction is None:
            os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        else:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_mem_fraction)
    else:
        os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)

    return {
        "platform": normalized_platform,
        "backend_platform": backend_platform,
        "precision": normalized_precision,
        "gpu_preallocate": (
            None if normalized_platform != "gpu" else effective_preallocate
        ),
        "gpu_mem_fraction": (
            None if normalized_platform != "gpu" else gpu_mem_fraction
        ),
        "env": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ),
        },
    }


def collect_runtime_report() -> dict[str, object]:
    """Collect backend/device information after JAX has been imported."""
    import jax

    return {
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "env": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ),
        },
    }


def write_runtime_report(path: str | Path, payload: dict[str, object]) -> Path:
    """Persist a runtime report to JSON."""
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return target
