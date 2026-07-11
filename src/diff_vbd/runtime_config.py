"""Runtime configuration helpers for JAX-backed entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
import json


def apply_runtime_config(
    *,
    platform: str,
    gpu_preallocate: bool | None = None,
    gpu_mem_fraction: float | None = None,
) -> dict[str, object]:
    """Apply JAX/XLA runtime environment variables before importing JAX."""
    normalized_platform = platform.lower()
    if normalized_platform not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported platform {platform!r}; expected 'cpu' or 'gpu'")

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
        "gpu_preallocate": (
            None if normalized_platform != "gpu" else effective_preallocate
        ),
        "gpu_mem_fraction": (
            None if normalized_platform != "gpu" else gpu_mem_fraction
        ),
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
