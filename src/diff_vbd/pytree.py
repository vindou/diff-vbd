"""Helpers for JAX pytree dataclasses."""

from dataclasses import dataclass
from typing import Any

import jax


def _tree_flatten_dataclass(instance: Any):
    children = tuple(getattr(instance, field) for field in instance.__dataclass_fields__)
    return children, None


def _tree_unflatten_dataclass(cls, aux_data, children):
    del aux_data
    field_names = tuple(cls.__dataclass_fields__)
    return cls(**dict(zip(field_names, children)))


def pytree_dataclass(cls):
    cls = dataclass(frozen=True)(cls)
    jax.tree_util.register_pytree_node(
        cls,
        _tree_flatten_dataclass,
        lambda aux_data, children: _tree_unflatten_dataclass(cls, aux_data, children),
    )
    return cls
