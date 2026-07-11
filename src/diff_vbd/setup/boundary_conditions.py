"""Boundary-condition assembly and evaluation helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from diff_vbd.model import BoundaryConditions, DirichletSpec, RigidRegionSpec
from diff_vbd.setup.selector_io import SelectorVertexMembership

_ALLOWED_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "abs": np.abs,
}
_ALLOWED_NAMES = frozenset({"t", "pi", "e", *tuple(_ALLOWED_FUNCTIONS)})
_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Constant,
)


@dataclass(frozen=True)
class CompiledComponentExpression:
    source: str
    code: object


@dataclass(frozen=True)
class CompiledDirichletSpec:
    selector_name: str
    mode: str
    components: tuple[CompiledComponentExpression, ...]


def build_free_mask_from_aabb(
    positions: jnp.ndarray, selector_min: jnp.ndarray, selector_max: jnp.ndarray
) -> jnp.ndarray:
    """Return a free/fixed mask using selector AABB inclusion."""
    inside = jnp.all((positions >= selector_min) & (positions <= selector_max), axis=1)
    return jnp.where(inside, 0.0, 1.0).astype(positions.dtype)


def build_fixed_boundary_conditions(
    rest_positions: jnp.ndarray, constrained_mask: jnp.ndarray
) -> BoundaryConditions:
    """Build fixed Dirichlet constraints from a dense constrained-vertex mask."""
    rest_positions = jnp.asarray(rest_positions)
    constrained_mask = jnp.asarray(constrained_mask).astype(bool)
    dirichlet_group_indices = jnp.where(
        constrained_mask,
        jnp.zeros((rest_positions.shape[0],), dtype=jnp.int32),
        -jnp.ones((rest_positions.shape[0],), dtype=jnp.int32),
    )
    return BoundaryConditions(
        dirichlet_mask=constrained_mask,
        dirichlet_group_indices=dirichlet_group_indices,
        reference_positions=rest_positions,
        dirichlet_specs=(
            DirichletSpec(
                selector_name="fixed",
                mode="position",
                components=("0.0", "0.0", "0.0"),
            ),
        )
        if bool(jnp.any(constrained_mask))
        else (),
    )


def _compile_expression(source: str) -> CompiledComponentExpression:
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Dirichlet expression {source!r}: {exc}") from exc

    for node in ast.walk(parsed):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(
                f"Unsupported syntax in Dirichlet expression {source!r}: {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(
                f"Unsupported symbol {node.id!r} in Dirichlet expression {source!r}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                raise ValueError(
                    f"Unsupported function call in Dirichlet expression {source!r}"
                )
    return CompiledComponentExpression(source=source, code=compile(parsed, "<dirichlet>", "eval"))


def _compile_dirichlet_spec(spec: DirichletSpec) -> CompiledDirichletSpec:
    if spec.mode not in {"position", "velocity"}:
        raise ValueError(f"Unsupported Dirichlet mode {spec.mode!r}")
    if len(spec.components) != 3:
        raise ValueError(
            f"Dirichlet spec for selector {spec.selector_name!r} must have 3 components"
        )
    return CompiledDirichletSpec(
        selector_name=spec.selector_name,
        mode=spec.mode,
        components=tuple(_compile_expression(component) for component in spec.components),
    )


def _build_rigid_region_spec(
    rest_positions: jnp.ndarray, membership: SelectorVertexMembership
) -> RigidRegionSpec:
    vertex_indices = np.asarray(membership.vertex_indices, dtype=np.int32)
    if vertex_indices.size < 3:
        raise ValueError(
            f"Rigid selector {membership.selector_name!r} must contain at least 3 vertices"
        )

    region_positions = np.asarray(rest_positions[vertex_indices], dtype=np.float64)
    reference_com = region_positions.mean(axis=0)
    reference_local_positions = region_positions - reference_com
    singular_values = np.linalg.svd(reference_local_positions, compute_uv=False)
    geometric_rank = int(np.sum(singular_values > 1.0e-8))
    if geometric_rank < 2:
        raise ValueError(
            f"Rigid selector {membership.selector_name!r} is degenerate; "
            "need at least 3 non-collinear vertices"
        )

    dtype = rest_positions.dtype
    return RigidRegionSpec(
        selector_name=membership.selector_name,
        vertex_indices=jnp.asarray(vertex_indices, dtype=jnp.int32),
        reference_local_positions=jnp.asarray(reference_local_positions, dtype=dtype),
        reference_com=jnp.asarray(reference_com, dtype=dtype),
    )


def assemble_boundary_conditions(
    rest_positions: jnp.ndarray,
    selector_memberships: Iterable[SelectorVertexMembership],
    dirichlet_specs: Iterable[DirichletSpec] = (),
    rigid_selector_names: Iterable[str] = (),
) -> BoundaryConditions:
    """Bind selector memberships to mixed Dirichlet and rigid-region constraints."""
    rest_positions = jnp.asarray(rest_positions)
    num_vertices = rest_positions.shape[0]
    selector_map = {membership.selector_name: membership for membership in selector_memberships}
    compiled_specs = tuple(_compile_dirichlet_spec(spec) for spec in dirichlet_specs)

    dirichlet_mask = np.zeros((num_vertices,), dtype=bool)
    dirichlet_group_indices = -np.ones((num_vertices,), dtype=np.int32)
    seen_dirichlet_selectors = set()
    for group_index, spec in enumerate(compiled_specs):
        if spec.selector_name in seen_dirichlet_selectors:
            raise ValueError(f"Duplicate Dirichlet spec for selector {spec.selector_name!r}")
        seen_dirichlet_selectors.add(spec.selector_name)
        if spec.selector_name not in selector_map:
            raise ValueError(f"Missing selector membership for {spec.selector_name!r}")
        membership_mask = np.asarray(selector_map[spec.selector_name].vertex_mask, dtype=bool)
        overlap = dirichlet_mask & membership_mask
        if np.any(overlap):
            raise ValueError(
                f"Dirichlet selectors overlap on vertices for selector {spec.selector_name!r}"
            )
        dirichlet_mask |= membership_mask
        dirichlet_group_indices[membership_mask] = group_index

    rigid_region_indices = -np.ones((num_vertices,), dtype=np.int32)
    rigid_region_specs = []
    seen_rigid_selectors = set()
    for rigid_region_index, selector_name in enumerate(rigid_selector_names):
        if selector_name in seen_rigid_selectors:
            raise ValueError(f"Duplicate rigid selector {selector_name!r}")
        seen_rigid_selectors.add(selector_name)
        if selector_name not in selector_map:
            raise ValueError(f"Missing selector membership for rigid selector {selector_name!r}")
        membership = selector_map[selector_name]
        membership_mask = np.asarray(membership.vertex_mask, dtype=bool)
        if np.any(dirichlet_mask & membership_mask):
            raise ValueError(
                f"Rigid selector {selector_name!r} overlaps Dirichlet-constrained vertices"
            )
        if np.any((rigid_region_indices >= 0) & membership_mask):
            raise ValueError(
                f"Rigid selectors overlap on vertices for selector {selector_name!r}"
            )
        rigid_region_indices[membership_mask] = rigid_region_index
        rigid_region_specs.append(_build_rigid_region_spec(rest_positions, membership))

    return BoundaryConditions(
        dirichlet_mask=jnp.asarray(dirichlet_mask),
        dirichlet_group_indices=jnp.asarray(dirichlet_group_indices, dtype=jnp.int32),
        reference_positions=rest_positions,
        dirichlet_specs=tuple(
            DirichletSpec(spec.selector_name, spec.mode, tuple(component.source for component in spec.components))
            for spec in compiled_specs
        ),
        rigid_region_indices=jnp.asarray(rigid_region_indices, dtype=jnp.int32),
        rigid_region_specs=tuple(rigid_region_specs),
    )


def assemble_dirichlet_boundary_conditions(
    rest_positions: jnp.ndarray,
    selector_memberships: Iterable[SelectorVertexMembership],
    dirichlet_specs: Iterable[DirichletSpec],
) -> BoundaryConditions:
    """Bind selector memberships to Dirichlet motion specs."""
    return assemble_boundary_conditions(
        rest_positions,
        selector_memberships,
        dirichlet_specs=dirichlet_specs,
        rigid_selector_names=(),
    )


def _evaluate_scalar_expression(source: str, time_value: float) -> float:
    compiled = _compile_expression(source)
    env = {"t": time_value, "pi": np.pi, "e": np.e, **_ALLOWED_FUNCTIONS}
    value = eval(compiled.code, {"__builtins__": {}}, env)
    return float(value)


def _evaluate_components(components: tuple[str, str, str], time_value: float) -> np.ndarray:
    return np.array(
        [_evaluate_scalar_expression(component, time_value) for component in components],
        dtype=np.float64,
    )


def _evaluate_velocity_displacement(
    components: tuple[str, str, str], time_value: float, dt: float
) -> np.ndarray:
    if dt <= 0.0:
        raise ValueError(f"Expected positive dt, got {dt}")
    step_count = int(round(time_value / dt))
    displacement = np.zeros((3,), dtype=np.float64)
    for step_index in range(step_count):
        sample_time = step_index * dt
        displacement += dt * _evaluate_components(components, sample_time)
    return displacement


def evaluate_dirichlet_targets(
    boundary_conditions: BoundaryConditions,
    time_value: float,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return target positions and velocities for constrained vertices."""
    reference_positions = np.asarray(boundary_conditions.reference_positions, dtype=np.float64)
    dirichlet_mask = np.asarray(boundary_conditions.dirichlet_mask, dtype=bool)
    group_indices = np.asarray(boundary_conditions.dirichlet_group_indices, dtype=np.int32)
    target_positions = np.array(reference_positions, copy=True)
    target_velocities = np.zeros_like(reference_positions)

    for group_index, spec in enumerate(boundary_conditions.dirichlet_specs):
        group_mask = group_indices == group_index
        if not np.any(group_mask):
            continue
        if spec.mode == "position":
            displacement_now = _evaluate_components(spec.components, time_value)
            displacement_next = _evaluate_components(spec.components, time_value + dt)
        elif spec.mode == "velocity":
            displacement_now = _evaluate_velocity_displacement(spec.components, time_value, dt)
            displacement_next = _evaluate_velocity_displacement(
                spec.components, time_value + dt, dt
            )
        else:
            raise ValueError(f"Unsupported Dirichlet mode {spec.mode!r}")

        target_positions[group_mask] = reference_positions[group_mask] + displacement_next
        target_velocities[group_mask] = (displacement_next - displacement_now) / dt

    target_positions[~dirichlet_mask] = reference_positions[~dirichlet_mask]
    target_velocities[~dirichlet_mask] = 0.0
    return (
        jnp.asarray(target_positions, dtype=boundary_conditions.reference_positions.dtype),
        jnp.asarray(target_velocities, dtype=boundary_conditions.reference_positions.dtype),
    )
