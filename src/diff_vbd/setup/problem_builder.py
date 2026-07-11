"""Validated problem assembly for solver-ready inputs."""

from typing import Iterable

import jax.numpy as jnp

from diff_vbd.model import (
    AccelerationOptions,
    BoundaryConditions,
    LineSearchOptions,
    MaterialParams,
    MeshData,
    SimulationProblem,
    SimulationState,
    SolverOptions,
    TopologyData,
)
from diff_vbd.setup.boundary_conditions import build_fixed_boundary_conditions
from diff_vbd.setup.topology import (
    build_incidence,
    build_lumped_masses,
    build_vertex_coloring,
)


def _as_vector3(value, dtype):
    vector = jnp.asarray(value, dtype=dtype)
    if vector.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {vector.shape}")
    return vector


def _validate_mesh(rest_positions: jnp.ndarray, tets: jnp.ndarray):
    if rest_positions.ndim != 2 or rest_positions.shape[1] != 3:
        raise ValueError(
            f"rest_positions must have shape (num_vertices, 3), got {rest_positions.shape}"
        )
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tets must have shape (num_tets, 4), got {tets.shape}")
    if rest_positions.shape[0] == 0 or tets.shape[0] == 0:
        raise ValueError("Mesh must contain at least one vertex and one tet")
    if not jnp.issubdtype(tets.dtype, jnp.integer):
        raise ValueError(f"tets must have an integer dtype, got {tets.dtype}")

    min_index = int(jnp.min(tets))
    max_index = int(jnp.max(tets))
    if min_index < 0 or max_index >= rest_positions.shape[0]:
        raise ValueError(
            f"tet indices must be in [0, {rest_positions.shape[0] - 1}], got [{min_index}, {max_index}]"
        )

    # The elastic energy is orientation sensitive: a tet whose vertices wind the wrong
    # way has det(F) < 0 in its own rest pose, so it is deformed before the sim starts.
    tet_positions = rest_positions[tets]
    edges = jnp.stack(
        [
            tet_positions[:, 1] - tet_positions[:, 0],
            tet_positions[:, 2] - tet_positions[:, 0],
            tet_positions[:, 3] - tet_positions[:, 0],
        ],
        axis=-1,
    )
    signed_volume = jnp.linalg.det(edges) / 6.0
    bad_tets = jnp.nonzero(signed_volume <= 0.0)[0]
    if bad_tets.shape[0] > 0:
        preview = [int(index) for index in bad_tets[:10]]
        raise ValueError(
            f"{bad_tets.shape[0]} tet(s) have non-positive rest volume; reorder their "
            f"vertices so each tet winds positively (first offenders: {preview})"
        )


def _expand_external_acceleration(external_acceleration, num_vertices: int, dtype):
    acceleration = jnp.asarray(external_acceleration, dtype=dtype)
    if acceleration.shape == (3,):
        return jnp.tile(acceleration[None, :], (num_vertices, 1))
    if acceleration.shape == (num_vertices, 3):
        return acceleration
    raise ValueError(
        "external_acceleration must have shape (3,) or (num_vertices, 3), "
        f"got {acceleration.shape}"
    )


def _validate_boundary_conditions(boundary_conditions: BoundaryConditions, num_vertices: int):
    dirichlet_mask = jnp.asarray(boundary_conditions.dirichlet_mask)
    if dirichlet_mask.ndim != 1 or dirichlet_mask.shape[0] != num_vertices:
        raise ValueError(
            "boundary_conditions.dirichlet_mask must have shape "
            f"({num_vertices},), got {dirichlet_mask.shape}"
        )
    if boundary_conditions.dirichlet_group_indices.shape != (num_vertices,):
        raise ValueError(
            "boundary_conditions.dirichlet_group_indices must have shape "
            f"({num_vertices},), got {boundary_conditions.dirichlet_group_indices.shape}"
        )
    if boundary_conditions.rigid_region_indices.shape != (num_vertices,):
        raise ValueError(
            "boundary_conditions.rigid_region_indices must have shape "
            f"({num_vertices},), got {boundary_conditions.rigid_region_indices.shape}"
        )
    if boundary_conditions.reference_positions.shape != (num_vertices, 3):
        raise ValueError(
            "boundary_conditions.reference_positions must have shape "
            f"({num_vertices}, 3), got {boundary_conditions.reference_positions.shape}"
        )
    for rigid_region in boundary_conditions.rigid_region_specs:
        if rigid_region.vertex_indices.ndim != 1:
            raise ValueError("rigid_region.vertex_indices must be rank 1")
        if rigid_region.reference_local_positions.shape != (
            rigid_region.vertex_indices.shape[0],
            3,
        ):
            raise ValueError(
                "rigid_region.reference_local_positions must have shape "
                f"({rigid_region.vertex_indices.shape[0]}, 3), got "
                f"{rigid_region.reference_local_positions.shape}"
            )
        if rigid_region.reference_com.shape != (3,):
            raise ValueError(
                f"rigid_region.reference_com must have shape (3,), got {rigid_region.reference_com.shape}"
            )

    restricted_mask = jnp.logical_or(
        boundary_conditions.dirichlet_mask,
        boundary_conditions.rigid_mask,
    )
    restricted_count = int(jnp.sum(restricted_mask))
    free_count = int(jnp.sum(~restricted_mask))
    if restricted_count == 0:
        raise ValueError("Boundary conditions must constrain at least one vertex or rigid region")
    if free_count == 0:
        raise ValueError("Boundary conditions must leave at least one vertex free")


def _validate_topology(
    incident_tets: jnp.ndarray,
    incident_mask: jnp.ndarray,
    color_groups: jnp.ndarray,
    color_group_mask: jnp.ndarray,
    mass: jnp.ndarray,
    num_vertices: int,
):
    if incident_tets.shape != incident_mask.shape:
        raise ValueError("incident_tets and incident_mask must have the same shape")
    if incident_tets.ndim != 2 or incident_tets.shape[0] != num_vertices:
        raise ValueError("incident_tets must have shape (num_vertices, max_incident)")
    if color_groups.shape != color_group_mask.shape:
        raise ValueError("color_groups and color_group_mask must have the same shape")
    if color_groups.ndim != 2:
        raise ValueError("color_groups must be rank 2")
    if mass.ndim != 1 or mass.shape[0] != num_vertices:
        raise ValueError(f"mass must have shape ({num_vertices},), got {mass.shape}")


def _validate_chebyshev_options(acceleration_enabled: bool, chebyshev_rho: float):
    if acceleration_enabled and not (0.0 < chebyshev_rho < 1.0):
        raise ValueError(
            "chebyshev_rho must satisfy 0 < chebyshev_rho < 1 when acceleration is enabled"
        )


def _validate_line_search_options(line_search_alphas):
    alphas = jnp.asarray(line_search_alphas)
    if alphas.ndim != 1:
        raise ValueError(f"line_search_alphas must be rank 1, got {alphas.shape}")
    if alphas.shape[0] == 0:
        raise ValueError("line_search_alphas must contain at least one alpha")
    if not bool(jnp.all((alphas > 0.0) & (alphas <= 1.0))):
        raise ValueError("line_search_alphas must satisfy 0 < alpha <= 1")


def assemble_problem(
    rest_positions: jnp.ndarray,
    tets: jnp.ndarray,
    boundary_data: jnp.ndarray | BoundaryConditions,
    *,
    dt=0.02,
    external_acceleration: Iterable[float] | jnp.ndarray = (0.0, 0.0, -9.81),
    mu=4.638e5,
    lam=4.174e6,
    density=1.0,
    eps=1.0e-6,
    num_iterations: int = 10,
    acceleration_enabled: bool = False,
    chebyshev_rho: float = 0.95,
    line_search_enabled: bool = True,
    line_search_alphas: Iterable[float] | jnp.ndarray = (1.0, 0.5, 0.25, 0.125),
) -> SimulationProblem:
    """Assemble a validated solver-ready problem from mesh and setup arrays."""
    rest_positions = jnp.asarray(rest_positions)
    dtype = rest_positions.dtype
    tets = jnp.asarray(tets, dtype=jnp.int32)

    _validate_mesh(rest_positions, tets)
    _validate_chebyshev_options(acceleration_enabled, chebyshev_rho)
    _validate_line_search_options(line_search_alphas)
    if isinstance(boundary_data, BoundaryConditions):
        boundary_conditions = boundary_data
    else:
        free_mask = jnp.asarray(boundary_data, dtype=dtype)
        if free_mask.ndim != 1 or free_mask.shape[0] != rest_positions.shape[0]:
            raise ValueError(
                f"free_mask must have shape ({rest_positions.shape[0]},), got {free_mask.shape}"
            )
        constrained_mask = free_mask == 0
        boundary_conditions = build_fixed_boundary_conditions(
            rest_positions, constrained_mask
        )
    _validate_boundary_conditions(boundary_conditions, rest_positions.shape[0])

    incident_tets, incident_mask = build_incidence(tets, rest_positions.shape[0])
    color_groups, color_group_mask, _ = build_vertex_coloring(
        tets, rest_positions.shape[0]
    )
    mass = build_lumped_masses(rest_positions, tets) * jnp.asarray(density, dtype=dtype)
    _validate_topology(
        incident_tets,
        incident_mask,
        color_groups,
        color_group_mask,
        mass,
        rest_positions.shape[0],
    )

    mesh = MeshData(rest_positions=rest_positions, tets=tets)
    topology = TopologyData(
        incident_tets=incident_tets,
        incident_mask=incident_mask,
        color_groups=color_groups,
        color_group_mask=color_group_mask,
        mass=mass,
    )
    material = MaterialParams(
        mu=jnp.asarray(mu, dtype=dtype),
        lam=jnp.asarray(lam, dtype=dtype),
        density=jnp.asarray(density, dtype=dtype),
    )
    solver = SolverOptions(
        dt=jnp.asarray(dt, dtype=dtype),
        external_acceleration=_expand_external_acceleration(
            external_acceleration, rest_positions.shape[0], dtype
        ),
        eps=jnp.asarray(eps, dtype=dtype),
        iteration_schedule=jnp.arange(num_iterations, dtype=jnp.int32),
        acceleration=AccelerationOptions(
            enabled=jnp.asarray(acceleration_enabled, dtype=jnp.bool_),
            chebyshev_rho=jnp.asarray(chebyshev_rho, dtype=dtype),
        ),
        line_search=LineSearchOptions(
            enabled=jnp.asarray(line_search_enabled, dtype=jnp.bool_),
            alphas=jnp.asarray(line_search_alphas, dtype=dtype),
        ),
    )

    return SimulationProblem(
        mesh=mesh,
        topology=topology,
        boundary_conditions=boundary_conditions,
        material=material,
        solver=solver,
    )


def initial_state(
    problem: SimulationProblem,
    position: jnp.ndarray | None = None,
    velocity: jnp.ndarray | None = None,
    time: jnp.ndarray | float | None = None,
) -> SimulationState:
    """Create a simulation state compatible with a problem definition."""
    if position is None:
        position = problem.mesh.rest_positions
    if velocity is None:
        velocity = jnp.zeros_like(problem.mesh.rest_positions)
    if time is None:
        time = jnp.array(0.0, dtype=problem.mesh.rest_positions.dtype)

    position = jnp.asarray(position, dtype=problem.mesh.rest_positions.dtype)
    velocity = jnp.asarray(velocity, dtype=problem.mesh.rest_positions.dtype)
    time = jnp.asarray(time, dtype=problem.mesh.rest_positions.dtype)

    expected_shape = problem.mesh.rest_positions.shape
    if position.shape != expected_shape:
        raise ValueError(
            f"position must have shape {expected_shape}, got {position.shape}"
        )
    if velocity.shape != expected_shape:
        raise ValueError(
            f"velocity must have shape {expected_shape}, got {velocity.shape}"
        )
    if time.shape != ():
        raise ValueError(f"time must be a scalar, got {time.shape}")

    return SimulationState(position=position, velocity=velocity, time=time)
