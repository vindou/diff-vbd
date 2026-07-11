"""Vertex block descent solver implementation."""

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from diff_vbd.model import SimulationProblem, SimulationState
from diff_vbd.setup.boundary_conditions import evaluate_dirichlet_targets
from diff_vbd.solver.materials import tet_energy


@jax.jit
def predict_inertial_target(
    problem: SimulationProblem, state: SimulationState
) -> jnp.ndarray:
    """Return the inertial target for all vertices."""
    return (
        state.position
        + problem.solver.dt * state.velocity
        + (problem.solver.dt**2) * problem.solver.external_acceleration
    )


@jax.jit
def vertex_local_objective(
    problem: SimulationProblem,
    block_position: jnp.ndarray,
    inertial_target: jnp.ndarray,
    vertex_index: jnp.ndarray,
    x_i: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the local objective for one vertex."""
    residual = x_i - inertial_target[vertex_index]
    inertia = (
        problem.topology.mass[vertex_index] / (2.0 * problem.solver.dt**2)
    ) * jnp.dot(residual, residual)

    vertex_incident_tets = problem.topology.incident_tets[vertex_index]
    vertex_incident_mask = problem.topology.incident_mask[vertex_index]

    def incident_tet_energy(tet_index):
        tet_vertices = problem.mesh.tets[tet_index]
        tet_positions = block_position[tet_vertices]
        local_vertex = jnp.argmax((tet_vertices == vertex_index).astype(jnp.int32))
        tet_positions = tet_positions.at[local_vertex].set(x_i)
        rest_tet_positions = problem.mesh.rest_positions[tet_vertices]
        return tet_energy(problem.material, rest_tet_positions, tet_positions)

    elastic_energies = jax.vmap(incident_tet_energy)(vertex_incident_tets)
    elastic = jnp.sum(
        elastic_energies * vertex_incident_mask.astype(elastic_energies.dtype)
    )
    return inertia + elastic


vertex_local_gradient = jax.jit(jax.grad(vertex_local_objective, argnums=4))
vertex_local_hessian = jax.jit(jax.hessian(vertex_local_objective, argnums=4))


@jax.jit
def clamped_hessian(hessian: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """Return the nearest positive-definite matrix with eigenvalues at least ``eps``.

    The stable Neo-Hookean energy is nonconvex, so the local Hessian can be indefinite
    under compression or inversion. Solving with an indefinite Hessian yields an ascent
    direction along its negative eigenvectors, so the eigenvalues are floored while the
    eigenvectors are left untouched.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    return (eigenvectors * jnp.maximum(eigenvalues, eps)) @ eigenvectors.T


@jax.jit
def _apply_line_search(
    problem: SimulationProblem,
    block_position: jnp.ndarray,
    inertial_target: jnp.ndarray,
    vertex_index: jnp.ndarray,
    x_i_iter: jnp.ndarray,
    delta_x: jnp.ndarray,
) -> jnp.ndarray:
    alphas = problem.solver.line_search.alphas

    def evaluate_alpha(alpha):
        candidate = x_i_iter + alpha * delta_x
        objective = vertex_local_objective(
            problem, block_position, inertial_target, vertex_index, candidate
        )
        return candidate, objective

    candidate_positions, objective_values = jax.vmap(evaluate_alpha)(alphas)
    best_index = jnp.argmin(objective_values)
    return candidate_positions[best_index]


@jax.jit
def solve_local_vertex_step(
    problem: SimulationProblem,
    block_position: jnp.ndarray,
    inertial_target: jnp.ndarray,
    vertex_index: jnp.ndarray,
) -> jnp.ndarray:
    """Apply one local VBD Newton step."""
    x_i_iter = block_position[vertex_index]
    gradient = vertex_local_gradient(
        problem, block_position, inertial_target, vertex_index, x_i_iter
    )
    hessian = vertex_local_hessian(
        problem, block_position, inertial_target, vertex_index, x_i_iter
    )
    regularized_hessian = clamped_hessian(hessian, problem.solver.eps)
    delta_x = jnp.linalg.solve(regularized_hessian, -gradient)

    def line_search_step(_):
        return _apply_line_search(
            problem, block_position, inertial_target, vertex_index, x_i_iter, delta_x
        )

    def full_newton_step(_):
        return x_i_iter + delta_x

    return jax.lax.cond(
        problem.solver.line_search.enabled,
        line_search_step,
        full_newton_step,
        operand=None,
    )


@jax.jit
def solve_color_block(
    problem: SimulationProblem,
    block_position: jnp.ndarray,
    inertial_target: jnp.ndarray,
    constrained_mask: jnp.ndarray,
    color_vertices: jnp.ndarray,
    color_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Solve all vertices in one color block from a frozen position snapshot."""

    def solve_one(vertex_index):
        return solve_local_vertex_step(
            problem, block_position, inertial_target, vertex_index
        )

    candidate_updates = jax.vmap(solve_one)(color_vertices)
    current_positions = block_position[color_vertices]
    valid_mask = color_mask.astype(bool)
    free_mask = jnp.logical_not(constrained_mask[color_vertices])
    update_mask = jnp.logical_and(valid_mask, free_mask)[:, None]
    return jnp.where(update_mask, candidate_updates, current_positions)


@jax.jit
def apply_color_block(
    problem: SimulationProblem,
    working_position: jnp.ndarray,
    inertial_target: jnp.ndarray,
    locked_mask: jnp.ndarray,
    color_index: jnp.ndarray,
) -> jnp.ndarray:
    """Apply one color update and scatter the results back together."""
    color_vertices = problem.topology.color_groups[color_index]
    color_mask = problem.topology.color_group_mask[color_index]
    updated_positions = solve_color_block(
        problem,
        working_position,
        inertial_target,
        locked_mask,
        color_vertices,
        color_mask,
    )
    valid_mask = color_mask.astype(bool)
    scatter_values = jnp.where(
        valid_mask[:, None], updated_positions, working_position[color_vertices]
    )
    return working_position.at[color_vertices].set(scatter_values)


@jax.jit
def project_rigid_regions(
    problem: SimulationProblem, position: jnp.ndarray
) -> jnp.ndarray:
    """Project each rigid selector onto its closest mass-weighted rigid motion."""
    projected_position = position
    for rigid_region in problem.boundary_conditions.rigid_region_specs:
        vertex_indices = rigid_region.vertex_indices
        reference_local_positions = rigid_region.reference_local_positions
        current_positions = projected_position[vertex_indices]
        weights = problem.topology.mass[vertex_indices]
        total_weight = jnp.sum(weights)
        current_com = jnp.sum(weights[:, None] * current_positions, axis=0) / total_weight
        current_centered = current_positions - current_com
        covariance = (weights[:, None] * reference_local_positions).T @ current_centered
        u, _, vh = jnp.linalg.svd(covariance, full_matrices=False)
        v = vh.T
        orientation = jnp.where(jnp.linalg.det(v @ u.T) < 0.0, -1.0, 1.0)
        correction = jnp.diag(jnp.array([1.0, 1.0, orientation], dtype=position.dtype))
        rotation = v @ correction @ u.T
        rigid_positions = current_com + reference_local_positions @ rotation.T
        projected_position = projected_position.at[vertex_indices].set(rigid_positions)
    return projected_position


@jax.jit
def _apply_position_constraints(
    problem: SimulationProblem,
    position: jnp.ndarray,
    prescribed_position: jnp.ndarray,
) -> jnp.ndarray:
    projected_position = project_rigid_regions(problem, position)
    dirichlet_mask = problem.boundary_conditions.dirichlet_mask.astype(bool)
    return jnp.where(dirichlet_mask[:, None], prescribed_position, projected_position)


@jax.jit
def _compute_initial_position(
    problem: SimulationProblem,
    state: SimulationState,
    inertial_target: jnp.ndarray,
    prescribed_position: jnp.ndarray,
) -> jnp.ndarray:
    dirichlet_mask = problem.boundary_conditions.dirichlet_mask.astype(bool)
    locked_mask = jnp.logical_or(dirichlet_mask, problem.boundary_conditions.rigid_mask)
    free_mask = jnp.logical_not(locked_mask)[:, None].astype(state.position.dtype)
    initial_position = inertial_target + free_mask * (state.position - inertial_target)
    initial_position = jnp.where(
        dirichlet_mask[:, None], prescribed_position, initial_position
    )
    return project_rigid_regions(problem, initial_position)


@jax.jit
def _raw_vbd_iteration(
    problem: SimulationProblem,
    position: jnp.ndarray,
    inertial_target: jnp.ndarray,
    prescribed_position: jnp.ndarray,
) -> jnp.ndarray:
    dirichlet_mask = problem.boundary_conditions.dirichlet_mask.astype(bool)
    locked_mask = jnp.logical_or(dirichlet_mask, problem.boundary_conditions.rigid_mask)
    num_colors = problem.topology.color_groups.shape[0]
    color_indices = jnp.arange(num_colors, dtype=jnp.int32)

    def color_step(position_in_color, color_index):
        return apply_color_block(
            problem,
            position_in_color,
            inertial_target,
            locked_mask,
            color_index,
        ), None

    next_position, _ = jax.lax.scan(color_step, position, color_indices)
    return _apply_position_constraints(problem, next_position, prescribed_position)


@jax.jit
def chebyshev_weight(
    iteration_index: jnp.ndarray,
    rho: jnp.ndarray,
    previous_weight: jnp.ndarray,
) -> jnp.ndarray:
    """Return the Chebyshev extrapolation weight for a 0-based sweep index."""
    rho_squared = rho * rho
    return jax.lax.cond(
        iteration_index == 0,
        lambda _: jnp.asarray(1.0, dtype=rho.dtype),
        lambda _: jax.lax.cond(
            iteration_index == 1,
            lambda __: 2.0 / (2.0 - rho_squared),
            lambda __: 4.0 / (4.0 - rho_squared * previous_weight),
            operand=None,
        ),
        operand=None,
    )


@jax.jit
def _sweep_positions_impl(
    problem: SimulationProblem,
    state: SimulationState,
    prescribed_position: jnp.ndarray,
) -> jnp.ndarray:
    """Perform fixed color-parallel iterations over the mesh."""
    inertial_target = predict_inertial_target(problem, state)
    initial_position = _compute_initial_position(
        problem, state, inertial_target, prescribed_position
    )

    def iteration_step(carry, iteration_index):
        current_position, previous_position, previous_weight = carry
        raw_position = _raw_vbd_iteration(
            problem, current_position, inertial_target, prescribed_position
        )

        def apply_acceleration(_):
            weight = chebyshev_weight(
                iteration_index,
                problem.solver.acceleration.chebyshev_rho,
                previous_weight,
            )
            accelerated_position = previous_position + weight * (
                raw_position - previous_position
            )
            accelerated_position = _apply_position_constraints(
                problem, accelerated_position, prescribed_position
            )
            return accelerated_position, weight

        def skip_acceleration(_):
            return raw_position, jnp.asarray(1.0, dtype=raw_position.dtype)

        next_position, next_weight = jax.lax.cond(
            problem.solver.acceleration.enabled,
            apply_acceleration,
            skip_acceleration,
            operand=None,
        )
        return (next_position, current_position, next_weight), None

    final_carry, _ = jax.lax.scan(
        iteration_step,
        (
            initial_position,
            initial_position,
            jnp.asarray(1.0, dtype=initial_position.dtype),
        ),
        problem.solver.iteration_schedule,
    )
    next_position, _, _ = final_carry
    return _apply_position_constraints(problem, next_position, prescribed_position)


def sweep_positions(
    problem: SimulationProblem,
    state: SimulationState,
    prescribed_position: jnp.ndarray,
) -> jnp.ndarray:
    """Perform fixed color-parallel iterations over the mesh."""
    return _sweep_positions_impl(problem, state, prescribed_position)


@jax.jit
def update_velocity(
    problem: SimulationProblem,
    state: SimulationState,
    next_position: jnp.ndarray,
    prescribed_velocity: jnp.ndarray,
) -> jnp.ndarray:
    """Update velocity from discrete displacement and Dirichlet prescriptions."""
    raw_velocity = (next_position - state.position) / problem.solver.dt
    return jnp.where(
        problem.boundary_conditions.dirichlet_mask[:, None],
        prescribed_velocity,
        raw_velocity,
    )


@jax.jit
def _advance_step(
    problem: SimulationProblem,
    state: SimulationState,
    prescribed_position: jnp.ndarray,
    prescribed_velocity: jnp.ndarray,
) -> SimulationState:
    """Advance the mesh by one timestep."""
    next_position = _sweep_positions_impl(problem, state, prescribed_position)
    next_velocity = update_velocity(
        problem, state, next_position, prescribed_velocity
    )
    return SimulationState(
        position=next_position,
        velocity=next_velocity,
        time=state.time + problem.solver.dt,
    )


def step(
    problem: SimulationProblem,
    state: SimulationState,
) -> SimulationState:
    """Advance the mesh by one timestep with Dirichlet targets applied."""
    prescribed_position, prescribed_velocity = evaluate_dirichlet_targets(
        problem.boundary_conditions, float(state.time), float(problem.solver.dt)
    )
    return _advance_step(problem, state, prescribed_position, prescribed_velocity)


def _stack_state_history(history: list[SimulationState]) -> SimulationState:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *history)


def simulate(
    problem: SimulationProblem,
    state: SimulationState,
    num_steps: int,
    *,
    show_progress: bool = True,
) -> tuple[SimulationState, SimulationState]:
    """Run a fixed number of timesteps and return final state plus history."""
    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")

    history = []
    current_state = state
    progress = tqdm(
        range(num_steps),
        desc="Simulating",
        unit="step",
        disable=not show_progress,
    )
    for _ in progress:
        current_state = step(problem, current_state)
        history.append(current_state)

    return current_state, _stack_state_history(history)
