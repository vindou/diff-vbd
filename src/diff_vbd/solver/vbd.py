"""Vertex block descent solver implementation."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from diff_vbd.model import ContactState, SimulationProblem, SimulationState
from diff_vbd.setup.boundary_conditions import evaluate_dirichlet_targets
from diff_vbd.solver.contact.ccd import (
    derive_detection_band,
    filter_sweep,
    vertex_time_of_impact,
)
from diff_vbd.solver.contact.detection import (
    build_contact_incidence,
    detect_contact_pairs,
)
from diff_vbd.solver.contact.friction import collider_friction_energy
from diff_vbd.solver.contact.potential import (
    collider_contact_energy,
    colliding_vertex_mask,
    incident_pair_energy,
    incident_pair_min_gap,
)
from diff_vbd.solver.materials import tet_energy

# Below this, the contact filter has throttled the sweep so hard that the mesh is not
# actually advancing, and the run is silently dead rather than slow. See `_audit_step`.
_MIN_USEFUL_TOI = 1.0e-6


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
    previous_position: jnp.ndarray,
    vertex_index: jnp.ndarray,
    x_i: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the local objective for one vertex.

    ``previous_position`` is the mesh at the start of the timestep. Friction needs it: the
    tangential slip is measured from there, and the lagged normal force and tangent basis
    are evaluated there and frozen.
    """
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

    # Contact is just another force element feeding the same accumulation. Because it is
    # summed into the objective, `vertex_local_gradient` and `vertex_local_hessian` -- both
    # plain autodiff of this function -- pick up the contact force and its 3x3 Hessian for
    # free, and `clamped_hessian` PSD-projects the sum. No hand-derivatives anywhere.
    contact = collider_contact_energy(
        problem.contact.params, problem.contact.colliders, x_i
    )
    friction = collider_friction_energy(
        problem.contact.params,
        problem.contact.colliders,
        x_i,
        previous_position[vertex_index],
        problem.solver.dt,
    )
    # Mesh-mesh and self-collision pairs: the same accumulation, one more force element.
    # Barrier and friction both, so self-contact is not silently frictionless.
    pairs = incident_pair_energy(
        problem.contact.params,
        problem.contact.state,
        problem.mesh.rest_positions,
        block_position,
        previous_position,
        problem.solver.dt,
        vertex_index,
        x_i,
    )
    return inertia + elastic + contact + friction + pairs


vertex_local_gradient = jax.jit(jax.grad(vertex_local_objective, argnums=5))
vertex_local_hessian = jax.jit(jax.hessian(vertex_local_objective, argnums=5))


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
    previous_position: jnp.ndarray,
    vertex_index: jnp.ndarray,
    x_i_iter: jnp.ndarray,
    delta_x: jnp.ndarray,
) -> jnp.ndarray:
    alphas = problem.solver.line_search.alphas

    def evaluate_alpha(alpha):
        candidate = x_i_iter + alpha * delta_x
        objective = vertex_local_objective(
            problem,
            block_position,
            inertial_target,
            previous_position,
            vertex_index,
            candidate,
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
    previous_position: jnp.ndarray,
    vertex_index: jnp.ndarray,
) -> jnp.ndarray:
    """Apply one local VBD Newton step."""
    x_i_iter = block_position[vertex_index]
    gradient = vertex_local_gradient(
        problem, block_position, inertial_target, previous_position, vertex_index, x_i_iter
    )
    hessian = vertex_local_hessian(
        problem, block_position, inertial_target, previous_position, vertex_index, x_i_iter
    )
    regularized_hessian = clamped_hessian(hessian, problem.solver.eps)
    delta_x = jnp.linalg.solve(regularized_hessian, -gradient)

    # Bound this vertex's step so it cannot cross an obstacle. The line search would
    # usually catch that on its own -- a candidate past the barrier has an astronomical
    # objective and loses the argmin -- but the full-Newton branch below does not evaluate
    # the objective at all, so without this it steps straight through.
    #
    # Both an obstacle it could cross and a mesh pair it belongs to. The pair half is not the
    # intersection-free guarantee -- the sweep filter is -- but without it a vertex deep in a
    # stiff barrier proposes an enormous raw Newton step, and the sweep filter then has to
    # scale the *whole mesh* back to accommodate that one vertex, freezing the solve. A bound
    # applied where the step is proposed keeps one bad vertex from throttling everybody.
    collider_toi = vertex_time_of_impact(
        problem.contact.colliders, x_i_iter, x_i_iter + delta_x
    )
    travel = jnp.sqrt(jnp.dot(delta_x, delta_x) + 1.0e-30)
    pair_gap = incident_pair_min_gap(
        problem.contact.state, block_position, vertex_index
    )
    # Distance is 1-Lipschitz in this vertex, so travelling less than `slack * gap` cannot
    # close any of its pairs. `inf / travel` when there are no pairs, which clips to 1.
    # Gated with the rest of the mesh-mesh guarantee: turning that off has to turn off *all*
    # of it, or the "barrier penalty only" mode is not the thing it says it is.
    pair_toi = jnp.where(
        problem.contact.ccd.enabled,
        jnp.clip(problem.contact.ccd.slack * pair_gap / travel, 0.0, 1.0),
        jnp.ones((), dtype=x_i_iter.dtype),
    )
    toi = jax.lax.stop_gradient(jnp.minimum(collider_toi, pair_toi))

    def line_search_step(_):
        return _apply_line_search(
            problem,
            block_position,
            inertial_target,
            previous_position,
            vertex_index,
            x_i_iter,
            toi * delta_x,
        )

    def full_newton_step(_):
        return x_i_iter + toi * delta_x

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
    previous_position: jnp.ndarray,
    constrained_mask: jnp.ndarray,
    color_vertices: jnp.ndarray,
    color_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Solve all vertices in one color block from a frozen position snapshot."""

    def solve_one(vertex_index):
        return solve_local_vertex_step(
            problem, block_position, inertial_target, previous_position, vertex_index
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
    previous_position: jnp.ndarray,
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
        previous_position,
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
    previous_position: jnp.ndarray,
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
            previous_position,
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
    # The displacement clamp binds only the vertices the solver is actually free to move.
    # A prescribed row is overwritten after the filter anyway, so clamping it would freeze
    # the mesh to no purpose; `_audit_step` on the host covers those instead.
    free_mask = ~problem.boundary_conditions.dirichlet_mask.astype(bool)

    # The *initial guess* has to be filtered too, and this is easy to miss. The guess is the
    # inertial target -- where the mesh would go with no forces at all -- and it is taken in
    # one jump, before a single sweep runs. At speed that jump is far larger than the gap: a
    # body moving at 20 m/s with dt = 5 ms starts the solve 0.1 units *inside* whatever it was
    # about to hit. Every filter downstream measures from this configuration, so if it is
    # already intersecting they are all certifying motion from an invalid state, and the
    # guarantee is void before the first iteration.
    initial_position, initial_toi = filter_sweep(
        problem.contact,
        problem.contact.state,
        state.position,
        state.position,
        initial_position,
        free_mask,
    )
    initial_position = _apply_position_constraints(
        problem, initial_position, prescribed_position
    )

    def iteration_step(carry, iteration_index):
        (
            current_position,
            previous_position,
            previous_weight,
            colliding,
            min_toi,
        ) = carry
        raw_position = _raw_vbd_iteration(
            problem,
            current_position,
            inertial_target,
            state.position,
            prescribed_position,
        )

        # Once a vertex has been in contact at any point this step, it stays flagged: a
        # vertex that is bouncing in and out of the activation band is exactly the one the
        # extrapolation would destabilise.
        colliding = colliding | colliding_vertex_mask(
            problem.contact.params,
            problem.contact.colliders,
            problem.contact.state,
            raw_position,
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
            # Colliding vertices take the un-extrapolated solve. This has to be a `where`
            # rather than a second `cond`: the outer cond is a global on/off flag, while
            # this gate is per vertex. The Chebyshev weight stays scalar and shared, so the
            # recurrence itself is untouched.
            accelerated_position = jnp.where(
                colliding[:, None], raw_position, accelerated_position
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

        # The guarantee. Bound the sweep's *aggregate* displacement, not each vertex's own
        # step: a vertex only ever certified its motion against a frozen snapshot, and this
        # is where the mesh actually ended up. It is also the only thing that sees the
        # Chebyshev extrapolation, which runs after every local solve has finished and can
        # otherwise fling a vertex clean through an obstacle the local solves respected.
        #
        # `state.position` -- the step start -- is not the same as `current_position`, the
        # start of this sweep. The displacement clamp inside measures from the former,
        # because the detection band it is defending was built there.
        next_position, toi = filter_sweep(
            problem.contact,
            problem.contact.state,
            state.position,
            current_position,
            next_position,
            free_mask,
        )
        min_toi = jnp.minimum(min_toi, toi)
        # Re-assert Dirichlet/rigid afterwards: the scaling above must not drag a
        # kinematically prescribed vertex off its target.
        next_position = _apply_position_constraints(
            problem, next_position, prescribed_position
        )
        return (
            next_position,
            current_position,
            next_weight,
            colliding,
            min_toi,
        ), None

    final_carry, _ = jax.lax.scan(
        iteration_step,
        (
            initial_position,
            initial_position,
            jnp.asarray(1.0, dtype=initial_position.dtype),
            jnp.zeros((initial_position.shape[0],), dtype=jnp.bool_),
            initial_toi,
        ),
        problem.solver.iteration_schedule,
    )
    next_position, _, _, _, min_toi = final_carry
    return (
        _apply_position_constraints(problem, next_position, prescribed_position),
        min_toi,
    )


def sweep_positions(
    problem: SimulationProblem,
    state: SimulationState,
    prescribed_position: jnp.ndarray,
) -> jnp.ndarray:
    """Perform fixed color-parallel iterations over the mesh."""
    positions, _ = _sweep_positions_impl(problem, state, prescribed_position)
    return positions


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
) -> tuple[SimulationState, jnp.ndarray]:
    """Advance the mesh by one timestep, and report the sweep's tightest time of impact.

    The time of impact comes back out because a collapsing bound is a *silent* failure: the
    mesh simply stops moving, and that is indistinguishable from "contact is slow". The host
    is the only place that can say so.
    """
    next_position, min_toi = _sweep_positions_impl(
        problem, state, prescribed_position
    )
    next_velocity = update_velocity(
        problem, state, next_position, prescribed_velocity
    )
    return (
        SimulationState(
            position=next_position,
            velocity=next_velocity,
            time=state.time + problem.solver.dt,
        ),
        min_toi,
    )


def redetect_contacts(
    problem: SimulationProblem,
    state: SimulationState,
    prescribed_position: jnp.ndarray | None = None,
) -> SimulationProblem:
    """Rebuild the mesh-mesh contact set on the host and return an updated problem.

    This is the boundary between the two layers. Detection is combinatorial -- integer
    indices, an active set, a classification -- so it runs here, in Python, between jitted
    sweeps, and everything it produces is frozen data by the time the device sees it.

    The rebuilt buffers keep the same shapes, so this is a jit cache hit rather than a
    recompile. That is the whole reason the capacity is fixed and overflow is an error: a
    capacity that grew with the contact count would recompile the solver every step.

    The detection *band* is derived here, from how far the mesh is about to move, and written
    into ``CcdParams`` alongside the displacement clamp that enforces it. The two are computed
    together, in this one place, because they are only sound as a matched pair: the band is
    wide enough to catch every pair that could reach contact **only because** the clamp stops
    any vertex travelling further than the band was sized for. Split them up and the guarantee
    quietly evaporates. Both are scalar array leaves, so refreshing them every step is a jit
    cache hit like the buffers themselves.
    """
    if not bool(problem.contact.params.enabled):
        return problem
    if problem.contact.surface_triangles.shape[0] == 0:
        return problem  # analytic colliders only: nothing combinatorial to do

    capacity = problem.contact.state.pair_vertices.shape[0]
    max_per_vertex = problem.contact.state.incident_contacts.shape[1]
    positions = np.asarray(state.position)
    d_hat = float(problem.contact.params.d_hat)

    # How far the mesh is about to move, in the only terms available before the solve runs.
    inertial_target = np.asarray(predict_inertial_target(problem, state))
    predicted = float(
        np.max(np.linalg.norm(inertial_target - positions, axis=-1), initial=0.0)
    )
    prescribed = 0.0
    if prescribed_position is not None:
        dirichlet = np.asarray(
            problem.boundary_conditions.dirichlet_mask
        ).astype(bool)
        if dirichlet.any():
            moved = np.asarray(prescribed_position)[dirichlet] - positions[dirichlet]
            prescribed = float(np.max(np.linalg.norm(moved, axis=-1), initial=0.0))

    band, max_displacement = derive_detection_band(d_hat, predicted, prescribed)

    pair_vertices, pair_type, pair_valid = detect_contact_pairs(
        positions,
        np.asarray(problem.contact.surface_triangles),
        np.asarray(problem.contact.surface_edges),
        d_hat=d_hat,
        capacity=capacity,
        band=band,
    )
    incidence, mask = build_contact_incidence(
        pair_vertices, pair_valid, positions.shape[0], max_per_vertex
    )

    contact_state = ContactState(
        pair_vertices=jnp.asarray(pair_vertices, dtype=jnp.int32),
        pair_type=jnp.asarray(pair_type, dtype=jnp.int32),
        pair_valid=jnp.asarray(pair_valid, dtype=jnp.bool_),
        incident_contacts=jnp.asarray(incidence, dtype=jnp.int32),
        incident_contact_mask=jnp.asarray(mask, dtype=jnp.bool_),
    )
    dtype = problem.contact.ccd.slack.dtype
    ccd = dataclasses.replace(
        problem.contact.ccd,
        max_displacement=jnp.asarray(max_displacement, dtype=dtype),
        detection_band=jnp.asarray(band, dtype=dtype),
    )
    contact = dataclasses.replace(problem.contact, state=contact_state, ccd=ccd)
    return dataclasses.replace(problem, contact=contact)


def _audit_step(
    problem: SimulationProblem,
    start_position: jnp.ndarray,
    end_position: jnp.ndarray,
    min_toi: jnp.ndarray,
) -> None:
    """E3 and E4: the two ways the mesh-mesh guarantee can fail without anyone noticing."""
    if not bool(problem.contact.ccd.enabled):
        return

    band = float(problem.contact.ccd.detection_band)
    d_hat = float(problem.contact.params.d_hat)
    travelled = np.linalg.norm(
        np.asarray(end_position) - np.asarray(start_position), axis=-1
    )
    worst = float(np.max(travelled, initial=0.0))

    # E3. The on-device clamp binds only the free vertices, because the position constraints
    # overwrite Dirichlet and rigid rows *after* the filter runs -- so a prescribed motion or
    # a rigid lever arm can still carry a vertex further than the band was sized for. This is
    # the compensating control for that one hole.
    if 2.0 * worst >= band - d_hat:
        vertex = int(np.argmax(travelled))
        raise ValueError(
            f"vertex {vertex} moved {worst:g} this step, but the contact pair set was built "
            f"with a band of {band:g} against an activation distance of {d_hat:g}. The "
            f"guarantee needs 2 * displacement < band - d_hat: a pair further apart than the "
            f"band is not in the set, so nothing is watching it, and at this speed one could "
            f"have closed to zero unseen. The usual cause is a prescribed (Dirichlet) or "
            f"rigid motion, which the sweep's clamp cannot bind because the constraints are "
            f"re-applied after it. Reduce solver.dt, or slow the prescribed motion."
        )

    # E4. A time of impact at zero is a mesh that has stopped moving -- which looks exactly
    # like "contact is just slow" and will be debugged as a performance problem for a day.
    if float(min_toi) < _MIN_USEFUL_TOI:
        raise ValueError(
            f"the contact time-of-impact filter collapsed to {float(min_toi):g} this step, so "
            f"the sweep made essentially no progress and the mesh is frozen. Either something "
            f"is being driven into an obstacle from ~zero distance, or ACCD is grinding: its "
            f"advance is proportional to the gap, so two surfaces sliding past each other fast "
            f"at a tiny separation need far more iterations than the budget allows. Reduce "
            f"solver.dt, raise contact.d_hat, or disable self_collision_ccd (which forfeits "
            f"the intersection-free guarantee)."
        )


def step(
    problem: SimulationProblem,
    state: SimulationState,
) -> SimulationState:
    """Advance the mesh by one timestep with Dirichlet targets applied."""
    prescribed_position, prescribed_velocity = evaluate_dirichlet_targets(
        problem.boundary_conditions, float(state.time), float(problem.solver.dt)
    )
    # Detection needs the prescribed targets: a Dirichlet vertex's motion is known exactly,
    # and it has to be inside the band the pair set is built with.
    problem = redetect_contacts(problem, state, prescribed_position)
    next_state, min_toi = _advance_step(
        problem, state, prescribed_position, prescribed_velocity
    )
    _audit_step(problem, state.position, next_state.position, min_toi)
    return next_state


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
