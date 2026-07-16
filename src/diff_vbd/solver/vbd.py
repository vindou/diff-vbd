"""Vertex block descent solver implementation."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from diff_vbd.model import ContactState, SimulationProblem, SimulationState
from diff_vbd.setup.boundary_conditions import evaluate_dirichlet_targets
from diff_vbd.solver.contact.bounds import (
    build_vertex_bounds,
    derive_bounds_band,
    redetection_threshold,
    truncate_to_bounds,
)
from diff_vbd.solver.contact.ccd import vertex_time_of_impact
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
def _accelerated_iteration(
    problem: SimulationProblem,
    current_position: jnp.ndarray,
    previous_position: jnp.ndarray,
    previous_weight: jnp.ndarray,
    colliding: jnp.ndarray,
    iteration_index: jnp.ndarray,
    inertial_target: jnp.ndarray,
    step_start: jnp.ndarray,
    prescribed_position: jnp.ndarray,
):
    """One VBD iteration plus its (optional) Chebyshev update, jitted as a unit.

    Returns ``(next_position, previous_for_next, next_weight, colliding)``. The host owns
    the loop over iterations -- it has to, because the conservative-bound scheme re-runs
    contact detection *mid-step* when enough vertices consume their bounds, and detection
    is host code. Everything per-iteration stays on the device.
    """
    raw_position = _raw_vbd_iteration(
        problem,
        current_position,
        inertial_target,
        step_start,
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
    return next_position, current_position, next_weight, colliding


def _locked_rows(problem: SimulationProblem) -> np.ndarray:
    """Rows the truncation kernel must not clip: Dirichlet *and* rigid-region vertices.

    Both are rewritten by ``_apply_position_constraints`` after any filter runs --
    Dirichlet rows to their prescribed targets, rigid rows to the mass-weighted best-fit
    rigid motion. Clipping a rigid row would not stick (the projection is dominated by
    the region's unclipped far-field rows and drags the clipped vertex right back), and
    counting its discarded clip toward the re-detection trigger would be noise. So
    locked rows pass through untruncated, and ``_audit_locked_bounds`` is the
    compensating control for **both** -- the review of this branch found rigid rows
    covered by neither, which silently voided the certificate for a rigid body driven
    into a deformable surface.
    """
    return np.asarray(problem.boundary_conditions.dirichlet_mask, dtype=bool) | (
        np.asarray(problem.boundary_conditions.rigid_region_indices) >= 0
    )


def _audit_locked_bounds(
    problem: SimulationProblem,
    anchor: jnp.ndarray,
    position: jnp.ndarray,
) -> None:
    """Raise if a locked (Dirichlet or rigid) vertex has outrun its conservative bound.

    Locked rows are the ones the truncation kernel cannot bind: their positions are
    rewritten after every filter -- prescribed rows to a boundary condition, rigid rows
    to the region's best-fit rigid motion. That leaves exactly one way the per-vertex
    certificate can fail: a locked vertex moving further from the anchor than its
    bound, where a pair involving it could close unseen. This is the compensating
    control, in the same register as the old E3 audit (which likewise named prescribed
    *and* rigid motion as the two ways past the on-device clamp).
    """
    if not bool(problem.contact.ccd.enabled):
        return
    locked = _locked_rows(problem)
    if not locked.any():
        return
    travelled = np.linalg.norm(
        np.asarray(position) - np.asarray(anchor), axis=-1
    )
    bounds = np.asarray(problem.contact.state.vertex_bounds)
    violating = locked & (travelled > bounds)
    if violating.any():
        vertex = int(np.argmax(np.where(violating, travelled - bounds, -np.inf)))
        kind = (
            "prescribed"
            if bool(problem.boundary_conditions.dirichlet_mask[vertex])
            else "rigid-region"
        )
        raise ValueError(
            f"{kind} vertex {vertex} moved {travelled[vertex]:g} since the last "
            f"contact detection, but its conservative bound is {bounds[vertex]:g}. The "
            f"per-vertex intersection-free certificate (Wu et al. 2020) needs every "
            f"vertex -- prescribed and rigid ones included -- to stay within its bound "
            f"between detections; the solver clips its own proposals but cannot clip a "
            f"boundary condition or a rigid projection. At this speed a contact pair "
            f"involving this vertex could have closed to zero unseen. Reduce solver.dt, "
            f"or slow the prescribed motion."
        )


def _sweep_positions_host(
    problem: SimulationProblem,
    state: SimulationState,
    prescribed_position: jnp.ndarray,
) -> tuple[SimulationProblem, jnp.ndarray]:
    """Run the step's iterations, enforcing per-vertex bounds, re-detecting as needed.

    The intersection-free guarantee now lives here, in three moves per iteration
    (Wu et al. 2020; OGC Algorithm 3):

    * every displacement the solver produces -- the initial guess included, since the
      inertial jump happens before a single sweep runs and can be far larger than any
      gap -- is truncated per vertex so no vertex strays further than its bound from the
      position recorded at the last detection. Truncation happens *after* the Chebyshev
      update, so it bounds where the mesh actually ended up;
    * displacement is cumulative from the last detection, not per iteration -- K
      iterations must not buy K times the certified distance;
    * when at least ``max(1, GAMMA_E * K)`` vertices have consumed their bounds,
      detection re-runs at the current positions, which re-anchors every vertex with
      fresh distances. A pinned vertex is *locally* pinned until then: nothing else in
      the mesh is slowed, which is the entire point of replacing the global filter.

    Returns the (possibly re-detected) problem alongside the positions, so the caller
    keeps the refreshed pair set for its end-of-step audit.
    """
    inertial_target = predict_inertial_target(problem, state)
    contact_active = bool(problem.contact.params.enabled)
    # The bound bookkeeping (exceeded counter, re-detection, locked-row audit) exists
    # only on the mesh-mesh path; collider-only problems keep the per-vertex collider
    # clip but skip the per-iteration host synchronisation the counter would force.
    bounds_active = contact_active and bool(problem.contact.ccd.enabled)
    threshold = redetection_threshold(int(state.position.shape[0]))
    anchor = state.position

    # Free-for-truncation excludes rigid rows as well as Dirichlet ones: both are
    # rewritten by the position constraints after every filter, so a clip would not
    # stick and its count would be noise. The locked-row audit covers them instead.
    truncation_mask = jnp.asarray(~_locked_rows(problem))

    dirichlet = problem.boundary_conditions.dirichlet_mask.astype(bool)
    num_iterations = int(problem.solver.iteration_schedule.shape[0])

    def prescribed_at(iteration):
        """The prescribed target, applied in per-iteration increments under bounds.

        A boundary condition cannot be truncated, and applying the whole step's
        prescribed jump at the initial guess means one increment of size ``v * dt`` has
        to fit inside a single conservative bound -- which caps scripted motion near
        contact at speeds far below what the old global filter handled. Interpolating
        the prescription across the sweep divides that requirement by the iteration
        count: each increment is checked against a (re-detectable, re-anchorable)
        budget, and the final iteration lands exactly on the full target. Without
        bounds in play the full target is applied immediately, exactly as before.
        """
        if not bounds_active:
            return prescribed_position
        fraction = min(1.0, (iteration + 1) / num_iterations)
        interpolated = state.position + fraction * (
            prescribed_position - state.position
        )
        return jnp.where(dirichlet[:, None], interpolated, prescribed_position)

    dirichlet_np = np.asarray(dirichlet)

    def ensure_prescribed_budget(problem, anchor, position, target, exceeded):
        """Re-detect (re-anchor) if the next prescribed increment would breach a bound.

        Runs strictly *before* the increment is applied, because afterwards is too late
        -- the motion would already have happened, and the next detection would find an
        intersected state and crash with a misleading E1 error instead of naming the
        real cause. One re-detection (at the still-valid current positions) buys a
        fresh anchor and fresh distances; if even a fresh budget cannot absorb a single
        increment, the raise here -- pre-emptive, with the vertex and the fix named --
        is the honest outcome.
        """
        if not bounds_active:
            return problem, anchor, exceeded

        def breached(current_problem, current_anchor):
            travelled = np.linalg.norm(
                np.asarray(target) - np.asarray(current_anchor), axis=-1
            )
            bounds = np.asarray(current_problem.contact.state.vertex_bounds)
            return dirichlet_np & (travelled > bounds), travelled, bounds

        rows, _, _ = breached(problem, anchor)
        if rows.any():
            problem = _redetect_at_positions(problem, position, prescribed_position)
            anchor = position
            exceeded = 0
            rows, travelled, bounds = breached(problem, anchor)
            if rows.any():
                vertex = int(
                    np.argmax(np.where(rows, travelled - bounds, -np.inf))
                )
                raise ValueError(
                    f"prescribed vertex {vertex} would move {travelled[vertex]:g} in "
                    f"one sweep iteration, but even a freshly detected conservative "
                    f"bound there is only {bounds[vertex]:g}. The per-vertex "
                    f"intersection-free certificate (Wu et al. 2020) needs every "
                    f"vertex to stay within its bound between detections, and a "
                    f"boundary condition cannot be truncated -- at this speed a "
                    f"contact pair involving this vertex could close to zero unseen. "
                    f"Reduce solver.dt, raise solver.num_iterations (the prescription "
                    f"is applied in per-iteration increments), or slow the prescribed "
                    f"motion."
                )
        return problem, anchor, exceeded

    # The budget check must precede _compute_initial_position: the guess is what first
    # writes the prescribed rows, and a veto after the fact is no veto at all.
    initial_prescribed = prescribed_at(0)
    exceeded = 0
    problem, anchor, exceeded = ensure_prescribed_budget(
        problem, anchor, state.position, initial_prescribed, exceeded
    )
    initial_position = _compute_initial_position(
        problem, state, inertial_target, initial_prescribed
    )

    position = initial_position
    if contact_active:
        position, count = truncate_to_bounds(
            problem.contact, anchor, state.position, initial_position, truncation_mask
        )
        if bounds_active:
            exceeded = int(count)
    position = _apply_position_constraints(problem, position, initial_prescribed)
    if bounds_active:
        _audit_locked_bounds(problem, anchor, position)

    previous_position = position
    previous_weight = jnp.asarray(1.0, dtype=position.dtype)
    colliding = jnp.zeros((position.shape[0],), dtype=jnp.bool_)

    for index in range(num_iterations):
        if bounds_active and exceeded >= threshold:
            problem = _redetect_at_positions(problem, position, prescribed_position)
            anchor = position
            exceeded = 0

        step_prescribed = prescribed_at(index)
        problem, anchor, exceeded = ensure_prescribed_budget(
            problem, anchor, position, step_prescribed, exceeded
        )

        next_position, previous_position, previous_weight, colliding = (
            _accelerated_iteration(
                problem,
                position,
                previous_position,
                previous_weight,
                colliding,
                jnp.asarray(index, dtype=jnp.int32),
                inertial_target,
                state.position,
                step_prescribed,
            )
        )

        if contact_active:
            next_position, count = truncate_to_bounds(
                problem.contact, anchor, position, next_position, truncation_mask
            )
            if bounds_active:
                exceeded += int(count)
        # Re-assert Dirichlet/rigid afterwards: the truncation must not drag a
        # kinematically prescribed vertex off its target.
        next_position = _apply_position_constraints(
            problem, next_position, step_prescribed
        )
        if bounds_active:
            _audit_locked_bounds(problem, anchor, next_position)
        position = next_position

    return problem, _apply_position_constraints(
        problem, position, prescribed_position
    )


def sweep_positions(
    problem: SimulationProblem,
    state: SimulationState,
    prescribed_position: jnp.ndarray,
) -> jnp.ndarray:
    """Perform fixed color-parallel iterations over the mesh."""
    _, positions = _sweep_positions_host(problem, state, prescribed_position)
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


def _advance_step(
    problem: SimulationProblem,
    state: SimulationState,
    prescribed_position: jnp.ndarray,
    prescribed_velocity: jnp.ndarray,
) -> SimulationState:
    """Advance the mesh by one timestep. Host code: the sweep inside re-runs detection."""
    problem, next_position = _sweep_positions_host(
        problem, state, prescribed_position
    )
    next_velocity = update_velocity(
        problem, state, next_position, prescribed_velocity
    )
    return SimulationState(
        position=next_position,
        velocity=next_velocity,
        time=state.time + problem.solver.dt,
    )


def _rebuild_contact_state(
    problem: SimulationProblem,
    positions: np.ndarray,
    band: float,
) -> SimulationProblem:
    """Detect pairs at ``positions`` and install the pair set, bounds and anchor."""
    capacity = problem.contact.state.pair_vertices.shape[0]
    max_per_vertex = problem.contact.state.incident_contacts.shape[1]
    d_hat = float(problem.contact.params.d_hat)

    pair_vertices, pair_type, pair_valid, pair_distances = detect_contact_pairs(
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
    surface_vertices = np.unique(np.asarray(problem.contact.surface_triangles))
    bounds = build_vertex_bounds(
        pair_vertices,
        pair_valid,
        pair_distances,
        positions.shape[0],
        band,
        surface_vertices,
    )

    dtype = problem.contact.ccd.slack.dtype
    contact_state = ContactState(
        pair_vertices=jnp.asarray(pair_vertices, dtype=jnp.int32),
        pair_type=jnp.asarray(pair_type, dtype=jnp.int32),
        pair_valid=jnp.asarray(pair_valid, dtype=jnp.bool_),
        incident_contacts=jnp.asarray(incidence, dtype=jnp.int32),
        incident_contact_mask=jnp.asarray(mask, dtype=jnp.bool_),
        vertex_bounds=jnp.asarray(bounds, dtype=dtype),
        bound_anchor=jnp.asarray(positions, dtype=dtype),
    )
    ccd = dataclasses.replace(
        problem.contact.ccd,
        detection_band=jnp.asarray(band, dtype=dtype),
    )
    contact = dataclasses.replace(problem.contact, state=contact_state, ccd=ccd)
    return dataclasses.replace(problem, contact=contact)


def redetect_contacts(
    problem: SimulationProblem,
    state: SimulationState,
    prescribed_position: jnp.ndarray | None = None,
) -> SimulationProblem:
    """Rebuild the mesh-mesh contact set on the host and return an updated problem.

    This is the boundary between the two layers. Detection is combinatorial -- integer
    indices, an active set, a classification, per-vertex bounds -- so it runs here, in
    Python, between jitted sweeps, and everything it produces is frozen data by the time
    the device sees it.

    The rebuilt buffers keep the same shapes, so this is a jit cache hit rather than a
    recompile. That is the whole reason the capacity is fixed and overflow is an error: a
    capacity that grew with the contact count would recompile the solver every step.

    The detection *band* is derived here, from how far the mesh is about to move. Under
    the conservative-bound scheme it is a performance parameter, not a guarantee one
    (see ``contact.bounds``): it sizes the far-field vertices' bounds so a step's
    expected motion fits in one detection interval, and it has to cover the prescribed
    (Dirichlet) motion because those rows cannot be truncated -- an undersized band
    would trip the prescribed-bounds audit rather than lose the guarantee.
    """
    if not bool(problem.contact.params.enabled):
        return problem
    if problem.contact.surface_triangles.shape[0] == 0:
        return problem  # analytic colliders only: nothing combinatorial to do

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

    band = derive_bounds_band(d_hat, predicted, prescribed)
    return _rebuild_contact_state(problem, positions, band)


def _redetect_at_positions(
    problem: SimulationProblem,
    positions: jnp.ndarray,
    prescribed_position: jnp.ndarray,
) -> SimulationProblem:
    """Mid-sweep re-detection: fresh pairs, bounds and anchor at the current iterate.

    Reuses the band the step started with: it was sized for the whole step's motion, so
    it over-covers any remaining fraction of it. Runs only when enough vertices have
    consumed their bounds, so its cost is proportional to how much contact is actually
    happening -- a quiet mesh re-detects once per step, exactly as before.
    """
    del prescribed_position  # the step-start band already covered the prescribed motion
    band = float(problem.contact.ccd.detection_band)
    return _rebuild_contact_state(problem, np.asarray(positions), band)


def step(
    problem: SimulationProblem,
    state: SimulationState,
) -> SimulationState:
    """Advance the mesh by one timestep with Dirichlet targets applied."""
    prescribed_position, prescribed_velocity = evaluate_dirichlet_targets(
        problem.boundary_conditions, float(state.time), float(problem.solver.dt)
    )
    # Detection needs the prescribed targets: a Dirichlet vertex's whole-step motion is
    # known exactly, and the band must be wide enough that its bound covers it.
    problem = redetect_contacts(problem, state, prescribed_position)
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
