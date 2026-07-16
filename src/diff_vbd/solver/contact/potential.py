"""The contact energy, defined once.

This module owns the per-primitive contact energy kernels. Two things consume them:

* ``vbd.vertex_local_objective``, which sums the primitives incident to one vertex and
  differentiates w.r.t. that vertex only (giving VBD its 3x3 local force and Hessian);
* ``contact_potential`` below, which sums every primitive once over the whole mesh.

Both call the same kernel, so the forward solve and any future tangent-stiffness or
adjoint path cannot silently disagree about what the contact energy *is*. They may
legitimately differ in how they *differentiate* it -- the forward solve uses a PSD-clamped
local Hessian, which is a solver device, not the true curvature -- but they must never
differ in the energy itself.

Nothing here differentiates through detection. The active set, the collider parameters and
any classification are frozen inputs.
"""

import jax
import jax.numpy as jnp

from diff_vbd.model import ColliderData, ContactParams, ContactState
from diff_vbd.solver.contact.barrier import activation_energy
from diff_vbd.solver.contact.colliders import collider_signed_distance
from diff_vbd.solver.contact.friction import (
    collider_friction_energy,
    pair_friction_energy,
)
from diff_vbd.solver.contact.distances import (
    distance_from_squared,
    pair_distance_sq,
    pair_gap_and_mollifier,
)


@jax.jit
def pair_contact_energy(
    params: ContactParams,
    pair_positions: jnp.ndarray,
    rest_pair_positions: jnp.ndarray,
    pair_type: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    """Return the barrier energy of one mesh-mesh primitive pair.

    ``pair_positions`` is (4, 3): for a vertex-triangle pair, the vertex followed by the
    triangle; for an edge-edge pair, the two edges back to back.

    The gap, the frozen classification and the edge-edge mollifier all come from
    ``pair_gap_and_mollifier`` -- the same function detection and CCD are built on, so the
    three can never disagree about how far apart two primitives are.
    """
    gap, mollifier = pair_gap_and_mollifier(
        pair_positions, rest_pair_positions, pair_type
    )

    energy = activation_energy(params.activation, gap, params.d_hat, valid)
    return params.kappa * mollifier * energy


@jax.jit
def collider_contact_energy(
    params: ContactParams,
    colliders: ColliderData,
    position: jnp.ndarray,
) -> jnp.ndarray:
    """Return the contact energy of a single vertex against every analytic collider.

    This is the kernel. ``position`` is the one live vertex; every collider parameter is
    frozen data. Summing over colliders (rather than over an incidence list) is exactly
    right for analytic colliders: there is no pair list to build, so the combinatorial
    layer has nothing to do and the active set is just a mask.
    """

    def one_collider(kind, normal, offset, center, radius, outside, enabled):
        gap = collider_signed_distance(
            position, kind, normal, offset, center, radius, outside
        )
        active = enabled & params.enabled
        return activation_energy(params.activation, gap, params.d_hat, active)

    energies = jax.vmap(one_collider)(
        colliders.kind,
        colliders.normal,
        colliders.offset,
        colliders.center,
        colliders.radius,
        colliders.outside,
        colliders.enabled,
    )
    return params.kappa * jnp.sum(energies)


@jax.jit
def incident_pair_energy(
    params: ContactParams,
    state: ContactState,
    rest_positions: jnp.ndarray,
    block_position: jnp.ndarray,
    previous_position: jnp.ndarray,
    dt: jnp.ndarray,
    vertex_index: jnp.ndarray,
    x_i: jnp.ndarray,
) -> jnp.ndarray:
    """Return the barrier *and friction* energy of every mesh-mesh pair incident to a vertex.

    Mirrors ``vertex_local_objective``'s tet accumulation exactly: the other participants
    are read from the frozen ``block_position`` snapshot and only ``x_i`` is live, so the
    Hessian w.r.t. ``x_i`` stays a clean 3x3.

    One difference from the tet path matters. The tet accumulation pads with tet index 0 and
    zeroes the garbage by *multiplying* by the mask, which is safe only because a tet's
    energy is always finite. A padded contact pair is typically coincident, and a barrier at
    zero distance is enormous -- so the mask has to be applied *inside* the kernel, before
    the barrier ever sees the distance. Multiplying afterwards would not save us:
    ``0 * inf`` is NaN.
    """

    # Analytic colliders only: there are no pair buffers at all. Shapes are static inside
    # jit, so this is a compile-time branch -- and it is necessary, because XLA rejects a
    # gather from a zero-length axis even when the vmap over it would iterate zero times.
    if state.pair_vertices.shape[0] == 0 or state.incident_contacts.shape[1] == 0:
        return jnp.zeros((), dtype=x_i.dtype)

    def one_pair(pair_index, is_valid):
        vertices = state.pair_vertices[pair_index]
        positions = block_position[vertices]
        # Substitute the live vertex wherever it appears in this primitive.
        matches = (vertices == vertex_index)[:, None]
        positions = jnp.where(matches, x_i[None, :], positions)
        rest = rest_positions[vertices]
        # The lagged reference is never substituted: this vertex's own step-start position is
        # already `previous_position[vertex_index]`, which is the row we would be writing.
        previous = previous_position[vertices]
        pair_type = state.pair_type[pair_index]
        active = is_valid & state.pair_valid[pair_index] & params.enabled

        barrier = pair_contact_energy(params, positions, rest, pair_type, active)
        # Friction costs about as much as the barrier does -- it differentiates the distance
        # function, and this whole kernel is already inside a Hessian -- so a frictionless run
        # should not pay for it. The predicate does not depend on `pair_index`, so it survives
        # the vmap as a real branch rather than collapsing into a select that evaluates both.
        friction = jax.lax.cond(
            params.friction_mu > 0.0,
            lambda: pair_friction_energy(
                params, positions, previous, rest, pair_type, active, dt
            ),
            lambda: jnp.zeros((), dtype=barrier.dtype),
        )
        return barrier + friction

    incident = state.incident_contacts[vertex_index]
    mask = state.incident_contact_mask[vertex_index]
    energies = jax.vmap(one_pair)(incident, mask)
    return jnp.sum(energies)


@jax.jit
def incident_pair_min_gap(
    state: ContactState,
    block_position: jnp.ndarray,
    vertex_index: jnp.ndarray,
) -> jnp.ndarray:
    """Return the distance from one vertex to its nearest incident mesh-mesh pair.

    The mesh-mesh half of the *local* step safeguard, and the counterpart of
    ``ccd.vertex_time_of_impact`` for analytic colliders. A primitive distance is 1-Lipschitz
    in each of its vertices, so moving this vertex alone by less than the returned gap cannot
    drive any of its pairs to zero -- the same one-line argument as ``ccd._lipschitz_toi``.

    This is a *heuristic*, not the guarantee (the sweep filter is that), and it exists for one
    specific reason: with the line search disabled, the solver takes the raw Newton step, and
    a vertex sitting in a stiff barrier will propose an enormous one. The sweep filter would
    then dutifully scale the *entire mesh* back by that one vertex's excess and the solve would
    grind to a halt -- correct, but useless. Bounding the step where it is proposed keeps one
    bad vertex from throttling everybody.

    Full per-vertex ACCD would be the rigorous version and is far too expensive: it would run
    the iterative kernel over up to ``max_per_vertex`` pairs inside the innermost loop, per
    vertex, per colour, per sweep. This costs one distance each.

    Returns ``inf`` when the vertex has no incident pairs, so it drops out of a ``minimum``.
    """
    infinity = jnp.asarray(jnp.inf, dtype=block_position.dtype)
    if state.pair_vertices.shape[0] == 0 or state.incident_contacts.shape[1] == 0:
        return infinity

    def one_pair(pair_index, is_valid):
        vertices = state.pair_vertices[pair_index]
        gap = distance_from_squared(
            pair_distance_sq(block_position[vertices], state.pair_type[pair_index])
        )
        active = is_valid & state.pair_valid[pair_index]
        return jnp.where(active, gap, infinity)

    gaps = jax.vmap(one_pair)(
        state.incident_contacts[vertex_index],
        state.incident_contact_mask[vertex_index],
    )
    return jnp.min(jnp.concatenate([gaps, infinity[None]]))


@jax.jit
def colliding_vertex_mask(
    params: ContactParams,
    colliders: ColliderData,
    state: ContactState,
    positions: jnp.ndarray,
) -> jnp.ndarray:
    """Return, per vertex, whether *any* contact is within the activation distance.

    Used to switch Chebyshev acceleration off for vertices in contact. The extrapolation
    is a global convergence trick that assumes a smooth, roughly linear iteration; a
    barrier is neither, and extrapolating across it drives a vertex straight at the
    obstacle faster than the local solves can push it away. The VBD paper skips
    acceleration for colliding vertices for exactly this reason.

    Mesh-mesh pairs count, not just analytic colliders. A vertex in self-contact is subject
    to precisely the same instability, and leaving it extrapolated was a live bug: the
    Chebyshev step is applied after every local solve has finished, so no per-vertex check
    inside the solve can see it coming.

    The criterion is the true distance against ``d_hat`` -- **not** "appears in the candidate
    list". Those are very different questions: the pair set is built with a band that grows
    with the mesh's speed and can be far larger than ``d_hat``, so the cheap test would flag
    every vertex merely *near* a contact and quietly disable acceleration across half the
    mesh. That failure costs performance rather than correctness, which is exactly why it
    would never be noticed.
    """

    def vertex_touches_collider(position):
        def one_collider(kind, normal, offset, center, radius, outside, enabled):
            gap = collider_signed_distance(
                position, kind, normal, offset, center, radius, outside
            )
            return enabled & (gap < params.d_hat)

        touching = jax.vmap(one_collider)(
            colliders.kind,
            colliders.normal,
            colliders.offset,
            colliders.center,
            colliders.radius,
            colliders.outside,
            colliders.enabled,
        )
        return jnp.any(touching) & params.enabled

    touching_collider = jax.vmap(vertex_touches_collider)(positions)

    # Static branch, as in `incident_pair_energy`: XLA rejects a gather from a zero-length
    # axis even where the vmap over it would iterate zero times.
    if state.pair_vertices.shape[0] == 0 or state.incident_contacts.shape[1] == 0:
        return touching_collider

    # Activity is computed once per *pair* (O(C) distances), then gathered through the
    # incidence table -- far cheaper than re-deriving it per incidence.
    def pair_is_active(vertices, pair_type, valid):
        gap = distance_from_squared(pair_distance_sq(positions[vertices], pair_type))
        return valid & (gap < params.d_hat)

    active = jax.vmap(pair_is_active)(
        state.pair_vertices, state.pair_type, state.pair_valid
    )
    incident = active[state.incident_contacts] & state.incident_contact_mask
    return touching_collider | (jnp.any(incident, axis=1) & params.enabled)


@jax.jit
def contact_potential(
    params: ContactParams,
    colliders: ColliderData,
    state: ContactState,
    rest_positions: jnp.ndarray,
    positions: jnp.ndarray,
    previous_positions: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Return the mesh's total contact energy, counting each PRIMITIVE exactly once.

    All four terms the local objective sums: collider barrier, collider friction, pair
    barrier, pair friction. This is the whole-mesh counterpart of ``vertex_local_objective``'s
    contact contribution, and the hook a sensitivity or adjoint path attaches to -- one energy
    definition, two consumers, so they cannot silently drift apart.

    **The counting is the entire point.** ``vertex_local_objective`` sums the primitives
    *incident to* a vertex, so summing it over every vertex counts a 4-vertex pair four times.
    This function sums over primitives, never over incidences:

        contact_potential          == sum_v [collider terms] + sum_c [pair terms]
        sum_v incident_pair_energy == 4 * sum_c [pair terms]

    and the factor is exactly 4, never anything else, because detection skips primitives that
    share a vertex so every valid pair has four distinct vertices. For analytic colliders each
    element touches exactly one vertex and the two counts coincide -- which is precisely why
    the invariant looked healthy while it was quietly false for pairs.

    The solver does **not** call this, and that is deliberate rather than neglect: VBD's whole
    premise is a 3x3 solve against a frozen block, and the global contact gradient is an object
    it never forms and would gain nothing from. It exists for the invariant tests and for a
    future adjoint path.

    ``previous_positions`` is the lagged reference the friction terms are measured from. Pass a
    *distinct* array -- passing ``positions`` itself puts the live positions on both sides of a
    ``stop_gradient``, and the gradient then legitimately disagrees with a finite difference.
    """
    collider_barrier = jnp.sum(
        jax.vmap(lambda x: collider_contact_energy(params, colliders, x))(positions)
    )
    collider_friction = jnp.sum(
        jax.vmap(
            lambda x, previous: collider_friction_energy(
                params, colliders, x, previous, dt
            )
        )(positions, previous_positions)
    )

    # Static branch, as in `incident_pair_energy`: an analytic-collider-only problem has no
    # pair buffers, and XLA rejects a gather from a zero-length axis.
    if state.pair_vertices.shape[0] == 0:
        return collider_barrier + collider_friction

    def one_pair(vertices, pair_type, valid):
        active = valid & params.enabled
        live = positions[vertices]
        previous = previous_positions[vertices]
        rest = rest_positions[vertices]
        return pair_contact_energy(
            params, live, rest, pair_type, active
        ) + pair_friction_energy(
            params, live, previous, rest, pair_type, active, dt
        )

    pair_energies = jax.vmap(one_pair)(
        state.pair_vertices, state.pair_type, state.pair_valid
    )
    return collider_barrier + collider_friction + jnp.sum(pair_energies)
