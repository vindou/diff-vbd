"""Lagged, smoothed Coulomb friction (IPC / VBD).

Coulomb friction is not a potential: the friction force depends on the normal force, which
depends on the configuration, and the stick/slip law has a kink at zero sliding velocity.
Two approximations turn it into something a Newton solver can actually descend.

**Lagging.** The normal force magnitude ``lambda`` and the tangent basis ``T`` are evaluated
once, at the start of the step, and then held fixed. With them frozen, what remains *is* a
genuine potential in the current position -- so friction becomes just another force element
with a gradient and a Hessian, and VBD's autodiff picks it up like everything else. This is
the same freeze-the-combinatorics discipline the rest of the contact stack runs on.

**Smoothing.** The Coulomb law is discontinuous at zero sliding speed (static friction can
take any value up to ``mu * lambda``). ``f1`` replaces that jump with a steep-but-smooth ramp
over a sliding speed of ``eps_v``: below it friction behaves like a stiff tangential spring
that holds the contact stuck, above it saturates at ``mu * lambda`` and the contact slides.
"""

import jax
import jax.numpy as jnp

from diff_vbd.model import ColliderData, ContactParams
from diff_vbd.solver.contact.barrier import barrier_energy, penalty_energy
from diff_vbd.solver.contact.colliders import collider_normal, collider_signed_distance
from diff_vbd.solver.contact.distances import pair_gap_and_mollifier

# Softening inside the norm of the tangential slip, so d|u|/du is finite at u == 0.
_SLIP_SOFTENING = 1.0e-24


@jax.jit
def contact_normal_force(
    params: ContactParams, gap: jnp.ndarray, active: jnp.ndarray
) -> jnp.ndarray:
    """Return the magnitude of the normal force the contact potential applies at ``gap``.

    Coulomb friction is bounded by ``mu`` times the normal force, so it has to be the force
    the solver *actually applies* -- which means dispatching on ``use_barrier`` exactly as
    the energy does. Reading it off the barrier while the solve is running the quadratic
    penalty would drive friction from a force that is not there.

    Both energies return an exactly-zero gradient when ``active`` is false, so an inactive or
    padded slot yields a zero force with no NaN.
    """
    force = jax.lax.cond(
        params.use_barrier,
        lambda: jax.grad(barrier_energy)(gap, params.d_hat, active),
        lambda: jax.grad(penalty_energy)(gap, params.d_hat, active),
    )
    return params.kappa * jnp.abs(force)


@jax.jit
def tangent_basis(normal: jnp.ndarray) -> jnp.ndarray:
    """Return a (3, 2) orthonormal basis for the plane perpendicular to ``normal``."""
    # Pick the axis least aligned with the normal, so the cross product is well conditioned.
    axis = jnp.where(
        jnp.abs(normal[0]) < 0.9,
        jnp.array([1.0, 0.0, 0.0], dtype=normal.dtype),
        jnp.array([0.0, 1.0, 0.0], dtype=normal.dtype),
    )
    first = jnp.cross(normal, axis)
    first = first / jnp.sqrt(jnp.dot(first, first) + 1.0e-24)
    second = jnp.cross(normal, first)
    return jnp.stack([first, second], axis=1)


@jax.jit
def smooth_friction_f0(slip: jnp.ndarray, eps_v_h: jnp.ndarray) -> jnp.ndarray:
    """Return the friction *potential* shape function, whose derivative is ``f1``.

    Piecewise, and C1 at the transition:

        f0(y) = -y^3 / (3 (eps_v h)^2) + y^2 / (eps_v h) + eps_v h / 3   for y < eps_v h
        f0(y) = y                                                        otherwise

    The quadratic leading term near zero is what makes this a *spring*: the gradient
    vanishes at zero slip, so a stuck contact has no spurious force, and it grows linearly
    with slip until friction saturates.
    """
    scaled = slip / eps_v_h
    sticking = eps_v_h * (
        -(scaled**3) / 3.0 + scaled**2 + 1.0 / 3.0
    )
    return jnp.where(slip < eps_v_h, sticking, slip)


@jax.jit
def smooth_friction_f1(slip: jnp.ndarray, eps_v_h: jnp.ndarray) -> jnp.ndarray:
    """Return the friction force shape function in [0, 1]: the derivative of ``f0``.

    Zero at zero slip, rising to 1 (fully sliding, force saturated at ``mu * lambda``) once
    the slip exceeds ``eps_v h``.
    """
    scaled = slip / eps_v_h
    return jnp.where(slip < eps_v_h, scaled * (2.0 - scaled), jnp.ones_like(scaled))


@jax.jit
def collider_friction_energy(
    params: ContactParams,
    colliders: ColliderData,
    position: jnp.ndarray,
    previous_position: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Return one vertex's friction potential against every analytic collider.

    ``previous_position`` is the vertex at the start of the step: it is the reference the
    tangential slip is measured from, and everything derived from it -- the gap, the normal
    force, the tangent basis -- is lagged and frozen.
    """

    def one_collider(kind, normal, offset, center, radius, outside, enabled):
        # --- lagged, frozen: evaluated at the step-start position, never differentiated ---
        previous_gap = collider_signed_distance(
            previous_position, kind, normal, offset, center, radius, outside
        )
        active = enabled & params.enabled & (previous_gap < params.d_hat)
        # The normal force magnitude is the contact potential's own force at the lagged gap.
        normal_force = contact_normal_force(params, previous_gap, active)
        contact_normal = collider_normal(
            previous_position, kind, normal, center, outside
        )
        basis = tangent_basis(contact_normal)
        normal_force, basis = jax.lax.stop_gradient((normal_force, basis))

        # --- smooth: a function of the live position, and the only part differentiated ---
        slip_vector = basis.T @ (position - previous_position)
        slip = jnp.sqrt(jnp.dot(slip_vector, slip_vector) + _SLIP_SOFTENING)
        eps_v_h = params.eps_v * dt
        energy = params.friction_mu * normal_force * smooth_friction_f0(slip, eps_v_h)
        return jnp.where(active, energy, jnp.zeros_like(energy))

    energies = jax.vmap(one_collider)(
        colliders.kind,
        colliders.normal,
        colliders.offset,
        colliders.center,
        colliders.radius,
        colliders.outside,
        colliders.enabled,
    )
    return jnp.sum(energies)


# Softening on the contact normal's normalisation. Only ever exercised by a degenerate
# (coincident) pair, where the contact normal genuinely does not exist.
_NORMAL_SOFTENING = 1.0e-24


def _pair_gap(pair_positions, rest_pair_positions, pair_type):
    gap, _ = pair_gap_and_mollifier(pair_positions, rest_pair_positions, pair_type)
    return gap


# The whole trick, in one line. The gradient of a pair's gap with respect to its four stacked
# points is `g_k = w_k * n`: the unit contact normal times the signed barycentric weights of
# the two closest points. So the normal *and* the weights both fall straight out of the
# distance function that detection and the barrier already use -- no new geometry code, and by
# construction the three cannot disagree about where the contact is.
#
# It is exact, not approximate. Each branch of `distances.py` is an unconstrained minimum over
# the sub-features' *affine hulls* (point-to-line, point-to-plane, line-to-line), so the
# envelope theorem applies with no boundary term and the closest-point parameters are
# stationary. Freezing the distance type is what makes each branch such a minimum -- the same
# structural fact that makes the barrier differentiable at all.
#
# Take the gradient of the *gap*, not of the squared distance: the latter is `2 * d * w_k * n`,
# which is zero exactly at contact and therefore useless. And the gap's softening means that as
# `d -> 0` the normal and the weights collapse to zero rather than becoming NaN -- friction
# vanishes precisely where the contact normal is genuinely undefined, and the barrier (which
# does not vanish there) is what handles that configuration.
_pair_gap_gradient = jax.grad(_pair_gap, argnums=0)


@jax.jit
def pair_friction_energy(
    params: ContactParams,
    pair_positions: jnp.ndarray,
    previous_pair_positions: jnp.ndarray,
    rest_pair_positions: jnp.ndarray,
    pair_type: jnp.ndarray,
    valid: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Return the friction potential of one mesh-mesh primitive pair.

    The mesh-mesh counterpart of ``collider_friction_energy``, with the same lagged structure:
    the normal force, the contact normal and the barycentric weights are evaluated once at the
    step-start configuration and frozen, leaving a genuine potential in the live positions that
    VBD's autodiff picks up like any other force element.

    One thing is **not** like the collider path, and it is the whole difference between them. A
    collider is static, so a vertex's slip is simply its own displacement. A mesh pair has two
    moving sides, so the slip is the *relative* displacement of the two closest points,
    ``sum_k w_k (x_k - x_k_prev)``. Copying the collider kernel per-vertex would measure
    absolute displacement instead, and would then put an enormous friction force on a pair that
    is merely translating rigidly through space -- which is not friction.

    NaN-safe by construction rather than by masking: every quantity that could be degenerate
    (the normal, the weights, the normal force, the tangent basis) is computed from the frozen
    previous positions and immediately ``stop_gradient``-ed, and the only live quantity enters
    *linearly*. A degenerate pair therefore yields frozen zeros multiplying a live linear term,
    which is zero -- never ``0 * inf``.
    """
    # --- lagged, frozen: evaluated at the step-start positions, never differentiated ------
    previous_gap, mollifier = pair_gap_and_mollifier(
        previous_pair_positions, rest_pair_positions, pair_type
    )
    active = valid & params.enabled & (previous_gap < params.d_hat)

    gradient = _pair_gap_gradient(
        previous_pair_positions, rest_pair_positions, pair_type
    )  # (4, 3); every row is parallel to the contact normal

    # |g_k| == |w_k|, and each side's weights sum to +-1, so the largest row always has
    # |w| >= 1/2: normalising *it* is unconditionally well conditioned, and needs no knowledge
    # of which rows belong to which primitive. Its sign is arbitrary and harmlessly so -- the
    # energy depends only on ||u||, which is invariant under n -> -n and under any orthonormal
    # basis of the same tangent plane.
    row = jnp.argmax(jnp.sum(gradient * gradient, axis=1))
    contact_normal = gradient[row] / jnp.sqrt(
        jnp.dot(gradient[row], gradient[row]) + _NORMAL_SOFTENING
    )
    weights = gradient @ contact_normal  # (4,)
    basis = tangent_basis(contact_normal)  # (3, 2)

    # The mollifier belongs here too. A near-parallel edge pair whose *barrier* has been
    # mollified to zero carries no normal force, and so must carry no friction either --
    # otherwise it feels a phantom tangential drag from a contact that is not pushing on it.
    normal_force = mollifier * contact_normal_force(params, previous_gap, active)
    normal_force, basis, weights = jax.lax.stop_gradient(
        (normal_force, basis, weights)
    )

    # --- smooth: a function of the live positions, and the only part differentiated --------
    displacement = pair_positions - previous_pair_positions  # (4, 3)
    relative = weights @ displacement  # (3,) relative motion of the two closest points
    slip_vector = basis.T @ relative  # (2,)
    slip = jnp.sqrt(jnp.dot(slip_vector, slip_vector) + _SLIP_SOFTENING)
    energy = (
        params.friction_mu
        * normal_force
        * smooth_friction_f0(slip, params.eps_v * dt)
    )
    return jnp.where(active, energy, jnp.zeros_like(energy))
