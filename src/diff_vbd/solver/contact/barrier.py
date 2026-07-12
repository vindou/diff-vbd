"""IPC log barrier (Li et al. 2020) and the quadratic-penalty fallback.

    b(g; d_hat) = -(g - d_hat)^2 * log(g / d_hat)   for 0 < g < d_hat,  else 0

``b``, ``b'`` and ``b''`` all vanish at ``g == d_hat``, so the energy is C2 across the
activation boundary and the solver never sees a kink switching a contact on or off.

The argument is a **gap in length units**, not a squared distance. That choice is what lets
an analytic collider hand the barrier a *signed* distance -- negative meaning the vertex is
on the wrong side -- while a mesh pair hands it an unsigned distance from
``distances.distance_from_squared``. Both then share a single ``d_hat`` and a single
``kappa``. Squaring would throw away the sign, and a vertex that tunnelled through a plane
would be pushed further out the wrong side rather than back.
"""

import jax
import jax.numpy as jnp

# Floor on g / d_hat inside the log. A gap at or below zero means the CCD filter has
# already failed; below the floor the barrier is continued linearly (see below).
_MIN_GAP_RATIO = 1.0e-9


@jax.jit
def barrier_energy(
    gap: jnp.ndarray,
    d_hat: jnp.ndarray,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Return the IPC barrier energy for a gap.

    ``active`` is the frozen mask: false for a padded slot, a primitive outside the
    activation range, or an otherwise degenerate pair.

    Two guards, and both are load bearing.

    The double ``where`` is not redundant: ``jnp.where`` evaluates *both* branches under
    ``jax.grad``, so the inner substitution is what stops an inactive slot from feeding
    ``log`` a non-positive number and back-propagating a NaN into a gradient that is then
    multiplied by zero -- and ``0 * NaN`` is NaN, not 0. Masking the *result* alone is not
    enough; the argument must be safe before the log ever sees it.

    Below the floor the barrier is continued **linearly**, not clamped. Clamping with a
    ``maximum`` would make the energy constant there, and the derivative of a constant is
    zero -- so a vertex that had already been pushed through would feel no restoring force
    at all and could never escape. That is exactly the failure the elastic energy in this
    repo used to have (a ``lax.cond`` returning a constant barrier whose gradient was
    identically zero). The linear continuation is C1 at the floor and carries the huge
    restoring gradient outward, so a penetrated vertex is driven back.
    """
    in_range = active & (gap < d_hat)
    floor = _MIN_GAP_RATIO * d_hat

    # Inactive slots are evaluated at d_hat, where the barrier and all of its derivatives
    # are exactly zero -- so nothing non-finite is produced, in value or in gradient.
    safe = jnp.where(in_range, gap, d_hat)

    # Smooth branch. A `where` rather than a `maximum`, because `maximum` splits the
    # gradient 50/50 exactly at the tie point.
    g_log = jnp.where(safe < floor, floor, safe)
    smooth = -((g_log - d_hat) ** 2) * jnp.log(g_log / d_hat)

    # Linear continuation below the floor (this is the branch a penetrated vertex takes).
    # Both coefficients are evaluated at the constant floor, so they are finite; `safe` is
    # what carries the gradient.
    log_floor = jnp.log(floor / d_hat)
    value_at_floor = -((floor - d_hat) ** 2) * log_floor
    slope_at_floor = (
        -2.0 * (floor - d_hat) * log_floor - ((floor - d_hat) ** 2) / floor
    )
    extrapolated = value_at_floor + slope_at_floor * (safe - floor)

    energy = jnp.where(safe < floor, extrapolated, smooth)
    return jnp.where(in_range, energy, jnp.zeros_like(energy))


@jax.jit
def penalty_energy(
    gap: jnp.ndarray,
    d_hat: jnp.ndarray,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Return the VBD quadratic-penalty energy: a cheap, non-intersection-free fallback.

    ``0.5 * (d_hat - g)^2`` inside the activation range. C1 but not C2 at the boundary, and
    it permits penetration -- which is the whole reason the barrier exists. Worth having
    when intersection-freedom is not required and the barrier's stiffness is not worth
    paying for.
    """
    in_range = active & (gap < d_hat)
    excess = d_hat - gap
    energy = 0.5 * excess * excess
    return jnp.where(in_range, energy, jnp.zeros_like(energy))


def barrier_stiffness(
    average_mass: float,
    dt: float,
    d_hat: float,
    *,
    scale: float = 1.0,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return a dimensionally consistent default barrier stiffness ``kappa``.

    ``b`` is quadratic in a length, so it carries units of length^2 and ``kappa`` carries
    energy / length^2. The inertia term of the local objective contributes a stiffness of
    ``m / dt^2``, in the same units. Matching them,

        kappa ~ scale * m / dt^2

    makes the barrier comparable in stiffness to inertia at the moment it activates: stiff
    enough to actually resist, soft enough not to swamp the local solve. This is IPC's
    adaptive-kappa idea in its simplest defensible form; ``scale`` and the clamps are the
    tuning knobs.
    """
    if d_hat <= 0.0:
        raise ValueError(f"d_hat must be positive, got {d_hat}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    kappa = scale * average_mass / (dt * dt)
    if minimum is not None:
        kappa = max(kappa, minimum)
    if maximum is not None:
        kappa = min(kappa, maximum)
    return float(kappa)
