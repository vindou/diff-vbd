"""Contact activation energies: the IPC log barrier, the quadratic penalty, and OGC's
two-stage function.

    barrier   (Li et al. 2020):   b(g) = -(g - d_hat)^2 * log(g / d_hat)   for 0 < g < d_hat
    penalty   (VBD paper):        p(g) = (d_hat - g)^2 / 2                 for     g < d_hat
    two-stage (Chen et al. 2025): g(d) = (d_hat - d)^2 / 2                 for tau <= d < d_hat
                                         k' * log(tau / d) + (d_hat - tau)^2 / 2   for d < tau

All three are *unit-stiffness*: the caller multiplies by ``kappa`` (``k_c`` in OGC's
notation), so one stiffness parameter serves every activation. ``ACTIVATION_KINDS`` maps
names to the int32 codes ``ContactParams.activation`` stores -- an int code rather than a
string for the same reason collider kinds are: every field of a ``pytree_dataclass`` is a
traced leaf.

Smoothness is where the three differ, and the differences are the point:

* the IPC barrier is C2 at ``d_hat`` (b, b', b'' all vanish there) but its curvature blows
  up like ``1/g`` as the gap closes, which is what makes it stiff;
* the penalty is C1 at ``d_hat`` and bounded everywhere -- and therefore permits penetration;
* the two-stage function is C2 at its interior stitch ``tau = d_hat / 2`` (that choice is
  exactly what makes ``g'' = k_c`` on both sides) but only **C1 at d_hat**: its curvature
  steps from ``k_c`` to 0 there, as any quadratic activation's must. What it buys is a
  *bounded* curvature over most of the active range -- ``g'' = k_c`` on the whole quadratic
  stage -- with the log stage's infinite force reserved for the last ``tau`` of gap. The
  C1 matching constants are derived in ``two_stage_energy``; note the paper's printed
  coefficient (its Eq. 19) is dimensionally inconsistent and is corrected there.

The argument is a **gap in length units**, not a squared distance. That choice is what lets
an analytic collider hand the activation a *signed* distance -- negative meaning the vertex
is on the wrong side -- while a mesh pair hands it an unsigned distance from
``distances.distance_from_squared``. Both then share a single ``d_hat`` and a single
``kappa``. Squaring would throw away the sign, and a vertex that tunnelled through a plane
would be pushed further out the wrong side rather than back.
"""

import jax
import jax.numpy as jnp

ACTIVATION_KINDS = {
    "BARRIER": 0,
    "PENALTY": 1,
    "TWO_STAGE": 2,
}

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


@jax.jit
def two_stage_energy(
    gap: jnp.ndarray,
    d_hat: jnp.ndarray,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Return OGC's two-stage activation energy (Chen et al. 2025, Eq. 18) for a gap.

    A quadratic near ``d_hat`` -- cheap, bounded curvature, fast convergence -- stitched
    C2-continuously at ``tau = d_hat / 2`` onto a pure log that carries the infinite
    force a non-penetration guarantee needs. In OGC's notation ``r = d_hat``,
    ``k_c = kappa`` (applied by the caller), and the stitch constants come from C1
    matching at ``tau``:

        g1'(tau) == g2'(tau):  -k_c (r - tau) == -k'_c / tau  =>  k'_c = tau k_c (r - tau)
        g1(tau)  == g2(tau):   b = k_c/2 (r - tau)^2 + k'_c log(tau)

    **The paper's printed Eq. 19 is ``k'_c = tau k_c (tau - r)^2``, and it is wrong** --
    dimensionally (energy*length, where ``k'_c`` must carry energy, since ``g''`` must
    carry energy/length^2) and by the paper's own C2 result: with the corrected ``k'_c``,
    ``g2''(tau) = k'_c / tau^2 = k_c (r - tau) / tau`` equals ``g1'' = k_c`` exactly at
    ``tau = r/2``, which is the choice the paper itself makes. The printed form does not
    reproduce that. The log is written as ``log(tau / d)`` -- a ratio, so ``b``'s "log of
    a length" never appears and the two forms are algebraically identical.

    At ``d_hat`` the function is only **C1**: the value and slope vanish, but the
    curvature steps from ``k_c`` to zero, as it must for any quadratic activation (the
    IPC barrier is C2 there; the two-stage function trades that for bounded curvature
    across the whole quadratic stage). The test pins the size of the step rather than
    pretending it away.

    Guard structure is copied from ``barrier_energy``, and both guards are load bearing
    there and here: inactive slots are evaluated at ``d_hat`` (where energy and slope are
    exactly zero) so no non-finite value is *produced*, and below the floor the log stage
    is continued **linearly** so a penetrated vertex keeps a huge but finite restoring
    gradient -- a clamp would zero it exactly where it is needed most.
    """
    tau = 0.5 * d_hat
    k_prime = tau * (d_hat - tau)  # the corrected Eq. 19; == d_hat^2 / 4 at tau = d_hat/2

    in_range = active & (gap < d_hat)
    floor = _MIN_GAP_RATIO * d_hat

    # Inactive slots are evaluated at d_hat, where the quadratic stage and its derivative
    # are exactly zero -- so nothing non-finite is produced, in value or in gradient.
    safe = jnp.where(in_range, gap, d_hat)

    quadratic = 0.5 * (d_hat - safe) ** 2

    # Log stage, with the argument floored by substitution (a `where`, not a `maximum`,
    # for the same gradient-splitting reason as in `barrier_energy`).
    g_log = jnp.where(safe < floor, floor, safe)
    log_stage = 0.5 * (d_hat - tau) ** 2 + k_prime * jnp.log(tau / g_log)

    # Linear continuation below the floor: C1 at the floor, slope -k'_c / floor, so a
    # vertex pushed through is driven back rather than feeling a constant's zero gradient.
    value_at_floor = 0.5 * (d_hat - tau) ** 2 + k_prime * jnp.log(tau / floor)
    extrapolated = value_at_floor - (k_prime / floor) * (safe - floor)

    energy = jnp.where(
        safe >= tau,
        quadratic,
        jnp.where(safe < floor, extrapolated, log_stage),
    )
    return jnp.where(in_range, energy, jnp.zeros_like(energy))


@jax.jit
def activation_energy(
    activation: jnp.ndarray,
    gap: jnp.ndarray,
    d_hat: jnp.ndarray,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Dispatch on the activation kind and return the unit-stiffness contact energy.

    The single switch every consumer goes through -- the pair kernel, the collider kernel
    and the friction normal force all dispatch here, so they cannot disagree about which
    activation the solver is running.
    """
    return jax.lax.switch(
        activation,
        (
            lambda: barrier_energy(gap, d_hat, active),
            lambda: penalty_energy(gap, d_hat, active),
            lambda: two_stage_energy(gap, d_hat, active),
        ),
    )


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
