"""Per-vertex conservative time of impact against the analytic colliders.

This is the combinatorial layer. Nothing here is differentiated -- a time of impact is a
step-size bound, not a force -- so it is wrapped in ``stop_gradient`` where it meets the
smooth layer.

``vertex_time_of_impact`` bounds a *single* vertex's motion segment against the static
colliders. That per-vertex certificate is sound on its own precisely because a collider
never moves: unlike a mesh-mesh pair, there is no other party whose simultaneous motion
the certificate would have to cover. It serves two consumers: the local Newton-step
safeguard (which stops the line-search-disabled path from proposing a step through an
obstacle), and the per-iteration truncation in ``contact.bounds`` (which clips every
vertex's realised motion, initial guess and Chebyshev extrapolation included).

The mesh-mesh guarantee does not live here any more. It used to be a global sweep filter
built on additive CCD over the pair set; it is now the per-vertex conservative bounds of
Wu et al. 2020 -- see ``contact.bounds`` for the scheme and for why the certificates
compose without any CCD at all.
"""

import jax
import jax.numpy as jnp

from diff_vbd.model import ColliderData

# Fraction of the available gap a step is allowed to consume. Strictly less than 1 so a
# vertex can approach an obstacle but never actually reach it -- which is what keeps the
# barrier's argument strictly positive and its logarithm finite.
_DEFAULT_SLACK = 0.9


@jax.jit
def _plane_toi(
    start: jnp.ndarray,
    end: jnp.ndarray,
    normal: jnp.ndarray,
    offset: jnp.ndarray,
    slack: jnp.ndarray,
) -> jnp.ndarray:
    """Exact time of impact against a plane.

    The signed distance is linear along the segment, so this is solved rather than bounded.
    Worth the special case: the conservative Lipschitz bound below would throttle motion
    *parallel* to a plane, which is precisely the sliding a resting body wants to do.
    """
    start_gap = jnp.dot(normal, start) - offset
    end_gap = jnp.dot(normal, end) - offset

    approaching = end_gap < start_gap
    # Allowed to close `slack` of the gap: gap(t) = (1 - slack) * start_gap.
    denominator = start_gap - end_gap
    safe_denominator = jnp.where(approaching, denominator, 1.0)
    toi = slack * start_gap / safe_denominator

    # Not approaching: the whole step is safe. Already through: only motion that increases
    # the gap is allowed, so a penetrated vertex can climb back out rather than freeze.
    toi = jnp.where(approaching, toi, 1.0)
    toi = jnp.where(start_gap <= 0.0, jnp.where(approaching, 0.0, 1.0), toi)
    return jnp.clip(toi, 0.0, 1.0)


@jax.jit
def _lipschitz_toi(
    start_gap: jnp.ndarray, displacement: jnp.ndarray, slack: jnp.ndarray
) -> jnp.ndarray:
    """Conservative time of impact for any 1-Lipschitz signed distance field.

    A true signed distance cannot fall faster than the distance travelled, so

        gap(t) >= gap(0) - t * |dx|

    and requiring the right-hand side to stay above ``(1 - slack) * gap(0)`` gives
    ``t <= slack * gap(0) / |dx|``. This is additive CCD in one line: safe for a sphere or
    any SDF, at the cost of assuming the motion is aimed straight at the obstacle.
    """
    travel = jnp.sqrt(jnp.dot(displacement, displacement) + 1.0e-30)
    toi = slack * start_gap / travel
    toi = jnp.where(start_gap <= 0.0, 0.0, toi)
    return jnp.clip(toi, 0.0, 1.0)


@jax.jit
def _collider_toi(
    start: jnp.ndarray,
    end: jnp.ndarray,
    kind: jnp.ndarray,
    normal: jnp.ndarray,
    offset: jnp.ndarray,
    center: jnp.ndarray,
    radius: jnp.ndarray,
    outside: jnp.ndarray,
    enabled: jnp.ndarray,
    slack: jnp.ndarray,
) -> jnp.ndarray:
    from diff_vbd.solver.contact.colliders import sphere_signed_distance

    def plane():
        return _plane_toi(start, end, normal, offset, slack)

    def sphere():
        start_gap = sphere_signed_distance(start, center, radius, outside)
        end_gap = sphere_signed_distance(end, center, radius, outside)
        toi = _lipschitz_toi(start_gap, end - start, slack)
        # Already through: only motion that increases the gap is allowed -- the same
        # escape clause `_plane_toi` documents. Without it a vertex that starts inside
        # the sphere freezes at toi = 0 *even for a step that climbs back out*, and
        # since the sweep filter takes a global minimum, one such vertex deadlocks the
        # entire mesh permanently. (The static solver hit exactly this: an indenting
        # sphere overlapping the initial state could never be escaped.)
        return jnp.where(
            start_gap <= 0.0,
            jnp.where(end_gap > start_gap, jnp.ones_like(toi), jnp.zeros_like(toi)),
            toi,
        )

    toi = jax.lax.switch(kind, (plane, sphere))
    return jnp.where(enabled, toi, jnp.ones_like(toi))


@jax.jit
def vertex_time_of_impact(
    colliders: ColliderData,
    start: jnp.ndarray,
    end: jnp.ndarray,
    slack: jnp.ndarray = jnp.asarray(_DEFAULT_SLACK),
) -> jnp.ndarray:
    """Return the largest safe fraction of a single vertex's step, in [0, 1]."""
    per_collider = jax.vmap(
        lambda k, n, o, c, r, out, en: _collider_toi(
            start, end, k, n, o, c, r, out, en, slack
        )
    )(
        colliders.kind,
        colliders.normal,
        colliders.offset,
        colliders.center,
        colliders.radius,
        colliders.outside,
        colliders.enabled,
    )
    # No colliders -> the reduction over an empty axis must yield 1.0, not +inf.
    return jnp.min(
        jnp.concatenate([per_collider, jnp.ones((1,), dtype=start.dtype)])
    )


