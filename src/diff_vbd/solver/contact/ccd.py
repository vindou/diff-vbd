"""Continuous collision detection: conservative time-of-impact filters.

This is the combinatorial layer. Nothing here is differentiated -- a time of impact is a
step-size bound, not a force -- so it is wrapped in ``stop_gradient`` where it meets the
smooth layer.

There are two filters and they do different jobs.

``vertex_time_of_impact`` bounds a *single* vertex's step against static geometry. It is a
local safeguard: it is what stops the line-search-disabled path (which otherwise takes the
raw Newton step) from jumping straight through an obstacle.

``sweep_time_of_impact`` bounds the *aggregate* displacement of a whole sweep with one
scalar, and it is the actual guarantee. The distinction matters because VBD solves a colour
of vertices in parallel from a single frozen snapshot: each vertex certifies "my own motion
is safe", but if two of them are coupled and share a colour, they both move, and neither
certificate covers their combined motion. A per-vertex bound cannot see that; a filter on
the whole displacement field can, because it looks at where the mesh actually ended up.
It also catches the Chebyshev extrapolation, which happens *after* the local solves have
finished and so is invisible to any per-vertex check.
"""

import jax
import jax.numpy as jnp

from diff_vbd.model import ColliderData, ContactState
from diff_vbd.solver.contact.distances import pair_distance_sq

# How many conservative advances a single pair's ACCD is allowed. Static, because this kernel
# runs under a vmap over the pair buffer inside a scan over sweeps inside a scan over colours
# -- and a `while_loop` under `vmap` runs to the batch-wide maximum anyway, so it would trade
# predictable cost for nothing. Running out of budget is *safe*: the accumulated time is a
# certified under-estimate, so the step is merely shorter than it needed to be, never longer.
_MAX_ACCD_ITERATIONS = 32

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
        return _lipschitz_toi(start_gap, end - start, slack)

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


@jax.jit
def collider_sweep_time_of_impact(
    colliders: ColliderData,
    start_positions: jnp.ndarray,
    end_positions: jnp.ndarray,
    slack: jnp.ndarray = jnp.asarray(_DEFAULT_SLACK),
) -> jnp.ndarray:
    """Return one conservative time of impact of a displacement field against the colliders.

    One of the three sources folded into ``sweep_time_of_impact`` below.
    """
    per_vertex = jax.vmap(
        lambda a, b: vertex_time_of_impact(colliders, a, b, slack)
    )(start_positions, end_positions)
    return jnp.min(
        jnp.concatenate(
            [per_vertex, jnp.ones((1,), dtype=start_positions.dtype)]
        )
    )


# --------------------------------------------------------------------------------------
# Mesh-mesh: additive CCD over the detected pair set.
# --------------------------------------------------------------------------------------


# Multiple of the predicted step displacement a vertex is actually allowed to travel. The
# prediction is inertial, and the solve can legitimately want more than it (a stiff elastic
# response, a Newton step), so the allowance has headroom.
_DEFAULT_DISPLACEMENT_FACTOR = 2.0

# Strict-inequality margin in the band derivation. In (0, 1); smaller is a wider band.
_DEFAULT_BAND_SAFETY = 0.9


def derive_detection_band(
    d_hat: float,
    predicted_displacement: float,
    prescribed_displacement: float = 0.0,
    displacement_factor: float = _DEFAULT_DISPLACEMENT_FACTOR,
    band_safety: float = _DEFAULT_BAND_SAFETY,
) -> tuple[float, float]:
    """Return ``(band, max_displacement)`` -- the two halves of the mesh-mesh guarantee.

    The pair set is built once per step, out to ``band``. A pair further apart than that is
    invisible to the CCD for the rest of the step, so the band is sound only if no such pair
    can possibly reach contact before the next detection. A primitive distance is 1-Lipschitz
    in each of its two sides, so if no vertex strays further than ``Δmax`` from the step
    start, an undetected pair stays above ``band - 2*Δmax``. Requiring that to remain above
    the activation distance -- so the pair never even enters the barrier's support, which also
    means the *energy* is complete and not just the collision check -- gives

        band >= d_hat + 2 * Δmax.

    ``band_safety`` makes that inequality strict. ``Δmax`` is floored at ``d_hat`` so a body at
    rest is not clamped to nothing, and scaled by ``displacement_factor`` above the inertial
    prediction because the prediction is only a prediction -- what makes it a *bound* is the
    clamp in ``displacement_clamp_time_of_impact``, which enforces it.

    Prescribed (Dirichlet) motion is included because those rows are written after the filter
    and so escape the clamp; the host audit is what actually catches them.

    The cost is real and it is the honest limit of this scheme: the band grows with speed, and
    candidates grow as its cube. Past roughly a body-length of travel per step the pair set
    stops being boundable, the capacity error fires, and the user has to choose a smaller
    timestep. That is by design -- the alternative is a band that quietly under-covers.
    """
    max_displacement = max(displacement_factor * predicted_displacement, d_hat)
    reach = max(max_displacement, prescribed_displacement)
    band = max(d_hat + (2.0 / band_safety) * reach, 2.0 * d_hat)
    return band, max_displacement


@jax.jit
def pair_time_of_impact(
    start_pair: jnp.ndarray,
    end_pair: jnp.ndarray,
    pair_type: jnp.ndarray,
    valid: jnp.ndarray,
    slack: jnp.ndarray = jnp.asarray(_DEFAULT_SLACK),
) -> jnp.ndarray:
    """Return the largest safe fraction of a linear motion of one primitive pair, in [0, 1].

    Additive CCD. Unlike a plane, a point-triangle or edge-edge distance is not linear along
    the motion, so there is nothing to solve -- it is *bounded*, then advanced conservatively
    until the bound runs out.

    **Mean-displacement subtraction.** The distance is invariant under a common translation
    of all four vertices, so with ``p_hat = p - mean(p)``, ``d(x + t*p) == d(x + t*p_hat)``
    for every ``t``. This is an exact identity, not an approximation, and it is what makes
    ACCD usable at all: two bodies flying side by side at 1000 m/s have ``p_hat ~ 0`` and
    certify the entire step in one iteration. Without it the Lipschitz bound below would be
    dominated by the shared velocity and the time of impact would collapse to nothing -- a
    silent, catastrophic throttle that merely looks like "the solver got slow". The probe
    configuration is therefore built from ``p_hat``, not ``p``.

    **The Lipschitz bound differs by pair type, and this is the sharp edge.** The closest
    point on a primitive is a convex combination of its own vertices, so its speed is at most
    the fastest of them; the distance closes at most as fast as the sum of the two:

        point-triangle:  l_p = |p0| + max(|p1|, |p2|, |p3|)
        edge-edge:       l_p = max(|p0|, |p1|) + max(|p2|, |p3|)

    Using the point-triangle form on an edge-edge pair *under-estimates* ``l_p`` and therefore
    permits too long a step. It is unsound, and it looks completely reasonable.

    **The advance is a segment certificate, not an endpoint check.** From an accepted time
    with gap ``g``, stepping ``slack * g / l_p`` keeps the gap above ``(1 - slack) * g`` for
    *every* ``t`` in the interval, because the gap cannot fall faster than ``l_p``. By
    induction the returned time bounds the whole trajectory, not just where it lands.
    """
    displacement = end_pair - start_pair
    relative = displacement - jnp.mean(displacement, axis=0)

    # The softening inflates l_p, which *shortens* the permitted step -- conservative, and so
    # safe. The same softening applied to the gap would lengthen it, and would not be.
    speeds = jnp.sqrt(jnp.sum(relative * relative, axis=1) + 1.0e-30)
    lipschitz = jax.lax.switch(
        pair_type,
        (
            lambda: speeds[0] + jnp.max(speeds[1:4]),  # point-triangle
            lambda: jnp.max(speeds[0:2]) + jnp.max(speeds[2:4]),  # edge-edge
        ),
    )

    # The RAW distance. `distance_from_squared`'s softening would inflate the gap by ~1e-12
    # and thereby permit a longer step: harmless for the barrier, unsound here.
    initial_gap = jnp.sqrt(jnp.maximum(pair_distance_sq(start_pair, pair_type), 0.0))

    moving = lipschitz > 1.0e-24
    separated = initial_gap > 0.0
    runnable = valid & moving & separated
    safe_lipschitz = jnp.where(moving, lipschitz, 1.0)
    threshold = (1.0 - slack) * initial_gap

    def advance(_, carry):
        time, gap, done = carry
        step = slack * gap / safe_lipschitz

        # Clamp the probe into the segment before evaluating it. Beyond t = 1 the motion is a
        # pure extrapolation that never happens, and the distance out there is meaningless --
        # a pair whose safe advance overshoots the segment has flown *through* the obstacle
        # and back out again by the probe, so its gap reads large-then-small and can trip the
        # convergence test below. That returns a time of impact of zero for a pair that was
        # never in danger, and one such pair drags the whole mesh's global minimum to zero.
        overshoots = (time + step) >= 1.0
        probe_time = jnp.minimum(time + step, 1.0)
        probe_gap = jnp.sqrt(
            jnp.maximum(
                pair_distance_sq(start_pair + probe_time * relative, pair_type), 0.0
            )
        )

        # Consumed most of the gap we started with: stop, and keep the *last accepted* time,
        # which is what preserves a margin rather than creeping right up to the obstacle.
        #
        # `time > 0` is load bearing. The first advance is safe *by construction* -- the gap
        # cannot fall faster than `lipschitz`, so it lands at exactly `(1 - slack) * gap`,
        # which is the threshold. In exact arithmetic that is not "below" it; in floating
        # point it lands on the knife edge and can tip either way. Without this guard, a pair
        # closing head-on at precisely the Lipschitz rate -- the most ordinary impact there
        # is -- would return a time of impact of zero and freeze the solve.
        converged = (time > 0.0) & (probe_gap < threshold)
        accept = overshoots | ~converged
        finished = done | overshoots | converged

        next_time = jnp.where(done | ~accept, time, probe_time)
        next_gap = jnp.where(finished, gap, probe_gap)
        return next_time, next_gap, finished

    toi, _, _ = jax.lax.fori_loop(
        0,
        _MAX_ACCD_ITERATIONS,
        advance,
        (
            jnp.zeros((), start_pair.dtype),
            initial_gap,
            jnp.asarray(False),
        ),
    )

    # A padded pair is four coincident points: no motion, no gap, and every division above was
    # guarded, so nothing non-finite was *produced* -- not merely masked. That matters because
    # a single NaN here would poison the global `min` and freeze the entire mesh.
    toi = jnp.where(runnable, toi, 1.0)
    # Already touching: the ACCD precondition is broken and its certificate would be vacuous.
    # Freeze rather than pretend. `detection` raises before this can happen in a real solve.
    toi = jnp.where(valid & moving & ~separated, 0.0, toi)
    return jnp.clip(toi, 0.0, 1.0)


@jax.jit
def contact_pair_sweep_time_of_impact(
    state: ContactState,
    start_positions: jnp.ndarray,
    end_positions: jnp.ndarray,
    slack: jnp.ndarray = jnp.asarray(_DEFAULT_SLACK),
) -> jnp.ndarray:
    """Return one conservative time of impact over every detected mesh-mesh pair."""
    # Static branch: an analytic-collider-only problem has no pair buffers, and XLA rejects a
    # gather from a zero-length axis even when the vmap over it would iterate zero times.
    if state.pair_vertices.shape[0] == 0:
        return jnp.ones((), start_positions.dtype)

    def one_pair(vertices, pair_type, valid):
        return pair_time_of_impact(
            start_positions[vertices],
            end_positions[vertices],
            pair_type,
            valid,
            slack,
        )

    per_pair = jax.vmap(one_pair)(
        state.pair_vertices, state.pair_type, state.pair_valid
    )
    return jnp.min(
        jnp.concatenate(
            [per_pair, jnp.ones((1,), dtype=start_positions.dtype)]
        )
    )


@jax.jit
def displacement_clamp_time_of_impact(
    step_start: jnp.ndarray,
    start_positions: jnp.ndarray,
    end_positions: jnp.ndarray,
    free_mask: jnp.ndarray,
    max_displacement: jnp.ndarray,
) -> jnp.ndarray:
    """Bound the sweep so no vertex strays further than ``max_displacement`` from ``x0``.

    This is what turns the detection band from a hope into a guarantee. The pair set is built
    once, at the start of the step, out to a radius ``band``. A pair further apart than that
    is not in the set, so the CCD above cannot see it -- for the whole step, no matter what
    happens. It could close to zero and interpenetrate with nothing watching.

    The band is chosen so that cannot happen: a pair outside it is more than ``band`` apart,
    a primitive distance is 1-Lipschitz in each side, so if no vertex moves further than
    ``Δmax`` the pair stays above ``band - 2*Δmax``, which the band derivation keeps above
    ``d_hat``. **This function is the clause that makes the "if" true.**

    Note the displacement is measured from ``step_start``, not from the previous sweep. A
    per-sweep clamp would let the error accumulate over K sweeps and the bound would be off by
    a factor of K.

    One scalar, folded into the same global minimum as the two time-of-impact filters -- not a
    per-vertex projection. Projecting each vertex back onto its own ball independently would
    change the *relative* motion of a pair the ACCD had just certified, invalidating the
    certificate the filter is about to apply.

    Per vertex this is a ray-vs-ball intersection, in closed form. With ``c = y - x0`` (where
    the vertex already is) and ``u`` (where this sweep wants to take it):

        A = |u|^2 >= 0,  B = c.u,  C = |c|^2 - Δmax^2 <= 0   (the invariant, by induction)
        t = (-B + sqrt(B^2 - A*C)) / A

    ``C <= 0`` makes the discriminant ``>= B^2 >= 0``, so **there is no negative-discriminant
    branch and no NaN to guard against** -- the geometry guarantees it.
    """
    offset = start_positions - step_start
    direction = end_positions - start_positions

    a = jnp.sum(direction * direction, axis=1)
    b = jnp.sum(offset * direction, axis=1)
    # Clamp C to <= 0. It is non-positive by induction, but a vertex sitting exactly on the
    # boundary can land a hair outside it in floating point; treating that as "on the
    # boundary" only ever *shrinks* t, so it stays sound.
    c = jnp.minimum(
        jnp.sum(offset * offset, axis=1) - max_displacement * max_displacement, 0.0
    )

    safe_a = jnp.where(a > 0.0, a, 1.0)
    root = (-b + jnp.sqrt(jnp.maximum(b * b - safe_a * c, 0.0))) / safe_a
    # A vertex that does not move is never the binding constraint.
    per_vertex = jnp.where(a > 0.0, root, 1.0)

    # Free vertices only. Dirichlet and rigid rows are overwritten by the position constraints
    # *after* this filter runs, so clamping them here would both freeze the mesh and be
    # silently undone. `_audit_step_displacement` on the host is what covers them instead.
    per_vertex = jnp.where(free_mask, per_vertex, 1.0)
    return jnp.clip(
        jnp.min(
            jnp.concatenate(
                [per_vertex, jnp.ones((1,), dtype=start_positions.dtype)]
            )
        ),
        0.0,
        1.0,
    )


@jax.jit
def sweep_time_of_impact(
    contact,
    state: ContactState,
    step_start: jnp.ndarray,
    start_positions: jnp.ndarray,
    end_positions: jnp.ndarray,
    free_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Return one conservative time of impact for an entire displacement field.

    A single scalar, deliberately. Scaling the whole update by one factor is what makes the
    bound hold no matter how the interior local solves behaved -- whether they raced each
    other within a colour, or were extrapolated afterwards by Chebyshev. A per-vertex bound
    cannot make that promise.

    A resting vertex does not throttle the mesh: its displacement is ~0, so its own time of
    impact is 1 and it drops out of the minimum. Only a vertex actually driving into an
    obstacle pulls the bound down.

    Three sources, one minimum: the analytic colliders, the detected mesh-mesh pairs, and the
    displacement clamp that keeps the pair set itself trustworthy.
    """
    slack = contact.ccd.slack
    collider = collider_sweep_time_of_impact(
        contact.colliders, start_positions, end_positions, slack
    )
    pair = jax.lax.cond(
        contact.ccd.enabled,
        lambda: contact_pair_sweep_time_of_impact(
            state, start_positions, end_positions, slack
        ),
        lambda: jnp.ones((), start_positions.dtype),
    )
    clamp = jax.lax.cond(
        contact.ccd.enabled,
        lambda: displacement_clamp_time_of_impact(
            step_start,
            start_positions,
            end_positions,
            free_mask,
            contact.ccd.max_displacement,
        ),
        lambda: jnp.ones((), start_positions.dtype),
    )
    return jnp.minimum(jnp.minimum(collider, pair), clamp)


@jax.jit
def filter_sweep(
    contact,
    state: ContactState,
    step_start: jnp.ndarray,
    start_positions: jnp.ndarray,
    end_positions: jnp.ndarray,
    free_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scale a whole sweep's displacement back to its conservative time of impact.

    Returns the filtered positions and the time of impact itself -- the latter so the host can
    see a collapsing bound and say so, rather than letting the mesh silently freeze.
    """
    toi = jax.lax.stop_gradient(
        sweep_time_of_impact(
            contact, state, step_start, start_positions, end_positions, free_mask
        )
    )
    return start_positions + toi * (end_positions - start_positions), toi
