"""Per-vertex conservative displacement bounds (Wu et al. 2020, via OGC section 3.7).

This replaces the global time-of-impact sweep filter, and the difference is exactly one
factor of one half. The old scheme's justification for a *global* scalar was that VBD
moves a whole colour simultaneously from one snapshot, so no single-vertex certificate
covers a pair's combined motion. Wu et al.'s bound answers that: give every vertex

    b_v = gamma_p * min(d_min_v, d_E_min_v, d_T_min_v),      0 < gamma_p < 0.5

-- its minimal distance to non-incident facets, its neighbour edges' minimal distances to
disjoint edges, and its neighbour facets' minimal distances to other vertices. If the
state at the last detection was intersection-free and **every** vertex has moved at most
``b_v`` since, the state is still intersection-free: any new intersection would have to
be preceded by some feature pair reaching distance zero, both sides of a pair at distance
``d`` have bounds at most ``gamma_p * d < d/2``, and a primitive distance is 1-Lipschitz
in each side -- so the pair cannot close. Both sides moving is priced in; that is why the
certificates compose where a naive per-vertex bound would not, and why no CCD and no
global reduction is needed.

What the pair set contributes here is *distances*, not safety: ``build_vertex_bounds``
takes the minimum of each vertex's candidate-pair distances, capped at the detection
band. A pair absent from the candidate set is more than ``band`` apart (the broadphase
guarantees completeness within it), and two bound-respecting endpoints close less than
``2 * gamma_p * band < band`` -- so unlike the old scheme, **the band is a performance
knob, not a soundness parameter**. A larger band means roomier bounds and rarer
re-detections; a smaller one means more detections; neither can admit an intersection.

Honesty about what this does not fix (OGC says it plainly, and so must we): a vertex
resting in contact sits at gap ~d_hat, so its bound is ~gamma_p * d_hat -- small. OGC
itself escapes that via a *large* contact radius made artifact-free by offset geometry,
which this repo does not implement. So these bounds convert **global** throttling into
**local** throttling: only vertices actually near contact are slowed, instead of the
whole mesh being scaled back by its worst offender. That is a strict improvement -- the
old filter's collapsing-TOI failure mode cannot exist because there is no global factor
-- but it is not a resolution of the underlying stiffness of near-contact stepping.
"""

import jax
import jax.numpy as jnp
import numpy as np

from diff_vbd.solver.contact.ccd import vertex_time_of_impact

# OGC's relaxation parameter: the fraction of its distance budget a vertex may consume
# per detection interval. Must be strictly below 0.5 -- that is the factor that makes two
# moving endpoints compose -- and OGC chooses 0.45 to leave a margin for floating point.
GAMMA_P = 0.45

# Fraction of the vertex count that must hit their bounds before detection re-runs
# (OGC's gamma_e). Displacement is tracked cumulatively from the last detection, so a
# handful of truncated vertices merely wait; once enough are pinned, re-detection
# re-anchors everyone with fresh distances.
GAMMA_E = 0.01

# Stand-in for "unbounded" (interior vertices, disabled contact). A finite large number
# rather than inf so the ray-ball arithmetic below never forms 0 * inf or inf - inf;
# squaring it stays comfortably inside float64.
UNBOUNDED = 1.0e30


def redetection_threshold(num_vertices: int) -> int:
    """Return how many bound-hits trigger a re-detection: ``max(1, gamma_e * K)``.

    The floor of 1 matters for small meshes, where ``gamma_e * K`` rounds to zero and
    would otherwise mean "re-detect every iteration even with nothing pinned".
    """
    return max(1, int(GAMMA_E * num_vertices))


def derive_bounds_band(
    d_hat: float,
    predicted_displacement: float,
    prescribed_displacement: float = 0.0,
) -> float:
    """Return the detection band for the bounds scheme.

    Sized so one detection interval can absorb the step's expected motion: a vertex far
    from everything gets ``b = GAMMA_P * band``, and that must comfortably exceed both
    the inertial prediction and any prescribed (Dirichlet) displacement -- prescribed
    rows cannot be truncated, so if their whole-step jump exceeded their bound the
    guarantee would be void and the audit would (rightly) raise. The 0.8 safety keeps
    those raises from triggering at the margin. Since the band is not a soundness
    parameter (see the module docstring), undersizing it costs re-detections, never
    correctness.
    """
    reach = max(2.0 * predicted_displacement, prescribed_displacement, d_hat)
    return d_hat + reach / (GAMMA_P * 0.8)


def build_vertex_bounds(
    pair_vertices: np.ndarray,
    pair_valid: np.ndarray,
    pair_distances: np.ndarray,
    num_vertices: int,
    band: float,
    surface_vertices: np.ndarray,
) -> np.ndarray:
    """Return per-vertex conservative bounds ``b_v`` from the detected candidate set.

    One scatter covers all three of Wu et al.'s minima: a vertex-triangle candidate's
    distance is ``d_min`` for its vertex and ``d_T_min`` for the triangle's three
    vertices; an edge-edge candidate's distance is ``d_E_min`` for all four endpoints.
    Any feature pair that could realise a first contact is itself a candidate (or is
    beyond the band, which the cap accounts for), so the scattered minimum is exactly
    the quantity the composition argument needs.

    Interior vertices get ``UNBOUNDED``: only surface primitives can intersect, so
    bounding interior motion would throttle the bulk of the mesh for nothing.
    """
    minima = np.full(num_vertices, UNBOUNDED, dtype=np.float64)
    surface = np.asarray(surface_vertices).reshape(-1)
    minima[surface] = float(band)

    live = np.asarray(pair_valid, dtype=bool)
    if live.any():
        vertices = np.asarray(pair_vertices)[live].reshape(-1)
        distances = np.repeat(np.asarray(pair_distances)[live], 4)
        np.minimum.at(minima, vertices, distances)

    # The sentinel stays the sentinel: an interior vertex is not "gamma_p of unbounded",
    # it is exempt, and scaling it would make "unbounded" mean two different numbers.
    return np.where(minima >= UNBOUNDED, UNBOUNDED, GAMMA_P * minima)


@jax.jit
def truncate_to_bounds(
    contact,
    anchor: jnp.ndarray,
    start_positions: jnp.ndarray,
    end_positions: jnp.ndarray,
    free_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Clip each vertex's motion to its bound-ball and its collider time of impact.

    Applied to every displacement the solver produces -- the initial guess (OGC Eq. 28)
    and each iteration's output, *after* any Chebyshev extrapolation, so wherever the
    mesh actually ended up is what gets certified. Per vertex, never a global factor:

    * the **ball**: ``|result - anchor| <= b_v``, where ``anchor`` is the vertex's
      position at the last detection. Cumulative, not per-iteration -- a per-iteration
      allowance would let K iterations drift K times further than the distances the
      bounds were computed from. The clip runs along the segment
      ``[start, end]``, using the same closed-form ray-vs-ball intersection the old
      displacement clamp used (C <= 0 by induction, so the discriminant is
      non-negative by geometry, not by clamping);
    * the **colliders**: the per-vertex conservative time of impact. Static obstacles
      need no composition argument -- the other party never moves -- so per-vertex
      clipping is sound for them, and it is what keeps the initial guess and the
      full-Newton path from stepping through a wall.

    Truncating along the segment (rather than projecting onto the ball surface) keeps
    every vertex on its own straight path, which is what the collider certificate was
    issued for.

    Returns the clipped positions and the number of *free* vertices whose ball bound was
    the binding constraint -- the counter the re-detection trigger watches. The whole
    thing is wrapped in ``stop_gradient`` by the caller's construction: bounds are
    combinatorial data, and a time of impact is a step-size bound, not a force.

    Dirichlet rows pass through untouched (the position constraints overwrite them after
    this filter anyway); the host-side audit is what covers prescribed motion.
    """
    bounds = contact.state.vertex_bounds

    collider_toi = jax.vmap(
        lambda a, b: vertex_time_of_impact(contact.colliders, a, b)
    )(start_positions, end_positions)

    offset = start_positions - anchor
    direction = end_positions - start_positions
    a = jnp.sum(direction * direction, axis=1)
    b = jnp.sum(offset * direction, axis=1)
    c = jnp.minimum(jnp.sum(offset * offset, axis=1) - bounds * bounds, 0.0)
    safe_a = jnp.where(a > 0.0, a, 1.0)
    root = (-b + jnp.sqrt(jnp.maximum(b * b - safe_a * c, 0.0))) / safe_a
    ball_toi = jnp.where(a > 0.0, root, 1.0)

    ball_toi = jax.lax.cond(
        contact.ccd.enabled,
        lambda: ball_toi,
        lambda: jnp.ones_like(ball_toi),
    )

    fraction = jnp.clip(jnp.minimum(collider_toi, ball_toi), 0.0, 1.0)
    fraction = jnp.where(free_mask, fraction, 1.0)
    fraction = jax.lax.stop_gradient(fraction)

    truncated = start_positions + fraction[:, None] * direction
    exceeded = free_mask & (ball_toi < 1.0) & (a > 0.0)
    return truncated, jnp.sum(exceeded)
