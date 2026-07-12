"""Primitive-pair distances, conditioned on a frozen distance type.

The unsigned distance between two primitives is realised by one of their *sub-features*
-- a vertex, an edge, or the interior of a face. Which sub-feature wins is a discrete
choice (the "distance type"), and it is the one genuinely non-smooth thing about contact
distance. We therefore split it out: ``classify_*`` picks the type in the combinatorial
layer, the type is frozen, and ``*_distance_sq`` is then a smooth function of position.

Everything here returns **squared** distance. A ``sqrt`` has an infinite derivative at
zero, and the whole point of the barrier is to operate where the distance is small, so a
squared distance is what is differentiated and the square root is taken (if at all) only
after the barrier has kept us away from zero.
"""

import jax
import jax.numpy as jnp

# Point-triangle: which sub-feature of the triangle realises the closest distance.
POINT_TRIANGLE_TYPES = {
    "P_T0": 0,  # closest to vertex t0
    "P_T1": 1,
    "P_T2": 2,
    "P_E0": 3,  # closest to edge (t0, t1)
    "P_E1": 4,  # closest to edge (t1, t2)
    "P_E2": 5,  # closest to edge (t2, t0)
    "P_T": 6,  # closest to the triangle interior
}

# Edge-edge: which sub-feature pair realises the closest distance.
EDGE_EDGE_TYPES = {
    "EA0_EB0": 0,
    "EA0_EB1": 1,
    "EA1_EB0": 2,
    "EA1_EB1": 3,
    "EA_EB0": 4,  # eb0 against the interior of edge a
    "EA_EB1": 5,
    "EA0_EB": 6,  # ea0 against the interior of edge b
    "EA1_EB": 7,
    "EA_EB": 8,  # interior-interior
}


# Softening added under the square root when converting a squared distance to a distance.
# In length^2 units; well below any physically meaningful gap.
_DISTANCE_SOFTENING = 1.0e-24


@jax.jit
def _safe_denominator(denominator: jnp.ndarray) -> jnp.ndarray:
    """Return ``denominator``, or 1 where it is zero, so a degenerate primitive is 0 / 1.

    Every distance below divides by the squared length of something -- an edge, a triangle
    normal, the cross product of two edges. For any real primitive that quantity is strictly
    positive and this substitution changes nothing, in value *or* in gradient. It exists for
    the one input that is not a real primitive: the **padded pair**, whose four vertices are
    all index 0 and therefore coincident, giving ``0 / 0``.

    That is not a hypothetical. A padded pair is masked with ``active=False``, so the barrier
    correctly returns an energy of ``0.0`` -- the *forward* pass looks perfectly clean. But
    ``jnp.where`` back-propagates a zero cotangent into a NaN numerator, and ``0 * NaN`` is
    NaN, so the **gradient** comes back poisoned while every value-based test stays green.
    This is the ``0 * inf = NaN`` trap in its most well-hidden form: the masking is right and
    the *distance* is wrong. Guarding the denominator is what makes the mask sufficient.

    ``_segment_projection_parameter`` below has always done this; the distance functions were
    simply never given the same treatment.
    """
    return jnp.where(denominator > 0.0, denominator, 1.0)


@jax.jit
def distance_from_squared(distance_sq: jnp.ndarray) -> jnp.ndarray:
    """Return ``sqrt(d^2 + softening)``: a distance whose derivative never blows up.

    The barrier is written in length units so that an analytic collider can hand it a
    *signed* distance (negative means penetrated) while a mesh pair hands it an unsigned
    one, and both share a single ``d_hat`` and a single ``kappa``.

    The softening is added *inside* the root rather than clamping the argument with a
    ``maximum``. Clamping would make the result constant below the floor, and the
    derivative of a constant is zero -- which would silently kill the contact force
    exactly where it is needed most. Adding a constant keeps the map strictly increasing,
    so ``d(sqrt)/d(d^2) = 1 / (2 sqrt(d^2 + s))`` is large but finite and never zero.
    """
    return jnp.sqrt(distance_sq + _DISTANCE_SOFTENING)


@jax.jit
def point_point_distance_sq(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Return |a - b|^2."""
    delta = a - b
    return jnp.dot(delta, delta)


@jax.jit
def point_edge_distance_sq(
    p: jnp.ndarray, e0: jnp.ndarray, e1: jnp.ndarray
) -> jnp.ndarray:
    """Return the squared distance from ``p`` to the infinite line through e0, e1.

    Valid as *the* point-edge distance only when the projection falls inside the segment,
    which is exactly what the ``P_E*`` distance types assert.
    """
    edge = e1 - e0
    offset = p - e0
    # |offset|^2 - (offset . edge)^2 / |edge|^2, written as a cross product so the
    # subtraction of two nearly-equal numbers never happens.
    cross = jnp.cross(edge, offset)
    return jnp.dot(cross, cross) / _safe_denominator(jnp.dot(edge, edge))


@jax.jit
def point_plane_distance_sq(
    p: jnp.ndarray, t0: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray
) -> jnp.ndarray:
    """Return the squared distance from ``p`` to the plane of the triangle."""
    normal = jnp.cross(t1 - t0, t2 - t0)
    signed = jnp.dot(p - t0, normal)
    return signed * signed / _safe_denominator(jnp.dot(normal, normal))


@jax.jit
def line_line_distance_sq(
    ea0: jnp.ndarray, ea1: jnp.ndarray, eb0: jnp.ndarray, eb1: jnp.ndarray
) -> jnp.ndarray:
    """Return the squared distance between the two infinite lines.

    Degenerate when the edges are parallel (``normal`` -> 0); that case is what the
    mollifier below exists to smooth, and the classifier routes it to a point-edge type.
    """
    normal = jnp.cross(ea1 - ea0, eb1 - eb0)
    signed = jnp.dot(eb0 - ea0, normal)
    return signed * signed / _safe_denominator(jnp.dot(normal, normal))


@jax.jit
def point_triangle_distance_sq(
    p: jnp.ndarray,
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    distance_type: jnp.ndarray,
) -> jnp.ndarray:
    """Return the squared point-triangle distance for a frozen ``distance_type``."""
    branches = (
        lambda: point_point_distance_sq(p, t0),
        lambda: point_point_distance_sq(p, t1),
        lambda: point_point_distance_sq(p, t2),
        lambda: point_edge_distance_sq(p, t0, t1),
        lambda: point_edge_distance_sq(p, t1, t2),
        lambda: point_edge_distance_sq(p, t2, t0),
        lambda: point_plane_distance_sq(p, t0, t1, t2),
    )
    return jax.lax.switch(distance_type, branches)


@jax.jit
def edge_edge_distance_sq(
    ea0: jnp.ndarray,
    ea1: jnp.ndarray,
    eb0: jnp.ndarray,
    eb1: jnp.ndarray,
    distance_type: jnp.ndarray,
) -> jnp.ndarray:
    """Return the squared edge-edge distance for a frozen ``distance_type``."""
    branches = (
        lambda: point_point_distance_sq(ea0, eb0),
        lambda: point_point_distance_sq(ea0, eb1),
        lambda: point_point_distance_sq(ea1, eb0),
        lambda: point_point_distance_sq(ea1, eb1),
        lambda: point_edge_distance_sq(eb0, ea0, ea1),
        lambda: point_edge_distance_sq(eb1, ea0, ea1),
        lambda: point_edge_distance_sq(ea0, eb0, eb1),
        lambda: point_edge_distance_sq(ea1, eb0, eb1),
        lambda: line_line_distance_sq(ea0, ea1, eb0, eb1),
    )
    return jax.lax.switch(distance_type, branches)


# --------------------------------------------------------------------------------------
# Classification. Combinatorial layer: integer-valued, never differentiated.
# --------------------------------------------------------------------------------------


@jax.jit
def classify_point_triangle(
    p: jnp.ndarray, t0: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray
) -> jnp.ndarray:
    """Return the point-triangle distance type (Ericson, Real-Time Collision Detection).

    Voronoi-region test on the barycentric coordinates of the projection of ``p``.
    """
    ab = t1 - t0
    ac = t2 - t0
    ap = p - t0

    d1 = jnp.dot(ab, ap)
    d2 = jnp.dot(ac, ap)
    # Vertex region t0.
    in_t0 = (d1 <= 0.0) & (d2 <= 0.0)

    bp = p - t1
    d3 = jnp.dot(ab, bp)
    d4 = jnp.dot(ac, bp)
    in_t1 = (d3 >= 0.0) & (d4 <= d3)

    cp = p - t2
    d5 = jnp.dot(ab, cp)
    d6 = jnp.dot(ac, cp)
    in_t2 = (d6 >= 0.0) & (d5 <= d6)

    vc = d1 * d4 - d3 * d2
    in_e0 = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)  # edge t0-t1

    va = d3 * d6 - d5 * d4
    in_e1 = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)  # edge t1-t2

    vb = d5 * d2 - d1 * d6
    in_e2 = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)  # edge t2-t0

    # Priority matches the region test: vertices, then edges, then the face interior.
    return jnp.int32(
        jnp.where(
            in_t0,
            POINT_TRIANGLE_TYPES["P_T0"],
            jnp.where(
                in_t1,
                POINT_TRIANGLE_TYPES["P_T1"],
                jnp.where(
                    in_t2,
                    POINT_TRIANGLE_TYPES["P_T2"],
                    jnp.where(
                        in_e0,
                        POINT_TRIANGLE_TYPES["P_E0"],
                        jnp.where(
                            in_e1,
                            POINT_TRIANGLE_TYPES["P_E1"],
                            jnp.where(
                                in_e2,
                                POINT_TRIANGLE_TYPES["P_E2"],
                                POINT_TRIANGLE_TYPES["P_T"],
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def _segment_projection_parameter(p: jnp.ndarray, e0: jnp.ndarray, e1: jnp.ndarray):
    """Return the parameter of p projected onto the line through e0, e1."""
    edge = e1 - e0
    length_sq = jnp.dot(edge, edge)
    safe = jnp.where(length_sq > 0.0, length_sq, 1.0)
    return jnp.dot(p - e0, edge) / safe


@jax.jit
def classify_edge_edge(
    ea0: jnp.ndarray, ea1: jnp.ndarray, eb0: jnp.ndarray, eb1: jnp.ndarray
) -> jnp.ndarray:
    """Return the edge-edge distance type.

    Rather than solving the line-line system and clamping ``s`` and ``t`` independently
    -- which is wrong, because clamping one parameter changes the optimal other one, and
    silently mis-classifies configurations that land exactly on an endpoint -- this scores
    every sub-feature pair and takes the smallest *valid* one.

    A sub-feature is valid when its own parameters lie inside their segments; the true
    segment-segment distance is realised by one of them, so the minimum over the valid set
    is exactly right, and its index is the distance type. Invalid candidates are pushed to
    +inf rather than dropped, which keeps the shape static and the whole thing jit-able.
    """
    u = ea1 - ea0
    v = eb1 - eb0
    w = ea0 - eb0

    a = jnp.dot(u, u)
    b = jnp.dot(u, v)
    c = jnp.dot(v, v)
    d = jnp.dot(u, w)
    e = jnp.dot(v, w)
    denom = a * c - b * b

    # Interior-interior is only a candidate when the lines are not parallel (the solve is
    # ill-conditioned there) and both parameters fall inside their segments.
    parallel = denom <= 1.0e-12 * a * c
    safe_denom = jnp.where(parallel, 1.0, denom)
    s = (b * e - c * d) / safe_denom
    t = (a * e - b * d) / safe_denom
    interior_valid = (
        jnp.logical_not(parallel)
        & (s >= 0.0)
        & (s <= 1.0)
        & (t >= 0.0)
        & (t <= 1.0)
    )

    # Point-edge candidates are valid only when the projection lands inside the segment.
    t_b0 = _segment_projection_parameter(eb0, ea0, ea1)
    t_b1 = _segment_projection_parameter(eb1, ea0, ea1)
    t_a0 = _segment_projection_parameter(ea0, eb0, eb1)
    t_a1 = _segment_projection_parameter(ea1, eb0, eb1)
    inside = lambda p: (p >= 0.0) & (p <= 1.0)

    infinity = jnp.asarray(jnp.inf, dtype=a.dtype)
    guard = lambda ok, value: jnp.where(ok, value, infinity)

    candidates = jnp.stack(
        [
            point_point_distance_sq(ea0, eb0),
            point_point_distance_sq(ea0, eb1),
            point_point_distance_sq(ea1, eb0),
            point_point_distance_sq(ea1, eb1),
            guard(inside(t_b0), point_edge_distance_sq(eb0, ea0, ea1)),
            guard(inside(t_b1), point_edge_distance_sq(eb1, ea0, ea1)),
            guard(inside(t_a0), point_edge_distance_sq(ea0, eb0, eb1)),
            guard(inside(t_a1), point_edge_distance_sq(ea1, eb0, eb1)),
            guard(interior_valid, line_line_distance_sq(ea0, ea1, eb0, eb1)),
        ]
    )
    return jnp.argmin(candidates).astype(jnp.int32)


# --------------------------------------------------------------------------------------
# Edge-edge mollifier.
# --------------------------------------------------------------------------------------


def edge_edge_mollifier_threshold(
    ea0_rest: jnp.ndarray,
    ea1_rest: jnp.ndarray,
    eb0_rest: jnp.ndarray,
    eb1_rest: jnp.ndarray,
    scale: float = 1.0e-3,
) -> jnp.ndarray:
    """Return the mollifier threshold ``eps_x`` from the *rest* edge lengths.

    Computed once from the rest pose (it is a constant of the pair, not of the current
    configuration) and frozen, so it never enters a gradient.
    """
    a = ea1_rest - ea0_rest
    b = eb1_rest - eb0_rest
    return scale * jnp.dot(a, a) * jnp.dot(b, b)


@jax.jit
def edge_edge_mollifier(
    ea0: jnp.ndarray,
    ea1: jnp.ndarray,
    eb0: jnp.ndarray,
    eb1: jnp.ndarray,
    eps_x: jnp.ndarray,
) -> jnp.ndarray:
    """Return the IPC edge-edge mollifier in [0, 1].

    Two edges that rotate through parallel swap which sub-feature is closest, so the
    edge-edge distance is discontinuous there *even at a fixed distance type*. The
    mollifier scales the barrier smoothly to zero as the edges approach parallel, which
    removes the discontinuity; it is C1 at ``c == eps_x`` and identically 1 beyond it.
    """
    cross = jnp.cross(ea1 - ea0, eb1 - eb0)
    c = jnp.dot(cross, cross)
    # Guarded for the same reason the distances above are: a padded pair's *rest* edges are
    # degenerate too, so `eps_x` is zero and this is 0 / 0. The forward value is masked away
    # downstream, but the NaN would still come back through the gradient.
    ratio = c / _safe_denominator(eps_x)
    mollified = ratio * (2.0 - ratio)  # -(c/eps)^2 + 2(c/eps)
    return jnp.where(c < eps_x, mollified, jnp.ones_like(mollified))


# --------------------------------------------------------------------------------------
# Pair-level entry points. One definition of "how far apart is this pair", shared by
# detection, the barrier, friction and CCD -- so they cannot drift apart.
# --------------------------------------------------------------------------------------


@jax.jit
def pair_distance_sq(
    pair_positions: jnp.ndarray, pair_type: jnp.ndarray
) -> jnp.ndarray:
    """Return the squared distance of one primitive pair, classification frozen.

    ``pair_positions`` is (4, 3): for a vertex-triangle pair the vertex then the triangle;
    for an edge-edge pair the two edges back to back. ``pair_type`` indexes
    ``detection.PAIR_TYPES``.

    Classifying *here* and immediately freezing the result with ``stop_gradient`` is what
    makes the distance differentiable: which sub-feature is closest is a discrete choice, and
    with it held fixed the distance is a smooth function of position.

    This is the **raw** squared distance. See ``pair_gap_and_mollifier`` for the softened gap
    the barrier wants, and read the warning there before choosing between them.
    """
    a, b, c, d = pair_positions

    def vertex_triangle():
        kind = jax.lax.stop_gradient(classify_point_triangle(a, b, c, d))
        return point_triangle_distance_sq(a, b, c, d, kind)

    def edge_edge():
        kind = jax.lax.stop_gradient(classify_edge_edge(a, b, c, d))
        return edge_edge_distance_sq(a, b, c, d, kind)

    return jax.lax.switch(pair_type, (vertex_triangle, edge_edge))


@jax.jit
def pair_gap_and_mollifier(
    pair_positions: jnp.ndarray,
    rest_pair_positions: jnp.ndarray,
    pair_type: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(gap, mollifier)`` for one primitive pair: what the *energy* consumes.

    The gap is softened by ``distance_from_squared``, so its derivative stays finite at zero
    distance. The mollifier is 1 for a vertex-triangle pair, and for edge-edge it scales the
    energy smoothly to zero as the two edges rotate through parallel -- where the distance
    jumps even at a fixed classification. Its threshold comes from the *rest* pose: it is a
    constant of the pair, not of the configuration.

    **Do not use this for CCD.** The softening *inflates* the gap by ~1e-12. For the barrier
    that is harmless -- it only ever makes the force finite. For a time of impact an inflated
    gap means a **longer permitted step**, which is unsound. CCD must use the raw
    ``pair_distance_sq`` above. Same operation, opposite sign of error.
    """
    a, b, c, d = pair_positions
    distance_sq = pair_distance_sq(pair_positions, pair_type)

    def no_mollifier():
        return jnp.ones((), a.dtype)

    def edge_edge_mollification():
        eps_x = jax.lax.stop_gradient(
            edge_edge_mollifier_threshold(*rest_pair_positions)
        )
        return edge_edge_mollifier(a, b, c, d, eps_x)

    mollifier = jax.lax.switch(
        pair_type, (no_mollifier, edge_edge_mollification)
    )
    return distance_from_squared(distance_sq), mollifier
