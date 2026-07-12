"""Host-side broad and narrow phase. The combinatorial layer.

Everything here is integer-valued and carries no gradient. It runs in the Python step loop,
between jitted sweeps, and its whole job is to emit **fixed-shape padded buffers** that the
device side can consume without ever changing shape.

Fixed shape is not a stylistic choice. Rebuilding the contact buffers each step with fresh
contents of the *same* shape is a jit cache hit, which is what makes host detection viable
at all. A changing capacity is a full recompile of a trace containing nested scans over
sweeps and colours -- every single step. So the capacity is chosen once, with headroom, and
overflow is a hard error rather than a silent drop of whichever pairs happened to be last.
"""

import numpy as np

PAIR_TYPES = {
    "VERTEX_TRIANGLE": 0,
    "EDGE_EDGE": 1,
}

# Padding points every slot at vertex 0 -- four coincident points -- and marks it invalid. The
# energy kernels must therefore mask on `valid` *before* the barrier can see the (meaningless)
# distance, because a barrier at zero distance is enormous. Masking the result afterwards is
# not enough: `0 * inf` is NaN.
#
# Padded slots are also typed VERTEX_TRIANGLE (code 0), which is not arbitrary. The distance
# functions guard their denominators, so either branch is safe -- but the vertex-triangle
# branch is the one whose degenerate case is most obviously benign, and `PaddedPairTests` pins
# the invariant so a future change cannot quietly route coincident points somewhere else.
_PAD = 0


def _encode_cells(cells: np.ndarray, origin: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Flatten integer grid cells to one int64 code each, injectively."""
    local = cells - origin
    return (local[:, 0] * span[1] + local[:, 1]) * span[2] + local[:, 2]


def _grid_candidates(
    query_points: np.ndarray, target_points: np.ndarray, cell_size: float
):
    """Return ``(query_index, target_index)`` for every pair sharing a 3x3x3 cell block.

    A uniform spatial hash, but expressed as a **sorted join** rather than a dict of Python
    lists. The dict version was O(candidates) in the interpreter and ran every single step;
    it is the thing that would not survive the enlarged detection band that the CCD guarantee
    requires, so the broadphase has to be array code all the way down.

    The join: bucket both point sets into cells, encode each cell as a single int64, sort the
    targets by code, and for each of the 27 neighbour offsets binary-search the queries into
    that sorted array. ``searchsorted`` gives a contiguous run of matching targets per query,
    and the runs are expanded to explicit index pairs with a ragged ``repeat``/``arange``.
    """
    query_cells = np.floor(query_points / cell_size).astype(np.int64)
    target_cells = np.floor(target_points / cell_size).astype(np.int64)

    # Room for the +-1 neighbourhood, so a shifted query can never encode outside the grid.
    origin = np.minimum(query_cells.min(axis=0), target_cells.min(axis=0)) - 1
    extent = np.maximum(query_cells.max(axis=0), target_cells.max(axis=0)) + 1
    span = extent - origin + 1

    if int(span[0]) * int(span[1]) * int(span[2]) > 2**62:
        raise ValueError(
            f"contact broadphase grid overflowed: {span.tolist()} cells of size "
            f"{cell_size:g} span the mesh. The detection band is far smaller than the "
            f"coordinate range -- rescale the mesh or raise d_hat."
        )

    target_codes = _encode_cells(target_cells, origin, span)
    order = np.argsort(target_codes, kind="stable")
    sorted_codes = target_codes[order]

    queries = []
    targets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                shifted = query_cells + np.array([dx, dy, dz], dtype=np.int64)
                codes = _encode_cells(shifted, origin, span)
                left = np.searchsorted(sorted_codes, codes, side="left")
                right = np.searchsorted(sorted_codes, codes, side="right")
                counts = right - left
                total = int(counts.sum())
                if total == 0:
                    continue
                # Expand each query's contiguous run [left, right) into explicit indices.
                query_index = np.repeat(np.arange(counts.size), counts)
                run_starts = np.repeat(left, counts)
                within = np.arange(total) - np.repeat(
                    np.cumsum(counts) - counts, counts
                )
                queries.append(query_index)
                targets.append(order[run_starts + within])

    if not queries:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return np.concatenate(queries), np.concatenate(targets)


def _narrow_phase_batch(positions, quads, kinds):
    """Squared distance for a *fixed-size* batch of candidate pairs.

    Calls the very same ``pair_distance_sq`` the barrier and the CCD are built on, so the
    three can never disagree about how far apart two primitives are.
    """
    import jax
    import jax.numpy as jnp

    from diff_vbd.solver.contact.distances import pair_distance_sq

    points = jnp.asarray(positions)[jnp.asarray(quads)]  # (B, 4, 3)
    return np.asarray(jax.vmap(pair_distance_sq)(points, jnp.asarray(kinds)))


def _next_batch_size(count: int) -> int:
    """Round a candidate count up to the next power of two, with a floor.

    The batch shape is what jit keys on. A batch that is exactly as long as the candidate
    list would be a *different shape every step*, and therefore a full recompile every step
    -- which is the same trap the fixed pair capacity exists to avoid, one level down.
    Bucketing to powers of two bounds the number of distinct traces to a handful.
    """
    size = 64
    while size < count:
        size *= 2
    return size


def _narrow_phase(positions, pair_vertices, pair_types, band):
    """Keep the candidate pairs whose true distance is within the band, and their distances.

    The distances are returned, not discarded: E1 below needs exactly these numbers, so the
    already-intersecting check costs nothing.
    """
    count = int(pair_types.shape[0])
    batch = _next_batch_size(count)

    quads = np.zeros((batch, 4), dtype=np.int32)
    kinds = np.zeros((batch,), dtype=np.int32)
    quads[:count] = pair_vertices
    kinds[:count] = pair_types

    squared = _narrow_phase_batch(positions, quads, kinds)[:count]
    keep = squared <= band * band
    return pair_vertices[keep], pair_types[keep], squared[keep]


def _reject_existing_intersections(pair_vertices, squared, d_hat):
    """E1. Raise if the step *begins* with a pair already at (or through) zero distance.

    This is the precondition the whole intersection-free argument rests on. ACCD certifies
    that a pair which starts separated stays separated; started already touching, its
    certificate is vacuous. The barrier, meanwhile, does not fail loudly either -- it hits the
    linear continuation below its floor and produces a large but perfectly plausible-looking
    restoring force, so the simulation runs on and quietly means nothing.

    Free to check: the narrow phase just computed every one of these distances.
    """
    floor = 1.0e-6 * d_hat
    intersecting = np.flatnonzero(squared <= floor * floor)
    if intersecting.size == 0:
        return
    first = int(intersecting[0])
    raise ValueError(
        f"{intersecting.size} contact pair(s) are already intersecting at the start of the "
        f"step: pair {pair_vertices[first].tolist()} is at distance "
        f"{float(np.sqrt(squared[first])):g} (the floor is {floor:g}). Intersection-freedom "
        f"is only maintained from an intersection-free state, so the guarantee cannot be "
        f"re-established from here. The usual causes are a mesh that self-intersects at "
        f"rest, a non-conforming tet split reporting interior faces as surface, or an "
        f"initial condition that starts two bodies overlapping."
    )


def detect_contact_pairs(
    positions: np.ndarray,
    surface_triangles: np.ndarray,
    surface_edges: np.ndarray,
    *,
    d_hat: float,
    capacity: int,
    band: float | None = None,
    inflation: float = 2.0,
):
    """Return padded vertex-triangle and edge-edge pair buffers within ``band``.

    Primitives that *share a vertex* are skipped. They are adjacent by construction -- a
    vertex always touches its own triangles -- so their distance is zero and the barrier
    would be infinite. This is the difference between self-collision and self-adjacency,
    and conflating them makes a mesh explode on the first step.

    ``band`` is the detection radius, and it is a **guarantee parameter, not a tuning knob**.
    A pair further apart than ``band`` at the start of the step is absent from the returned
    set, and therefore invisible to the CCD for the whole step -- so it could close to zero
    and tunnel with nothing watching. The band is only sound if no vertex can move far enough
    to bring such a pair into contact, which is the inequality ``band >= d_hat + 2 * Δmax``
    that ``ccd.derive_detection_band`` computes and the sweep's displacement clamp enforces.
    Pass the derived band; the ``inflation * d_hat`` fallback is for callers that have no
    motion (detection at rest, and the tests).
    """
    positions = np.asarray(positions, dtype=np.float64)
    triangles = np.asarray(surface_triangles, dtype=np.int64).reshape(-1, 3)
    edges = np.asarray(surface_edges, dtype=np.int64).reshape(-1, 2)

    band = inflation * d_hat if band is None else float(band)
    cell = max(band, 1.0e-12)

    quads: list[np.ndarray] = []
    kinds: list[np.ndarray] = []

    # --- vertex vs triangle ---------------------------------------------------------
    surface_vertices = np.unique(triangles)
    triangle_centroids = positions[triangles].mean(axis=1)
    triangle_radius = np.linalg.norm(
        positions[triangles] - triangle_centroids[:, None, :], axis=-1
    ).max(axis=1)

    if surface_vertices.size and triangles.shape[0]:
        # The 3x3x3 neighbourhood finds every target within `cell_size` of a query, so the
        # cell must be at least as large as the widest gap the cull below can accept -- here
        # `band + triangle_radius`, since a vertex has no radius of its own. A cell any
        # smaller silently *misses* pairs, which is the worst possible broadphase failure:
        # an undetected pair has no barrier and no CCD, so it is free to interpenetrate.
        cell_size = cell + float(triangle_radius.max(initial=0.0))
        vertex_index, triangle_index = _grid_candidates(
            positions[surface_vertices], triangle_centroids, max(cell_size, cell)
        )
        vertices = surface_vertices[vertex_index]
        candidate_triangles = triangles[triangle_index]

        # Adjacent, not colliding: a vertex always touches its own triangles, so their
        # distance is zero and the barrier would be infinite. Conflating self-collision with
        # self-adjacency makes a mesh explode on the first step.
        adjacent = (candidate_triangles == vertices[:, None]).any(axis=1)
        gap = np.linalg.norm(
            positions[vertices] - triangle_centroids[triangle_index], axis=-1
        )
        keep = ~adjacent & (gap <= band + triangle_radius[triangle_index])

        quads.append(
            np.concatenate(
                [vertices[keep, None], candidate_triangles[keep]], axis=1
            )
        )
        kinds.append(
            np.full(int(keep.sum()), PAIR_TYPES["VERTEX_TRIANGLE"], dtype=np.int64)
        )

    # --- edge vs edge ---------------------------------------------------------------
    edge_midpoints = positions[edges].mean(axis=1)
    edge_radius = (
        np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=-1) / 2.0
    )

    if edges.shape[0]:
        # Both sides have a radius here, so the widest acceptable gap is
        # `band + r_a + r_b <= band + 2 * max_radius` -- note the **2**. Sizing this cell at
        # `band + max_radius` (as the point-triangle case correctly does, because a vertex is
        # a point) under-covers by one edge radius and drops real edge-edge pairs.
        edge_cell = cell + 2.0 * float(edge_radius.max(initial=0.0))
        a_index, b_index = _grid_candidates(
            edge_midpoints, edge_midpoints, max(edge_cell, cell)
        )
        edges_a = edges[a_index]
        edges_b = edges[b_index]

        unordered = b_index > a_index  # each unordered pair once, and never a self-pair
        shares_vertex = (edges_a[:, :, None] == edges_b[:, None, :]).any(axis=(1, 2))
        gap = np.linalg.norm(
            edge_midpoints[a_index] - edge_midpoints[b_index], axis=-1
        )
        keep = (
            unordered
            & ~shares_vertex
            & (gap <= band + edge_radius[a_index] + edge_radius[b_index])
        )

        quads.append(np.concatenate([edges_a[keep], edges_b[keep]], axis=1))
        kinds.append(np.full(int(keep.sum()), PAIR_TYPES["EDGE_EDGE"], dtype=np.int64))

    pair_vertices = (
        np.concatenate(quads, axis=0)
        if quads
        else np.empty((0, 4), dtype=np.int64)
    )
    pair_types = (
        np.concatenate(kinds, axis=0) if kinds else np.empty((0,), dtype=np.int64)
    )

    # --- narrow phase ---------------------------------------------------------------
    # The broadphase culls on bounding radii, which is loose: it admits pairs whose
    # *bounds* overlap the band even when the primitives themselves are far apart. Without
    # this step a thin mesh reports hundreds of "contacts" at rest. Reuse the very same
    # distance kernel the barrier uses, so detection and energy cannot disagree about how
    # far apart two primitives are.
    if pair_types.size:
        pair_vertices, pair_types, squared = _narrow_phase(
            positions, pair_vertices, pair_types, band
        )
        _reject_existing_intersections(pair_vertices, squared, d_hat)

    found = int(pair_types.shape[0])
    if found > capacity:
        raise ValueError(
            f"contact pair capacity exceeded: found {found} pairs within {band:g} of each "
            f"other but the buffers hold {capacity}.\n"
            f"The band is {band:g} against an activation distance of {d_hat:g}. A band far "
            f"larger than d_hat means the band is being driven by *speed*, not proximity: it "
            f"has to cover everything the mesh could reach this step, or a pair could close "
            f"from outside the set and tunnel unseen. So the usual fix is not a bigger "
            f"buffer -- it is less motion per step. Reduce solver.dt, or accept the cost and "
            f"raise contact.capacity (which recompiles the solver, so choose it once with "
            f"headroom)."
        )

    padded_vertices = np.full((capacity, 4), _PAD, dtype=np.int32)
    padded_types = np.zeros((capacity,), dtype=np.int32)
    valid = np.zeros((capacity,), dtype=bool)
    if found:
        padded_vertices[:found] = np.asarray(pair_vertices, dtype=np.int32)
        padded_types[:found] = np.asarray(pair_types, dtype=np.int32)
        valid[:found] = True

    return padded_vertices, padded_types, valid


def build_contact_incidence(
    pair_vertices: np.ndarray,
    pair_valid: np.ndarray,
    num_vertices: int,
    max_per_vertex: int,
):
    """Return padded per-vertex contact incidence, mirroring ``build_incidence`` for tets.

    ``max_per_vertex`` is fixed rather than data-derived, for the same reason the pair
    capacity is: it is an array dimension, and a dimension that changes between steps is a
    recompile. Overflow is a hard error -- never a silent drop of whichever pairs happened to
    be last, which would quietly delete contact forces from the most crowded vertex on the
    mesh, exactly where they matter most.

    Vectorised, because this runs every step: the sort-and-rank below is O(live pairs), while
    the obvious loop is O(capacity) in the interpreter and pays for all 4096 padded slots even
    when three pairs are live.
    """
    incidence = np.zeros((num_vertices, max_per_vertex), dtype=np.int32)
    mask = np.zeros((num_vertices, max_per_vertex), dtype=bool)
    if max_per_vertex == 0:
        return incidence, mask

    live = np.flatnonzero(pair_valid)
    if live.size == 0:
        return incidence, mask

    # One (vertex, pair) row per incidence. A valid pair always has four distinct vertices
    # (detection skips primitives that share one), so no de-duplication is needed here.
    vertices = np.asarray(pair_vertices)[live].reshape(-1).astype(np.int64)
    pairs = np.repeat(live.astype(np.int64), 4)

    # Group by vertex, then rank within the group: sorting makes each vertex's incidences
    # contiguous, so a slot index is just the offset from its group's start.
    order = np.lexsort((pairs, vertices))
    vertices = vertices[order]
    pairs = pairs[order]

    counts = np.bincount(vertices, minlength=num_vertices)
    starts = np.zeros(num_vertices, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    slots = np.arange(vertices.size, dtype=np.int64) - starts[vertices]

    if slots.max(initial=-1) >= max_per_vertex:
        busiest = int(counts.argmax())
        raise ValueError(
            f"contact incidence overflow: vertex {busiest} is touched by "
            f"{int(counts.max())} contact pairs but max_per_vertex is {max_per_vertex}. "
            f"Raise contact.max_per_vertex to at least {int(counts.max())} (note that it is "
            f"an array dimension, so changing it recompiles the solver -- choose it once "
            f"with headroom). Flat-on-flat contact between two coincident faces is the usual "
            f"cause: it generates a large edge-edge fan at a single vertex."
        )

    incidence[vertices, slots] = pairs
    mask[vertices, slots] = True
    return incidence, mask
