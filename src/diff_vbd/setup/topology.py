"""Topology and precompute builders.

Everything here is setup-time code, but setup time is not free time: the solver is
GPU-parallel and these builders are what stand between a mesh file and the first step. The
original implementations were Python loops — one ``.at[].add()`` dispatch *per tet* for the
masses, a dict-of-lists pass per face for the surface, a set rebuilt per vertex for the
colouring — which at paper-scale meshes (hundreds of thousands of tets) is the wall. Each
builder below is array code: the combinatorics are expressed as sorts, ``unique`` and
scatters, so the cost is a handful of O(T log T) passes rather than O(T) interpreter
round-trips. Outputs are pinned element-wise to the old implementations by the tests
(``build_vertex_coloring`` excepted — it may colour differently, and the test asserts
validity instead).
"""

import jax
import jax.numpy as jnp
import numpy as np

from diff_vbd.solver.kinematics import tet_volume


def build_incidence(tets: jnp.ndarray, num_vertices: int):
    """Build padded incident tet lists for each vertex.

    Argsort-based: flatten the tet array to (vertex, tet) records, stable-sort by vertex,
    and each vertex's incidences become one contiguous run whose slot index is the offset
    from the run's start. The *stable* sort is what preserves the original tet-major order
    within each vertex, so the output is element-wise identical to the old per-tet loop,
    not merely equivalent.
    """
    tets_np = np.asarray(jax.device_get(tets), dtype=np.int64)
    num_tets = tets_np.shape[0]

    vertices = tets_np.reshape(-1)
    tet_ids = np.repeat(np.arange(num_tets, dtype=np.int64), 4)

    order = np.argsort(vertices, kind="stable")
    vertices = vertices[order]
    tet_ids = tet_ids[order]

    counts = np.bincount(vertices, minlength=num_vertices)
    max_incident = int(counts.max(initial=0))
    starts = np.zeros(num_vertices, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    slots = np.arange(vertices.size, dtype=np.int64) - starts[vertices]

    incident_tets = np.zeros((num_vertices, max_incident), dtype=np.int32)
    incident_mask = np.zeros((num_vertices, max_incident), dtype=bool)
    incident_tets[vertices, slots] = tet_ids
    incident_mask[vertices, slots] = True

    return jnp.asarray(incident_tets), jnp.asarray(incident_mask)


def _unique_tet_edges(tets_np: np.ndarray) -> np.ndarray:
    """Return the deduplicated undirected edges induced by tetrahedra, as (E, 2) int64.

    Each canonicalised edge is packed into one int64 so the deduplication is a 1-D
    ``unique`` (a plain sort) rather than ``unique(axis=0)``, whose void-view row sort is
    several times slower and was the dominant cost of both colouring and adjacency. The
    packing is injective while ``max_index**2`` fits in an int64, i.e. for meshes below
    ~3e9 vertices — beyond any mesh this solver will ever hold in memory.
    """
    pair_slots = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edges = tets_np[:, np.asarray(pair_slots)].reshape(-1, 2)
    base = np.int64(tets_np.max(initial=0)) + 1
    codes = np.unique(edges.min(axis=1) * base + edges.max(axis=1))
    return np.stack([codes // base, codes % base], axis=1)


def build_vertex_adjacency(tets: jnp.ndarray, num_vertices: int):
    """Build the vertex adjacency graph induced by tetrahedra.

    Kept for API compatibility (the output is a list of Python sets, which is itself the
    slow part); the colouring below no longer consumes it. The construction goes through
    the deduplicated edge array first, so the interpreter loop runs once per unique edge
    rather than six times per tet with repeated set insertions.
    """
    adjacency = [set() for _ in range(num_vertices)]
    edges = _unique_tet_edges(np.asarray(jax.device_get(tets), dtype=np.int64))
    # `.tolist()` first: iterating numpy rows yields numpy scalars, and converting those
    # one at a time costs more than the whole set-building loop.
    for a, b in edges.tolist():
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def build_vertex_coloring(tets: jnp.ndarray, num_vertices: int):
    """Build the same greedy vertex coloring as ever, in vectorised waves.

    The old implementation walked vertices in index order, colouring each with the
    smallest colour absent among its already-coloured neighbours. That order-dependence
    is not cosmetic: the colour classes set the Gauss-Seidel sweep order, and a
    different (equally *valid*) colouring measurably changes how fast a sweep converges
    -- switching to Jones-Plassmann flipped a resting frictionless block from a clean
    slide into a perpetual bounce at the same iteration budget. So the rewrite keeps
    the colouring **bit-identical** and only changes how it is computed: since vertex
    ``v``'s colour depends exactly on its lower-indexed neighbours, the dependency
    graph is a DAG whose levels can be coloured in parallel waves -- ``level[v] =
    1 + max(level[u])`` over neighbours ``u < v``, computed by iterating a vectorised
    ``maximum.at`` to its fixed point (one round per level), then one
    smallest-missing-colour pass per wave over each wave's incoming edges. Interpreter
    work is O(depth) rounds of array ops instead of O(V) per-vertex set rebuilds.
    """
    tets_np = np.asarray(jax.device_get(tets), dtype=np.int64)
    edges = _unique_tet_edges(tets_np)
    low = edges.min(axis=1)
    high = edges.max(axis=1)

    # Wave levels: fixed point of level[high] = max(level[high], level[low] + 1).
    levels = np.zeros(num_vertices, dtype=np.int64)
    while True:
        proposed = levels[low] + 1
        before = levels.copy()
        np.maximum.at(levels, high, proposed)
        if np.array_equal(levels, before):
            break

    colors = np.full(num_vertices, -1, dtype=np.int64)
    order_by_level = np.argsort(levels, kind="stable")
    wave_starts = np.flatnonzero(
        np.concatenate([[True], np.diff(levels[order_by_level]) > 0])
    )
    wave_bounds = np.concatenate([wave_starts, [num_vertices]])

    # Incoming (lower-indexed) edges grouped by their higher endpoint, once.
    order_by_high = np.argsort(high, kind="stable")
    sorted_high = high[order_by_high]
    sorted_low = low[order_by_high]
    incoming_counts = np.bincount(high, minlength=num_vertices)
    incoming_starts = np.zeros(num_vertices + 1, dtype=np.int64)
    incoming_starts[1:] = np.cumsum(incoming_counts)

    for wave_index in range(wave_bounds.size - 1):
        wave = order_by_level[
            wave_bounds[wave_index] : wave_bounds[wave_index + 1]
        ]
        wave = np.sort(wave)
        degrees = incoming_counts[wave]
        total = int(degrees.sum())
        colors[wave] = 0  # no lower neighbours: colour 0
        if total == 0:
            continue
        seg_off = np.zeros(wave.size + 1, dtype=np.int64)
        seg_off[1:] = np.cumsum(degrees)
        flat = np.repeat(incoming_starts[wave], degrees) + (
            np.arange(total, dtype=np.int64) - np.repeat(seg_off[:-1], degrees)
        )
        owner = np.repeat(np.arange(wave.size, dtype=np.int64), degrees)
        held = colors[sorted_low[flat]]
        # Lower-indexed neighbours are all coloured (their level is strictly smaller).
        # Smallest colour absent from each owner's used set: sort and dedupe the
        # (owner, colour) pairs; the answer is the first rank where the sorted colour
        # run departs from 0, 1, 2, ... (or the run length if it never does).
        sort_order = np.lexsort((held, owner))
        owner, held = owner[sort_order], held[sort_order]
        first = np.ones(owner.size, dtype=bool)
        first[1:] = (owner[1:] != owner[:-1]) | (held[1:] != held[:-1])
        owner, held = owner[first], held[first]
        seg_starts = np.flatnonzero(
            np.concatenate([[True], owner[1:] != owner[:-1]])
        )
        seg_lengths = np.diff(np.concatenate([seg_starts, [owner.size]]))
        rank = np.arange(owner.size) - np.repeat(seg_starts, seg_lengths)
        gap = np.where(held != rank, rank, owner.size)
        first_gap = np.minimum.reduceat(gap, seg_starts)
        colors[wave[owner[seg_starts]]] = np.where(
            first_gap < owner.size, first_gap, seg_lengths
        )

    num_colors = int(colors.max()) + 1
    counts = np.bincount(colors, minlength=num_colors)
    max_group_size = int(counts.max())

    color_groups = np.zeros((num_colors, max_group_size), dtype=np.int32)
    color_group_mask = np.zeros((num_colors, max_group_size), dtype=bool)
    group_order = np.argsort(colors, kind="stable")
    starts = np.zeros(num_colors, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    slots = np.arange(num_vertices, dtype=np.int64) - starts[colors[group_order]]
    color_groups[colors[group_order], slots] = group_order
    color_group_mask[colors[group_order], slots] = True

    return (
        jnp.asarray(color_groups),
        jnp.asarray(color_group_mask),
        jnp.asarray(colors, dtype=jnp.int32),
    )


def build_surface_topology(tets: jnp.ndarray):
    """Return the boundary triangles and boundary edges, in *global* vertex indices.

    A face is on the boundary exactly when it belongs to a single tet; an interior face is
    shared by two. The winding is kept outward, which contact needs — an inverted triangle
    reports its normal backwards and the barrier then pushes the wrong way.

    Sort + ``unique`` on canonicalised faces, not a dict count: every face key is a sorted
    vertex triple, ``np.unique`` counts the duplicates in one pass, and the *original*
    (outward-wound) rows are recovered by indexing back into the uncanonicalised array —
    canonicalisation is only ever a grouping key, never the stored value, or the winding
    would be destroyed exactly where it matters.

    Note this deliberately does *not* reuse ``export.extract_surface_mesh``: that one
    re-indexes into a compact surface-local numbering for writing meshes out, whereas
    contact has to index straight into ``position``, ``mass`` and the incidence tables, all
    of which are in global indices.

    Surface **edges** exist nowhere else in the codebase and are what edge-edge primitives
    are built from.
    """
    tets_np = np.asarray(jax.device_get(tets), dtype=np.int64)
    # The four faces of a tet, wound so their normals point out of the tet.
    face_offsets = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))

    faces = tets_np[:, face_offsets].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    # Group identical keys by lexsort + run detection rather than `unique(axis=0)`: the
    # void-view row sort inside the latter is several times slower, and only the per-face
    # duplicate *count* is needed, never the unique rows themselves.
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    new_group = np.ones(sorted_keys.shape[0], dtype=bool)
    new_group[1:] = (sorted_keys[1:] != sorted_keys[:-1]).any(axis=1)
    group_of_sorted = np.cumsum(new_group) - 1
    group_counts = np.bincount(group_of_sorted)
    group_of_face = np.empty(faces.shape[0], dtype=np.int64)
    group_of_face[order] = group_of_sorted

    triangles = faces[group_counts[group_of_face] == 1]
    # Lexicographic row order, matching the old list-of-tuples sort exactly.
    triangles = triangles[
        np.lexsort((triangles[:, 2], triangles[:, 1], triangles[:, 0]))
    ]

    edge_pairs = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    base = np.int64(tets_np.max(initial=0)) + 1
    edge_codes = np.unique(
        edge_pairs.min(axis=1) * base + edge_pairs.max(axis=1)
    )
    edges = np.stack([edge_codes // base, edge_codes % base], axis=1)

    surface_vertices = np.unique(triangles)

    return (
        jnp.asarray(triangles, dtype=jnp.int32).reshape(-1, 3),
        jnp.asarray(edges, dtype=jnp.int32).reshape(-1, 2),
        jnp.asarray(surface_vertices, dtype=jnp.int32).reshape(-1),
    )


def build_lumped_masses(rest_positions: jnp.ndarray, tets: jnp.ndarray):
    """Build per-vertex lumped masses from tet rest volumes with density 1.

    One vmapped volume kernel and one scatter-add, instead of one device dispatch per tet.
    ``at[].add`` sums duplicate indices, which is exactly the lumping. This stays JAX code
    rather than dropping to numpy like the combinatorial builders above, deliberately: the
    masses are a *differentiable* function of the rest positions, and a caller
    differentiating w.r.t. the rest shape (an adjoint, shape optimisation) needs the
    volume-to-mass path in the graph, not frozen on the host.
    """
    volumes = jax.vmap(tet_volume)(rest_positions[tets])
    contribution = jnp.broadcast_to((volumes / 4.0)[:, None], tets.shape)
    masses = jnp.zeros((rest_positions.shape[0],), dtype=rest_positions.dtype)
    return masses.at[tets].add(contribution)
