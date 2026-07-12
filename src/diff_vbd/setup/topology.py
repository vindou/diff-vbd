"""Topology and precompute builders."""

import jax
import jax.numpy as jnp

from diff_vbd.solver.kinematics import tet_volume


def build_incidence(tets: jnp.ndarray, num_vertices: int):
    """Build padded incident tet lists for each vertex."""
    incident_lists = [[] for _ in range(num_vertices)]
    for tet_index, tet in enumerate(jax.device_get(tets)):
        for vertex in tet:
            incident_lists[int(vertex)].append(tet_index)

    max_incident = max(len(entries) for entries in incident_lists)
    incident_tets = []
    incident_mask = []
    for entries in incident_lists:
        pad = max_incident - len(entries)
        incident_tets.append(entries + [0] * pad)
        incident_mask.append([True] * len(entries) + [False] * pad)

    return jnp.array(incident_tets, dtype=jnp.int32), jnp.array(incident_mask)


def build_vertex_adjacency(tets: jnp.ndarray, num_vertices: int):
    """Build the vertex adjacency graph induced by tetrahedra."""
    adjacency = [set() for _ in range(num_vertices)]
    for tet in jax.device_get(tets):
        tet_vertices = [int(v) for v in tet]
        for i, src in enumerate(tet_vertices):
            for dst in tet_vertices[i + 1 :]:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
    return adjacency


def build_vertex_coloring(tets: jnp.ndarray, num_vertices: int):
    """Build a deterministic greedy vertex coloring from tet adjacency."""
    adjacency = build_vertex_adjacency(tets, num_vertices)
    colors = [-1] * num_vertices
    for vertex in range(num_vertices):
        used = {colors[nbr] for nbr in adjacency[vertex] if colors[nbr] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[vertex] = color

    num_colors = max(colors) + 1
    color_lists = [[] for _ in range(num_colors)]
    for vertex, color in enumerate(colors):
        color_lists[color].append(vertex)

    max_group_size = max(len(group) for group in color_lists)
    color_groups = []
    color_group_mask = []
    for group in color_lists:
        pad = max_group_size - len(group)
        color_groups.append(group + [0] * pad)
        color_group_mask.append([True] * len(group) + [False] * pad)

    return (
        jnp.array(color_groups, dtype=jnp.int32),
        jnp.array(color_group_mask),
        jnp.array(colors, dtype=jnp.int32),
    )


def build_surface_topology(tets: jnp.ndarray):
    """Return the boundary triangles and boundary edges, in *global* vertex indices.

    A face is on the boundary exactly when it belongs to a single tet; an interior face is
    shared by two. The winding is kept outward, which contact needs -- an inverted triangle
    reports its normal backwards and the barrier then pushes the wrong way.

    Note this deliberately does *not* reuse ``export.extract_surface_mesh``: that one
    re-indexes into a compact surface-local numbering for writing meshes out, whereas
    contact has to index straight into ``position``, ``mass`` and the incidence tables, all
    of which are in global indices.

    Surface **edges** exist nowhere else in the codebase and are what edge-edge primitives
    are built from.
    """
    # The four faces of a tet, wound so their normals point out of the tet.
    face_offsets = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))

    face_counts: dict[tuple[int, ...], int] = {}
    face_table: dict[tuple[int, ...], tuple[int, ...]] = {}
    for tet in jax.device_get(tets):
        vertices = [int(v) for v in tet]
        for offsets in face_offsets:
            face = tuple(vertices[o] for o in offsets)
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1
            face_table[key] = face

    triangles = [face_table[key] for key, count in face_counts.items() if count == 1]
    triangles.sort()

    edges = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))

    surface_vertices = sorted({v for triangle in triangles for v in triangle})

    return (
        jnp.asarray(triangles, dtype=jnp.int32).reshape(-1, 3),
        jnp.asarray(sorted(edges), dtype=jnp.int32).reshape(-1, 2),
        jnp.asarray(surface_vertices, dtype=jnp.int32).reshape(-1),
    )


def build_lumped_masses(rest_positions: jnp.ndarray, tets: jnp.ndarray):
    """Build per-vertex lumped masses from tet rest volumes with density 1."""
    masses = jnp.zeros((rest_positions.shape[0],), dtype=rest_positions.dtype)
    for tet in jax.device_get(tets):
        tet_vertices = jnp.array(tet, dtype=jnp.int32)
        tet_rest_positions = rest_positions[tet_vertices]
        contribution = tet_volume(tet_rest_positions) / 4.0
        masses = masses.at[tet_vertices].add(contribution)
    return masses
