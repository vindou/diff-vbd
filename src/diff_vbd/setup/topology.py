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


def build_lumped_masses(rest_positions: jnp.ndarray, tets: jnp.ndarray):
    """Build per-vertex lumped masses from tet rest volumes with density 1."""
    masses = jnp.zeros((rest_positions.shape[0],), dtype=rest_positions.dtype)
    for tet in jax.device_get(tets):
        tet_vertices = jnp.array(tet, dtype=jnp.int32)
        tet_rest_positions = rest_positions[tet_vertices]
        contribution = tet_volume(tet_rest_positions) / 4.0
        masses = masses.at[tet_vertices].add(contribution)
    return masses
