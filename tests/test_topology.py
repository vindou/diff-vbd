"""Setup-builder equivalence gates (M5).

The topology builders were rewritten from Python loops into array code (argsort, unique,
scatter-add) because setup was the scaling wall: `build_lumped_masses` alone dispatched one
device op per tet. A rewrite like that has exactly one honest gate: **the outputs are the
same arrays**, element for element, against the old implementations — which are preserved
verbatim below as `_reference_*`. "Looks right" is not a gate; a transposed incidence table
or a face that lost its outward winding produces a solver that runs fine and is quietly
wrong.

The single deliberate exception is `build_vertex_coloring`: it moved from a serial greedy
pass to Jones–Plassmann, whose colouring is *valid but different*. Nothing downstream
depends on which valid colouring it gets — a colour is only ever "no two adjacent vertices
solve simultaneously" — so the gate there is validity (checked against the exact adjacency
relation), determinism (the schedule must not change run to run), and the structural
invariants of the padded group buffers.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from diff_vbd.setup.topology import (
    build_incidence,
    build_lumped_masses,
    build_surface_topology,
    build_vertex_adjacency,
    build_vertex_coloring,
)
from diff_vbd.solver.kinematics import tet_volume


# --------------------------------------------------------------------------------------
# The pre-rewrite implementations, verbatim. These are the reference the gate compares
# against; do not "improve" them.
# --------------------------------------------------------------------------------------


def _reference_incidence(tets, num_vertices):
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


def _reference_adjacency(tets, num_vertices):
    adjacency = [set() for _ in range(num_vertices)]
    for tet in jax.device_get(tets):
        tet_vertices = [int(v) for v in tet]
        for i, src in enumerate(tet_vertices):
            for dst in tet_vertices[i + 1 :]:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
    return adjacency


def _reference_surface_topology(tets):
    face_offsets = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))

    face_counts = {}
    face_table = {}
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


def _reference_greedy_coloring(tets, num_vertices):
    adjacency = _reference_adjacency(tets, num_vertices)
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


def _reference_lumped_masses(rest_positions, tets):
    masses = jnp.zeros((rest_positions.shape[0],), dtype=rest_positions.dtype)
    for tet in jax.device_get(tets):
        tet_vertices = jnp.array(tet, dtype=jnp.int32)
        tet_rest_positions = rest_positions[tet_vertices]
        contribution = tet_volume(tet_rest_positions) / 4.0
        masses = masses.at[tet_vertices].add(contribution)
    return masses


# --------------------------------------------------------------------------------------
# Fixtures. Kuhn-split grids of a few shapes (the mesh family every other test uses),
# plus the degenerate corners a vectorised rewrite is most likely to fumble: a single
# tet, and a vertex that belongs to no tet at all.
# --------------------------------------------------------------------------------------


def _kuhn_grid(nx, ny, nz, spacing=0.25):
    grid = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1
    )
    positions = grid.reshape(-1, 3).astype(np.float64) * spacing
    index = lambda i, j, k: (i * ny + j) * nz + k
    tets = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c = [
                    index(i + a, j + b, k + d)
                    for a in (0, 1)
                    for b in (0, 1)
                    for d in (0, 1)
                ]
                v000, v001, v010, v011, v100, v101, v110, v111 = c
                tets += [
                    [v000, v100, v110, v111],
                    [v000, v101, v100, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v001, v101, v111],
                    [v000, v011, v001, v111],
                ]
    return (
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.asarray(np.array(tets), dtype=jnp.int32),
    )


def _single_tet():
    positions = jnp.asarray(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        dtype=jnp.float64,
    )
    tets = jnp.asarray(np.array([[0, 1, 2, 3]]), dtype=jnp.int32)
    return positions, tets


def _tet_with_isolated_vertex():
    """One tet plus a fifth vertex no tet touches: every builder must pad, not crash."""
    positions, tets = _single_tet()
    positions = jnp.concatenate(
        [positions, jnp.asarray([[9.0, 9.0, 9.0]], dtype=jnp.float64)]
    )
    return positions, tets


def _fixtures():
    yield "single_tet", _single_tet()
    yield "isolated_vertex", _tet_with_isolated_vertex()
    yield "kuhn_2x2x2", _kuhn_grid(2, 2, 2)
    yield "kuhn_3x3x3", _kuhn_grid(3, 3, 3)
    yield "kuhn_5x4x3", _kuhn_grid(5, 4, 3)


class BuilderEquivalenceTests(unittest.TestCase):
    def test_incidence_is_element_wise_identical(self):
        """Identical, not equivalent: the stable argsort preserves the old tet-major
        slot order within each vertex, so even the padding layout must match."""
        for name, (positions, tets) in _fixtures():
            with self.subTest(fixture=name):
                num_vertices = positions.shape[0]
                ref_tets, ref_mask = _reference_incidence(tets, num_vertices)
                new_tets, new_mask = build_incidence(tets, num_vertices)
                np.testing.assert_array_equal(np.asarray(ref_tets), np.asarray(new_tets))
                np.testing.assert_array_equal(np.asarray(ref_mask), np.asarray(new_mask))

    def test_adjacency_is_identical(self):
        for name, (positions, tets) in _fixtures():
            with self.subTest(fixture=name):
                self.assertEqual(
                    _reference_adjacency(tets, positions.shape[0]),
                    build_vertex_adjacency(tets, positions.shape[0]),
                )

    def test_surface_topology_is_element_wise_identical(self):
        """Triangles must keep their outward winding *and* the old lexicographic row
        order — contact indexes these buffers, and a re-ordered triangle array is a
        different (if valid) pair set, which the equivalence gate deliberately rejects."""
        for name, (positions, tets) in _fixtures():
            with self.subTest(fixture=name):
                ref = _reference_surface_topology(tets)
                new = build_surface_topology(tets)
                for ref_part, new_part, label in zip(
                    ref, new, ("triangles", "edges", "surface_vertices")
                ):
                    np.testing.assert_array_equal(
                        np.asarray(ref_part), np.asarray(new_part), err_msg=label
                    )

    def test_lumped_masses_are_element_wise_identical(self):
        """Exact equality, which holds because the scatter-add accumulates each vertex's
        contributions in the same tet order the old sequential loop did. If a future XLA
        changes scatter ordering this becomes a 1-2 ulp difference — loosen to
        rtol=1e-15 *with a comment*, not silently."""
        for name, (positions, tets) in _fixtures():
            with self.subTest(fixture=name):
                ref = np.asarray(_reference_lumped_masses(positions, tets))
                new = np.asarray(build_lumped_masses(positions, tets))
                np.testing.assert_array_equal(ref, new)


class ColoringValidityTests(unittest.TestCase):
    """The colouring is pinned element-wise to the old greedy pass, not merely valid.

    An earlier revision of this branch replaced greedy with Jones–Plassmann and gated
    on validity alone — and a valid-but-different colouring changed the Gauss–Seidel
    sweep order enough to flip a resting frictionless block from a clean slide into a
    perpetual bounce at the same iteration budget (see FINDINGS.md). The sweep order is
    solver behaviour, so the colouring is held bit-identical and only its computation
    was vectorised (dependency waves over the lower-indexed-neighbour DAG)."""

    def test_coloring_is_element_wise_identical_to_the_greedy_reference(self):
        for name, (positions, tets) in _fixtures():
            with self.subTest(fixture=name):
                num_vertices = positions.shape[0]
                for ref_part, new_part, label in zip(
                    _reference_greedy_coloring(tets, num_vertices),
                    build_vertex_coloring(tets, num_vertices),
                    ("groups", "mask", "colors"),
                ):
                    np.testing.assert_array_equal(
                        np.asarray(ref_part), np.asarray(new_part), err_msg=label
                    )

    def test_no_edge_joins_two_vertices_of_one_color(self):
        for name, (positions, tets) in _fixtures():
            with self.subTest(fixture=name):
                num_vertices = positions.shape[0]
                _, _, colors = build_vertex_coloring(tets, num_vertices)
                colors = np.asarray(colors)
                adjacency = _reference_adjacency(tets, num_vertices)
                for vertex, neighbours in enumerate(adjacency):
                    for neighbour in neighbours:
                        self.assertNotEqual(
                            colors[vertex],
                            colors[neighbour],
                            msg=f"{name}: vertices {vertex} and {neighbour} share a colour",
                        )

    def test_coloring_is_deterministic(self):
        """The colouring sizes and orders `color_groups`, which the jitted sweep scans;
        a run-to-run difference would silently re-trace the solver."""
        _, tets = _kuhn_grid(3, 3, 3)
        first = build_vertex_coloring(tets, 27)
        second = build_vertex_coloring(tets, 27)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_groups_partition_the_vertices_exactly_once(self):
        for name, (positions, tets) in _fixtures():
            with self.subTest(fixture=name):
                num_vertices = positions.shape[0]
                groups, mask, colors = build_vertex_coloring(tets, num_vertices)
                groups = np.asarray(groups)
                mask = np.asarray(mask)
                colors = np.asarray(colors)

                live = groups[mask]
                self.assertEqual(live.size, num_vertices)
                np.testing.assert_array_equal(np.sort(live), np.arange(num_vertices))
                # Each group row holds exactly the vertices of its colour.
                for color_index in range(groups.shape[0]):
                    members = groups[color_index][mask[color_index]]
                    np.testing.assert_array_equal(
                        np.sort(members), np.flatnonzero(colors == color_index)
                    )


if __name__ == "__main__":
    unittest.main()
