# Results — feature/infrastructure

Measured numbers per milestone, plus what a reviewer would attack first. All CPU timings on
the development machine (macOS, CPU-only JAX 0.10.2, float64) unless stated otherwise.

## M5 — Setup performance

Mesh: Kuhn-split 27x27x27 grid — 19,683 vertices, 105,456 tets. Best of one run each
(the old `build_lumped_masses` is too slow to repeat); new-implementation times include
their host->device conversions.

| builder | old (s) | new (s) | speedup |
|---|---|---|---|
| build_lumped_masses | 113.9 | 0.49 | ~230x |
| build_incidence | 3.09 | 0.08 | ~40x |
| build_surface_topology | 1.21 | 0.16 | ~7x |
| build_vertex_coloring | 0.67 | 0.22 | ~3x |
| build_vertex_adjacency | 0.41 | 0.13 | ~3x |

Equivalence: `build_incidence`, `build_surface_topology`, `build_vertex_adjacency` and
`build_lumped_masses` are **element-wise identical** to the old implementations on all
fixtures (masses bitwise-exact — the scatter-add accumulates in the same order the serial
loop did). `build_vertex_coloring` moved from serial greedy to Jones–Plassmann and colours
differently: the gate is validity + determinism (see `tests/test_topology.py`). On the
fixtures JP uses 0–2 more colours than greedy (e.g. 7 vs 5 on the 3x3x3 grid); each extra
colour adds one color-block scan per sweep, which is noise next to the per-vertex solve cost.

What a reviewer would attack first: the coloring speedup is modest (3x) because the numpy
Jones–Plassmann loop still runs O(log V) interpreter rounds; its advantage over the greedy
pass grows with mesh size (the greedy pass is a pure O(V·deg) interpreter loop and falls
hopelessly behind at millions of vertices), but no million-vertex measurement is reported
here. And the masses gate asserts *bitwise* equality, which leans on XLA's current scatter
ordering; if a future backend reorders it, the test should be loosened to rtol=1e-15 with a
comment, not deleted.
