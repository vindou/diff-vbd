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
| build_vertex_coloring | 0.48 | 0.19 | ~2.6x |
| build_vertex_adjacency | 0.41 | 0.13 | ~3x |

Equivalence: **every builder is element-wise identical** to the old implementation on all
fixtures — the masses bitwise-exact (the scatter-add accumulates in the same order the
serial loop did), and the colouring bit-identical by construction: it is the same greedy
colouring, computed in vectorised dependency waves instead of a per-vertex interpreter
loop. The colouring identity is not pedantry: an earlier revision used Jones–Plassmann
gated on validity alone, and the different (equally valid) colour classes changed the
Gauss–Seidel sweep order enough to flip a resting frictionless block from a clean slide
at exactly g·sin(theta)·cos(theta) into a perpetual bounce at the same iteration budget
(caught by the incline friction test; see FINDINGS.md). Sweep order is solver behaviour.

What a reviewer would attack first: the coloring speedup is modest (~2.6x) because the
wave decomposition still runs one interpreter round per DAG level (~mesh diameter); a
truly parallel colouring would change the sweep order and therefore solver behaviour, so
it would need to be gated on physics, not validity. And the masses gate asserts *bitwise*
equality, which leans on XLA's current scatter ordering; if a future backend reorders it,
the test should be loosened to rtol=1e-15 with a comment, not deleted.

## M2 — Two-stage activation vs the IPC log barrier

Scene: the resting-contact block-on-plane (3x3x3 Kuhn block, mu=lam=50, gravity,
d_hat=1e-3), settled 200 steps; conditioning from the dense free-DOF Hessian of the
static potential (81x81 — assembled for measurement only), excluding the free block's
three rigid zero modes; convergence from the excess variational energy per VBD iteration
at a contact-dominated step (the block kicked downward at 2 units/s).

**At equal accuracy (kappa tuned so both rest at the same gap), the two activations are
indistinguishable in this solver.** Barrier at its derived kappa=11.6 rests at gap
4.74e-5; the two-stage needs kappa=56.8 to match (4.91e-5), which puts the contact deep
in its *log stage*, where the two energies have the same local shape:

| | resting gap | lambda_max | cond+ | iters to 1e-5 excess energy |
|---|---|---|---|---|
| barrier (kappa 11.6) | 4.74e-5 | 5.75e3 | 3.26e3 | 9 |
| two-stage (kappa 56.8, log stage) | 4.91e-5 | 5.95e3 | 3.39e3 | 9 |

So: **OGC's 2x convergence and 5x conditioning did not reproduce at IPC's operating
point**, and the reason is structural, not a bug: those numbers belong to OGC's own
operating point, where kappa is large enough that contacts resolve in the *quadratic*
stage. Measuring there:

| | resting gap | lambda_max | cond+ |
|---|---|---|---|
| barrier (kappa 11.6) | 4.74e-5 (0.047 d_hat) | 5.75e3 | 3.26e3 |
| two-stage (kappa 900, quadratic stage) | 6.81e-4 (0.68 d_hat) | 9.69e2 | 5.47e2 |

That is a **5.9x lower peak stiffness and 6.0x better conditioning — OGC's ~5x figure
reproduces** — but at a resting gap 14x larger. The quadratic stage's win is only
available if a large standoff (equivalently, a large contact radius) is acceptable,
which is exactly why OGC pairs this activation with offset geometry (M6, not
implemented) and why M3's bounds alone do not resolve near-contact stiffness. The
per-iteration convergence comparison showed no measurable difference in VBD's
line-searched 3x3 local solves at either operating point; OGC measured theirs in a
Newton solver, where the Hessian conditioning bites directly.

## M3 — Per-vertex bounds vs the global sweep filter

Scene (in `LocalThrottlingTests`): two disconnected block pairs, both upper blocks
descending at 1 unit/s with dt = 5 ms (a 5e-3 inertial step); one upper block hovers
0.6e-3 above its lower block, the other has 30 units of clearance.

| | far body's step | near body's interface approach |
|---|---|---|
| old global sweep filter (analysis) | scaled by the near body's TOI: ~2.7e-4 of its wanted 5e-3 (~5%) | ~2.7e-4 |
| per-vertex bounds (measured) | >= 0.95 x 5e-3 (full inertial step) | < 0.6e-3 (bound-limited) |

The far body is not throttled at all; the near body's *interface layer* is bound-limited
while its own top layer keeps descending as the block compresses — the throttling is
local even within one body. The intersection audit (`min` pair distance after every step
of the fast-impact scenes, line search off, Chebyshev on) stays strictly positive; the
negative-twin test (guarantee off) still tunnels, so the mechanism is load-bearing.

What this does NOT fix, measured: a vertex resting at gap g keeps a budget of 0.45 g per
detection interval, so near-contact stepping is locally slow exactly as before — see the
M2 table for why the full escape needs OGC's large contact radius (M6, not implemented).

## M4 — Static equilibrium and the adjoint

Gate scene: 3x3x3 slab, base clamped, rigid sphere held at 0.02 indentation, gravity on,
barrier active at the solution (min gap 6.7e-4, d_hat 1e-3), residual tolerance 1e-9
(reached in 9 Newton iterations from a feasible initial guess).

Adjoint vs central differences (float64, loss = fixed random weighting of the
equilibrium positions):

| parameter | relative error |
|---|---|
| mu | 8.5e-6 |
| lam | 5.8e-5 |
| density (through the lumped masses) | 4.3e-5 |
| collider radius | 1.2e-5 |
| collider centre height | 9.6e-5 |
| body force (accel z) | 5.3e-6 |
| rest positions (directional) | 2.7e-6 |

All within the 1e-4 gate. The unconverged-adjoint request raises (tested).

## M1 — Hertz validation

Committed test (`tests/test_hertz.py`): graded Kuhn slab, R = 1, deltas 2.5e-3..1e-2
(factor 4), d_hat = delta_min/100 = 2.5e-5, nu = 0.3, E* = 2857.1, theory coefficient
(4/3) E* sqrt(R) = 3809.5. Protocol: sphere fixed per indentation, feasible
column-compressed start, static Newton-CG settle, then dynamic certification
(velocity-zeroed steps + one free step; KE/PE < 1e-4 and |load - reaction| < 5%
asserted). Mesh-refinement study (barrier activation):

| fine cell h | vertices | a_min/h | fitted exponent | fitted coeff / theory | per-delta P/P_theory |
|---|---|---|---|---|---|
| 0.050 | 2,601 | 1.0 | 1.5074 | 1.203 | 1.131, 1.166, 1.196, 1.134 |
| 0.035 | 6,348 | 1.4 | 1.4896 | 1.080 | 1.167, 1.112, 1.148, 1.137 |
| 0.025 | 10,206 | 2.0 | 1.5292 | 1.311* | 1.098, 1.124, 1.126, 1.148 |

*The fitted coefficient is exponent-correlated (extrapolation to delta = 1); the
per-delta ratios are the physical content, and they converge monotonically downward:
mean 1.157 -> 1.141 -> 1.124. The residual ~10-15% at the finest mesh is consistent
with linear-tet contact resolution plus the bonded base at depth exactly 10*a_max (a
known ~5-8% stiffening relative to a true half-space). The exponent is within 0.03 of
3/2 at every resolution — the contact model's force-vs-indentation law is right, and
the stable Neo-Hookean small-strain reduction (the alpha remap) carries the right
prefactor within discretization accuracy.

d_hat sensitivity at delta_min: doubling d_hat (2.5e-5 -> 5e-5) changes P by **1.06%**
(measured by the test) — the standoff bias is bounded by d_hat/delta as predicted, and
at d_hat = delta_min/100 it is comfortably below every other error source.

Committed-test outcome (both activations, coarse mesh, ~8-15 min wall):
barrier: exponent 1.5267, per-delta ratios [1.131, 1.163, 1.198, 1.167], geomean 1.164;
two_stage: exponent 1.4971, per-delta ratios [1.141, 1.144, 1.169, 1.127], geomean 1.145.
The two-stage activation *passing the same Hertz gate* is M2's end-to-end evidence.

The committed suite test runs the coarse (h = 0.05) mesh for BOTH activations
(barrier and two_stage) with gates: |exponent - 1.5| < 0.075; per-delta load ratios
within 25% of theory as a geometric mean (35% individually). The gate is on the loads
at the measured indentations, never on the fitted intercept: the intercept extrapolates
five log-decades to delta = 1, so +-0.03 of exponent noise swings it ~15% while the
physics stands still — the finest mesh has the best per-delta ratios and the worst
intercept, which is the correlation in action. The 25% band is the measured
discretization band plus margin (see DECISIONS.md for why that is calibration, not
tuning).

What a reviewer would attack first: the 25% coefficient gate is wide. It is wide
because the committed mesh is the one the CPU suite can afford; the refinement study
above is the sharp evidence, and re-running `hertz` scripts at h <= 0.025 (or on a GPU)
is the way to tighten it. The finite-depth correction could be removed analytically
(Hayes-type bonded-layer factors) but that would be tuning the theory to the mesh
rather than reporting the discrepancy.

## Final verification

`pytest -q` on the final tree: **198 passed, 271 subtests, 0 failed**, 1h05m wall on the
development machine's CPU. The Hertz numbers above were re-captured on the final tree
after the colouring fix and are identical to the digits shown (the static-settle
protocol does not depend on the sweep order). The first full-suite run is what caught
the colouring regression (FINDINGS.md) — the suite earned its keep on its own branch.
