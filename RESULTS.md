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

## traced-static — the traced solve and adjoint (branch feature/traced-static)

The branch adds a `lax.while_loop` sibling of the host-side static solve so a batch of
independent solves can run under `jax.vmap` — the host driver's Python `while` cannot be
batched, which had pinned a Fisher-information design loop at batch 4. Same kernels,
same acceptance rules; only the driver moved. The host drivers and their call sites
are unchanged and their behaviour is bit-identical; the two shared CG kernels gained
optional default-None preconditioner arguments, which is the whole of the host-path
diff. Gate scene
throughout: the M4 seated fixture (3x3x3 slab, clamped base, sphere indenter, barrier
active).

### Equivalence gates: swept, survivor-filtered, portable

The first gate design pinned parameters "verified to converge under the exact compiled
program the test runs." That pins the machine: on an x86_64 Linux box (same JAX 0.10.2,
CPU) 11 of 23 gates failed, every failure a convergence precondition on the *unchanged
host driver* — the loud-fixture design working, the pinning conclusion wrong. The gates
now sweep a 27-point (mu, collider-height) grid, filter on the `converged` certificates
both drivers already return, assert the property on **every** survivor, and demand at
least 8 survivors. Equivalence is asserted everywhere it is defined, and a platform
where most of the grid stops converging fails loudly about the solver.

Measured on the authoring machine (survivor worst cases; all gates <= 1e-10 relative
on positions, float64):

| gate | result |
|---|---|
| agreement, eager, contact sweep (gap in barrier band asserted per survivor) | 15/27 survive at tol 1e-11; worst rel diff 5.4e-13 class (per-point spot checks 9.6e-14) |
| agreement, contact-free (mu sweep, tol 1e-10 — its floor is ~1e-11) | rel diff 7.1e-14 class |
| batching: vmap lanes vs sequential host solves | survivors match elementwise, same tolerance class |
| adjoint under vmap vs central differences (tol per gate's noise budget) | mu: 4/7 survive, 7.4e-6..1.2e-5; radius: 4/5 survive, 5.0e-6..3.3e-5; rest-directional: 3/7 survive, 2.7e-6..4.3e-6 (gate 1e-4) |
| refusal, constructed asymmetry (tol 1e-6 descent-phase lane + penetrating zigzag lane) | converged=[T,F]; unconverged lane all-NaN gradient; healthy lane equal to solo within 4e-5 (adjoint-CG 1e-5 tolerance class) |
| transcription vs line-for-line host replica (forced rejections, damping ladder 0→1e-4→1e-3, timid accept found at runtime, NaN direction) | bitwise equal, every pass |

Two FD-methodology notes a reviewer should check rather than trust: the comparison
solves run at 1e-11/1e-10 because two correct drivers may stop anywhere below tol,
which alone separates positions by ~tol/lambda_min (≈5e-10 at tol=1e-9 — larger than
the gate with both answers right); and each adjoint gate's solve tolerance and FD step
are sized so probe stop-noise sits well under the difference signal (mu: h=1e-3 at tol
1e-10, its signal is ~2e-4/unit; rest positions: tol 1e-9 suffices, its signal is
orders larger — and 1e-10 starved the sweep to one survivor on the Linux box).

### The stall pockets: propensity trends with depth, the trigger is a lottery, masking is unsafe

135-point grid (mu in 30..70, indentation 0.01..0.03 with a per-depth feasible start,
kappa in 500..2000), tol=1e-9, max_iterations=100, five compilation contexts
(`examples/characterize_stall_pockets.py`):

| context | overall stall rate | vs indentation (0.01 → 0.03) | vs kappa (500 → 2000) |
|---|---|---|---|
| traced eager | 10.4% (14/135) | 4% → 15% | 4% → 16% |
| traced eager + block-Jacobi | 5.9% (8/135) | 0% → 0% (peak 19% at 0.025) | 9% → 4% |
| traced vmap | 7.4% (10/135) | 4% → 11% | 11% → 9% |
| traced vmap + block-Jacobi | 9.6% (13/135) | 0% → 7% (peak 22% at 0.025) | 9% → 4% |
| host | 5.9% (8/135) | 4% → 7% | 7% → 4% |

Stall-location overlap between contexts: Jaccard 0.00–0.20 (even eager-vs-vmap of the
same driver: 0.14; host vs traced vmap: 0.00).

Reading: stall *probability* rises with indentation in every unpreconditioned
context (3.75x eager, 2.75x vmap, 1.75x host, shallow to deep) and with kappa in the
eager context only (4% -> 16%; the kappa marginal is flat-to-falling elsewhere) —
while stall *location* is nearly uncorrelated across compilations of the same math.
Statistical honesty about the effect sizes: each marginal rests on small event counts
(n=27 per indentation level, n=45 per kappa level, 0–7 stalls per cell), so no single
marginal is individually significant — the finding is the *direction's consistency*
across three independently compiled contexts, not any one ratio. For the caller the
asymmetry settles the masking question anyway: **masking unconverged lanes is
unsafe.** Masking is unbiased only if stall propensity is independent of the
parameters, and the data points the other way in every unpreconditioned context along
the indentation axis — the dropped samples over-represent deep-indentation problems,
so a masked Monte-Carlo estimate of E[logdet I] under-weights the most informative
loads, with no symptom. Per-lane rates of
5–10% put P(at least one stalled lane) at ~99.9% for B=128 — every gradient step of the
intended caller sees the refusal unless it mitigates (see the caller contract in
`static.py`).

A second consequence, found by the benchmark: **convergence certificates do not
transfer between compiled programs.** Re-batching the B=64 draw's 50 both-converged
lanes as a B=50 program flipped 7 of them back into stalls (FINDINGS.md). "Keep the
convergent subset and batch it" is not a strategy.

### Block-Jacobi preconditioning: kills the easy pockets, halves the eager rate, washes out under vmap

Per-vertex 3x3 static-Hessian blocks (incident-tet elastic + exact collider barrier),
PSD-floored, shifted by the operator's own Levenberg damping, rebuilt every pass;
optional on both the Newton CG and the adjoint CG, default off. Gated by a
preconditioned agreement sweep (same equilibrium as the plain host driver on every
both-converged point) and a preconditioned adjoint-vs-FD sweep (rel errors in the same
1e-6..1e-5 band as plain).

Measured: on the fixed-scene pocket prototype (indentation 0.02, kappa 1e3) it is a
clean kill at tol=1e-9 — 5/16 known pocket points stalled plain, 0/16 preconditioned,
all 10–14 iterations. On the full characterization grid it halves the eager stall rate
(10.4% → 5.9%) but is neutral-to-noise under vmap (7.4% → 9.6%), and its surviving
stalls concentrate at the mid-depth indentations (peak 19% eager / 22% vmap at 0.025,
with clean endpoints at 0.01 and 0.03); at tol=1e-11 pockets shuffle rather than
vanish (9/16 → 3/16 at different points). The honest summary: the preconditioner fixes
the conditioning-driven ~1e-9 plateaus in the prototype's regime and the B<=16
benchmark draws, is cheap, and is worth switching on for batched work — and it does
**not** release the caller from reading `converged`, because mid-depth stalls and the
vmap lottery survive it. If those must die too, the
structural fix remains M6 (offset geometry, the large-contact-radius escape from
barrier stiffness): with M3 only halving the tax, M6 is no longer a stretch goal but
the critical path for any caller that cannot tolerate certificate-reading.

### Batch scaling (CPU, macOS dev machine, float64; best-of-3 steady-state, compile excluded)

Fixed-seed mu draw in [30, 70], tol 1e-9, max_iterations 100; "conv" = converged lanes
(traced/host; bj = block-Jacobi traced). Two runs of the same protocol are shown for
the plain columns because laptop run-to-run variance is large — treat ratios, not
absolute times, and treat even ratios as one-significant-figure:

| B | host s/solve (run1 / run2) | plain traced speedup (run1 / run2) | bj speedup | bj conv |
|---|---|---|---|---|
| 1 | 0.030 / 0.061 | 21.7x / 18.7x | 7.0x | 1/1 |
| 4 | 0.015 / 0.082 | 0.40x / 2.07x | **21.4x** | 4/4 |
| 16 | 0.037 / 0.033 | 0.94x / 0.56x | **7.0x** | 16/16 |
| 64 | 0.030 / 0.035 | 0.79x / 0.55x | 0.64x | 53/64 |

Reading: plain traced batching pays only at B=1 (the compiled loop vs ~40 host syncs
per solve on a 54-DOF mesh); at B>=4 the random draw's stalled lanes (5–25% of lanes
across the recorded runs)
run to max_iterations=100 and a batched while_loop pays max-over-lanes, so one stalled
lane drags every healthy one through ~90 masked passes. With block-Jacobi the B=4 and
B=16 draws converge on every lane and batching pays 7–21x; at B=64 that compilation
re-rolled 11 stalled lanes and the drag returned (0.64x). The stall economics, not the
masking overhead, decide whether batching pays: `max_iterations` is a price every
healthy lane pays for the worst one. GPU scaling is untested — linbox01's RTX 4090 was
at 100% utilisation on both attempts; the CPU numbers establish the mechanism (masking
works; drag is max-over-lanes; stall-free batches win) and a GPU would change
constants, not structure.

### Final verification

Local (macOS dev machine, CPU, JAX 0.10.2): full suite **214 passed, 337 subtests, 0
failed**, 56m36s, on the tree at the M4 commit; the subsequent review-round-2 delta is
documentation plus gate-file additions, and the gate file was re-verified green on the
final tree (18 passed, 66 subtests, including two new preconditioner-wiring tests).

Portability (linbox01: x86_64 Linux, JAX 0.9.2, CPU): **every gate this branch wrote
passes on a machine that did not author it** — the swept design's first contact went
15/16 with the one failure being a survivor floor at an over-tight tolerance, fixed by
that gate's own noise budget, then fully green. The full suite there is **211 passed /
3 failed**: all three failures are the *parent branch's* pinned
`StaticAdjointTests` FD gates (mu, lam, body_force), and running the parent commit's
own tree on the same box reproduces them with bit-identical assertion values — the
parent's M4 gates carry the same machine-pinned-fixture disease this branch's rewrite
fixed for its own gates, with silently-stalled FD probes biasing the difference
quotient by ~4e-3. The fix pattern (sweep, filter on certificates, survivor floor,
probe-convergence asserts) is demonstrated in `test_static_traced.py` and left to the
parent branch's owner rather than smuggled into this one.

A second adversarial review (5 lenses, 3 refuters per finding) ran over the M2/M3/M4
additions and confirmed a batch of documentation-accuracy defects — a stale docstring
claiming grid-level stall elimination the tables refute, a kappa trend generalized
beyond its one measured context, missing small-sample caveats, an adjoint-gate table
quoting the retired protocol's numbers — plus one real test gap (nothing would have
caught a silently no-op'd preconditioner switch) and one real contract trap
(`solve_result` certificates come from a different compiled program than the
gradients; the poison itself is the exact certificate). All fixed: the documents now
carry the measured numbers with their uncertainty, and the gate file gained wiring
tests that fail if the preconditioner stops reaching the kernels.

What a reviewer would attack next, ranked: (1) the parent branch's own gates are not
portable and this branch proved it — that is now the suite's weakest point; (2) all
stall statistics rest on one small mesh and small event counts — direction is
credible, effect sizes are not; (3) the GPU story is still untested (the box was
occupied on every attempt), and the batching economics that matter to the Fisher
caller will be decided there; (4) M6 (offset geometry) is the structural escape from
barrier stiffness, and with M3 only halving the stall tax, it is the critical path for
any caller that cannot tolerate certificate-reading.
