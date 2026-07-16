# Decisions — feature/infrastructure

Format: date · decision · rationale. Append-only.

- **2026-07-16 · Verified the OGC Eq. 19 correction before writing code.** The paper prints
  `k'_c = tau * k_c * (tau - r)^2` (energy·length — dimensionally wrong). C1 matching of
  `g1' = -k_c (r - d)` and `g2' = -k'_c / d` at `d = tau` gives `k'_c = tau * k_c * (r - tau)`,
  and only that form reproduces the paper's own C2 result: `g2''(tau) = k'_c / tau^2
  = k_c (r - tau) / tau`, equal to `g1'' = k_c` iff `tau = r/2`, where `k'_c = k_c r^2 / 4`.
  The paper's Eq. 20 for `b` is used as printed.

- **2026-07-16 · Implementation order: M5 before M1, everything else per brief (M1, M2, M4, M3, M6?).**
  The Hertz half-space mesh needs ~80K tets to satisfy the brief's own modelling requirements
  (extent >= 10a, cell ~ a_min/2.5), and the current `build_lumped_masses` dispatches one op per
  tet — setup alone would take minutes per test run. M5 is mechanical and gated on output
  equality, so it goes first. M1 is still the first milestone that touches *physics*, and it
  lands before any contact-model change (M2/M3), which is what "M1 first" is for.

- **2026-07-16 · M2 keeps `use_barrier` working as a legacy spelling.** The enum
  (`contact_activation: barrier | penalty | two_stage`) replaces the bool everywhere
  internally; the bool maps onto it at the `assemble_problem` boundary, and specifying
  both is an error rather than a precedence rule. Existing configs must not change
  behaviour underneath their owners.

- **2026-07-16 · M2's continuity gate asserts what is mathematically true, not what the
  brief's shorthand says.** The two-stage activation is C2 at the stitch `tau` and only
  C1 at `d_hat`: `g''` steps from `k_c` to 0 there, as it must for any quadratic
  activation (the paper's own C2 claim is about `tau`). The test pins the size of that
  step to exactly `k_c` instead of pretending it is zero — hiding it would be tuning a
  gate to pass.

- **2026-07-16 · M4 static solve refuses friction, rigid regions and self-collision.**
  Friction: the lagged Coulomb force is not the gradient of any potential, so a state
  held by friction is not a stationary point of Pi and the adjoint identity is false
  there. Rigid regions: the projection is a constraint outside the energy. Self-collision:
  the pair set is complete only within a detection band sized for one dynamic step, and a
  static Newton step respects no band. Each raises with the reason and the remedy, in
  `_audit_step`'s register. Analytic colliders have no pair set and are exact.

- **2026-07-16 · M4 requires a feasible initial state, like every interior-point method.**
  The barrier is infinite at zero gap; a start that already penetrates has no finite-energy
  path back through it (the linear continuation exists to recover from *rounding*, not
  from initialisation). The gate test builds a feasible initial guess explicitly (column
  compression under the indenter) rather than teaching the solver to un-penetrate, which
  would be a lie about what barrier methods can do.

- **2026-07-16 · M4's collider safeguard clips per vertex, not by a global factor.** The
  first implementation capped the whole Newton direction by the minimum per-vertex
  collider time of impact; one vertex creeping toward the indenter then limited the whole
  mesh to ~1% of its step per iteration and the solve crawled — the sweep filter's
  global-throttling failure mode, reproduced in statics. Per-vertex clipping is sound
  against *static* colliders (no other party moves), and with it plus adaptive Levenberg
  damping (the log barrier's quadratic model overshoots wildly near contact) the gate
  scene converges in 9 Newton iterations to a 4e-12 residual.
- **2026-07-16 · M6 (OGC offset geometry) not attempted.** The brief marks it stretch,
  all-or-nothing, and conditional on M1–M4 being green with budget to spare. The
  remaining budget went to making M1's protocol honest (the first two attempts produced
  frozen-mesh garbage worth understanding rather than papering over) and to verifying
  M3 end to end. A half-implemented contact model is worse than none — Polyhedral Gauss
  Maps, the constructive block definitions, feasible-region checks and the edge-only
  manifold are a large, coupled lift — so per the brief's own rule it was not started.
  What is lost without it is quantified rather than hand-waved: the M2 measurement in
  RESULTS.md shows the two-stage activation only pays off at a large standoff, and the
  M3 caveat (bounds ~ gamma_p * d_hat for resting vertices) is stated in the README.
- **2026-07-16 · M1's load gate is per-indentation, not the fitted intercept.** The
  log-log intercept extrapolates to delta = 1, five decades from the data, so the
  +-0.03 exponent noise of a coarse contact patch swings it ~15% run to run — it failed
  once on pure fit noise while every per-delta load was inside the calibrated band. The
  gate now takes the loads at the measured indentations (geometric-mean ratio within
  25%, each within 35%); the intercept is printed for information.

- **2026-07-16 · M1's load band (25%) is calibrated by a refinement study, not
  tuned to a failing number.** At the committed CPU-budget resolution (fine cell 0.05,
  a_min/h = 1) the measured load sits 12–20% above Hertz; refining to h = 0.035 and
  0.025 moves the band to 11–17% and 10–15% — monotone convergence toward theory, with
  the residual consistent with the bonded base at depth exactly 10·a_max (a known ~5–8%
  stiffening Hertz's half-space does not have). The full three-resolution table is in
  RESULTS.md. The gate is the measured discretization band plus margin; the failure the
  test exists to catch (a wrong contact model) presents as a ~1.3x+ coefficient or a
  wrong exponent, both far outside it. Two earlier driving mechanisms (collider ramp,
  velocity-zeroed relaxation) produced frozen-mesh garbage and non-converging far
  fields respectively; both are documented in the test docstring, and the final
  protocol is static-settle plus dynamic certification.

## feature/traced-static

- **2026-07-16 · The traced driver is a sibling, not a flag; the host path is untouched
  byte-for-byte.** `solve_static_equilibrium_traced` and `StaticAdjointTraced` sit next
  to their host twins rather than behind a `traced=True` switch, because the two differ
  in failure contract, not just execution: the host adjoint raises on an unconverged
  state, the traced one poisons the sample's gradient with NaN. A flag that silently
  swaps "raises on bad input" for "returns NaN on bad input" is a behaviour change that
  should be visible at the call site. The backward-pass math is transcribed rather than
  refactored into a shared helper so the host path's bytes (and tests) cannot move.

- **2026-07-16 · The traced line search is a nested `lax.while_loop`, not vbd's alpha
  ladder.** `vbd._apply_line_search` evaluates every rung and takes `argmin`; the static
  solver takes the *first Armijo-acceptable* step of a halving ladder. Those are
  different acceptance rules — argmin can pick a smaller step than first-acceptable, or
  a rung first-acceptable never visits — and swapping one for the other is a solver
  behaviour change, not a refactor. The nested while_loop preserves the host rule
  exactly (1–5 objective evaluations typical) at the cost of batched rungs running to
  the slowest element; the agreement gate (traced == host to 1e-10) is what enforces
  this choice stays honest.

- **2026-07-16 · The traced carry gains a `stagnated` flag beyond the brief's six
  fields, and the pass cap is `13 * max_iterations + 13`.** The host loop's two `break`s
  (no descent direction after both clip fallbacks; a rejection with the Levenberg shift
  already at its 1e6 ceiling) cannot stop a `lax.while_loop` from inside the body — the
  predicate sees only carried state, so "stop this element" must be data. The pass cap
  bounds the retry ladder: a rejection raises the shift 0 → 1e-4 → … → 1e6 in at most
  11 steps and the next rejection sets `stagnated`, so 13 passes per accepted Newton
  step plus one final ladder bounds every trajectory the host loop could take; the cap
  exists so a pathological element cannot spin a batch forever, and it can never bind
  first on a healthy solve.

- **2026-07-16 · Unconverged traced gradients are NaN-poisoned, not checkify'd.** The
  backward pass multiplies the sample's cotangent by `where(converged, 1, NaN)`. NaN
  propagates through every downstream reduction, cannot be silently consumed, stays
  confined to its own batch element, and needs no cooperation from the caller's jit
  boundary; `checkify` would give a real exception but demands the caller thread and
  check an error value, and a forgotten check is exactly the silent-wrong-gradient
  failure this refusal exists to prevent. `assert_converged` restores the host
  adjoint's loud error at any batch boundary for callers who want it.

- **2026-07-16 · Agreement/batching fixtures solve to 1e-11 so the 1e-10 gate is
  honest; fixture parameters are pinned to verified-convergent values.** Both drivers
  stop at the first iterate below `tol`, so two correct solves can disagree by
  ~`tol/lambda_min` — measured 1.5e-10 at tol=1e-9 on the gate scene, *outside* the
  1e-10 gate with both answers right. Solving the comparison fixtures to 1e-11 keeps
  the gate at 1e-10 (tightened solve, untouched gate — the opposite of loosening).
  Pinning matters because the near-contact Newton iteration has zigzag stall pockets
  (FINDINGS.md) and *which* parameters stall differs between compilations of the same
  math; every comparison fixture asserts convergence first, so rot fails loudly as a
  fixture problem rather than silently as a gradient error.

- **2026-07-16 · `warm_start_steps` is absent from the traced path.** It runs
  `vbd.step`, which is host-side (concrete `float(state.time)` reads, host audits) and
  is documented as forward-only relaxation. Tracing it would mean tracing the dynamic
  stepper — out of scope and of dubious value, since the traced path's caller supplies
  per-batch `initial_position` anyway. The traced docstring says so and points the
  caller at `initial_position`.

- **2026-07-16 · Pinned gate fixtures replaced by swept both-converged property tests.**
  The first design pinned parameters "verified to converge under the exact compiled
  program the test runs" — which, it turned out, pins them to the machine that ran the
  verification: on x86_64 Linux (same JAX 0.10.2, CPU) 11 of 23 gates failed, every one
  a convergence *precondition* on the unchanged host driver. The loud-fixture design
  worked; the pinning conclusion was wrong. The gates now sweep a 27-point (mu, collider
  height) grid, filter on the `converged` certificates both drivers already return,
  assert the gate property on every survivor, and demand `survivors >= 8`. That is
  portable by construction and a *stronger* claim than the pinned version — equivalence
  holds everywhere it is defined — and it hides nothing: a platform where most of the
  grid stops converging fails loudly about the solver, which is what we want to hear.
  Same pattern for the adjoint gates (survivor = finite vmapped gradient + both FD
  probes converged, minimum survivors per parameter class). No tolerance moved.

- **2026-07-16 · The refusal test's asymmetry is constructed, not harvested.** It
  previously relied on the easy lane converging to 1e-9 within 25 iterations — pocket
  geography again. The refusal adjoint now runs at tol=1e-6, which sits in the globally
  convergent descent phase (~8 iterations), three-plus orders above where the zigzag
  pockets live, so the easy lane converges on any machine or the machine has a real
  solver problem; the unconverged lane keeps the 5mm-lowered sphere whose penetrating
  start zigzags at a residual of order one — six orders above tolerance, a physical
  plateau no compilation flips. The refusal semantics do not depend on the tolerance's
  value, only on the certificate, so nothing is weakened.

- **2026-07-16 · Transcription coverage that depends on the energy landscape is found
  at runtime, never pinned.** The timid-accept branch (accepted alpha < 1/2 must raise
  the shift) is now forced by scanning `-c * gradient` scales at runtime for the first
  c whose Armijo-accepted alpha lands below 1/2 on the executing machine's actual
  objective, and handing exactly that direction to both drivers. The natural zigzag
  trajectory is kept as an equality-only trace with no coverage promise. Rejections and
  the NaN direction were already forced deterministically.

- **2026-07-16 · Block-Jacobi preconditioning is built from per-vertex static Hessian
  blocks, damped like the operator, PSD-floored, and defaults off.** The stall pockets
  are barrier-conditioning zigzags, and against an analytic collider the barrier's
  Hessian is exactly per-vertex, so the natural preconditioner is the per-vertex 3x3
  diagonal block: incident-tet elastic curvature (VBD's own local-objective pattern,
  minus inertia — no timestep in Pi — minus friction and pairs, both refused in
  statics) plus the exact collider barrier block. `clamped_hessian` floors the
  eigenvalues first (an indefinite block makes a "preconditioner" that is not one),
  the blocks are shifted by the same Levenberg damping as the CG operator before
  inversion (a preconditioner for H under a solve of H + damping·I disagrees with its
  system exactly when the shift is large — exactly when the solve is struggling), and
  they are rebuilt at every Newton pass because the barrier curvature moves fastest
  where the preconditioner matters most. Default off because a preconditioner changes
  CG's iterates: equivalence with the host driver holds at the equilibrium (gated by
  the preconditioned agreement sweep), not decision-for-decision along the path.
  Prototype measurement before wiring: at the caller's tol=1e-9, 5/16 pocket points
  stalled plain and 0/16 preconditioned (10–14 iterations each); at 1e-11 the pockets
  shuffle rather than vanish (9/16 → 3/16, at different points) — the fix addresses
  the conditioning-driven ~1e-9 plateaus, not the near-floor terminal zigzag. The
  kernels gained an optional `preconditioner=None` argument; the host driver's calls
  are unchanged and its behaviour is bit-identical.
