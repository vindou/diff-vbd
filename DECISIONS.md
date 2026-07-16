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
