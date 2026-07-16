# Findings — feature/infrastructure

Things noticed on the way that are not milestones. Append-only.

- **2026-07-16 · A vertex penetrating a *sphere* collider deadlocks the mesh permanently;
  a plane does not.** `_plane_toi` has an explicit escape clause ("a penetrated vertex can
  climb back out rather than freeze"), but the sphere branch of `_collider_toi` went
  through the bare Lipschitz bound, which returns time-of-impact 0 for any non-positive
  start gap *regardless of direction*. Because the sweep filter takes a global minimum,
  one such vertex froze the entire mesh, forever — the M4 static solver hit this
  immediately when a scene started with the indenter overlapping the slab. Fixed by
  mirroring the plane's escape clause (motion that increases the gap gets toi 1); found by
  the static solver, but the dynamic path had the same hole.

- **2026-07-16 · The test suite was already order-dependent on `main`.**
  `DiffVbdTests::test_apply_runtime_config_disables_gpu_preallocation_by_default` calls
  `apply_runtime_config(platform="gpu")`, whose default `precision="float32"` flips the
  process-global `jax_enable_x64` flag **off** and leaves it off. Every float64 test that
  runs after it silently executes in float32. In pytest's default alphabetical file order
  (`test_diff_vbd` before `test_potential`) this fails three `ElasticPotentialTests`
  gradient/stress checks on unmodified `main`; each passes in isolation, which is exactly
  how it went unnoticed. Fixed on this branch by save/restoring the flag (and the env vars
  the call rewrites) in the test itself. The library behaviour is arguably a sharp edge too
  — `apply_runtime_config` is documented as a process-level entrypoint helper, so the test,
  not the library, was the wrong party.

- **2026-07-16 · Adversarial review of the branch confirmed four defects, all fixed
  before the M3 commit.** (1) *Critical:* rigid-region rows escaped both the per-vertex
  truncation (the rigid projection rewrites them afterwards, dominated by the region's
  unclipped rows) and the prescribed audit (which filtered on the Dirichlet mask only) —
  a rigid indenter driven into a deformable surface could void the intersection-free
  certificate silently. Fixed by excluding all locked rows (Dirichlet | rigid) from
  truncation and auditing both kinds; parity with the old E3 audit, which named rigid
  motion explicitly. (2) Scripted Dirichlet presses raised at speeds `main` handled,
  because the whole prescribed jump landed at the initial guess and was audited against
  one bound. Fixed by applying the prescription in per-iteration increments with
  budget pre-checks and on-demand re-anchoring re-detection (and the pre-check must run
  *before* the initial guess writes the prescribed rows, or the next detection finds an
  intersected state and E1 crashes with a misleading message — found the hard way).
  (3) The bounds composability fuzz did not actually pin `gamma_p < 0.5`: it moved only
  one side of each pair and stayed green with `gamma_p` patched up to 0.8. Fixed with a
  true worst-case pass — all four vertices of each pair spend their full bounds along
  the gap gradient's descent directions, which closes a pair to exactly zero at 0.5 —
  plus a direct pin of the constant. (4) `test_rest_positions_directional` could pass
  vacuously if the finite-difference derivative were zero; a nonzero guard was added.

- **2026-07-16 · A valid-but-different vertex colouring is a behaviour change, not an
  implementation detail.** The full suite caught two contact tests failing after M5
  replaced the serial greedy colouring with Jones–Plassmann (validity-gated, as the brief
  allowed): the colour classes set VBD's Gauss–Seidel sweep order, and under JP a
  frictionless block on a 20-degree incline stopped sliding — it bounced perpetually and
  even crept uphill, while under the greedy colouring (and on `main`) it slides at exactly
  g·sin·cos with zero throttle. Bisected to the M5 commit, confirmed by hot-patching the
  colouring alone. Fix: keep the greedy colouring bit-identical, computed in vectorised
  dependency waves over the lower-indexed-neighbour DAG (2.6x faster, and the equivalence
  test now asserts identity rather than validity). Two dead ends are worth recording:
  the per-vertex truncation was fully ablated with zero effect on the trajectory, and a
  tangential-preserving plane clamp (kept — it is strictly better than scalar
  time-of-impact clipping for planes) also changed nothing here.

- **2026-07-16 · (traced-static) The near-contact static Newton iteration has zigzag
  stall pockets, and which parameters fall in them changes with the compilation, not
  the math.** On the M4 gate scene the host solve at mu=45 accepts 200 straight steps
  and stalls at residual 5.6e-9 while mu=40 and mu=50 converge to 4e-12 in 10
  iterations; the traced driver at mu=45 converges in 12 while *its* pockets sit at
  mu=40 (dz=0) and mu=30 (dz=0.004), where the host is fine. An eager Python replica of
  the traced body's exact decision structure reproduces the host loop decision-for-
  decision, so the drivers are semantically identical — the pocket membership is decided
  by last-ulp differences (CG's threshold stop, Armijo at a knife edge) that differ
  between XLA compilations of the same operations, then get amplified by the barrier's
  overshoot-oscillation mode into converge-vs-stall. Consequences: (1) iteration counts
  are not comparable across drivers (9 vs 11 on the gate scene, positions equal to
  2e-13); (2) any test comparing the two drivers must pin parameters verified to
  converge on *both, under the exact compiled program that runs in the test* — vmap
  batching is a third compilation context with its own pockets; (3) a caller squeezing
  tolerances near 1e-9 on contact problems should expect occasional stalls and use the
  `converged` certificate rather than trusting the iteration budget.

- **2026-07-16 · (traced-static) The residual floor of the gravity-only slab is ~1e-11,
  and a solve whose tolerance sits at its own floor stalls unconverged by definition.**
  At tol=1e-11 the host driver reached 1.0e-11 (barely under) while the traced driver
  landed at 1.1e-11 and then zigzagged for 190 masked iterations without crossing —
  positions agreeing with the host to 7e-14 the whole time. Roundoff in the energy
  gradient sets a per-problem floor; the no-contact agreement fixture therefore solves
  at 1e-10 (a decade of headroom) and any future fixture at a new scale should measure
  its floor first.

- **2026-07-16 · (traced-static) FD gates on implicit gradients inherit the *solve
  tolerance* as noise, not just FD truncation.** The first mu-gate failure at mu=60 was
  neither a wrong adjoint nor a stall: the probe at mu-1e-4 stopped at residual 9.4e-10
  (legitimately under tol=1e-9), and the ~tol-scale position offset of that stop fed a
  2.3e-11 error into a 2.6e-8 central-difference numerator — 9e-4 relative, over the
  1e-4 gate, with both numbers individually fine. The adjoint-gate fixtures now solve
  probes and forwards at 1e-10 so stop noise sits two decades under the FD signal. The
  host M4 gates never saw this because their probe solves happened to stop at 4e-12.

- **2026-07-16 · (traced-static) Adversarial review of the branch confirmed two code
  defects and one test gap, all fixed before commit.** (1) *Major:* the traced body
  encoded the host loop's `if slope >= 0: break` as `slope_ok = slope < 0`, which is not
  its complement when the slope is NaN (a CG breakdown handing back a NaN direction —
  the exact pathology the Levenberg shift exists for). The host runs the Armijo ladder,
  fails every trial, raises the shift and retries — often recovering; the traced body
  classified the same state as terminal stagnation with the shift frozen. Fixed by
  writing every predicate against the host's break conditions (`blocked = slope >= 0.0`)
  rather than their complements; NaN now takes the retry path on both drivers. (2)
  `assert_converged` applied `flatnonzero`'s flat indices to an unraveled residual
  array, so a nested-vmap (rank >= 2) result would report a wrong worst residual or die
  in IndexError before the intended ValueError; fixed by raveling both arrays. (3) No
  gate exercised the rejection/retry/damping transcription — healthy fixtures never
  reject a line search. Fixed by hoisting the loop body to module-level
  `_traced_newton_step` and adding transcription tests that drive it eagerly, pass by
  pass, against a line-for-line host-loop replica, forcing rejections (directions
  scaled by 2^60) and the NaN direction via a wrapped `_newton_direction`; agreement is
  asserted bitwise, per pass. Seven further review findings were refuted by the
  adversarial verification panel (3 refuters per finding).
