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
