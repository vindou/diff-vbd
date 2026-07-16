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
