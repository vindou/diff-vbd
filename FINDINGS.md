# Findings — feature/infrastructure

Things noticed on the way that are not milestones. Append-only.

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
