"""Batch-scaling benchmark: traced+vmapped static solves vs sequential host solves.

The traced driver's reason to exist is throughput on *batches* of small solves (the
host driver is the right tool for one large mesh -- see solver/static.py). This script
measures the number that justifies or indicts it: wall-clock per solve at B = 1, 4, 16,
64, traced+vmapped against B sequential host solves, on the M4 gate scene (3x3x3
Kuhn slab, clamped base, sphere indenter, barrier active).

Method notes, so the table can be read honestly:

* Batch content is a fixed-seed mu sweep in [30, 70]. Some elements may land in the
  zigzag stall pockets documented in FINDINGS.md and run to max_iterations; the batch
  is identical for both drivers, and convergence counts are reported next to the
  timings so a stall-dominated row is visible rather than silently biasing either side.
* The traced timing is steady-state: the vmapped program is compiled once per batch
  shape (reported separately) and then reused, which is exactly the Fisher-loop usage
  pattern -- many gradient steps over the same batch shape. Host solves have no
  compile step to amortise beyond the per-kernel jit cache, which is warm.
* A batched while_loop runs until its *slowest* element finishes, so the traced
  per-solve number improves with B only while the batch stays iteration-homogeneous.

Run: PYTHONPATH=src:tests python examples/bench_traced_static.py
"""

import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import jax
import jax.numpy as jnp
import numpy as np

from test_static import _TOL, _feasible_initial, _seated_problem
from diff_vbd import (
    StaticParams,
    apply_static_params,
    solve_static_equilibrium,
    solve_static_equilibrium_traced,
)

_MAX_ITERATIONS = 100
_BATCH_SIZES = (1, 4, 16, 64)
_REPEATS = 3  # best-of for the steady-state timings


def main():
    template, positions = _seated_problem()
    initial = _feasible_initial(positions)
    rng = np.random.default_rng(7)
    all_mus = rng.uniform(30.0, 70.0, size=max(_BATCH_SIZES))

    def solve_one_traced(params):
        problem = apply_static_params(template, params)
        return solve_static_equilibrium_traced(
            problem, tol=_TOL, initial_position=initial, max_iterations=_MAX_ITERATIONS
        )

    def solve_one_traced_bj(params):
        problem = apply_static_params(template, params)
        return solve_static_equilibrium_traced(
            problem, tol=_TOL, initial_position=initial,
            max_iterations=_MAX_ITERATIONS, preconditioner="block_jacobi",
        )

    print(f"mesh: {positions.shape[0]} vertices, "
          f"{int(np.sum(np.asarray(positions)[:, 2] > 1e-9)) * 3} free DOF; "
          f"tol={_TOL:g}, max_iterations={_MAX_ITERATIONS}, best of {_REPEATS}")
    print(f"{'B':>4} {'host seq (s)':>14} {'traced vmap (s)':>16} {'host s/solve':>13} "
          f"{'traced s/solve':>15} {'speedup':>8} {'conv':>9} {'compile+1st (s)':>16}")

    for batch_size in _BATCH_SIZES:
        mus = jnp.asarray(all_mus[:batch_size])
        batch = StaticParams(mu=mus)

        # --- host sequential ---
        host_positions = []
        host_converged = 0
        host_best = np.inf
        for _ in range(_REPEATS):
            t0 = time.perf_counter()
            host_positions, host_converged = [], 0
            for b in range(batch_size):
                result = solve_static_equilibrium(
                    apply_static_params(template, StaticParams(mu=mus[b])),
                    tol=_TOL,
                    initial_position=initial,
                    max_iterations=_MAX_ITERATIONS,
                )
                host_positions.append(np.asarray(result.position))
                host_converged += int(result.converged)
            host_best = min(host_best, time.perf_counter() - t0)

        # --- traced, vmapped: compile once per batch shape, then steady-state ---
        solve_batch = jax.jit(jax.vmap(solve_one_traced))
        t0 = time.perf_counter()
        batched = solve_batch(batch)
        batched.position.block_until_ready()
        compile_and_first = time.perf_counter() - t0
        traced_best = np.inf
        for _ in range(_REPEATS):
            t0 = time.perf_counter()
            batched = solve_batch(batch)
            batched.position.block_until_ready()
            traced_best = min(traced_best, time.perf_counter() - t0)

        traced_converged = int(np.sum(np.asarray(batched.converged)))
        worst_rel = max(
            float(
                np.linalg.norm(np.asarray(batched.position[b]) - host_positions[b])
                / np.linalg.norm(host_positions[b])
            )
            for b in range(batch_size)
        )

        # -- traced + block-Jacobi: the M3 variant, same protocol --------------------
        solve_batch_bj = jax.jit(jax.vmap(solve_one_traced_bj))
        bj = solve_batch_bj(batch)
        bj.position.block_until_ready()
        bj_best = np.inf
        for _ in range(_REPEATS):
            t0 = time.perf_counter()
            bj = solve_batch_bj(batch)
            bj.position.block_until_ready()
            bj_best = min(bj_best, time.perf_counter() - t0)
        bj_converged = int(np.sum(np.asarray(bj.converged)))

        print(f"{batch_size:>4} {host_best:>14.3f} {traced_best:>16.3f} "
              f"{host_best / batch_size:>13.4f} {traced_best / batch_size:>15.4f} "
              f"{host_best / traced_best:>7.2f}x {traced_converged:>3}/{host_converged:<3} "
              f"{compile_and_first:>12.1f}   "
              f"bj: {bj_best:>7.3f}s {host_best / bj_best:>5.2f}x {bj_converged:>3}/{batch_size}")
        if worst_rel > 1e-8:
            print(f"     [!] worst traced-vs-host position rel diff {worst_rel:.1e} "
                  f"(expected ~tol/lambda_min for elements stopping at different "
                  f"residuals below tol; see test_static_traced.py)")

    # ---- stall-free batch: isolates the batching economics from the pocket drag ----
    # The random draw above hits zigzag stall pockets (FINDINGS.md): one lane running
    # to max_iterations drags the whole batch through that many masked passes, so the
    # per-solve number above conflates masking overhead with stall drag. This batch is
    # the largest draw's own survivors -- the lanes whose `converged` certificate holds
    # on BOTH drivers -- discovered at runtime, so the measurement is portable rather
    # than pinned to the authoring machine's pocket geography. It is also exactly the
    # sample a masking caller would keep, so this row is the honest ceiling of the
    # mask-and-batch strategy on this hardware.
    solve_batch_full = jax.jit(jax.vmap(solve_one_traced))
    full_batch = StaticParams(mu=jnp.asarray(all_mus))
    full = solve_batch_full(full_batch)
    host_converged_mask = []
    for b in range(len(all_mus)):
        result = solve_static_equilibrium(
            apply_static_params(template, StaticParams(mu=jnp.asarray(all_mus[b]))),
            tol=_TOL,
            initial_position=initial,
            max_iterations=_MAX_ITERATIONS,
        )
        host_converged_mask.append(bool(result.converged))
    survivors = [
        b
        for b in range(len(all_mus))
        if host_converged_mask[b] and bool(full.converged[b])
    ]
    if len(survivors) < 2:
        print(f"\nstall-free batch skipped: only {len(survivors)} survivor lanes")
        return

    survivor_mus = jnp.asarray(all_mus[survivors])
    batch = StaticParams(mu=survivor_mus)
    batch_size = len(survivors)

    host_best = np.inf
    for _ in range(_REPEATS):
        t0 = time.perf_counter()
        for b in range(batch_size):
            solve_static_equilibrium(
                apply_static_params(template, StaticParams(mu=survivor_mus[b])),
                tol=_TOL,
                initial_position=initial,
                max_iterations=_MAX_ITERATIONS,
            )
        host_best = min(host_best, time.perf_counter() - t0)

    solve_batch = jax.jit(jax.vmap(solve_one_traced))
    batched = solve_batch(batch)
    batched.position.block_until_ready()
    traced_best = np.inf
    for _ in range(_REPEATS):
        t0 = time.perf_counter()
        batched = solve_batch(batch)
        batched.position.block_until_ready()
        traced_best = min(traced_best, time.perf_counter() - t0)
    converged = int(np.sum(np.asarray(batched.converged)))
    print(f"\nstall-free survivor batch (B={batch_size} of {len(all_mus)} draw lanes "
          f"converged on both drivers): host {host_best:.3f}s, traced {traced_best:.3f}s, "
          f"speedup {host_best / traced_best:.2f}x, converged {converged}/{batch_size}")


if __name__ == "__main__":
    main()
