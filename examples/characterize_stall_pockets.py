"""Are the zigzag stall pockets parameter-correlated, or a last-ulp lottery?

The question decides whether a Monte-Carlo caller may *mask* stalled lanes
(``g = sum(where(conv, grad, 0)) / sum(conv)``). If stalls are missing-at-random, the
masked estimate stays unbiased; if stiffer problems stall more often, masking silently
drops the hard end of the distribution and the caller optimises over the wrong
``p(phi)`` with no symptom. Both can be true at once — a systematic *propensity*
(conditioning) triggered by a last-ulp *coin flip* — and that combination is still
biased, so the measurement has to separate rate-vs-parameter from location-vs-context.

Method: a grid over the three knobs that plausibly control conditioning — ``mu``
(material stiffness), indentation depth (normal load, via the collider height, with a
*per-depth* feasible start so every solve is legitimate), and ``kappa`` (barrier
stiffness) — solved at the caller's tolerance in five compilation contexts: the
traced driver eager and under per-kappa ``vmap`` slices, each with and without the
block-Jacobi preconditioner, plus the host driver.
Reported per axis: stall rate marginals (the propensity question); across contexts:
the overlap of stall *locations* (the lottery question).

Run: PYTHONPATH=src:tests python examples/characterize_stall_pockets.py
"""

import os
import sys
from itertools import product

os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import jax
import jax.numpy as jnp
import numpy as np

from test_static import _CENTER, _D_HAT, _R, _TOP, _seated_problem
from diff_vbd import (
    StaticParams,
    apply_static_params,
    solve_static_equilibrium,
    solve_static_equilibrium_traced,
)

_TOL = 1.0e-9  # the caller's tolerance, not the gates' tightened one
_MAX_ITERATIONS = 100

_MUS = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
_INDENTATIONS = [0.010, 0.015, 0.020, 0.025, 0.030]  # rest overlap of the sphere
_KAPPAS = [5.0e2, 1.0e3, 2.0e3]


def _feasible_initial_for(positions, center):
    """The gate scene's column-compression start, generalised to a moved sphere."""
    rest = np.asarray(positions)
    init = rest.copy()
    for cx, cy in {(x, y) for x, y in rest[:, :2]}:
        r2 = (cx - center[0]) ** 2 + (cy - center[1]) ** 2
        if r2 < _R * _R:
            surface = center[2] - np.sqrt(_R * _R - r2) - _D_HAT / 2
            cap = min(_TOP, surface)
            column = (rest[:, 0] == cx) & (rest[:, 1] == cy)
            init[column, 2] = rest[column, 2] * (cap / _TOP)
    return jnp.asarray(init)


def main():
    grid = list(product(_MUS, _INDENTATIONS, _KAPPAS))
    print(f"{len(grid)} grid points, tol={_TOL:g}, max_iterations={_MAX_ITERATIONS}")

    problems, initials = [], []
    for mu, indentation, kappa in grid:
        template, positions = _seated_problem(contact_kappa=kappa)
        center = np.array([_CENTER[0], _CENTER[1], _TOP + _R - indentation])
        problems.append(
            apply_static_params(
                template,
                StaticParams(
                    mu=jnp.asarray(mu), collider_center=jnp.asarray(center[None, :])
                ),
            )
        )
        initials.append(_feasible_initial_for(positions, center))

    outcomes = {}

    # -- contexts 1 & 4: traced eager, plain and block-Jacobi ---------------------------
    for precond in ("none", "block_jacobi"):
        conv = []
        for problem, initial in zip(problems, initials):
            result = solve_static_equilibrium_traced(
                problem, tol=_TOL, initial_position=initial,
                max_iterations=_MAX_ITERATIONS, preconditioner=precond,
            )
            conv.append(bool(result.converged))
        label = "traced eager" if precond == "none" else "traced eager + bj"
        outcomes[label] = np.array(conv)

    # -- contexts 2 & 5: traced, one vmap over the kappa-slices -------------------------
    # kappa lives outside StaticParams (it is not a differentiable input), so the vmap
    # batches (mu, center, initial) per kappa slice -- still one compiled program per
    # slice, which is the caller's shape.
    for precond in ("none", "block_jacobi"):
        conv = np.zeros(len(grid), dtype=bool)
        for kappa in _KAPPAS:
            rows = [i for i, (_, _, k) in enumerate(grid) if k == kappa]
            template, _ = _seated_problem(contact_kappa=kappa)

            def solve_one(params, initial):
                problem = apply_static_params(template, params)
                return solve_static_equilibrium_traced(
                    problem, tol=_TOL, initial_position=initial,
                    max_iterations=_MAX_ITERATIONS, preconditioner=precond,
                )

            batch = StaticParams(
                mu=jnp.asarray([grid[i][0] for i in rows]),
                collider_center=jnp.asarray(
                    np.stack(
                        [
                            np.array(
                                [[_CENTER[0], _CENTER[1], _TOP + _R - grid[i][1]]]
                            )
                            for i in rows
                        ]
                    )
                ),
            )
            stacked_initials = jnp.stack([initials[i] for i in rows])
            batched = jax.vmap(solve_one)(batch, stacked_initials)
            conv[rows] = np.asarray(batched.converged)
        label = "traced vmap" if precond == "none" else "traced vmap + bj"
        outcomes[label] = conv

    # -- context 3: host ----------------------------------------------------------------
    conv = []
    for problem, initial in zip(problems, initials):
        result = solve_static_equilibrium(
            problem, tol=_TOL, initial_position=initial, max_iterations=_MAX_ITERATIONS
        )
        conv.append(bool(result.converged))
    outcomes["host"] = np.array(conv)

    # -- report ---------------------------------------------------------------------------
    axes = {
        "mu": (_MUS, [g[0] for g in grid]),
        "indentation": (_INDENTATIONS, [g[1] for g in grid]),
        "kappa": (_KAPPAS, [g[2] for g in grid]),
    }
    for context, converged in outcomes.items():
        stalled = ~converged
        print(f"\n[{context}] overall stall rate: {stalled.mean():.1%} "
              f"({stalled.sum()}/{len(grid)})")
        for axis, (levels, values) in axes.items():
            values = np.array(values)
            rates = [f"{stalled[values == level].mean():.0%}" for level in levels]
            print(f"  stall rate vs {axis:12s}: "
                  + "  ".join(f"{level:g}:{rate}" for level, rate in zip(levels, rates)))

    # location overlap across contexts: the lottery question
    names = list(outcomes)
    print("\nstall-location overlap (Jaccard) between contexts:")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            sa, sb = ~outcomes[names[a]], ~outcomes[names[b]]
            union = np.logical_or(sa, sb).sum()
            jaccard = np.logical_and(sa, sb).sum() / union if union else float("nan")
            print(f"  {names[a]} vs {names[b]}: {jaccard:.2f} "
                  f"({np.logical_and(sa, sb).sum()} shared of {union} union)")


if __name__ == "__main__":
    main()
