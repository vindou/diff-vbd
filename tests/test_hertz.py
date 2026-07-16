"""M1: the forward contact model against Hertz's closed-form solution.

A rigid sphere (radius R) indenting an elastic half-space carries, at indentation
``delta`` and with ``E* = E / (1 - nu^2)``:

    P = (4/3) E* sqrt(R) delta^(3/2)

Fitting ``log P`` against ``log delta`` pins two independent things at once: the
**exponent** (3/2 — the contact model), and the **coefficient** (that stable Neo-Hookean
really reduces to linear elasticity at small strain, as the alpha remap claims and
nothing else in this suite verifies). Nothing checked the forward contact model against
anything closed-form before this file.

Driving mechanism (the brief left the choice open): the sphere collider is **held fixed
at each indentation**; the slab starts from a feasible, column-compressed state that
clears the sphere by ``d_hat / 2`` everywhere, is settled by the static Newton-CG solver
(M4), and the measurement state is then **certified by the dynamic solver itself**:
velocity-zeroed dynamic steps followed by one free step, with kinetic energy and the
contact-load-vs-base-reaction mismatch asserted small. Why not a ramp: the barrier is an
interior-point method, and with ``d_hat <= delta_min / 100`` (required below, so the
standoff biases delta by under 1%) a ramped approach must either creep at ~``d_hat`` per
step (hundreds of steps per indentation) or carry the surface through the whole
activation band in one step, landing it penetrated where the barrier's linear
continuation produces astronomical gradients and the line search rightly refuses every
positive step — a frozen mesh measuring garbage. Why not settle dynamically: the deep
stress field converges at the slab's softest-mode rate (~0.5% per step), needing many
hundreds of steps per indentation. The contact is frictionless, so the equilibrium is
path-independent and any settler that reaches it measures the same physics; the
certification asserts are what make the dynamic solver, not the static one, the thing
being validated — if the static warm start delivered a wrong state, the dynamic steps
would drift and the load/reaction check would fail.

Modelling requirements, asserted rather than assumed:
- half-space validity: lateral half-extent and depth >= 10 * a_max (asserted);
- small strain: a_max / R <= 0.1 (asserted);
- frictionless: contact_friction_mu = 0;
- barrier standoff: d_hat <= delta_min / 100 (asserted), with the sensitivity to d_hat
  measured by re-running the smallest indentation at 2 * d_hat and reported;
- this is a dynamic solver: kinetic energy is asserted negligible against the elastic
  energy after a free (non-zeroed) step, or the measured load means nothing;
- float64 throughout (conftest).

Tolerances: exponent within 0.075 of 1.5 (the brief's "a few percent"); the measured
loads within 25% of theory as a geometric mean over the four indentations (each also
within 35% individually). The coefficient is checked **at the measured indentations,
never as the fitted intercept**: the intercept extrapolates to delta = 1, five
log-decades from the data, so the +-0.03 exponent noise the coarse contact patch
produces swings it by ~15% — a gate on it flakes on fit noise while the physics stands
still (observed directly: the finest-mesh run has the best per-delta ratios and the
worst fitted intercept). The 25% band is calibrated by a mesh-refinement study
(RESULTS.md), not tuned to pass: at fine cells h = 0.05 / 0.035 / 0.025 the per-delta
loads sit 13-20% / 11-17% / 10-15% above theory, converging downward — linear-tet
resolution of the contact patch (a_min / h = 1 at the committed CPU-budget resolution),
plus the bonded base at depth exactly 10 * a_max (a known ~5-8% stiffening Hertz's
half-space does not have), plus the d_hat standoff (~1%). A real contact-model error
presents as a ~1.3x+ load ratio or a wrong exponent, both far outside the band.
**A mismatch beyond these is a finding about the contact model, not a test bug.**
Measured values are printed and recorded in RESULTS.md.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from diff_vbd import (
    assemble_problem,
    elastic_potential,
    initial_state,
    solve_static_equilibrium,
)
from diff_vbd.solver import vbd
from diff_vbd.solver.contact.colliders import sphere_signed_distance
from diff_vbd.solver.contact.potential import collider_contact_energy

_R = 1.0
_DELTAS = (2.5e-3, 4.0e-3, 6.3e-3, 1.0e-2)
_D_HAT = min(_DELTAS) / 100.0
_MU, _LAM = 1000.0, 1500.0  # nu = 0.3
_DENSITY = 500.0  # keeps E dt^2 / (rho h^2) ~ 1, so sweeps actually converge
_DT = 0.02
_H_FINE = 0.05  # a_min / h ~ 1: the CPU-budget resolution; see module docstring
_GRADE_RATIO = 1.5


def _graded_half_axis(fine_half, h_fine, extent, ratio):
    xs = [0.0]
    while xs[-1] < fine_half - 1e-12:
        xs.append(xs[-1] + h_fine)
    h = h_fine
    while xs[-1] < extent - 1e-12:
        h *= ratio
        xs.append(xs[-1] + h)
    xs[-1] = extent
    return np.array(xs)


def _graded_slab(fine_half, h_fine, half_extent, depth, ratio):
    """Kuhn-split slab, fine near the contact zone, geometrically graded outward.

    The Kuhn 6-tet split conforms on any structured hexahedral grid, uniform or not, so
    grading costs nothing in mesh validity — and it is what makes a >= 10 * a_max
    half-space affordable: the resolution requirement is set by the contact radius, the
    extent requirement by the far field, and they differ by an order of magnitude.
    """
    half = _graded_half_axis(fine_half, h_fine, half_extent, ratio)
    lateral = np.concatenate([-half[1:][::-1], half])
    zs = -_graded_half_axis(fine_half, h_fine, depth, ratio)[::-1]
    nx, ny, nz = len(lateral), len(lateral), len(zs)
    X, Y, Z = np.meshgrid(lateral, lateral, zs, indexing="ij")
    positions = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    index = lambda i, j, k: (i * ny + j) * nz + k
    tets = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c = [
                    index(i + a, j + b, k + d)
                    for a in (0, 1)
                    for b in (0, 1)
                    for d in (0, 1)
                ]
                v000, v001, v010, v011, v100, v101, v110, v111 = c
                tets += [
                    [v000, v100, v110, v111],
                    [v000, v101, v100, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v001, v101, v111],
                    [v000, v011, v001, v111],
                ]
    return (
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.asarray(np.array(tets), dtype=jnp.int32),
    )


def _feasible_initial(positions, center_z, d_hat):
    """Column-compress the slab so every vertex clears the sphere by d_hat / 2."""
    rest = np.asarray(positions)
    init = rest.copy()
    r2 = rest[:, 0] ** 2 + rest[:, 1] ** 2
    inside = r2 < _R * _R
    surface = center_z - np.sqrt(np.maximum(_R * _R - r2, 0.0)) - 0.5 * d_hat
    cap = np.minimum(0.0, surface)
    bottom = rest[:, 2].min()
    scale = np.where(inside, (cap - bottom) / (0.0 - bottom), 1.0)
    init[:, 2] = bottom + (rest[:, 2] - bottom) * scale
    return jnp.asarray(init)


@jax.jit
def _contact_load(problem, positions):
    """Magnitude of the total vertical force the sphere exerts on the slab."""

    def energy(x):
        return jnp.sum(
            jax.vmap(
                lambda p: collider_contact_energy(
                    problem.contact.params, problem.contact.colliders, p
                )
            )(x)
        )

    return jnp.abs(jnp.sum(-jax.grad(energy)(positions)[:, 2]))


@jax.jit
def _base_reaction(problem, positions):
    """Vertical elastic reaction at the clamped base: equals the load at equilibrium."""
    gradient = jax.grad(
        lambda x: elastic_potential(
            problem.material, problem.mesh.tets, problem.mesh.rest_positions, x
        )
    )(positions)
    fixed = problem.boundary_conditions.dirichlet_mask
    return jnp.abs(jnp.sum(jnp.where(fixed[:, None], gradient, 0.0)[:, 2]))


@jax.jit
def _min_gap(problem, positions):
    colliders = problem.contact.colliders
    return jnp.min(
        jax.vmap(
            lambda p: sphere_signed_distance(
                p, colliders.center[0], colliders.radius[0], colliders.outside[0]
            )
        )(positions)
    )


class HertzContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a_max = np.sqrt(_R * max(_DELTAS))
        cls.half_extent = 10.0 * a_max
        cls.depth = 10.0 * a_max
        cls.positions, cls.tets = _graded_slab(
            1.5 * a_max, _H_FINE, cls.half_extent, cls.depth, _GRADE_RATIO
        )
        base = np.asarray(cls.positions)[:, 2] < (-cls.depth + 1e-9)
        cls.free_mask = jnp.asarray((~base).astype(np.float64))

        E = _MU * (3 * _LAM + 2 * _MU) / (_LAM + _MU)
        nu = _LAM / (2 * (_LAM + _MU))
        cls.e_star = E / (1 - nu**2)
        cls.coefficient_theory = (4.0 / 3.0) * cls.e_star * np.sqrt(_R)

    def _problem(self, delta, d_hat, activation):
        return assemble_problem(
            self.positions,
            self.tets,
            self.free_mask,
            dt=_DT,
            external_acceleration=(0.0, 0.0, 0.0),
            mu=_MU,
            lam=_LAM,
            density=_DENSITY,
            eps=1.0e-10,
            num_iterations=10,
            line_search_enabled=True,
            acceleration_enabled=True,
            chebyshev_rho=0.95,
            colliders=[
                {
                    "kind": "sphere",
                    "center": (0.0, 0.0, _R - delta),
                    "radius": _R,
                    "outside": True,
                }
            ],
            contact_d_hat=d_hat,
            contact_kappa=1.0e3,
            contact_friction_mu=0.0,
            contact_activation=activation,
        )

    def _measure(self, delta, d_hat, activation):
        """Settle to the seated equilibrium at one indentation and measure the load."""
        problem = self._problem(delta, d_hat, activation)
        seated = solve_static_equilibrium(
            problem,
            tol=1.0e-6,
            initial_position=_feasible_initial(self.positions, _R - delta, d_hat),
            max_iterations=200,
            cg_max_iterations=300,
        )
        state = initial_state(problem, position=seated.position)
        # Polish with velocity-zeroed dynamic steps until the contact load and the base
        # reaction agree — the effort adapts to the certification criterion (the log
        # stage of the two-stage activation seats ~4x deeper than the barrier at equal
        # kappa and needs more of it), but the criterion itself never loosens: if the
        # cap is hit unconverged, the assertion below fails and says so.
        for _ in range(12):
            for _ in range(5):
                state = vbd.step(
                    problem,
                    dataclasses.replace(
                        state, velocity=jnp.zeros_like(state.velocity)
                    ),
                )
            load = float(_contact_load(problem, state.position))
            reaction = float(_base_reaction(problem, state.position))
            if load > 0 and abs(load - reaction) < 0.04 * load:
                break
        # One free (non-zeroed) step: the kinetic-energy assertion below has to hold on
        # the solver's own dynamics, not because the harness deleted the velocity.
        state = vbd.step(problem, state)

        kinetic = float(
            0.5
            * jnp.sum(problem.topology.mass * jnp.sum(state.velocity**2, axis=-1))
        )
        elastic = float(
            elastic_potential(
                problem.material,
                problem.mesh.tets,
                problem.mesh.rest_positions,
                state.position,
            )
        )
        load = float(_contact_load(problem, state.position))
        reaction = float(_base_reaction(problem, state.position))
        gap = float(_min_gap(problem, state.position))
        return dict(
            load=load,
            reaction=reaction,
            gap=gap,
            kinetic=kinetic,
            elastic=elastic,
        )

    def test_load_displacement_curve_matches_hertz(self):
        a_max = np.sqrt(_R * max(_DELTAS))
        # Modelling requirements first: a fit on an invalid model would be worse than
        # no test, because it would get tuned until it passed.
        self.assertGreaterEqual(self.half_extent, 10.0 * a_max)
        self.assertGreaterEqual(self.depth, 10.0 * a_max)
        self.assertLessEqual(a_max / _R, 0.1)
        self.assertLessEqual(_D_HAT, min(_DELTAS) / 100.0)

        for activation in ("barrier", "two_stage"):
            with self.subTest(activation=activation):
                loads = []
                for delta in _DELTAS:
                    result = self._measure(delta, _D_HAT, activation)
                    # Quasi-static: the measurement means nothing while the mesh rings.
                    self.assertLess(
                        result["kinetic"], 1.0e-4 * result["elastic"],
                        msg=f"KE/PE at delta={delta}",
                    )
                    # Seated, not penetrated and not separated.
                    self.assertGreater(result["gap"], 0.0)
                    self.assertLess(result["gap"], _D_HAT)
                    # Settled: contact load and base reaction agree at equilibrium.
                    self.assertLess(
                        abs(result["load"] - result["reaction"]),
                        0.05 * result["load"],
                        msg=f"equilibrium residual at delta={delta}",
                    )
                    loads.append(result["load"])

                exponent, log_coefficient = np.polyfit(
                    np.log(np.asarray(_DELTAS)), np.log(np.asarray(loads)), 1
                )
                theory = self.coefficient_theory * np.asarray(_DELTAS) ** 1.5
                ratios = np.asarray(loads) / theory
                mean_ratio = float(np.exp(np.mean(np.log(ratios))))
                print(
                    f"\nHertz [{activation}]: exponent={exponent:.4f} (theory 1.5), "
                    f"per-delta P/P_theory={np.round(ratios, 3).tolist()}, "
                    f"geometric mean {mean_ratio:.3f} "
                    f"(fitted intercept {float(np.exp(log_coefficient)):.1f} vs "
                    f"{self.coefficient_theory:.1f}, reported not gated)"
                )
                self.assertLess(abs(exponent - 1.5), 0.075)
                self.assertLess(abs(np.log(mean_ratio)), np.log(1.25))
                self.assertLess(float(np.max(np.abs(np.log(ratios)))), np.log(1.35))

    def test_load_is_insensitive_to_the_barrier_standoff(self):
        """Contact force begins at gap d_hat, so contact effectively starts ~d_hat early
        and delta is biased by up to d_hat. With d_hat = delta_min/100 that is a <=1%
        bias on delta and ~1.5% on P; doubling d_hat doubles it. The doubled-d_hat load
        is required to sit within 5% of the reference — sensitivity beyond that would
        mean the barrier, not the elasticity, is setting the measured stiffness."""
        delta = min(_DELTAS)
        reference = self._measure(delta, _D_HAT, "barrier")["load"]
        doubled = self._measure(delta, 2.0 * _D_HAT, "barrier")["load"]
        print(
            f"\nHertz d_hat sensitivity at delta={delta}: P({_D_HAT:g})={reference:.4f}, "
            f"P({2 * _D_HAT:g})={doubled:.4f}, relative change "
            f"{abs(doubled - reference) / reference:.4f}"
        )
        self.assertLess(abs(doubled - reference), 0.05 * reference)


if __name__ == "__main__":
    unittest.main()
