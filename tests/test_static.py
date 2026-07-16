"""Static equilibrium and implicit-adjoint gates (M4).

The one non-negotiable idea under test: **implicit differentiation is only true at a
stationary point.** ``du*/dp = -H^{-1} dg/dp`` is the derivative of the optimality
condition ``g = grad_u Pi = 0``; evaluated anywhere else it returns a plausible-looking
gradient of nothing, and no forward test can catch it. So the gates here are exactly the
brief's: adjoint-vs-central-differences at a *seated* configuration with the barrier
demonstrably active (``0 < min gap < d_hat`` is asserted, because at a contact-free state
the collider sensitivities are identically zero and the check would compare noise against
noise and certify nothing), and a hard *raise* when the adjoint is requested on a state
whose residual exceeds tolerance.

The scene is a soft slab with its base clamped and a rigid sphere held at a fixed
indentation -- the static twin of the Hertz test's setup. Barrier solvers need a feasible
(non-penetrating) start, so the initial guess pre-compresses each vertex column to clear
the sphere; starting from the rest positions (which the sphere overlaps) is exactly the
infeasible-start case interior-point methods refuse.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from diff_vbd import (
    StaticAdjoint,
    StaticParams,
    assemble_problem,
    body_force_potential,
    solve_static_equilibrium,
    static_potential,
)
from diff_vbd.solver.contact.colliders import sphere_signed_distance

_R = 1.0
_DELTA = 0.02  # sphere indentation into the rest slab
_TOP = 0.5
_D_HAT = 1.0e-3
_CENTER = np.array([0.25, 0.25, _TOP + _R - _DELTA])
_TOL = 1.0e-9


def _kuhn_block(n=3, spacing=0.25):
    grid = np.stack(
        np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1
    ).reshape(-1, 3).astype(np.float64) * spacing
    index = lambda i, j, k: (i * n + j) * n + k
    tets = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
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
        jnp.asarray(grid, dtype=jnp.float64),
        jnp.asarray(np.array(tets), dtype=jnp.int32),
    )


def _seated_problem(**kwargs):
    """Clamped-base slab with a rigid sphere held at indentation ``_DELTA``."""
    positions, tets = _kuhn_block()
    free = (np.asarray(positions)[:, 2] > 1e-9).astype(np.float64)
    defaults = dict(
        dt=0.02,
        external_acceleration=(0.0, 0.0, -9.81),
        mu=50.0,
        lam=75.0,
        density=1.0,
        num_iterations=20,
        colliders=[
            {
                "kind": "sphere",
                "center": tuple(_CENTER),
                "radius": _R,
                "outside": True,
            }
        ],
        contact_d_hat=_D_HAT,
        contact_kappa=1.0e3,
    )
    defaults.update(kwargs)
    problem = assemble_problem(positions, tets, jnp.asarray(free), **defaults)
    return problem, positions


def _feasible_initial(positions):
    """Compress each vertex column so its top clears the sphere by d_hat / 2.

    The sphere overlaps the *rest* slab by ``_DELTA``, so the rest positions are an
    infeasible start -- the one thing a barrier method cannot recover from gracefully.
    """
    rest = np.asarray(positions)
    init = rest.copy()
    for cx, cy in {(x, y) for x, y in rest[:, :2]}:
        r2 = (cx - _CENTER[0]) ** 2 + (cy - _CENTER[1]) ** 2
        if r2 < _R * _R:
            surface = _CENTER[2] - np.sqrt(_R * _R - r2) - _D_HAT / 2
            cap = min(_TOP, surface)
            column = (rest[:, 0] == cx) & (rest[:, 1] == cy)
            init[column, 2] = rest[column, 2] * (cap / _TOP)
    return jnp.asarray(init)


def _min_gap(problem, positions):
    colliders = problem.contact.colliders
    gaps = jax.vmap(
        lambda q: sphere_signed_distance(
            q, colliders.center[0], colliders.radius[0], colliders.outside[0]
        )
    )(positions)
    return float(jnp.min(gaps))


class BodyForcePotentialTests(unittest.TestCase):
    def test_negative_gradient_is_mass_times_acceleration(self):
        """The defining property: -dU/dx == m a, exactly, per vertex and axis."""
        problem, positions = _seated_problem()
        force = -jax.grad(
            lambda x: body_force_potential(
                problem.topology.mass, problem.solver.external_acceleration, x
            )
        )(positions)
        expected = (
            np.asarray(problem.topology.mass)[:, None]
            * np.asarray(problem.solver.external_acceleration)
        )
        np.testing.assert_allclose(np.asarray(force), expected, rtol=1e-12)

    def test_static_potential_includes_the_body_force(self):
        """Pi = potential_energy + body force: at the rest state (no contact active for
        an interior vertex) the gradient must be exactly the negative gravity force on
        free interior vertices -- if this drifts, the static problem being solved is not
        the one advertised."""
        problem, positions = _seated_problem(colliders=None, contact_enabled=False)
        gradient = jax.grad(lambda x: static_potential(problem, x))(positions)
        interior = 13  # centre vertex of the 3x3x3 grid
        expected = -np.asarray(problem.topology.mass)[interior] * np.array(
            [0.0, 0.0, -9.81]
        )
        np.testing.assert_allclose(
            np.asarray(gradient)[interior], expected, rtol=1e-10
        )


class StaticSolveTests(unittest.TestCase):
    def test_converges_to_residual_tolerance_at_a_seated_state(self):
        """Terminates on the residual, and the barrier is genuinely active there.

        Both halves of the assertion carry weight: convergence without contact would
        test nothing about the barrier, and ``0 < gap < d_hat`` is the precondition for
        every adjoint check below to be meaningful rather than a comparison of zeros.
        """
        problem, positions = _seated_problem()
        result = solve_static_equilibrium(
            problem,
            tol=_TOL,
            initial_position=_feasible_initial(positions),
            max_iterations=200,
        )
        self.assertTrue(bool(result.converged))
        self.assertLessEqual(float(result.residual_norm), _TOL)
        gap = _min_gap(problem, result.position)
        self.assertGreater(gap, 0.0)
        self.assertLess(gap, _D_HAT)

    def test_friction_is_refused(self):
        problem, _ = _seated_problem(contact_friction_mu=0.3)
        with self.assertRaisesRegex(ValueError, "not the gradient of any potential"):
            solve_static_equilibrium(problem, tol=_TOL)

    def test_self_collision_is_refused(self):
        positions, tets = _kuhn_block()
        free = np.ones(positions.shape[0])
        free[0] = 0.0
        problem = assemble_problem(
            positions,
            tets,
            jnp.asarray(free),
            mu=50.0,
            lam=75.0,
            self_collision=True,
        )
        with self.assertRaisesRegex(ValueError, "detection band"):
            solve_static_equilibrium(problem, tol=_TOL)


class StaticAdjointTests(unittest.TestCase):
    """du*/dp against central differences, one parameter class at a time.

    float64, small mesh, seated configuration with the barrier active. The 1e-4 relative
    gate is the brief's; the measured errors are one to two orders below it, so a failure
    here is a regression, not noise. Each central difference re-runs the full solve at
    p +- h, which is what makes this class slow -- and is also exactly why the adjoint
    exists: one backward solve replaces 2 x dim(p) forward solves.
    """

    @classmethod
    def setUpClass(cls):
        cls.problem, cls.positions = _seated_problem()
        cls.initial = _feasible_initial(cls.positions)
        cls.adjoint = StaticAdjoint(
            cls.problem,
            tol=_TOL,
            warm_start_steps=0,
            initial_position=cls.initial,
            max_iterations=200,
        )
        weights = np.random.default_rng(0).normal(size=cls.positions.shape)
        cls.weights = jnp.asarray(weights)

    def _loss(self, params):
        return jnp.sum(self.weights * self.adjoint(params))

    def _check(self, make_params, value, h):
        gradient = float(
            jax.grad(lambda v: self._loss(make_params(v)))(jnp.asarray(value))
        )
        plus = float(self._loss(make_params(jnp.asarray(value + h))))
        minus = float(self._loss(make_params(jnp.asarray(value - h))))
        numeric = (plus - minus) / (2.0 * h)
        self.assertGreater(abs(numeric), 0.0)
        self.assertLessEqual(
            abs(gradient - numeric),
            1.0e-4 * abs(numeric),
            msg=f"adjoint {gradient} vs FD {numeric}",
        )

    def test_material_mu(self):
        self._check(lambda v: StaticParams(mu=v), 50.0, 1e-4)

    def test_material_lam(self):
        self._check(lambda v: StaticParams(lam=v), 75.0, 1e-4)

    def test_density_flows_through_the_lumped_masses(self):
        """density enters Pi only via mass = build_lumped_masses(rest) * density; a
        stale mass in apply_static_params would zero this gradient, not merely bias it."""
        self._check(lambda v: StaticParams(density=v), 1.0, 1e-5)

    def test_collider_radius(self):
        self._check(
            lambda v: StaticParams(collider_radius=jnp.asarray([v])), 1.0, 1e-6
        )

    def test_collider_center_height(self):
        self._check(
            lambda v: StaticParams(
                collider_center=jnp.asarray(
                    [[_CENTER[0], _CENTER[1], v]]
                )
            ),
            float(_CENTER[2]),
            1e-6,
        )

    def test_body_force(self):
        self._check(
            lambda v: StaticParams(
                external_acceleration=jnp.asarray([0.0, 0.0, v])
            ),
            -9.81,
            1e-5,
        )

    def test_rest_positions_directional(self):
        """Directional derivative along one fixed random perturbation of the rest shape
        (the full (N,3) Jacobian would need 3N solves for its FD twin). The direction
        leaves the clamped base untouched so the perturbed problem keeps its geometry."""
        rest = np.asarray(self.positions)
        rng = np.random.default_rng(1)
        direction = rng.normal(size=rest.shape) * 0.01
        direction[rest[:, 2] <= 1e-9] = 0.0
        direction = jnp.asarray(direction)

        def loss_of_scale(s):
            return self._loss(
                StaticParams(rest_positions=jnp.asarray(rest) + s * direction)
            )

        gradient = float(jax.grad(loss_of_scale)(jnp.asarray(0.0)))
        h = 1e-5
        numeric = (
            float(loss_of_scale(jnp.asarray(h)))
            - float(loss_of_scale(jnp.asarray(-h)))
        ) / (2.0 * h)
        # A zero numeric derivative would make the relative check below pass vacuously
        # (0 <= 0) -- exactly the failure this test exists to rule out, since a broken
        # rest-position path (e.g. stale masses) zeroes the gradient rather than
        # biasing it.
        self.assertGreater(abs(numeric), 0.0)
        self.assertLessEqual(abs(gradient - numeric), 1.0e-4 * abs(numeric))

    def test_unconverged_adjoint_raises(self):
        """The refuse-to-be-wrong gate. One Newton iteration cannot reach 1e-30, so the
        forward state is genuinely unconverged; asking for its gradient must raise, not
        return the implicit formula's plausible nonsense."""
        stalled = StaticAdjoint(
            self.problem,
            tol=1e-30,
            warm_start_steps=0,
            initial_position=self.initial,
            max_iterations=1,
        )
        with self.assertRaisesRegex(ValueError, "residual"):
            jax.grad(lambda v: jnp.sum(stalled(StaticParams(mu=v))))(
                jnp.asarray(50.0)
            )


if __name__ == "__main__":
    unittest.main()
