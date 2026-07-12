"""Contact tests.

The gates here are organised around the four ways contact fails *silently* -- each one
produces plausible-looking output rather than an error, so each needs a test that would go
red rather than merely look wrong:

* a NaN gradient leaking out of an inactive or degenerate primitive (`0 * inf == nan`);
* a barrier that is constant where it should be steep, so a penetrated vertex feels no
  restoring force and can never escape;
* a float type that cannot resolve the gap, so a positive distance rounds negative;
* a solver path that bypasses the barrier entirely and tunnels straight through.
"""

import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from diff_vbd import assemble_problem, simulate
from diff_vbd.model import SimulationState
from diff_vbd.setup.topology import build_surface_topology
from diff_vbd.solver import vbd
from diff_vbd.solver.contact import distances as D
from diff_vbd.solver.contact.barrier import barrier_energy, penalty_energy
from diff_vbd.solver.contact.ccd import (
    collider_sweep_time_of_impact,
    derive_detection_band,
    displacement_clamp_time_of_impact,
    pair_time_of_impact,
    vertex_time_of_impact,
)
from diff_vbd.solver.contact.colliders import (
    plane_signed_distance,
    sphere_signed_distance,
)
from diff_vbd.solver.contact.detection import (
    build_contact_incidence,
    detect_contact_pairs,
)
from diff_vbd.solver.contact.distances import pair_distance_sq, pair_gap_and_mollifier
from diff_vbd.solver.contact.friction import (
    collider_friction_energy,
    pair_friction_energy,
    smooth_friction_f0,
    smooth_friction_f1,
    tangent_basis,
)
from diff_vbd.solver.contact.potential import (
    collider_contact_energy,
    colliding_vertex_mask,
    contact_potential,
    incident_pair_energy,
    pair_contact_energy,
)

TRUE = jnp.asarray(True)
FALSE_ = jnp.asarray(False)
SLACK = jnp.asarray(0.9)
FALSE = jnp.asarray(False)


def _block(nx=3, ny=3, nz=3, z0=0.5, spacing=0.25):
    """A free tetrahedral block, positively wound, with no Dirichlet vertices."""
    grid = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1
    )
    positions = grid.reshape(-1, 3).astype(np.float64) * spacing
    positions[:, 2] += z0
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
                # Kuhn (Freudenthal) 6-tet split, not the usual 5-tet one. The 5-tet split
                # only conforms if adjacent cells alternate on a checkerboard: apply one
                # pattern everywhere and neighbouring cells disagree about which way to cut
                # each shared face, so interior faces never pair up and are reported as
                # *surface* faces -- and self-collision then finds phantom contacts inside
                # the solid at rest. Kuhn routes every tet through the same main diagonal
                # and so conforms unconditionally.
                tets += [
                    [v000, v100, v110, v111],
                    [v000, v101, v100, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v001, v101, v111],
                    [v000, v011, v001, v111],
                ]
    # float64: contact is a float64 feature. See tests/conftest.py and the conditioning
    # validator in problem_builder.
    return (
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.asarray(np.array(tets), dtype=jnp.int32),
    )


def _ground_plane():
    return [{"kind": "plane", "normal": (0.0, 0.0, 1.0), "offset": 0.0}]


def _two_blocks(
    *,
    gap=0.5e-3,
    n=2,
    spacing=0.25,
    velocity=(0.0, 0.0, 0.0),
    d_hat=1.0e-3,
    friction_mu=0.0,
    self_collision_ccd=True,
    num_iterations=20,
    external_acceleration=(0.0, 0.0, 0.0),
    **kwargs,
):
    """Two stacked blocks in one tet array: a Dirichlet-fixed lower one, a free upper one.

    The *only* contact in the system is mesh-mesh -- there are no analytic colliders at all,
    so nothing here can pass by accident on the collider path.

    Returns ``(problem, state, n_lower)``; vertices ``[:n_lower]`` are the fixed block.
    """
    lower_positions, lower_tets = _block(nx=n, ny=n, nz=n, z0=0.0, spacing=spacing)
    top = float(spacing * (n - 1))
    upper_positions, upper_tets = _block(
        nx=n, ny=n, nz=n, z0=top + gap, spacing=spacing
    )
    n_lower = int(lower_positions.shape[0])

    positions = jnp.concatenate([lower_positions, upper_positions])
    tets = jnp.concatenate([lower_tets, upper_tets + n_lower])
    free = jnp.concatenate(
        [
            jnp.zeros((n_lower,), dtype=positions.dtype),
            jnp.ones((upper_positions.shape[0],), dtype=positions.dtype),
        ]
    )

    with warnings.catch_warnings():
        # `self_collision_ccd=False` warns by design; the negative tests want it.
        warnings.simplefilter("ignore", RuntimeWarning)
        problem = assemble_problem(
            positions,
            tets,
            free,
            dt=0.005,
            num_iterations=num_iterations,
            mu=50.0,
            lam=50.0,
            density=1.0,
            eps=1.0e-8,
            external_acceleration=external_acceleration,
            contact_d_hat=d_hat,
            contact_friction_mu=friction_mu,
            self_collision=True,
            self_collision_ccd=self_collision_ccd,
            contact_capacity=16384,
            contact_max_per_vertex=512,
            **kwargs,
        )

    velocities = jnp.concatenate(
        [
            jnp.zeros((n_lower, 3), dtype=positions.dtype),
            jnp.tile(
                jnp.asarray(velocity, dtype=positions.dtype)[None, :],
                (upper_positions.shape[0], 1),
            ),
        ]
    )
    state = SimulationState(
        position=positions,
        velocity=velocities,
        time=jnp.asarray(0.0, dtype=positions.dtype),
    )
    return problem, state, n_lower


def _block_on_plane(*, d_hat=1.0e-3, num_iterations=30, **kwargs):
    positions, tets = _block()
    free_mask = jnp.ones((positions.shape[0],), dtype=positions.dtype)
    return assemble_problem(
        positions,
        tets,
        free_mask,
        dt=0.02,
        external_acceleration=(0.0, 0.0, -9.81),
        num_iterations=num_iterations,
        mu=50.0,
        lam=50.0,
        eps=1.0e-8,
        density=1.0,
        colliders=_ground_plane(),
        contact_d_hat=d_hat,
        **kwargs,
    )


class BarrierTests(unittest.TestCase):
    def setUp(self):
        self.d_hat = jnp.asarray(1.0e-2)

    def _barrier(self, gap, active=TRUE):
        return barrier_energy(jnp.asarray(gap), self.d_hat, active)

    def test_barrier_and_two_derivatives_vanish_at_the_activation_distance(self):
        """C2 across d_hat, so switching a contact on is not a kink the solver can see."""
        first = jax.grad(lambda g: barrier_energy(g, self.d_hat, TRUE))
        second = jax.grad(first)
        at_hat = self.d_hat
        self.assertAlmostEqual(float(self._barrier(at_hat)), 0.0, places=12)
        self.assertAlmostEqual(float(first(at_hat)), 0.0, places=10)
        self.assertAlmostEqual(float(second(at_hat)), 0.0, places=8)

    def test_barrier_is_zero_beyond_the_activation_distance(self):
        self.assertEqual(float(self._barrier(self.d_hat * 2.0)), 0.0)

    def test_barrier_derivative_matches_central_differences(self):
        first = jax.grad(lambda g: barrier_energy(g, self.d_hat, TRUE))
        for gap in (1.0e-3, 3.0e-3, 9.0e-3):
            with self.subTest(gap=gap):
                h = 1.0e-9
                finite = (self._barrier(gap + h) - self._barrier(gap - h)) / (2.0 * h)
                self.assertAlmostEqual(
                    float(first(jnp.asarray(gap))) / float(finite), 1.0, places=5
                )

    def test_inactive_primitive_has_zero_energy_and_zero_gradient(self):
        """The `0 * inf == nan` trap.

        A padded or out-of-range slot must contribute nothing -- and crucially its
        *gradient* must be 0.0 and not NaN. `jnp.where` evaluates both branches under
        `jax.grad`, so masking the result alone leaves a NaN to propagate. Gaps at and
        below zero are included because that is exactly what a degenerate padded pair
        looks like.
        """
        grad = jax.grad(lambda g: barrier_energy(g, self.d_hat, FALSE))
        for gap in (1.0e-3, 0.0, -1.0e-9, -1.0e3):
            with self.subTest(gap=gap):
                self.assertEqual(float(self._barrier(gap, active=FALSE)), 0.0)
                derivative = float(grad(jnp.asarray(gap)))
                self.assertFalse(np.isnan(derivative), "inactive slot leaked a NaN")
                self.assertEqual(derivative, 0.0)

    def test_penetrated_vertex_feels_a_restoring_force(self):
        """The constant-plateau trap, in its contact incarnation.

        Clamping the log's argument with a `maximum` would make the barrier *constant*
        below the floor, and the derivative of a constant is zero -- so a vertex that had
        been pushed through would feel no push-out force whatsoever and could never
        recover. That is the same failure the elastic energy in this repo used to have.
        The energy must keep rising, and the gradient must stay strongly negative, all the
        way through zero and beyond.
        """
        first = jax.grad(lambda g: barrier_energy(g, self.d_hat, TRUE))
        previous_energy = -np.inf
        for gap in (1.0e-3, 1.0e-6, 1.0e-12, 0.0, -1.0e-6):
            with self.subTest(gap=gap):
                energy = float(self._barrier(gap))
                derivative = float(first(jnp.asarray(gap)))
                self.assertTrue(np.isfinite(energy))
                self.assertTrue(np.isfinite(derivative))
                self.assertGreater(energy, previous_energy)  # monotone as the gap closes
                self.assertLess(derivative, 0.0)  # pushes the vertex back out
                previous_energy = energy

    def test_penalty_fallback_is_active_only_inside_the_range(self):
        self.assertEqual(float(penalty_energy(self.d_hat * 2, self.d_hat, TRUE)), 0.0)
        self.assertGreater(float(penalty_energy(self.d_hat / 2, self.d_hat, TRUE)), 0.0)
        self.assertEqual(float(penalty_energy(self.d_hat / 2, self.d_hat, FALSE)), 0.0)


class DistanceTests(unittest.TestCase):
    def _brute_point_triangle(self, p, t0, t1, t2, n=200):
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        a, b = i[keep] / n, j[keep] / n
        c = 1.0 - a - b
        q = a[:, None] * t0 + b[:, None] * t1 + c[:, None] * t2
        return float((((p - q) ** 2).sum(-1)).min())

    def _brute_edge_edge(self, a0, a1, b0, b1, n=600):
        s = np.linspace(0.0, 1.0, n)[:, None]
        a = a0 + (a1 - a0) * s
        b = b0 + (b1 - b0) * s
        return float((((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())

    def test_point_triangle_classification_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for trial in range(60):
            with self.subTest(trial=trial):
                p, t0, t1, t2 = (rng.normal(size=3) for _ in range(4))
                args = [jnp.asarray(x) for x in (p, t0, t1, t2)]
                kind = D.classify_point_triangle(*args)
                got = float(D.point_triangle_distance_sq(*args, kind))
                # The brute-force grid can only over-estimate the true minimum.
                self.assertLessEqual(got, self._brute_point_triangle(p, t0, t1, t2) + 1e-9)

    def test_edge_edge_classification_matches_brute_force(self):
        rng = np.random.default_rng(1)
        for trial in range(60):
            with self.subTest(trial=trial):
                a0, a1, b0, b1 = (rng.normal(size=3) for _ in range(4))
                args = [jnp.asarray(x) for x in (a0, a1, b0, b1)]
                kind = D.classify_edge_edge(*args)
                got = float(D.edge_edge_distance_sq(*args, kind))
                self.assertLessEqual(got, self._brute_edge_edge(a0, a1, b0, b1) + 1e-9)

    def test_edge_edge_handles_touching_endpoints(self):
        """Clamping s and t independently mis-classifies this and reports a nonzero gap.

        Clamping one parameter changes the optimal other one, so they must not be clamped
        independently. Two edges sharing an endpoint land exactly on the boundary, which is
        where that mistake shows up as a *positive* distance between touching primitives --
        a contact the solver would never see.
        """
        rng = np.random.default_rng(2)
        for trial in range(30):
            with self.subTest(trial=trial):
                a0, a1, b1 = (rng.normal(size=3) for _ in range(3))
                b0 = a1  # the two edges touch, exactly
                args = [jnp.asarray(x) for x in (a0, a1, b0, b1)]
                kind = D.classify_edge_edge(*args)
                self.assertAlmostEqual(
                    float(D.edge_edge_distance_sq(*args, kind)), 0.0, places=9
                )

    def test_distance_gradient_matches_central_differences(self):
        rng = np.random.default_rng(3)
        t0, t1, t2 = (jnp.asarray(rng.normal(size=3)) for _ in range(3))
        p = np.array([0.3, 0.2, 0.9])
        kind = D.classify_point_triangle(jnp.asarray(p), t0, t1, t2)
        f = lambda q: D.point_triangle_distance_sq(q, t0, t1, t2, kind)
        analytic = np.asarray(jax.grad(f)(jnp.asarray(p)))
        h = 1.0e-5
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            finite = (float(f(jnp.asarray(p + step))) - float(f(jnp.asarray(p - step)))) / (
                2.0 * h
            )
            self.assertAlmostEqual(analytic[axis], finite, places=4)

    def test_mollifier_vanishes_smoothly_at_parallel(self):
        """Two edges rotating through parallel swap which sub-feature is closest, so the
        edge-edge distance jumps there even at a fixed distance type. The mollifier is what
        removes that discontinuity: it must reach zero at parallel and saturate at 1."""
        a0 = jnp.array([0.0, 0.0, 0.0])
        a1 = jnp.array([1.0, 0.0, 0.0])
        eps_x = D.edge_edge_mollifier_threshold(a0, a1, a0, a1)

        def mollifier(theta):
            b0 = jnp.array([0.5, -0.5, 0.4])
            b1 = b0 + jnp.stack([jnp.cos(theta), jnp.sin(theta), jnp.zeros(())])
            return D.edge_edge_mollifier(a0, a1, b0, b1, eps_x)

        self.assertAlmostEqual(float(mollifier(jnp.asarray(0.0))), 0.0, places=9)
        self.assertAlmostEqual(float(mollifier(jnp.asarray(0.5))), 1.0, places=9)
        # Continuous and non-decreasing on the way out of the degenerate configuration.
        values = [float(mollifier(jnp.asarray(t))) for t in np.linspace(0.0, 0.05, 12)]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(values, values[1:])))
        self.assertTrue(all(np.isfinite(values)))

    def test_distance_from_squared_never_has_a_zero_derivative(self):
        """A `maximum`-style clamp here would kill the contact force at zero distance."""
        grad = jax.grad(D.distance_from_squared)
        for squared in (1.0, 1.0e-12, 0.0):
            with self.subTest(squared=squared):
                derivative = float(grad(jnp.asarray(squared)))
                self.assertTrue(np.isfinite(derivative))
                self.assertGreater(derivative, 0.0)


class ColliderTests(unittest.TestCase):
    def test_plane_distance_is_signed(self):
        normal = jnp.array([0.0, 0.0, 1.0])
        offset = jnp.asarray(0.0)
        above = plane_signed_distance(jnp.array([0.0, 0.0, 0.25]), normal, offset)
        below = plane_signed_distance(jnp.array([0.0, 0.0, -0.25]), normal, offset)
        # The sign is the whole point: squaring it would push a tunnelled vertex further
        # out the wrong side instead of back through.
        self.assertAlmostEqual(float(above), 0.25)
        self.assertAlmostEqual(float(below), -0.25)

    def test_sphere_distance_respects_inside_and_outside(self):
        center = jnp.zeros(3)
        radius = jnp.asarray(1.0)
        point = jnp.array([2.0, 0.0, 0.0])
        self.assertAlmostEqual(
            float(sphere_signed_distance(point, center, radius, TRUE)), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(sphere_signed_distance(point, center, radius, FALSE)), -1.0, places=6
        )


def _potential(problem, positions, previous=None):
    """Whole-mesh contact energy at `positions`, lagged against `previous`."""
    return contact_potential(
        problem.contact.params,
        problem.contact.colliders,
        problem.contact.state,
        problem.mesh.rest_positions,
        positions,
        problem.mesh.rest_positions if previous is None else previous,
        problem.solver.dt,
    )


class ContactPotentialTests(unittest.TestCase):
    def test_potential_equals_the_sum_of_the_local_contributions(self):
        """The shared-definition gate.

        `vertex_local_objective` and `contact_potential` must agree about what the contact
        energy *is*, or the forward solve and any future sensitivity path are optimising
        different things.
        """
        problem = _block_on_plane()
        positions = problem.mesh.rest_positions
        total = float(_potential(problem, positions))
        summed = float(
            sum(
                collider_contact_energy(
                    problem.contact.params, problem.contact.colliders, positions[i]
                )
                + collider_friction_energy(
                    problem.contact.params,
                    problem.contact.colliders,
                    positions[i],
                    positions[i],
                    problem.solver.dt,
                )
                for i in range(positions.shape[0])
            )
        )
        self.assertAlmostEqual(total, summed, places=6)

    def test_potential_counts_each_pair_once_not_once_per_incidence(self):
        """The counting gate, and it is only non-trivial once mesh-mesh pairs exist.

        `incident_pair_energy` sums the pairs *touching* a vertex, so summing it over every
        vertex counts each 4-vertex pair four times. `contact_potential` sums over primitives.
        The factor is exactly 4 -- detection skips primitives that share a vertex, so a valid
        pair always has four distinct vertices. With analytic colliders alone the two counts
        coincide, which is exactly why the discrepancy could hide.
        """
        problem, state, _ = _two_blocks(gap=0.5e-3)
        problem = vbd.redetect_contacts(problem, state)
        positions = state.position
        self.assertGreater(
            int(np.asarray(problem.contact.state.pair_valid).sum()),
            0,
            "no pairs detected -- the test would be vacuous",
        )

        per_incidence = float(
            sum(
                incident_pair_energy(
                    problem.contact.params,
                    problem.contact.state,
                    problem.mesh.rest_positions,
                    positions,
                    positions,
                    problem.solver.dt,
                    jnp.int32(i),
                    positions[i],
                )
                for i in range(positions.shape[0])
            )
        )
        # `contact_potential` has no colliders here, so all of it is the pair term.
        per_primitive = float(_potential(problem, positions, previous=positions))
        self.assertGreater(per_primitive, 0.0)
        self.assertAlmostEqual(
            per_incidence, 4.0 * per_primitive, delta=1e-6 * abs(per_incidence)
        )

    def test_potential_gradient_matches_central_differences(self):
        """Real central differences, on a problem where pairs and friction are both live.

        The mesh is *jittered* first, and that is not incidental. Two perfectly aligned flat
        faces put a great many pairs exactly on a distance-type classification boundary -- a
        vertex sitting precisely over a triangle's edge, say. The type is frozen inside the
        energy (which is what makes the barrier differentiable at all), so at those
        configurations the potential has a genuine kink, and a finite difference straddles it
        and reports a slope the gradient does not have. That is the measure-zero boundary this
        design accepts by construction, not a defect in the gradient. Jittering moves the test
        off it and onto a generic configuration, where the two must agree.
        """
        rng = np.random.default_rng(0)
        problem, state, _ = _two_blocks(gap=1.0e-3, friction_mu=0.3, d_hat=4.0e-3)
        jitter = jnp.asarray(
            rng.normal(scale=2.0e-4, size=state.position.shape),
            dtype=state.position.dtype,
        )
        positions = state.position + jitter
        state = SimulationState(
            position=positions, velocity=state.velocity, time=state.time
        )
        problem = vbd.redetect_contacts(problem, state)
        self.assertGreater(
            int(np.asarray(problem.contact.state.pair_valid).sum()),
            0,
            "no pairs detected -- the test would be vacuous",
        )

        # A *distinct* lagged reference. Passing `positions` itself would put the live
        # positions on both sides of a stop_gradient, and AD would then legitimately disagree
        # with a finite difference -- failing for the wrong reason.
        previous = positions - 1.0e-4

        energy = lambda x: _potential(problem, x, previous=previous)
        gradient = np.asarray(jax.grad(energy)(positions))
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertGreater(np.abs(gradient).max(), 0.0)

        h = 1.0e-7
        scale = float(np.abs(gradient).max())
        probes = rng.choice(positions.shape[0], size=6, replace=False)
        for vertex in probes:
            for axis in range(3):
                shift = np.zeros_like(np.asarray(positions))
                shift[vertex, axis] = h
                plus = float(energy(positions + shift))
                minus = float(energy(positions - shift))
                numeric = (plus - minus) / (2.0 * h)
                self.assertAlmostEqual(
                    gradient[vertex, axis], numeric, delta=1e-4 * scale
                )


class TimeOfImpactTests(unittest.TestCase):
    def setUp(self):
        self.problem = _block_on_plane()
        self.colliders = self.problem.contact.colliders

    def test_step_straight_into_a_plane_is_cut_short(self):
        start = jnp.array([0.0, 0.0, 0.1])
        end = jnp.array([0.0, 0.0, -1.0])  # would end up well below the plane
        toi = float(vertex_time_of_impact(self.colliders, start, end))
        self.assertLess(toi, 1.0)
        landed = start + toi * (end - start)
        self.assertGreater(float(landed[2]), 0.0)  # never actually reaches the plane

    def test_step_away_from_a_plane_is_unrestricted(self):
        start = jnp.array([0.0, 0.0, 0.1])
        end = jnp.array([0.0, 0.0, 5.0])
        self.assertAlmostEqual(
            float(vertex_time_of_impact(self.colliders, start, end)), 1.0, places=9
        )

    def test_sliding_along_a_plane_is_not_throttled(self):
        """The reason the plane gets an exact solve rather than the Lipschitz bound.

        The conservative bound assumes the motion is aimed straight at the obstacle, so it
        would throttle motion *parallel* to the plane -- which is exactly the sliding a
        resting body needs to do.
        """
        start = jnp.array([0.0, 0.0, 1.0e-4])
        end = jnp.array([10.0, 0.0, 1.0e-4])  # a long way, but never any closer
        self.assertAlmostEqual(
            float(vertex_time_of_impact(self.colliders, start, end)), 1.0, places=9
        )

    def test_resting_vertex_does_not_throttle_the_whole_sweep(self):
        """The global filter is one scalar for the entire mesh, so a vertex already resting
        in contact must not drag it to zero and freeze everything."""
        positions = jnp.array([[0.0, 0.0, 1.0e-5], [0.0, 0.0, 1.0]])
        # The resting vertex barely moves; the free one moves a long way, away from the plane.
        moved = jnp.array([[0.0, 0.0, 1.0e-5], [0.0, 0.0, 2.0]])
        toi = float(collider_sweep_time_of_impact(self.colliders, positions, moved))
        self.assertAlmostEqual(toi, 1.0, places=6)

    def test_no_colliders_yields_an_unrestricted_step(self):
        positions, tets = _block()
        free = jnp.ones((positions.shape[0],), dtype=positions.dtype)
        free = free.at[0].set(0.0)  # needs a Dirichlet vertex when there is no collider
        problem = assemble_problem(positions, tets, free)
        toi = float(
            collider_sweep_time_of_impact(
                problem.contact.colliders, positions, positions + 100.0
            )
        )
        self.assertEqual(toi, 1.0)


class RestingContactTests(unittest.TestCase):
    def test_block_settles_on_the_plane_without_penetrating(self):
        """A body held up by nothing but the barrier. Note there is no Dirichlet vertex at
        all: resting on the ground is the only thing stopping it falling forever."""
        d_hat = 1.0e-3
        problem = _block_on_plane(d_hat=d_hat)
        final, history = simulate(
            problem, _initial(problem), num_steps=250, show_progress=False
        )
        heights = np.asarray(history.position)[:, :, 2]

        self.assertTrue(np.isfinite(heights).all())
        self.assertGreater(heights.min(), 0.0, "the block penetrated the plane")
        final_gap = float(heights[-1].min())
        self.assertGreater(final_gap, 0.0)
        self.assertLess(final_gap, d_hat, "the block never came to rest on the plane")

    def test_result_is_invariant_to_num_iterations(self):
        """The convergence property must survive the barrier. A solver whose answer depends
        on how long you run it is not converging -- and a stiff barrier is exactly the kind
        of thing that would break that."""
        few = _block_on_plane(num_iterations=30)
        many = _block_on_plane(num_iterations=90)
        a, _ = simulate(few, _initial(few), num_steps=120, show_progress=False)
        b, _ = simulate(many, _initial(many), num_steps=120, show_progress=False)
        drift = float(
            np.abs(np.asarray(a.position) - np.asarray(b.position)).max()
        )
        self.assertLess(drift, 1.0e-4, f"answer moved by {drift:.3e} when sweeps grew")


class TunnellingTests(unittest.TestCase):
    """Every solver path must respect the barrier, including the ones that never evaluate it.

    The local line search rejects a penetrating candidate on its own, because a candidate
    past the barrier has an astronomical objective and loses the argmin. But two paths never
    consult the objective: the full-Newton branch (taken when the line search is disabled)
    and the Chebyshev extrapolation (applied *after* every local solve has finished). Both
    tunnelled before the CCD filter existed -- the first to NaN, the second clean through
    the floor.
    """

    def _drive_at_plane(self, speed, **kwargs):
        problem = _block_on_plane(**kwargs)
        positions = problem.mesh.rest_positions
        state = SimulationState(
            position=positions,
            velocity=jnp.tile(
                jnp.asarray([0.0, 0.0, speed], dtype=positions.dtype),
                (positions.shape[0], 1),
            ),
            time=jnp.asarray(0.0, dtype=positions.dtype),
        )
        _, history = simulate(problem, state, num_steps=100, show_progress=False)
        return np.asarray(history.position)[:, :, 2]

    def _assert_held(self, heights):
        self.assertTrue(np.isfinite(heights).all(), "solver produced NaN at impact")
        self.assertGreater(heights.min(), 0.0, "the block tunnelled through the plane")

    def test_fast_impact_with_line_search(self):
        self._assert_held(self._drive_at_plane(-50.0))

    def test_fast_impact_without_line_search(self):
        # The full-Newton branch never evaluates the objective, so only the per-vertex time
        # of impact stands between it and the floor.
        self._assert_held(self._drive_at_plane(-50.0, line_search_enabled=False))

    def test_fast_impact_with_chebyshev_acceleration(self):
        # Chebyshev extrapolates after the local solves have finished, so no per-vertex
        # check can see it. Only the sweep-level filter and the skip mask catch this.
        self._assert_held(
            self._drive_at_plane(
                -50.0, acceleration_enabled=True, chebyshev_rho=0.95
            )
        )

    def test_extreme_impact_with_every_accelerator_enabled(self):
        heights = self._drive_at_plane(
            -2000.0,  # 40 units of travel per step, against a 1e-3 activation distance
            line_search_enabled=False,
            acceleration_enabled=True,
            chebyshev_rho=0.95,
        )
        self._assert_held(heights)


class FrictionTests(unittest.TestCase):
    def test_shape_functions_are_consistent_and_c1(self):
        """f1 must be the derivative of f0, and both must be continuous at the transition,
        or the 'friction potential' is not a potential and the solver is descending
        something other than what it thinks."""
        eps_v_h = jnp.asarray(1.0e-3)
        derivative = jax.grad(lambda y: smooth_friction_f0(y, eps_v_h))
        for slip in (1.0e-5, 5.0e-4, 1.0e-3, 5.0e-3):
            with self.subTest(slip=slip):
                self.assertAlmostEqual(
                    float(derivative(jnp.asarray(slip))),
                    float(smooth_friction_f1(jnp.asarray(slip), eps_v_h)),
                    places=6,
                )
        # The two branches must agree at the transition itself. (Comparing f0 either side
        # of it would not show this: f0 has slope 1 there, so nearby values differ by the
        # step size no matter how continuous it is.)
        self.assertAlmostEqual(
            float(smooth_friction_f0(eps_v_h, eps_v_h)), float(eps_v_h), places=12
        )
        # ...and so must the slopes, which is what makes it C1.
        self.assertAlmostEqual(
            float(derivative(eps_v_h * 0.999)), 1.0, places=4
        )
        self.assertAlmostEqual(
            float(derivative(eps_v_h * 1.001)), 1.0, places=9
        )

    def test_friction_force_saturates_at_full_sliding(self):
        eps_v_h = jnp.asarray(1.0e-3)
        self.assertAlmostEqual(
            float(smooth_friction_f1(jnp.asarray(0.0), eps_v_h)), 0.0, places=9
        )
        self.assertAlmostEqual(
            float(smooth_friction_f1(jnp.asarray(1.0), eps_v_h)), 1.0, places=9
        )

    def test_tangent_basis_is_orthonormal_and_perpendicular_to_the_normal(self):
        for normal in (
            jnp.array([0.0, 0.0, 1.0]),
            jnp.array([1.0, 0.0, 0.0]),
            jnp.asarray(np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])),
        ):
            with self.subTest(normal=tuple(np.asarray(normal))):
                basis = tangent_basis(normal)
                self.assertTrue(
                    np.allclose(np.asarray(basis.T @ basis), np.eye(2), atol=1e-9)
                )
                self.assertTrue(
                    np.allclose(np.asarray(basis.T @ normal), np.zeros(2), atol=1e-9)
                )

    def test_no_friction_force_without_slip(self):
        """A stuck contact must feel no tangential force, or a body at rest would creep."""
        problem = _block_on_plane(contact_friction_mu=0.5)
        position = jnp.array([0.0, 0.0, 1.0e-4])
        force = jax.grad(
            lambda x: collider_friction_energy(
                problem.contact.params,
                problem.contact.colliders,
                x,
                position,  # identical: zero slip
                problem.solver.dt,
            )
        )(position)
        self.assertTrue(np.allclose(np.asarray(force), np.zeros(3), atol=1e-9))

    def test_friction_opposes_sliding(self):
        problem = _block_on_plane(contact_friction_mu=0.5)
        previous = jnp.array([0.0, 0.0, 1.0e-4])
        slid = previous + jnp.array([0.01, 0.0, 0.0])  # slid along +x
        force = np.asarray(
            jax.grad(
                lambda x: collider_friction_energy(
                    problem.contact.params,
                    problem.contact.colliders,
                    x,
                    previous,
                    problem.solver.dt,
                )
            )(slid)
        )
        # grad of the potential is +force resisting motion, so the energy rises with slip.
        self.assertGreater(force[0], 0.0)
        self.assertAlmostEqual(force[2], 0.0, places=9)  # purely tangential

    def test_friction_holds_a_block_on_an_incline_and_lets_it_slide(self):
        """The physical gate: below the friction cone the block sticks, above it slides."""
        angle = np.deg2rad(20.0)
        normal = (float(np.sin(angle)), 0.0, float(np.cos(angle)))
        tan_theta = float(np.tan(angle))

        def travel(mu):
            positions, tets = _block(z0=0.02, spacing=0.1)
            free = jnp.ones((positions.shape[0],), dtype=positions.dtype)
            problem = assemble_problem(
                positions,
                tets,
                free,
                dt=0.005,
                external_acceleration=(0.0, 0.0, -9.81),
                num_iterations=40,
                mu=200.0,
                lam=200.0,
                eps=1.0e-8,
                density=1.0,
                colliders=[{"kind": "plane", "normal": normal, "offset": 0.0}],
                contact_d_hat=1.0e-3,
                contact_friction_mu=mu,
                contact_eps_v=1.0e-4,
            )
            from diff_vbd import initial_state

            final, _ = simulate(
                problem, initial_state(problem), num_steps=200, show_progress=False
            )
            start = np.asarray(positions).mean(axis=0)
            end = np.asarray(final.position).mean(axis=0)
            return float(abs(end[0] - start[0]))  # downhill drift

        sticking = travel(mu=4.0 * tan_theta)  # well inside the cone
        sliding = travel(mu=0.0)  # frictionless
        self.assertTrue(np.isfinite(sticking) and np.isfinite(sliding))
        self.assertLess(
            sticking,
            0.5 * sliding,
            f"friction did not hold the block (stuck {sticking:.4g} vs free {sliding:.4g})",
        )


class SurfaceTopologyTests(unittest.TestCase):
    def test_extracted_surface_is_a_closed_manifold(self):
        """Euler characteristic V - E + F == 2 for a closed genus-0 surface.

        This is the check that catches a *non-conforming* tet mesh, where neighbouring cells
        disagree about how to cut a shared face. Interior faces then never pair up, get
        reported as boundary faces, and self-collision finds phantom contacts deep inside
        the solid at rest. A raw triangle count would look perfectly reasonable; the Euler
        characteristic does not.
        """
        for size in (2, 3, 5):
            with self.subTest(size=size):
                _, tets = _block(nx=size, ny=size, nz=size)
                triangles, edges, vertices = build_surface_topology(tets)
                euler = vertices.shape[0] - edges.shape[0] + triangles.shape[0]
                self.assertEqual(euler, 2, "surface is not a closed manifold")

    def test_surface_triangles_are_wound_outward(self):
        """A backwards triangle reports its normal inverted, and the barrier then pushes
        the wrong way."""
        positions, tets = _block(nx=3, ny=3, nz=3)
        triangles, _, _ = build_surface_topology(tets)
        points = np.asarray(positions)[np.asarray(triangles)]
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        outward = (normals * (points.mean(1) - np.asarray(positions).mean(0))).sum(-1)
        self.assertTrue((outward > 0).all())

    def test_surface_indices_are_global(self):
        """Contact indexes straight into `position` and `mass`, so surface-local numbering
        (which is what the exporter produces) would silently address the wrong vertices."""
        positions, tets = _block()
        triangles, edges, vertices = build_surface_topology(tets)
        self.assertLess(int(triangles.max()), positions.shape[0])
        self.assertLess(int(edges.max()), positions.shape[0])
        # A block's interior vertices are not on the surface, so the surface is a strict
        # subset -- which is only meaningful if these are global indices.
        self.assertLess(vertices.shape[0], positions.shape[0])


class ConditioningTests(unittest.TestCase):
    def test_d_hat_below_the_float_resolution_is_rejected(self):
        """The float32 trap, caught at assembly instead of as a NaN a thousand steps in.

        At coordinate 100 the absolute float32 resolution is ~1.2e-5, so a gap of 1e-5 comes
        out ~24% wrong and a small positive gap can round negative -- at which point
        log(g/d_hat) is NaN and intersection-freedom is lost to rounding alone.
        """
        positions, tets = _block(spacing=50.0, z0=0.0)  # coordinates out to 100
        positions = jnp.asarray(positions, dtype=jnp.float32)
        free = jnp.ones((positions.shape[0],), dtype=positions.dtype)
        with self.assertRaisesRegex(ValueError, "below what .* can resolve"):
            assemble_problem(
                positions,
                tets,
                free,
                colliders=_ground_plane(),
                contact_d_hat=1.0e-6,
            )

    def test_a_resolvable_d_hat_is_accepted(self):
        problem = _block_on_plane(d_hat=1.0e-3)
        self.assertAlmostEqual(float(problem.contact.params.d_hat), 1.0e-3, places=9)

    def test_non_positive_d_hat_is_rejected(self):
        positions, tets = _block()
        free = jnp.ones((positions.shape[0],), dtype=positions.dtype)
        with self.assertRaisesRegex(ValueError, "d_hat must be positive"):
            assemble_problem(
                positions, tets, free, colliders=_ground_plane(), contact_d_hat=0.0
            )


class ContactAssemblyTests(unittest.TestCase):
    def test_a_body_supported_only_by_a_collider_needs_no_dirichlet_vertex(self):
        problem = _block_on_plane()
        self.assertEqual(int(jnp.sum(problem.boundary_conditions.dirichlet_mask)), 0)

    def test_an_unconstrained_body_with_no_collider_is_still_rejected(self):
        positions, tets = _block()
        free = jnp.ones((positions.shape[0],), dtype=positions.dtype)
        with self.assertRaisesRegex(ValueError, "constrain at least one vertex"):
            assemble_problem(positions, tets, free)

    def test_contact_is_absent_by_default(self):
        positions, tets = _block()
        free = jnp.ones((positions.shape[0],), dtype=positions.dtype).at[0].set(0.0)
        problem = assemble_problem(positions, tets, free)
        self.assertEqual(problem.contact.colliders.kind.shape, (0,))
        self.assertFalse(bool(problem.contact.params.enabled))

    def test_unknown_collider_kind_is_rejected(self):
        positions, tets = _block()
        free = jnp.ones((positions.shape[0],), dtype=positions.dtype)
        with self.assertRaisesRegex(ValueError, "unknown kind"):
            assemble_problem(
                positions, tets, free, colliders=[{"kind": "banana"}]
            )

    def test_rebuilding_contact_does_not_trigger_recompilation(self):
        """Host-side detection rebuilds the contact buffers every step. That is only viable
        because fresh buffers of the *same* shape are a cache hit -- JAX keys on structure,
        shape and dtype, not on contents. A changing capacity would recompile a trace
        containing nested scans over sweeps and colours, every single step."""
        import dataclasses

        problem = _block_on_plane()
        traces = {"count": 0}

        @jax.jit
        def consume(contact):
            traces["count"] += 1
            return jnp.sum(contact.colliders.offset) + contact.params.d_hat

        for _ in range(5):
            rebuilt = dataclasses.replace(
                problem.contact,
                colliders=dataclasses.replace(
                    problem.contact.colliders,
                    offset=problem.contact.colliders.offset + 0.0,
                ),
            )
            consume(rebuilt).block_until_ready()

        self.assertEqual(traces["count"], 1)


# =======================================================================================
# Mesh-mesh / self-collision. Everything above this line is analytic colliders.
# =======================================================================================


def _quad(*points):
    return jnp.asarray(np.array(points, dtype=np.float64))


def _pair_gap(points, pair_type):
    """The true (unsoftened) distance of a pair, as CCD sees it."""
    return float(
        jnp.sqrt(jnp.maximum(pair_distance_sq(points, jnp.int32(pair_type)), 0.0))
    )


_TRIANGLE = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])


class PairTimeOfImpactTests(unittest.TestCase):
    """ACCD: the mesh-mesh intersection-free guarantee, at the kernel level."""

    def _toi(self, start, end, pair_type, valid=TRUE):
        return float(
            pair_time_of_impact(start, end, jnp.int32(pair_type), valid, SLACK)
        )

    def test_a_head_on_impact_is_cut_short_and_keeps_a_margin(self):
        start = _quad([0.3, 0.3, 0.1], *_TRIANGLE)
        end = _quad([0.3, 0.3, -0.5], *_TRIANGLE)
        toi = self._toi(start, end, 0)
        self.assertGreater(toi, 0.0)
        self.assertLess(toi, 1.0)

        initial = _pair_gap(start, 0)
        landed = _pair_gap(start + toi * (end - start), 0)
        # Not merely "did not intersect": it stopped with room to spare, which is what keeps
        # the barrier's argument away from zero on the next step.
        self.assertGreaterEqual(landed, (1.0 - 0.9) * initial - 1e-12)

    def test_the_whole_certified_interval_is_safe_not_just_its_endpoint(self):
        """A segment certificate. Checking only where the step lands would miss a dip."""
        start = _quad([0.3, 0.3, 0.1], *_TRIANGLE)
        end = _quad([0.3, 0.3, -0.5], *_TRIANGLE)
        toi = self._toi(start, end, 0)
        for t in np.linspace(0.0, toi, 200):
            self.assertGreater(_pair_gap(start + t * (end - start), 0), 0.0)

    def test_the_time_of_impact_never_over_estimates(self):
        start = _quad([0.3, 0.3, 0.1], *_TRIANGLE)
        end = _quad([0.3, 0.3, -0.5], *_TRIANGLE)
        toi = self._toi(start, end, 0)
        # Dense sampling for the first crossing: ACCD must stop strictly before it.
        crossing = 1.0
        for t in np.linspace(0.0, 1.0, 20001):
            if _pair_gap(start + t * (end - start), 0) <= 0.0:
                crossing = t
                break
        self.assertLessEqual(toi, crossing + 1e-9)

    def test_a_shared_translation_does_not_throttle_the_step(self):
        """The mean-subtraction gate.

        Two primitives flying through space together are not approaching each other at all.
        Without subtracting the mean displacement, the Lipschitz bound is dominated by the
        shared velocity and the time of impact collapses -- a silent, catastrophic throttle
        that presents only as "the solver got slow".
        """
        start = _quad([0.3, 0.3, 0.1], *_TRIANGLE)
        end = _quad([0.3, 0.3, -0.5], *_TRIANGLE)
        toi = self._toi(start, end, 0)

        drift = _quad(*([[1.0e3, 0.0, 0.0]] * 4))
        drifted = self._toi(start + drift, end + drift, 0)
        self.assertAlmostEqual(toi, drifted, places=12)

    def test_a_rigidly_translating_pair_is_never_throttled(self):
        start = _quad([0.3, 0.3, 0.01], *_TRIANGLE)
        end = start + _quad(*([[9.0, 9.0, 9.0]] * 4))
        self.assertEqual(self._toi(start, end, 0), 1.0)

    def test_edge_edge_uses_the_edge_edge_lipschitz_bound(self):
        """Two edges whose endpoints fly apart in opposite directions.

        The point-triangle bound (`|p0| + max of the other three`) *under-estimates* the rate
        at which two edges can close, and an under-estimated Lipschitz constant permits too
        long a step. It is unsound, and it looks entirely reasonable.
        """
        start = _quad(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -0.5, 0.01], [0.5, 0.5, 0.01]
        )
        end = _quad(
            [0.0, 0.0, 0.5],
            [1.0, 0.0, -0.5],
            [0.5, -0.5, -0.49],
            [0.5, 0.5, 0.51],
        )
        toi = self._toi(start, end, 1)
        for t in np.linspace(0.0, toi, 300):
            self.assertGreater(_pair_gap(start + t * (end - start), 1), 0.0)

    def test_a_safe_advance_that_overshoots_the_segment_certifies_all_of_it(self):
        """Regression: a distant pair must return 1.0, not 0.0.

        When the conservative advance runs past t = 1 the probe lands outside the motion that
        actually happens -- the primitive has sailed *through* the obstacle and out the far
        side -- so its gap reads small again and can trip the convergence test. Returning the
        last accepted time then yields 0 for a pair that was never in any danger, and a single
        such pair drags the whole mesh's global minimum to zero and freezes the solve.
        """
        start = _quad([0.3, 0.3, 0.30], *_TRIANGLE)
        end = _quad([0.3, 0.3, 0.255], *_TRIANGLE)
        self.assertEqual(self._toi(start, end, 0), 1.0)

    def test_separating_motion_is_unrestricted(self):
        start = _quad([0.3, 0.3, 0.1], *_TRIANGLE)
        end = _quad([0.3, 0.3, 5.0], *_TRIANGLE)
        self.assertEqual(self._toi(start, end, 0), 1.0)

    def test_a_padded_pair_is_finite_and_unrestricted(self):
        """A padded slot is four coincident points: zero gap, zero motion, every division 0/0.

        It must produce 1.0 *without producing a NaN anywhere*, not merely mask one away --
        a single NaN here would poison the global minimum and freeze the entire mesh.
        """
        padded = jnp.zeros((4, 3))
        for pair_type in (0, 1):
            for end in (padded, padded + 1.0):
                toi = self._toi(padded, end, pair_type, valid=FALSE_)
                self.assertTrue(np.isfinite(toi))
                self.assertEqual(toi, 1.0)

    def test_an_already_touching_pair_freezes_rather_than_pretending(self):
        start = _quad([0.3, 0.3, 0.0], *_TRIANGLE)
        end = _quad([0.3, 0.3, -1.0], *_TRIANGLE)
        toi = self._toi(start, end, 0)
        self.assertTrue(np.isfinite(toi))
        self.assertEqual(toi, 0.0)

    def test_fuzz_is_finite_in_range_and_never_certifies_an_intersection(self):
        rng = np.random.default_rng(7)
        batched = jax.jit(
            jax.vmap(pair_time_of_impact, in_axes=(0, 0, 0, 0, None))
        )
        squared = jax.jit(jax.vmap(pair_distance_sq))

        for pair_type in (0, 1):
            with self.subTest(pair_type=pair_type):
                count = 1000
                start = jnp.asarray(rng.normal(scale=0.5, size=(count, 4, 3)))
                end = start + jnp.asarray(rng.normal(size=(count, 4, 3)))
                types = jnp.full((count,), pair_type, dtype=jnp.int32)
                valid = jnp.asarray(rng.random(count) > 0.2)

                toi = batched(start, end, types, valid, SLACK)
                self.assertTrue(bool(jnp.isfinite(toi).all()))
                self.assertTrue(bool(((toi >= 0.0) & (toi <= 1.0)).all()))

                initial = jnp.sqrt(jnp.maximum(squared(start, types), 0.0))
                live = np.asarray(valid & (initial > 1e-6))
                # Nothing that started apart may be certified into an intersection...
                for fraction in np.linspace(0.0, 1.0, 25):
                    probe = start + (fraction * toi)[:, None, None] * (end - start)
                    gap = np.asarray(
                        jnp.sqrt(jnp.maximum(squared(probe, types), 0.0))
                    )
                    self.assertEqual(int(((gap <= 0.0) & live).sum()), 0)
                # ...and nothing that started apart may be spuriously frozen, either.
                self.assertEqual(
                    int(((np.asarray(toi) < 1e-12) & live).sum()), 0
                )


class DisplacementClampTests(unittest.TestCase):
    """The clause that makes the detection band trustworthy."""

    def test_the_band_always_dominates_twice_the_allowed_displacement(self):
        """`band >= d_hat + 2 * max_displacement` -- the inequality the guarantee rests on."""
        for d_hat in (1e-4, 1e-3, 1e-2):
            for predicted in (0.0, 1e-3, 0.05, 1.0, 40.0):
                with self.subTest(d_hat=d_hat, predicted=predicted):
                    band, allowed = derive_detection_band(d_hat, predicted)
                    self.assertGreater(band - 2.0 * allowed, d_hat)

    def test_a_step_that_would_leave_the_ball_is_cut_back(self):
        start = jnp.zeros((3, 3))
        end = jnp.asarray([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        free = jnp.asarray([True, True, True])
        toi = float(
            displacement_clamp_time_of_impact(
                start, start, end, free, jnp.asarray(1.0)
            )
        )
        self.assertAlmostEqual(toi, 0.1, places=9)
        moved = np.asarray(start + toi * (end - start))
        self.assertLessEqual(np.linalg.norm(moved, axis=-1).max(), 1.0 + 1e-9)

    def test_a_step_inside_the_ball_is_unrestricted(self):
        start = jnp.zeros((2, 3))
        end = jnp.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
        free = jnp.asarray([True, True])
        toi = float(
            displacement_clamp_time_of_impact(
                start, start, end, free, jnp.asarray(1.0)
            )
        )
        self.assertEqual(toi, 1.0)

    def test_a_constrained_vertex_does_not_bind_the_clamp(self):
        """Dirichlet rows are rewritten *after* the filter, so clamping them would freeze the
        mesh for nothing. The host audit covers them instead."""
        start = jnp.zeros((2, 3))
        end = jnp.asarray([[99.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        free = jnp.asarray([False, True])
        toi = float(
            displacement_clamp_time_of_impact(
                start, start, end, free, jnp.asarray(1.0)
            )
        )
        self.assertEqual(toi, 1.0)


class PairFrictionTests(unittest.TestCase):
    """Mesh-mesh friction. Self-contact was silently frictionless before this."""

    def setUp(self):
        problem = _block_on_plane(contact_friction_mu=0.5, d_hat=1.0e-2)
        self.params = problem.contact.params
        self.dt = problem.solver.dt
        self.start = _quad([1 / 3, 1 / 3, 5.0e-3], *_TRIANGLE)

    def _force(self, live, previous, pair_type=0):
        energy = lambda x: pair_friction_energy(
            self.params, x, previous, previous, jnp.int32(pair_type), TRUE, self.dt
        )
        return np.asarray(jax.grad(energy)(live))

    def test_the_gap_gradient_is_the_normal_times_the_barycentric_weights(self):
        """The identity the whole kernel is built on: `grad(gap)[k] == w_k * n`.

        Both the contact normal *and* the closest-point weights fall out of the distance
        function that detection and the barrier already use -- so no new geometry code, and the
        three cannot drift apart.
        """
        gradient = jax.grad(lambda x: pair_gap_and_mollifier(x, x, jnp.int32(0))[0])
        matrix = np.asarray(gradient(self.start))

        singular = np.linalg.svd(matrix, compute_uv=False)
        # Rank one: every row is parallel to one vector, so a single normal exists.
        self.assertLess(singular[1] / singular[0], 1e-9)
        # Weights sum to zero: the distance is invariant to translating the whole pair.
        self.assertLess(np.abs(matrix.sum(axis=0)).max(), 1e-9)

        normal = matrix[0] / np.linalg.norm(matrix[0])
        weights = matrix @ normal
        np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(
            weights, [1.0, -1 / 3, -1 / 3, -1 / 3], atol=1e-9
        )

    def test_a_rigidly_translating_pair_feels_no_friction(self):
        """The single most important test here.

        It is the only one that separates the correct design from the obvious wrong one. A
        collider is static, so a vertex's slip is its own displacement -- but a mesh pair has
        two moving sides, so slip is the *relative* motion of the two closest points. Copying
        the collider kernel per-vertex would measure absolute displacement and put an enormous
        friction force on a pair merely drifting through space, which is not friction.
        """
        drift = _quad(*([[7.0, -3.0, 2.0]] * 4))
        force = self._force(self.start + drift, self.start)

        sliding = self._force(
            self.start + _quad([1.0, 0.0, 0.0], [0.0] * 3, [0.0] * 3, [0.0] * 3),
            self.start,
        )
        scale = np.abs(sliding).max()
        self.assertGreater(scale, 0.0)  # there IS a real force to be had here...
        self.assertLess(np.abs(force).max(), 1e-6 * scale)  # ...and drifting gets none of it

    def test_no_friction_force_without_slip(self):
        # On the force, not the energy: f0(0) is `eps_v * dt / 3`, not zero.
        force = self._force(self.start, self.start)
        self.assertLess(np.abs(force).max(), 1e-12)

    def test_a_pure_normal_approach_produces_no_friction(self):
        approach = self.start + _quad(
            [0.0, 0.0, -1.0e-3], [0.0] * 3, [0.0] * 3, [0.0] * 3
        )
        force = self._force(approach, self.start)
        self.assertLess(np.abs(force).max(), 1e-9)

    def test_friction_conserves_momentum_and_splits_by_barycentric_weight(self):
        slid = self.start + _quad(
            [1.0e-2, 0.0, 0.0], [0.0] * 3, [0.0] * 3, [0.0] * 3
        )
        force = self._force(slid, self.start)

        # Every force the contact applies has an equal and opposite reaction.
        self.assertLess(np.abs(force.sum(axis=0)).max(), 1e-9)
        # Purely tangential: the friction term must not push along the normal.
        self.assertLess(abs(force[:, 2]).max(), 1e-9)
        # The point is over the centroid, so each triangle vertex takes exactly a third.
        for row in range(1, 4):
            self.assertAlmostEqual(force[row, 0], -force[0, 0] / 3.0, places=9)

    def test_friction_saturates_at_mu_times_the_lagged_normal_force(self):
        from diff_vbd.solver.contact.friction import contact_normal_force

        lagged_gap = _pair_gap(self.start, 0)
        expected = float(self.params.friction_mu) * float(
            contact_normal_force(self.params, jnp.asarray(lagged_gap), TRUE)
        )
        slid = self.start + _quad([1.0, 0.0, 0.0], [0.0] * 3, [0.0] * 3, [0.0] * 3)
        force = self._force(slid, self.start)
        self.assertAlmostEqual(
            float(np.linalg.norm(force[0])), expected, delta=1e-6 * expected
        )

    def test_a_coincident_pair_yields_zero_friction_and_no_nan(self):
        coincident = jnp.zeros((4, 3))
        force = self._force(coincident + 1.0e-3, coincident)
        self.assertTrue(np.all(np.isfinite(force)))
        self.assertLess(np.abs(force).max(), 1e-9)


class PaddedPairTests(unittest.TestCase):
    """C4, at the level that actually matters: the production call path."""

    def setUp(self):
        self.problem, self.state, _ = _two_blocks(gap=0.5e-3, friction_mu=0.3)
        self.params = self.problem.contact.params
        self.dt = self.problem.solver.dt
        # Exactly what detection emits for an unused slot: four coincident points at vertex 0.
        self.padded = jnp.zeros((4, 3))
        self.rest = jnp.zeros((4, 3))

    def test_padding_always_types_a_pair_vertex_triangle(self):
        """An invariant the NaN-safety of every padded slot quietly depends on."""
        problem = vbd.redetect_contacts(self.problem, self.state)
        pair_type = np.asarray(problem.contact.state.pair_type)
        valid = np.asarray(problem.contact.state.pair_valid)
        self.assertTrue(np.all(pair_type[~valid] == 0))

    def test_a_padded_pair_has_zero_energy_and_an_exactly_zero_gradient(self):
        for pair_type in (0, 1):
            with self.subTest(pair_type=pair_type):
                barrier = lambda x: pair_contact_energy(
                    self.params, x, self.rest, jnp.int32(pair_type), FALSE_
                )
                friction = lambda x: pair_friction_energy(
                    self.params,
                    x,
                    self.padded,
                    self.rest,
                    jnp.int32(pair_type),
                    FALSE_,
                    self.dt,
                )
                for energy in (barrier, friction):
                    self.assertEqual(float(energy(self.padded)), 0.0)
                    gradient = np.asarray(jax.grad(energy)(self.padded))
                    # Not merely "not NaN": exactly zero. `0 * inf` is NaN, and masking the
                    # *result* would not have saved us -- the argument had to be safe first.
                    self.assertFalse(np.isnan(gradient).any())
                    np.testing.assert_array_equal(gradient, np.zeros((4, 3)))

    def test_the_gradient_through_the_real_incidence_path_is_not_a_nan(self):
        """The production call. A NaN here poisons the entire solve, silently."""
        problem = vbd.redetect_contacts(self.problem, self.state)
        state = problem.contact.state
        positions = self.state.position

        # Point vertex 0's whole incidence row at padded slots, and mark them live: the
        # masking inside the kernel is all that stands between us and a NaN.
        capacity = int(state.pair_valid.shape[0])
        padded_slot = int(np.flatnonzero(~np.asarray(state.pair_valid))[0])
        self.assertLess(padded_slot, capacity)

        forced = type(state)(
            pair_vertices=state.pair_vertices,
            pair_type=state.pair_type,
            pair_valid=state.pair_valid,
            incident_contacts=state.incident_contacts.at[0].set(padded_slot),
            incident_contact_mask=jnp.ones_like(state.incident_contact_mask).at[0].set(
                True
            ),
        )

        energy = lambda x: incident_pair_energy(
            problem.contact.params,
            forced,
            problem.mesh.rest_positions,
            positions,
            positions,
            problem.solver.dt,
            jnp.int32(0),
            x,
        )
        self.assertEqual(float(energy(positions[0])), 0.0)
        gradient = np.asarray(jax.grad(energy)(positions[0]))
        self.assertFalse(np.isnan(gradient).any())
        np.testing.assert_array_equal(gradient, np.zeros((3,)))


class SelfCollisionDetectionTests(unittest.TestCase):
    def test_a_conforming_block_at_rest_detects_zero_pairs(self):
        """The phantom-contact gate.

        A non-conforming tet split reports interior faces as *surface* faces, and
        self-collision then finds contacts deep inside a solid body at rest -- pushing it apart
        from within. A raw triangle count looks perfectly reasonable while being wrong.
        """
        for size in (2, 3, 5):
            with self.subTest(size=size):
                positions, tets = _block(nx=size, ny=size, nz=size)
                triangles, edges, _ = build_surface_topology(np.asarray(tets))
                _, _, valid = detect_contact_pairs(
                    np.asarray(positions),
                    np.asarray(triangles),
                    np.asarray(edges),
                    d_hat=1.0e-3,
                    capacity=8192,
                )
                self.assertEqual(int(valid.sum()), 0)

    def test_a_non_conforming_mesh_is_rejected_at_assembly(self):
        """The 5-tet cube split only conforms if adjacent cells alternate on a checkerboard."""
        size, spacing = 3, 0.25
        grid = np.stack(
            np.meshgrid(*(np.arange(size),) * 3, indexing="ij"), -1
        )
        positions = grid.reshape(-1, 3).astype(np.float64) * spacing
        index = lambda i, j, k: (i * size + j) * size + k
        tets = []
        for i in range(size - 1):
            for j in range(size - 1):
                for k in range(size - 1):
                    c = [
                        index(i + a, j + b, k + d)
                        for a in (0, 1)
                        for b in (0, 1)
                        for d in (0, 1)
                    ]
                    v000, v001, v010, v011, v100, v101, v110, v111 = c
                    tets += [
                        [v000, v100, v010, v001],
                        [v100, v110, v010, v111],
                        [v100, v101, v001, v111],
                        [v010, v011, v001, v111],
                        [v100, v010, v001, v111],
                    ]

        # Wind every tet positively, so the *volume* validator does not fire first and mask
        # the thing we are actually testing: the split is still non-conforming.
        tets = np.array(tets, dtype=np.int64)
        corners = positions[tets]
        volume = np.einsum(
            "ij,ij->i",
            np.cross(
                corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
            ),
            corners[:, 3] - corners[:, 0],
        )
        flipped = volume < 0.0
        tets[flipped] = tets[flipped][:, [0, 2, 1, 3]]

        with self.assertRaisesRegex(ValueError, "closed surface"):
            assemble_problem(
                jnp.asarray(positions),
                jnp.asarray(tets, dtype=jnp.int32),
                jnp.ones((positions.shape[0],), dtype=jnp.float64),
                self_collision=True,
            )

    def test_two_approaching_blocks_detect_pairs_that_all_straddle_them(self):
        problem, state, n_lower = _two_blocks(gap=0.5e-3)
        problem = vbd.redetect_contacts(problem, state)
        pairs = np.asarray(problem.contact.state.pair_vertices)
        valid = np.asarray(problem.contact.state.pair_valid)

        self.assertGreater(int(valid.sum()), 0)
        for quad in pairs[valid]:
            sides = {bool(v < n_lower) for v in quad}
            # A pair with all four vertices in one body is a phantom intra-block contact.
            self.assertEqual(len(sides), 2)

    def test_an_already_intersecting_mesh_is_rejected(self):
        """E1. The precondition every downstream guarantee silently assumes."""
        problem, state, _ = _two_blocks(gap=0.5e-3)
        overlapped = SimulationState(
            position=state.position.at[8:].add(
                jnp.asarray([0.0, 0.0, -0.5], dtype=state.position.dtype)
            ),
            velocity=state.velocity,
            time=state.time,
        )
        with self.assertRaisesRegex(ValueError, "already intersecting"):
            vbd.redetect_contacts(problem, overlapped)

    def test_incidence_overflow_is_an_error_naming_what_to_raise_it_to(self):
        pair_vertices = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
        pair_valid = np.array([True, True])
        with self.assertRaisesRegex(ValueError, "max_per_vertex"):
            build_contact_incidence(pair_vertices, pair_valid, 8, max_per_vertex=1)


class SelfCollisionTunnellingTests(unittest.TestCase):
    """C1, for real: the guarantee applied to contact *pairs*, not just colliders.

    Both bypass paths are live -- the line search is off (so the raw Newton step is taken
    without ever evaluating the objective) and Chebyshev is on (so the mesh is extrapolated
    after every local solve has finished, where no per-vertex check can see it).

    The negative twin is the point. "It did not penetrate" proves nothing until the same
    scenario *does* penetrate with the mechanism removed.
    """

    def _run(self, *, self_collision_ccd, speed=-20.0, steps=25):
        problem, state, n_lower = _two_blocks(
            velocity=(0.0, 0.0, speed),
            self_collision_ccd=self_collision_ccd,
            line_search_enabled=False,
            acceleration_enabled=True,
        )
        worst = np.inf
        for _ in range(steps):
            state = vbd.step(problem, state)
            position = np.asarray(state.position)
            if not np.isfinite(position).all():
                return -np.inf  # diverged: the guarantee is gone
            separation = (
                position[n_lower:, 2].mean() - position[:n_lower, 2].mean()
            )
            worst = min(worst, float(separation))
        return worst

    def test_self_collision_tunnels_without_the_filter(self):
        """The negative twin. If this ever stops failing, the positive test proves nothing."""
        self.assertLess(self._run(self_collision_ccd=False), 0.0)

    def test_self_collision_holds_with_the_filter(self):
        separation = self._run(self_collision_ccd=True)
        self.assertTrue(np.isfinite(separation))
        self.assertGreater(separation, 0.0)

    def test_no_detected_pair_ever_reaches_zero_distance(self):
        """The guarantee itself, stated in the terms the CCD actually controls."""
        problem, state, _ = _two_blocks(
            velocity=(0.0, 0.0, -20.0),
            line_search_enabled=False,
            acceleration_enabled=True,
        )
        squared = jax.jit(jax.vmap(pair_distance_sq))
        for _ in range(20):
            state = vbd.step(problem, state)
            self.assertTrue(np.isfinite(np.asarray(state.position)).all())

            current = vbd.redetect_contacts(problem, state)
            pair_state = current.contact.state
            valid = np.asarray(pair_state.pair_valid)
            if not valid.any():
                continue
            gaps = np.asarray(
                jnp.sqrt(
                    jnp.maximum(
                        squared(
                            state.position[pair_state.pair_vertices],
                            pair_state.pair_type,
                        ),
                        0.0,
                    )
                )
            )
            self.assertGreater(float(gaps[valid].min()), 0.0)


class CollidingMaskTests(unittest.TestCase):
    def test_a_vertex_in_self_contact_is_flagged_with_no_colliders_present(self):
        problem, state, _ = _two_blocks(gap=0.5e-3)
        problem = vbd.redetect_contacts(problem, state)
        mask = np.asarray(
            colliding_vertex_mask(
                problem.contact.params,
                problem.contact.colliders,
                problem.contact.state,
                state.position,
            )
        )
        self.assertEqual(int(problem.contact.colliders.kind.shape[0]), 0)
        self.assertGreater(int(mask.sum()), 0)

    def test_a_pair_in_the_candidate_list_but_outside_d_hat_does_not_flag(self):
        """The criterion is the true distance, not "appears in the list".

        The candidate band grows with the mesh's speed and is far wider than `d_hat`, so
        flagging everything in the list would disable Chebyshev across half the mesh -- a
        silent performance failure, which is exactly the kind nobody notices.
        """
        # Moving, so the band grows to cover the step and picks these pairs up as *candidates*
        # while they are still far outside the activation distance.
        problem, state, _ = _two_blocks(
            gap=8.0e-3, d_hat=1.0e-3, velocity=(0.0, 0.0, -2.0)
        )
        problem = vbd.redetect_contacts(problem, state)
        self.assertGreater(
            int(np.asarray(problem.contact.state.pair_valid).sum()),
            0,
            "no candidates -- the test would be vacuous",
        )
        mask = np.asarray(
            colliding_vertex_mask(
                problem.contact.params,
                problem.contact.colliders,
                problem.contact.state,
                state.position,
            )
        )
        self.assertEqual(int(mask.sum()), 0)


class ContactRecompilationTests(unittest.TestCase):
    """C3, through the real solver rather than a toy stand-in."""

    def test_a_changing_pair_set_does_not_recompile_the_solver(self):
        # Approaching from clear air, so the pair set genuinely grows from step to step.
        problem, state, _ = _two_blocks(gap=0.05, velocity=(0.0, 0.0, -2.0))

        live = lambda s: int(
            np.asarray(vbd.redetect_contacts(problem, s).contact.state.pair_valid).sum()
        )

        counts = [live(state)]  # out of range at t = 0; in contact a few steps later
        state = vbd.step(problem, state)  # warm the trace cache
        counts.append(live(state))

        before = vbd._advance_step._cache_size()
        for _ in range(6):
            state = vbd.step(problem, state)
            counts.append(live(state))
        after = vbd._advance_step._cache_size()

        # The pair set must actually change across these steps, or the test is vacuous: a
        # buffer whose contents never move could not recompile anything either way.
        self.assertGreater(len(set(counts)), 1, f"pair count never changed: {counts}")
        # A fresh ContactState pytree every step, with fresh contents and identical shapes:
        # a jit cache *hit*, not a recompile of a trace containing nested scans.
        self.assertEqual(after, before)

    def test_a_changing_pair_count_never_changes_a_buffer_shape(self):
        # Approaching from clear air, so the pair set genuinely grows from step to step.
        problem, state, _ = _two_blocks(gap=0.05, velocity=(0.0, 0.0, -2.0))
        shapes = None
        for _ in range(5):
            state = vbd.step(problem, state)
            contact = vbd.redetect_contacts(problem, state).contact.state
            current = tuple(
                leaf.shape
                for leaf in (
                    contact.pair_vertices,
                    contact.pair_type,
                    contact.pair_valid,
                    contact.incident_contacts,
                    contact.incident_contact_mask,
                )
            )
            if shapes is None:
                shapes = current
            self.assertEqual(current, shapes)


class MeshMeshSimulationTests(unittest.TestCase):
    def test_a_block_settles_on_a_block_without_interpenetrating(self):
        """The stacked-block scenario, run to completion.

        Flat-on-flat contact between two coincident faces generates a large edge-edge fan at a
        single vertex, which is what exhausted the per-vertex incidence buffer before.
        """
        problem, state, n_lower = _two_blocks(
            n=3,
            gap=2.0e-3,
            external_acceleration=(0.0, 0.0, -9.81),
            num_iterations=20,
        )
        squared = jax.jit(jax.vmap(pair_distance_sq))
        for _ in range(60):
            state = vbd.step(problem, state)  # must not raise on capacity
        position = np.asarray(state.position)
        self.assertTrue(np.isfinite(position).all())

        # It came to rest ON the lower block, rather than through it.
        self.assertGreater(
            position[n_lower:, 2].mean() - position[:n_lower, 2].mean(), 0.0
        )
        settled = vbd.redetect_contacts(problem, state).contact.state
        valid = np.asarray(settled.pair_valid)
        self.assertGreater(int(valid.sum()), 0, "the blocks never made contact")
        gaps = np.asarray(
            jnp.sqrt(
                jnp.maximum(
                    squared(state.position[settled.pair_vertices], settled.pair_type),
                    0.0,
                )
            )
        )
        self.assertGreater(float(gaps[valid].min()), 0.0)

    def test_mesh_mesh_friction_resists_sliding(self):
        """The physical gate, and the only contact in the system is mesh-mesh.

        No analytic colliders at all, so nothing here can pass on the collider friction path.
        """

        def drift(friction_mu):
            problem, state, n_lower = _two_blocks(
                gap=0.5e-3,
                friction_mu=friction_mu,
                velocity=(1.0, 0.0, 0.0),
                external_acceleration=(0.0, 0.0, -9.81),
            )
            start = float(np.asarray(state.position)[n_lower:, 0].mean())
            for _ in range(60):
                state = vbd.step(problem, state)
            position = np.asarray(state.position)
            self.assertTrue(np.isfinite(position).all())
            return abs(float(position[n_lower:, 0].mean()) - start)

        free = drift(0.0)
        held = drift(1.0)
        self.assertGreater(free, 1.0e-3, "the frictionless block never slid")
        self.assertLess(held, 0.5 * free)


def _initial(problem):
    from diff_vbd import initial_state

    return initial_state(problem)


if __name__ == "__main__":
    unittest.main()
