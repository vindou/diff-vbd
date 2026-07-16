"""Traced static solve and adjoint gates (feature/traced-static).

The traced driver exists for exactly one reason: a host-side Newton loop cannot be
``vmap``ped, so a caller who needs a *batch* of independent static solves (Monte Carlo
over loads, a design sweep) was forced to run them sequentially. These gates therefore
test equivalence and batchability, not new physics:

* **Agreement** -- traced and host drivers produce the same equilibrium on the same
  problem, on a contact-active fixture (``0 < min gap < d_hat`` asserted, else the
  comparison is between two trivial states) and a contact-free one.
* **Batching** -- ``vmap`` over ``StaticParams`` matches B sequential host solves
  elementwise. This is the gate the branch exists for.
* **Adjoint under vmap** -- per-sample implicit gradients against central differences.
* **Refusal under vmap** -- an unconverged element's gradient is NaN while its converged
  neighbours' gradients are untouched. The *asymmetric* case is the whole point: an
  all-unconverged batch would prove nothing about containment.
* **Transcription** -- the traced step function, driven eagerly pass-by-pass against a
  line-for-line replica of the host loop, on the paths healthy fixtures never visit:
  rejected line searches, the damping-retry ladder, and a NaN direction from CG.

Two fixture-design facts, learned the hard way and load-bearing for every tolerance in
this file:

* Both drivers stop at the first iterate whose residual clears ``tol``, and two valid
  stops can sit anywhere below it, so positions may legitimately disagree by
  ~``tol / lambda_min`` -- for this scene about ``5e-10`` at ``tol = 1e-9``, *larger*
  than the 1e-10 agreement gate. The agreement and batching fixtures therefore solve to
  ``1e-11`` (contact) / ``1e-10`` (contact-free, whose measured residual floor is
  ~1e-11): the gate stays 1e-10, the fixtures just earn it honestly.
* Near contact the Newton iteration has zigzag pockets -- parameter values where it
  stalls around ``1e-9`` instead of converging quadratically to ``1e-13`` (FINDINGS.md;
  e.g. host mu=45 stalls at 5.6e-9 while mu=40 and 50 reach 4e-12 in 10 iterations) --
  and *which* pockets stall differs between compilations of the same math. Every fixture
  parameter here is pinned to values verified to converge on both drivers, and every
  test asserts convergence before comparing, so a fixture rotting into a pocket fails
  loudly as a fixture problem, not silently as a wrong gradient.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from test_static import (
    _CENTER,
    _D_HAT,
    _TOL,
    _feasible_initial,
    _min_gap,
    _seated_problem,
)

from diff_vbd import (
    StaticAdjointTraced,
    StaticParams,
    apply_static_params,
    assert_converged,
    solve_static_equilibrium,
    solve_static_equilibrium_traced,
)
from diff_vbd.solver import static as static_module

_TIGHT = 1.0e-11  # agreement/batching solve tolerance; see module docstring
_AGREEMENT_RTOL = 1.0e-10  # the gate
# Adjoint-gate solves (forward and FD probes) run at 1e-10 rather than 1e-9: a probe
# that legitimately stops just under 1e-9 feeds ~tol-scale position noise into a ~2e-8
# central-difference numerator -- 9e-4 relative, over the gate, with the adjoint right
# (FINDINGS.md). One decade of tolerance puts stop noise two decades under the signal.
_ADJOINT_TOL = 1.0e-10

# (mu, collider dz) pairs verified to converge to _TIGHT on BOTH drivers *under this
# exact batch* -- vmap is its own compilation context with its own zigzag pockets (an
# 8-pair subset of this very list stalls one lane that the full batch does not), so the
# verified set is used whole. See the module docstring for why pairs are pinned.
_BATCH_PAIRS = [
    (30.0, 0.000),
    (30.0, 0.002),
    (35.0, 0.002),
    (40.0, 0.002),
    (40.0, 0.004),
    (50.0, 0.000),
    (50.0, 0.004),
    (55.0, 0.002),
    (55.0, 0.004),
    (60.0, 0.000),
    (65.0, 0.000),
    (65.0, 0.002),
    (65.0, 0.004),
    (70.0, 0.000),
    (70.0, 0.004),
]


def _relative_difference(candidate, reference):
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def _center_at(dz):
    center = np.array(_CENTER, dtype=np.float64).copy()
    center[2] += dz
    return center[None, :]


class TracedAgreementTests(unittest.TestCase):
    """Same problem, both drivers, positions within 1e-10 relative."""

    @classmethod
    def setUpClass(cls):
        cls.problem, cls.positions = _seated_problem()
        cls.initial = _feasible_initial(cls.positions)

    def test_agreement_with_contact_active(self):
        host = solve_static_equilibrium(
            self.problem, tol=_TIGHT, initial_position=self.initial, max_iterations=200
        )
        traced = solve_static_equilibrium_traced(
            self.problem, tol=_TIGHT, initial_position=self.initial, max_iterations=200
        )
        self.assertTrue(bool(host.converged))
        self.assertTrue(bool(traced.converged))
        # Without an active barrier this would compare two trivial states.
        for result in (host, traced):
            gap = _min_gap(self.problem, result.position)
            self.assertGreater(gap, 0.0)
            self.assertLess(gap, _D_HAT)
        self.assertLessEqual(
            _relative_difference(traced.position, host.position), _AGREEMENT_RTOL
        )

    def test_agreement_without_contact(self):
        problem, _ = _seated_problem(colliders=None, contact_enabled=False)
        # 1e-10, not _TIGHT: this scene's measured residual floor is ~1e-11, and a
        # fixture solving into its own roundoff floor stalls unconverged (docstring).
        host = solve_static_equilibrium(problem, tol=1.0e-10, max_iterations=200)
        traced = solve_static_equilibrium_traced(problem, tol=1.0e-10, max_iterations=200)
        self.assertTrue(bool(host.converged))
        self.assertTrue(bool(traced.converged))
        self.assertLessEqual(
            _relative_difference(traced.position, host.position), _AGREEMENT_RTOL
        )


class TracedBatchingTests(unittest.TestCase):
    """vmap over StaticParams == B sequential host solves, elementwise.

    The batch is deliberately heterogeneous (10-14 Newton iterations across elements) so
    early-converging elements spend real loop trips masked while their neighbours still
    run -- the case where a masking bug in the carry (a damping raise or a counter tick
    leaking into a frozen element) would move an answer.
    """

    def test_vmap_matches_sequential_host_solves(self):
        template, positions = _seated_problem()
        initial = _feasible_initial(positions)
        mus = jnp.asarray([mu for mu, _ in _BATCH_PAIRS])
        centers = jnp.asarray(np.stack([_center_at(dz) for _, dz in _BATCH_PAIRS]))
        batch = StaticParams(mu=mus, collider_center=centers)

        def solve_one(params):
            problem = apply_static_params(template, params)
            return solve_static_equilibrium_traced(
                problem, tol=_TIGHT, initial_position=initial, max_iterations=200
            )

        batched = jax.vmap(solve_one)(batch)

        for index, (mu, dz) in enumerate(_BATCH_PAIRS):
            with self.subTest(mu=mu, dz=dz):
                self.assertTrue(bool(batched.converged[index]))
                single = StaticParams(
                    mu=jnp.asarray(mu), collider_center=jnp.asarray(_center_at(dz))
                )
                problem = apply_static_params(template, single)
                host = solve_static_equilibrium(
                    problem, tol=_TIGHT, initial_position=initial, max_iterations=200
                )
                self.assertTrue(bool(host.converged))
                gap = _min_gap(problem, batched.position[index])
                self.assertGreater(gap, 0.0)
                self.assertLess(gap, _D_HAT)
                self.assertLessEqual(
                    _relative_difference(batched.position[index], host.position),
                    _AGREEMENT_RTOL,
                )


class TracedAdjointUnderVmapTests(unittest.TestCase):
    """Per-sample du*/dtheta under vmap against central differences, 1e-4 relative.

    The differences are taken through the *traced* solve -- the adjoint's own forward
    operator -- at the adjoint's tolerance; every probe solve asserts convergence so a
    zigzag-pocket stall reads as a fixture failure, not a gradient error. mu and
    collider radius cover a material and a contact parameter per the brief;
    rest_positions is the design-parameter path and goes through the lumped-mass
    rebuild in apply_static_params.
    """

    @classmethod
    def setUpClass(cls):
        cls.template, cls.positions = _seated_problem()
        cls.initial = _feasible_initial(cls.positions)
        cls.adjoint = StaticAdjointTraced(
            cls.template,
            tol=_ADJOINT_TOL,
            initial_position=cls.initial,
            max_iterations=200,
        )
        cls.weights = jnp.asarray(
            np.random.default_rng(0).normal(size=cls.positions.shape)
        )

    def _loss(self, params):
        return jnp.sum(self.weights * self.adjoint(params))

    def _fd_loss(self, params):
        """The same loss through a fresh forward solve, for the difference quotient."""
        problem = apply_static_params(self.template, params)
        result = solve_static_equilibrium_traced(
            problem, tol=_ADJOINT_TOL, initial_position=self.initial, max_iterations=200
        )
        self.assertTrue(
            bool(result.converged),
            msg="an FD probe solve stalled; fix the fixture, do not loosen the gate",
        )
        return float(jnp.sum(self.weights * result.position))

    def _check_batch(self, make_params, values, h):
        gradients = jax.vmap(
            jax.grad(lambda v: self._loss(make_params(v)))
        )(jnp.asarray(values))
        for index, value in enumerate(values):
            with self.subTest(value=value):
                numeric = (
                    self._fd_loss(make_params(jnp.asarray(value + h)))
                    - self._fd_loss(make_params(jnp.asarray(value - h)))
                ) / (2.0 * h)
                self.assertGreater(abs(numeric), 0.0)
                gradient = float(gradients[index])
                self.assertLessEqual(
                    abs(gradient - numeric),
                    1.0e-4 * abs(numeric),
                    msg=f"vmapped adjoint {gradient} vs FD {numeric}",
                )

    def test_barrier_is_active_at_the_gate_scene(self):
        """Collider sensitivities at a contact-free state are identically zero, so the
        radius check below would compare noise against noise without this."""
        result = self.adjoint.solve_result(StaticParams(mu=jnp.asarray(50.0)))
        self.assertTrue(bool(result.converged))
        gap = _min_gap(self.template, result.position)
        self.assertGreater(gap, 0.0)
        self.assertLess(gap, _D_HAT)

    def test_material_mu(self):
        # 65, not 60: the mu = 60 - 1e-4 FD probe sits in a traced-driver zigzag
        # pocket at this tolerance (module docstring) and stalls.
        self._check_batch(lambda v: StaticParams(mu=v), [50.0, 65.0], 1e-4)

    def test_collider_radius(self):
        self._check_batch(
            lambda v: StaticParams(collider_radius=jnp.reshape(v, (1,))),
            [1.0, 0.995],
            1e-6,
        )

    def test_rest_positions_directional(self):
        """Directional derivative along a fixed perturbation of the rest shape, one
        scale per batch element (the full Jacobian would need 3N solves for its FD
        twin). The direction leaves the clamped base untouched. Scales pinned to
        (0.004, 0.005): 0.002 and 0.003 sit in vmapped-forward zigzag pockets at this
        tolerance, and their stalled solves NaN-poison the gradient -- the refusal
        doing its job on a bad fixture (module docstring)."""
        rest = np.asarray(self.positions)
        rng = np.random.default_rng(1)
        direction = rng.normal(size=rest.shape) * 0.01
        direction[rest[:, 2] <= 1e-9] = 0.0
        direction = jnp.asarray(direction)
        rest = jnp.asarray(rest)

        self._check_batch(
            lambda s: StaticParams(rest_positions=rest + s * direction),
            [0.004, 0.005],
            1e-5,
        )


class TracedRefusalTests(unittest.TestCase):
    """The refusal, surviving vmap: unconverged gradients are NaN, per element.

    The host adjoint raises; per-sample control flow does not exist under vmap, so the
    traced adjoint poisons instead. The load-bearing case is the *mixed* batch: one
    element unconverged, its neighbour untouched -- an all-unconverged batch would pass
    even if the poison leaked across elements.
    """

    @classmethod
    def setUpClass(cls):
        cls.template, cls.positions = _seated_problem()
        cls.initial = _feasible_initial(cls.positions)
        cls.weights = jnp.asarray(
            np.random.default_rng(0).normal(size=cls.positions.shape)
        )
        # Element 0 converges in ~11 iterations. Element 1 lowers the sphere 5mm, which
        # puts the shared initial guess inside the barrier's linear continuation; the
        # solve un-penetrates but then zigzags (residual ~0.7 after 200 host iterations),
        # so at max_iterations=25 it is genuinely, reproducibly unconverged.
        cls.centers = np.stack([_center_at(0.0), _center_at(-0.005)])
        cls.adjoint = StaticAdjointTraced(
            cls.template,
            tol=_TOL,
            initial_position=cls.initial,
            max_iterations=25,
        )

    def _loss(self, center):
        return jnp.sum(
            self.weights * self.adjoint(StaticParams(collider_center=center))
        )

    def test_one_unconverged_element_poisons_only_itself(self):
        batched = jax.vmap(self.adjoint.solve_result)(
            StaticParams(collider_center=jnp.asarray(self.centers))
        )
        converged = np.asarray(batched.converged)
        # The asymmetry is the precondition: if both elements converged (or neither),
        # this test would prove nothing about per-element containment.
        self.assertTrue(bool(converged[0]))
        self.assertFalse(bool(converged[1]))

        gradients = jax.vmap(jax.grad(self._loss))(jnp.asarray(self.centers))
        poisoned = np.asarray(gradients[1])
        healthy = np.asarray(gradients[0])
        self.assertTrue(np.all(np.isnan(poisoned)))
        self.assertTrue(np.all(np.isfinite(healthy)))

        # "Unaffected" means equal to the same element's gradient solved alone, not
        # merely finite. 1e-4, not machine precision: the backward pass solves
        # H lambda = w by CG at jax.scipy's default 1e-5 relative tolerance, and the
        # batched and solo programs are different compilations of it, so their lambdas
        # legitimately differ by ~1e-5 (measured 4e-5 here) -- the same noise class the
        # FD gates bound. A poison leak or a masking bug is NaN or a wrong solve, both
        # far outside this.
        solo = np.asarray(jax.grad(self._loss)(jnp.asarray(self.centers[0])))
        np.testing.assert_allclose(healthy, solo, rtol=1e-4)

    def test_single_unconverged_solve_poisons(self):
        """The traced twin of the host adjoint's raise test: one Newton iteration
        cannot reach 1e-30, and the gradient of that state must be NaN, not plausible."""
        stalled = StaticAdjointTraced(
            self.template,
            tol=1e-30,
            initial_position=self.initial,
            max_iterations=1,
        )
        gradient = jax.grad(
            lambda v: jnp.sum(self.weights * stalled(StaticParams(mu=v)))
        )(jnp.asarray(50.0))
        self.assertTrue(bool(jnp.isnan(gradient)))

    def test_assert_converged_is_the_loud_version(self):
        batched = jax.vmap(self.adjoint.solve_result)(
            StaticParams(collider_center=jnp.asarray(self.centers))
        )
        with self.assertRaisesRegex(ValueError, "plausible-looking gradient"):
            assert_converged(batched, tol=_TOL)

        healthy = self.adjoint.solve_result(
            StaticParams(collider_center=jnp.asarray(self.centers[0]))
        )
        assert_converged(healthy, tol=_TOL)  # converged: silent


class TracedTranscriptionTests(unittest.TestCase):
    """Decision-for-decision equality of ``_traced_newton_step`` with the host loop.

    The equivalence gates above only ever visit the smooth path (alpha = 1 accepts,
    plus the zigzag's timid accepts): a healthy fixture never rejects a line search, so
    the hardest part of the transcription -- the rejected-search retry, the damping
    ladder, and the host loop's NaN-slope behaviour -- would otherwise ship ungated.
    This class drives the module-level step function *eagerly*, pass by pass, against a
    replica of ``solve_static_equilibrium``'s loop transcribed line-for-line below, and
    forces the rare branches by wrapping ``_newton_direction``. Both sides call the
    same jitted kernels on the same inputs, so agreement is *bitwise* -- any drift is a
    transcription defect, not compilation noise (contrast the module docstring's
    pocket story, which is about two different compilations).

    If this class fails after a deliberate host-loop change, update the replica to
    match the source -- keeping them in sync is the price of gating the transcription.
    """

    #: passes are normalised to the traced accounting: one entry per executed body
    #: pass, (accepted, damping_after, newton_iters_after, stagnated_after).
    _MAX_PASSES = 25

    @classmethod
    def setUpClass(cls):
        cls.problem, cls.positions = _seated_problem()
        cls.initial = _feasible_initial(cls.positions)
        from diff_vbd.setup.boundary_conditions import evaluate_dirichlet_targets

        prescribed, _ = evaluate_dirichlet_targets(
            cls.problem.boundary_conditions, 0.0, float(cls.problem.solver.dt)
        )
        dirichlet = np.flatnonzero(
            np.asarray(cls.problem.boundary_conditions.dirichlet_mask, dtype=bool)
        )
        cls.free = jnp.asarray(static_module._free_indices(cls.problem), dtype=jnp.int32)
        cls.pinned = (
            jnp.asarray(cls.initial)
            .at[dirichlet]
            .set(jnp.asarray(prescribed)[dirichlet])
        )

    # -- the two drivers ---------------------------------------------------------------

    def _run_host_replica(self, problem, pinned, tol):
        """solve_static_equilibrium's loop, transcribed with decision recording.

        Differences from the source are bookkeeping only: the two `break`s record a
        final pass entry with stagnated=True (the traced carry's accounting), because
        a while_loop body has no mid-body exit.
        """
        free = self.free
        u = pinned[free]
        gradient = static_module._gradient(problem, pinned, free, u)
        residual = float(jnp.linalg.norm(gradient))
        initial_residual = max(residual, 1e-300)
        damping_rel, iterations, passes = 0.0, 0, 0
        trace, states = [], []
        while residual > tol and iterations < 200 and passes < self._MAX_PASSES:
            forcing = min(0.5, float(np.sqrt(residual / initial_residual)))
            curvature_scale = float(
                jnp.linalg.norm(static_module._hvp_at(problem, pinned, free, u, gradient))
            ) / max(residual, 1e-300)
            direction = static_module._newton_direction(
                problem, pinned, free, u, gradient,
                jnp.asarray(forcing), jnp.asarray(damping_rel * curvature_scale),
                maxiter=250,
            )
            if float(jnp.sum(direction * gradient)) >= 0.0:
                direction = -gradient
            direction = static_module._clip_direction_to_colliders(
                problem, pinned, free, u, direction
            )
            if float(jnp.sum(direction * gradient)) >= 0.0:
                direction = static_module._clip_direction_to_colliders(
                    problem, pinned, free, u, -gradient
                )
            current = float(static_module._objective(problem, pinned, free, u))
            slope = float(jnp.sum(direction * gradient))
            passes += 1
            if slope >= 0.0:
                trace.append((False, damping_rel, iterations, True))
                states.append(np.asarray(u))
                break
            alpha, accepted = 1.0, False
            for _ in range(40):
                candidate = u + alpha * direction
                if float(static_module._objective(problem, pinned, free, candidate)) <= (
                    current + 1.0e-4 * alpha * slope
                ):
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                if damping_rel < 1.0e6:
                    damping_rel = max(10.0 * damping_rel, 1.0e-4)
                    trace.append((False, damping_rel, iterations, False))
                    states.append(np.asarray(u))
                    continue
                trace.append((False, damping_rel, iterations, True))
                states.append(np.asarray(u))
                break
            if alpha >= 0.5:
                damping_rel /= 3.0
                if damping_rel < 1.0e-7:
                    damping_rel = 0.0
            else:
                damping_rel = max(10.0 * damping_rel, 1.0e-4)
            u = candidate
            gradient = static_module._gradient(problem, pinned, free, u)
            residual = float(jnp.linalg.norm(gradient))
            iterations += 1
            trace.append((True, damping_rel, iterations, False))
            states.append(np.asarray(u))
        return trace, states, residual

    def _run_traced_step(self, problem, pinned, tol):
        free = self.free
        u0 = pinned[free]
        gradient0 = static_module._gradient(problem, pinned, free, u0)
        residual0 = jnp.linalg.norm(gradient0)
        initial_residual = jnp.maximum(residual0, 1e-300)
        carry = (
            u0,
            gradient0,
            residual0,
            jnp.zeros((), dtype=u0.dtype),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=bool),
        )
        trace, states = [], []
        while (
            float(carry[2]) > tol
            and int(carry[4]) < 200
            and int(carry[5]) < self._MAX_PASSES
            and not bool(carry[6])
        ):
            previous_iterations = int(carry[4])
            carry = static_module._traced_newton_step(
                problem, pinned, free, initial_residual, 250, carry
            )
            trace.append(
                (
                    int(carry[4]) > previous_iterations,
                    float(carry[3]),
                    int(carry[4]),
                    bool(carry[6]),
                )
            )
            states.append(np.asarray(carry[0]))
        return trace, states, float(carry[2])

    def _assert_drivers_agree(self, wrapper_factory=None, problem=None):
        """Run both drivers (optionally with _newton_direction wrapped identically)
        and demand bitwise-equal decisions and iterates, pass for pass."""
        problem = self.problem if problem is None else problem
        original = static_module._newton_direction
        results = []
        for run in (self._run_host_replica, self._run_traced_step):
            if wrapper_factory is not None:
                static_module._newton_direction = wrapper_factory(original)
            try:
                results.append(run(problem, self.pinned, _TOL))
            finally:
                static_module._newton_direction = original
        (host_trace, host_states, host_residual) = results[0]
        (step_trace, step_states, step_residual) = results[1]
        self.assertEqual(host_trace, step_trace)
        self.assertEqual(len(host_states), len(step_states))
        for index, (host_u, step_u) in enumerate(zip(host_states, step_states)):
            np.testing.assert_array_equal(
                host_u, step_u, err_msg=f"iterates diverged at pass {index}"
            )
        self.assertEqual(host_residual, step_residual)
        return host_trace

    # -- scenarios ---------------------------------------------------------------------

    def test_smooth_path(self):
        """Sanity anchor: the all-accepts trajectory, alpha = 1 throughout."""
        trace = self._assert_drivers_agree()
        self.assertTrue(all(accepted for accepted, *_ in trace))

    def test_zigzag_timid_accepts_raise_the_shift(self):
        """The barrier's zigzag mode (sphere lowered 5mm, penetrating start): accepted
        steps with alpha < 1/2, which must *raise* the shift on both drivers -- the
        select arm no smooth fixture ever exercises."""
        center = np.array(_CENTER, dtype=np.float64)
        center[2] -= 0.005
        problem = apply_static_params(
            self.problem, StaticParams(collider_center=jnp.asarray(center[None, :]))
        )
        trace = self._assert_drivers_agree(problem=problem)
        timid = [
            entry
            for previous, entry in zip([(False, 0.0, 0, False)] + trace, trace)
            if entry[0] and entry[1] > previous[1]
        ]
        self.assertGreater(len(timid), 0, msg="no timid accepts; fixture lost its zigzag")

    def test_forced_rejections_walk_the_damping_ladder(self):
        """The retry: hand back ``-gradient * 2^60`` for two calls, so every Armijo
        trial overshoots astronomically and the line search rejects outright. On the
        *contact-free* problem, deliberately: with a collider present the clip caps
        every vertex at ~its own gap regardless of the pre-scale -- that is its job --
        and the clipped step becomes sane enough that Armijo accepts (two earlier
        versions of this test, one scaling the CG direction and one scaling the
        gradient, were both neutralized that way). The host loop must raise the shift
        and retry the same iterate; the traced step's masked non-update must do
        exactly the same, including not counting the pass as a Newton iteration."""

        def factory(original):
            calls = {"n": 0}

            def wrapper(problem, pinned, free, u, gradient, cg_tol, damping, maxiter):
                calls["n"] += 1
                if calls["n"] <= 2:
                    return -gradient * (2.0**60)
                return original(
                    problem, pinned, free, u, gradient, cg_tol, damping, maxiter=maxiter
                )

            return wrapper

        no_contact, _ = _seated_problem(colliders=None, contact_enabled=False)
        trace = self._assert_drivers_agree(factory, problem=no_contact)
        rejected = [entry for entry in trace if not entry[0] and not entry[3]]
        self.assertGreaterEqual(len(rejected), 2, msg="the forced rejections never fired")
        # The ladder: first rejection lifts the shift from 0 to the floor, the second
        # multiplies it by 10 -- and neither counts a Newton iteration.
        self.assertEqual(rejected[0][1], 1.0e-4)
        self.assertEqual(rejected[1][1], 1.0e-3)
        self.assertEqual(rejected[0][2], rejected[1][2])
        self.assertTrue(any(entry[0] for entry in trace), msg="never recovered")

    def test_nan_direction_retries_like_the_host(self):
        """The review catch, pinned: a NaN direction (CG breakdown) gives a NaN slope,
        which is NOT the host loop's `slope >= 0` break -- the host runs the ladder,
        fails every trial, raises the shift, and typically recovers next pass. The
        traced step must classify it as a retry, not terminal stagnation."""

        def factory(original):
            calls = {"n": 0}

            def wrapper(problem, pinned, free, u, gradient, cg_tol, damping, maxiter):
                calls["n"] += 1
                direction = original(
                    problem, pinned, free, u, gradient, cg_tol, damping, maxiter=maxiter
                )
                if calls["n"] == 1:
                    return direction * jnp.nan
                return direction

            return wrapper

        trace = self._assert_drivers_agree(factory)
        accepted, damping_after, iterations_after, stagnated = trace[0]
        self.assertFalse(accepted)
        self.assertFalse(stagnated, msg="NaN slope must retry, not stagnate")
        self.assertEqual(damping_after, 1.0e-4)
        self.assertEqual(iterations_after, 0)
        self.assertTrue(any(entry[0] for entry in trace[1:]), msg="never recovered")


if __name__ == "__main__":
    unittest.main()
