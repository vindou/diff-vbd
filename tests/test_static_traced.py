"""Traced static solve and adjoint gates (feature/traced-static).

The traced driver exists for exactly one reason: a host-side Newton loop cannot be
``vmap``ped, so a caller who needs a *batch* of independent static solves (Monte Carlo
over loads, a design sweep) was forced to run them sequentially. These gates therefore
test equivalence and batchability, not new physics:

* **Agreement** -- traced and host drivers produce the same equilibrium wherever both
  converge, on a contact-active parameter sweep (``0 < min gap < d_hat`` asserted per
  survivor, else the comparison is between trivial states) and a contact-free one.
* **Batching** -- ``vmap`` over the sweep matches sequential host solves elementwise on
  every lane where both converged. This is the gate the branch exists for.
* **Adjoint under vmap** -- per-sample implicit gradients against central differences,
  on every swept sample whose forward and probe solves all converged.
* **Refusal under vmap** -- an unconverged element's gradient is NaN while its converged
  neighbours' gradients are untouched. The asymmetry is *constructed*, not harvested.
* **Transcription** -- the traced step function, driven eagerly pass-by-pass against a
  line-for-line replica of the host loop, on the paths healthy fixtures never visit:
  rejected line searches, the damping ladder, timid accepts, a NaN direction from CG.

Three fixture-design facts, learned the hard way and load-bearing for every tolerance
and every sweep in this file:

* Both drivers stop at the first iterate whose residual clears ``tol``, and two valid
  stops can sit anywhere below it, so positions may legitimately disagree by
  ~``tol / lambda_min`` -- for this scene about ``5e-10`` at ``tol = 1e-9``, *larger*
  than the 1e-10 agreement gate. The agreement and batching sweeps therefore solve to
  ``1e-11`` (contact) / ``1e-10`` (contact-free, whose measured residual floor is
  ~1e-11): the gate stays 1e-10, the fixtures just earn it honestly.
* Near contact the Newton iteration has zigzag stall pockets -- parameter values where
  it stalls around ``1e-9`` instead of converging quadratically (FINDINGS.md) -- and
  *which* points stall differs between compilations of the same math: host vs traced,
  eager vs vmap, and -- the fact that killed this file's first design -- **machine vs
  machine**. A version of these gates pinned to parameters "verified to converge" on
  the authoring laptop failed 11 of 23 on an x86_64 Linux box with the same JAX
  version, every failure a convergence precondition on the *unchanged host driver*.
  Pinning parameters pins the machine. So these gates sweep instead: run both drivers
  over a grid, keep the points where **both** ``converged`` certificates hold, compare
  the survivors, and demand a minimum survivor count. That is portable by construction
  and *stronger* than the pinned claim -- equivalence is asserted everywhere it is
  defined, and a platform where too few points converge fails loudly about the solver
  rather than silently about the fixture.
* The rare loop branches (rejections, timid accepts, NaN directions) cannot be swept
  into existence -- healthy solves never visit them -- so the transcription tests
  *force* them, deterministically, through a wrapped ``_newton_direction``, and where a
  forcing scale is machine-dependent it is found at runtime by direct evaluation on
  the same machine that runs the assertion.
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

# The equivalence sweep: a grid over the two parameters the seated scene is most
# sensitive to. 27 points; on the authoring machine 15 survive both drivers at 1e-11,
# so a floor of 8 leaves room for pocket geography to move between machines while
# still failing loudly if half the grid stops converging anywhere.
_SWEEP_MUS = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
_SWEEP_DZS = [0.0, 0.002, 0.004]
_SWEEP_TOL = 1.0e-11
_SWEEP_MAX_ITERATIONS = 60  # healthy points need <= 14; stalled points plateau long before 60
_MIN_SURVIVORS = 8

_AGREEMENT_RTOL = 1.0e-10  # the gate

# Adjoint-gate solves (forward and FD probes) run at 1e-10 rather than 1e-9: a probe
# that legitimately stops just under 1e-9 feeds ~tol-scale position noise into a ~2e-8
# central-difference numerator -- 9e-4 relative, over the gate, with the adjoint right
# (FINDINGS.md). One decade of tolerance puts stop noise two decades under the signal.
_ADJOINT_TOL = 1.0e-10


def _relative_difference(candidate, reference):
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def _center_at(dz):
    center = np.array(_CENTER, dtype=np.float64).copy()
    center[2] += dz
    return center[None, :]


def _sweep_params():
    return [
        StaticParams(mu=jnp.asarray(mu), collider_center=jnp.asarray(_center_at(dz)))
        for mu in _SWEEP_MUS
        for dz in _SWEEP_DZS
    ]


def _sweep_batch():
    mus = jnp.asarray([mu for mu in _SWEEP_MUS for _ in _SWEEP_DZS])
    centers = jnp.asarray(
        np.stack([_center_at(dz) for _ in _SWEEP_MUS for dz in _SWEEP_DZS])
    )
    return StaticParams(mu=mus, collider_center=centers)


class SweptEquivalenceTests(unittest.TestCase):
    """Traced == host on every sweep point where both converge, eager and vmapped.

    The host sweep runs once here and feeds both tests. Survivor filtering uses only
    the ``converged`` certificates the solvers already return -- no tolerance is
    loosened, no point is special-cased, and a platform where fewer than
    ``_MIN_SURVIVORS`` of the 27 points converge on both drivers fails loudly.
    """

    @classmethod
    def setUpClass(cls):
        cls.template, cls.positions = _seated_problem()
        cls.initial = _feasible_initial(cls.positions)
        cls.host_results = []
        for params in _sweep_params():
            problem = apply_static_params(cls.template, params)
            cls.host_results.append(
                (
                    problem,
                    solve_static_equilibrium(
                        problem,
                        tol=_SWEEP_TOL,
                        initial_position=cls.initial,
                        max_iterations=_SWEEP_MAX_ITERATIONS,
                    ),
                )
            )

    def _assert_survivor(self, problem, traced_position, host_result):
        gap = _min_gap(problem, traced_position)
        self.assertGreater(gap, 0.0)
        self.assertLess(gap, _D_HAT)
        self.assertLessEqual(
            _relative_difference(traced_position, host_result.position),
            _AGREEMENT_RTOL,
        )

    def test_agreement_eager(self):
        survivors = 0
        for params, (problem, host) in zip(_sweep_params(), self.host_results):
            traced = solve_static_equilibrium_traced(
                problem,
                tol=_SWEEP_TOL,
                initial_position=self.initial,
                max_iterations=_SWEEP_MAX_ITERATIONS,
            )
            if not (bool(host.converged) and bool(traced.converged)):
                continue
            with self.subTest(mu=float(np.asarray(params.mu))):
                self._assert_survivor(problem, traced.position, host)
            survivors += 1
        self.assertGreaterEqual(
            survivors,
            _MIN_SURVIVORS,
            msg="too few sweep points converge on both drivers: a solver problem, "
            "not a fixture one -- investigate before touching this floor",
        )

    def test_batching_vmap_matches_sequential_host(self):
        template, initial = self.template, self.initial

        def solve_one(params):
            problem = apply_static_params(template, params)
            return solve_static_equilibrium_traced(
                problem,
                tol=_SWEEP_TOL,
                initial_position=initial,
                max_iterations=_SWEEP_MAX_ITERATIONS,
            )

        batched = jax.vmap(solve_one)(_sweep_batch())
        survivors = 0
        for index, (problem, host) in enumerate(self.host_results):
            if not (bool(host.converged) and bool(batched.converged[index])):
                continue
            with self.subTest(lane=index):
                self._assert_survivor(problem, batched.position[index], host)
            survivors += 1
        self.assertGreaterEqual(
            survivors,
            _MIN_SURVIVORS,
            msg="too few vmap lanes converge alongside their host twins",
        )

    def test_agreement_eager_preconditioned(self):
        """Block-Jacobi changes CG's iterates, not the equilibrium: wherever the
        preconditioned traced solve and the plain host solve both converge, they must
        land on the same positions. This is the gate that decides whether the
        preconditioner changed the math or only the conditioning."""
        survivors = 0
        for problem, host in self.host_results:
            traced = solve_static_equilibrium_traced(
                problem,
                tol=_SWEEP_TOL,
                initial_position=self.initial,
                max_iterations=_SWEEP_MAX_ITERATIONS,
                preconditioner="block_jacobi",
            )
            if not (bool(host.converged) and bool(traced.converged)):
                continue
            with self.subTest(survivor=survivors):
                self._assert_survivor(problem, traced.position, host)
            survivors += 1
        self.assertGreaterEqual(
            survivors,
            _MIN_SURVIVORS,
            msg="too few sweep points converge preconditioned alongside the host",
        )


class SweptNoContactAgreementTests(unittest.TestCase):
    """The contact-free twin, swept over mu at this scene's honest tolerance."""

    def test_agreement(self):
        problem_base, _ = _seated_problem(colliders=None, contact_enabled=False)
        survivors = 0
        for mu in [40.0, 50.0, 60.0, 70.0]:
            problem = apply_static_params(problem_base, StaticParams(mu=jnp.asarray(mu)))
            host = solve_static_equilibrium(problem, tol=1.0e-10, max_iterations=60)
            traced = solve_static_equilibrium_traced(
                problem, tol=1.0e-10, max_iterations=60
            )
            if not (bool(host.converged) and bool(traced.converged)):
                continue
            with self.subTest(mu=mu):
                self.assertLessEqual(
                    _relative_difference(traced.position, host.position),
                    _AGREEMENT_RTOL,
                )
            survivors += 1
        self.assertGreaterEqual(survivors, 2)


class SweptAdjointUnderVmapTests(unittest.TestCase):
    """Per-sample du*/dtheta under vmap against central differences, 1e-4 relative.

    Swept, filtered on certificates: a sample survives if its vmapped gradient is
    finite (a stalled forward would be NaN-poisoned -- the refusal doubling as the
    filter) and both its FD probe solves converged. Every survivor must pass the gate
    and carry an active barrier; the sweep must produce a minimum number of survivors.
    The differences go through the *traced* solve -- the adjoint's own forward
    operator -- at the adjoint's tolerance.
    """

    @classmethod
    def setUpClass(cls):
        cls.template, cls.positions = _seated_problem()
        cls.initial = _feasible_initial(cls.positions)
        cls.adjoint = StaticAdjointTraced(
            cls.template,
            tol=_ADJOINT_TOL,
            initial_position=cls.initial,
            max_iterations=_SWEEP_MAX_ITERATIONS,
        )
        cls.preconditioned_adjoint = StaticAdjointTraced(
            cls.template,
            tol=_ADJOINT_TOL,
            initial_position=cls.initial,
            max_iterations=_SWEEP_MAX_ITERATIONS,
            preconditioner="block_jacobi",
        )
        cls.weights = jnp.asarray(
            np.random.default_rng(0).normal(size=cls.positions.shape)
        )

    def _fd_solve(self, params, preconditioner="none"):
        problem = apply_static_params(self.template, params)
        return solve_static_equilibrium_traced(
            problem,
            tol=_ADJOINT_TOL,
            initial_position=self.initial,
            max_iterations=_SWEEP_MAX_ITERATIONS,
            preconditioner=preconditioner,
        )

    def _check_swept(
        self,
        make_params,
        values,
        h,
        min_survivors,
        contact_active=True,
        preconditioner="none",
    ):
        adjoint = (
            self.preconditioned_adjoint
            if preconditioner == "block_jacobi"
            else self.adjoint
        )
        gradients = jax.vmap(
            jax.grad(
                lambda v: jnp.sum(self.weights * adjoint(make_params(v)))
            )
        )(jnp.asarray(values))
        survivors = 0
        for index, value in enumerate(values):
            gradient = float(gradients[index])
            if not np.isfinite(gradient):
                continue  # forward stalled under the vmapped program: not a survivor
            plus = self._fd_solve(make_params(jnp.asarray(value + h)), preconditioner)
            minus = self._fd_solve(make_params(jnp.asarray(value - h)), preconditioner)
            if not (bool(plus.converged) and bool(minus.converged)):
                continue
            with self.subTest(value=value):
                if contact_active:
                    # At a contact-free state every collider sensitivity is
                    # identically zero and the check would compare noise with noise.
                    problem = apply_static_params(
                        self.template, make_params(jnp.asarray(value))
                    )
                    gap = _min_gap(problem, plus.position)
                    self.assertGreater(gap, 0.0)
                    self.assertLess(gap, _D_HAT)
                numeric = (
                    float(jnp.sum(self.weights * plus.position))
                    - float(jnp.sum(self.weights * minus.position))
                ) / (2.0 * h)
                self.assertGreater(abs(numeric), 0.0)
                self.assertLessEqual(
                    abs(gradient - numeric),
                    1.0e-4 * abs(numeric),
                    msg=f"vmapped adjoint {gradient} vs FD {numeric}",
                )
            survivors += 1
        self.assertGreaterEqual(
            survivors,
            min_survivors,
            msg="too few adjoint sweep samples survived their convergence filters",
        )

    def test_material_mu(self):
        # h = 1e-3, not the host gates' 1e-4: the FD probes stop anywhere below
        # _ADJOINT_TOL, feeding ~1e-11 of stop noise into the loss difference, and at
        # the sweep's weak end (dL/dmu ~ 2e-4 at mu=40) a 2h = 2e-4 step leaves that
        # noise a quarter of the 1e-4 gate. Ten times the step is still only 2.5e-5
        # of mu -- truncation stays negligible -- and puts the noise two decades under
        # the gate across the whole sweep, not just at the strongest sample.
        self._check_swept(
            lambda v: StaticParams(mu=v),
            [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
            1e-3,
            min_survivors=3,
        )

    def test_material_mu_preconditioned(self):
        """The same sweep through the block-Jacobi adjoint: the preconditioned
        forward's u* and the preconditioned backward CG's lambda must produce the
        same implicit gradient the finite differences see. Radius and rest positions
        go through the identical code path, so mu stands for all three."""
        self._check_swept(
            lambda v: StaticParams(mu=v),
            [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
            1e-3,
            min_survivors=3,
            preconditioner="block_jacobi",
        )

    def test_collider_radius(self):
        self._check_swept(
            lambda v: StaticParams(collider_radius=jnp.reshape(v, (1,))),
            [1.0, 0.9975, 0.995, 0.9925, 0.99],
            1e-6,
            min_survivors=2,
        )

    def test_rest_positions_directional(self):
        """Directional derivative along a fixed perturbation of the rest shape, one
        scale per swept sample (the full Jacobian would need 3N solves for its FD
        twin). The direction leaves the clamped base untouched."""
        rest = np.asarray(self.positions)
        rng = np.random.default_rng(1)
        direction = rng.normal(size=rest.shape) * 0.01
        direction[rest[:, 2] <= 1e-9] = 0.0
        direction = jnp.asarray(direction)
        rest = jnp.asarray(rest)

        self._check_swept(
            lambda s: StaticParams(rest_positions=rest + s * direction),
            [0.002, 0.003, 0.004, 0.005, 0.006],
            1e-5,
            min_survivors=2,
        )


class TracedRefusalTests(unittest.TestCase):
    """The refusal, surviving vmap: unconverged gradients are NaN, per element.

    The host adjoint raises; per-sample control flow does not exist under vmap, so the
    traced adjoint poisons instead. The load-bearing case is the *mixed* batch: one
    element unconverged, its neighbour untouched -- an all-unconverged batch would pass
    even if the poison leaked across elements.

    The asymmetry is constructed, not harvested. tol = 1e-6 sits in the globally
    convergent descent phase, far above the ~1e-9..1e-11 region where the zigzag
    pockets live (FINDINGS.md), so element 0 converges on any machine in ~8 iterations
    -- if it does not, that is a solver alarm worth hearing. Element 1 lowers the
    sphere 5mm, which puts the shared initial guess inside the barrier's linear
    continuation; the solve un-penetrates and then zigzags at a residual of order one
    -- six orders above the tolerance, a physical plateau no compilation flips -- so
    it is unconverged everywhere, deterministically.
    """

    _REFUSAL_TOL = 1.0e-6

    @classmethod
    def setUpClass(cls):
        cls.template, cls.positions = _seated_problem()
        cls.initial = _feasible_initial(cls.positions)
        cls.weights = jnp.asarray(
            np.random.default_rng(0).normal(size=cls.positions.shape)
        )
        cls.centers = np.stack([_center_at(0.0), _center_at(-0.005)])
        cls.adjoint = StaticAdjointTraced(
            cls.template,
            tol=cls._REFUSAL_TOL,
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
            assert_converged(batched, tol=self._REFUSAL_TOL)

        healthy = self.adjoint.solve_result(
            StaticParams(collider_center=jnp.asarray(self.centers[0]))
        )
        assert_converged(healthy, tol=self._REFUSAL_TOL)  # converged: silent


class TracedTranscriptionTests(unittest.TestCase):
    """Decision-for-decision equality of ``_traced_newton_step`` with the host loop.

    The equivalence sweeps above only ever visit the smooth path: a healthy fixture
    never rejects a line search, so the hardest part of the transcription -- the
    rejected-search retry, the damping ladder, timid accepts, and the host loop's
    NaN-slope behaviour -- would otherwise ship ungated. This class drives the
    module-level step function *eagerly*, pass by pass, against a replica of
    ``solve_static_equilibrium``'s loop transcribed line-for-line below, and forces
    the rare branches by wrapping ``_newton_direction``. Both sides call the same
    jitted kernels on the same inputs, so agreement is *bitwise* -- any drift is a
    transcription defect, not compilation noise. Where a forcing scale depends on the
    machine's energy landscape (the timid accept), it is found at runtime by direct
    evaluation, never pinned.

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
        cls.no_contact, _ = _seated_problem(colliders=None, contact_enabled=False)

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
        """Sanity anchor: the natural trajectory from the feasible seated start. The
        only coverage this promises is >= 1 accepted step -- whether the machine's
        trajectory is all-accepts is pocket geography, and the substance is the
        pass-for-pass equality either way."""
        trace = self._assert_drivers_agree()
        self.assertTrue(any(accepted for accepted, *_ in trace))

    def test_zigzag_trajectory(self):
        """A messy real trajectory (penetrating start, barrier zigzag) traced for
        equality. Coverage here is opportunistic -- on the authoring machine it
        includes timid accepts -- but nothing is promised beyond equality; the timid
        branch has its own constructed test below."""
        center = np.array(_CENTER, dtype=np.float64)
        center[2] -= 0.005
        problem = apply_static_params(
            self.problem, StaticParams(collider_center=jnp.asarray(center[None, :]))
        )
        trace = self._assert_drivers_agree(problem=problem)
        self.assertGreater(len(trace), 0)

    def test_constructed_timid_accept_raises_the_shift(self):
        """An accepted step with alpha < 1/2 must *raise* the shift -- the select arm
        a confident trajectory never exercises. The forcing scale is found at runtime:
        scan c in powers of two until ``-c * gradient`` first-accepts below 1/2 on
        this machine's actual energy, then hand exactly that direction to both
        drivers. Contact-free problem so the collider clip cannot rescale the probe
        (it caps every vertex at ~its gap regardless of the pre-scale)."""
        problem = self.no_contact
        free = self.free
        u = self.pinned[free]
        gradient = static_module._gradient(problem, self.pinned, free, u)
        current = float(static_module._objective(problem, self.pinned, free, u))

        chosen = None
        for scale in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0):
            direction = -scale * gradient
            slope = float(jnp.sum(direction * gradient))
            alpha, accepted = 1.0, False
            for _ in range(40):
                trial = float(
                    static_module._objective(problem, self.pinned, free, u + alpha * direction)
                )
                if trial <= current + 1.0e-4 * alpha * slope:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted and alpha < 0.5:
                chosen = scale
                break
        self.assertIsNotNone(
            chosen, msg="no scale produced a timid accept on this energy landscape"
        )

        def factory(original):
            calls = {"n": 0}

            def wrapper(problem_, pinned, free_, u_, gradient_, cg_tol, damping,
                        maxiter, preconditioner=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    return -chosen * gradient_
                return original(
                    problem_, pinned, free_, u_, gradient_, cg_tol, damping,
                    maxiter=maxiter, preconditioner=preconditioner,
                )

            return wrapper

        trace = self._assert_drivers_agree(factory, problem=problem)
        accepted, damping_after, iterations_after, stagnated = trace[0]
        self.assertTrue(accepted)
        self.assertFalse(stagnated)
        # A timid accept from zero shift lands exactly on the floor.
        self.assertEqual(damping_after, 1.0e-4)
        self.assertEqual(iterations_after, 1)

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

            def wrapper(problem, pinned, free, u, gradient, cg_tol, damping,
                        maxiter, preconditioner=None):
                calls["n"] += 1
                if calls["n"] <= 2:
                    return -gradient * (2.0**60)
                return original(
                    problem, pinned, free, u, gradient, cg_tol, damping,
                    maxiter=maxiter, preconditioner=preconditioner,
                )

            return wrapper

        trace = self._assert_drivers_agree(factory, problem=self.no_contact)
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

            def wrapper(problem, pinned, free, u, gradient, cg_tol, damping,
                        maxiter, preconditioner=None):
                calls["n"] += 1
                direction = original(
                    problem, pinned, free, u, gradient, cg_tol, damping,
                    maxiter=maxiter, preconditioner=preconditioner,
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
