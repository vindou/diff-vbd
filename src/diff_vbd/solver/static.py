"""Static equilibrium and the implicit-function adjoint.

``potential_energy`` has advertised itself as "the hook a future adjoint path attaches to"
since it was written. This module is that path, and its design is driven by one fact that
is easy to state and easy to silently violate:

**Implicit differentiation requires an actual stationary point.** The sensitivities below
come from differentiating the optimality condition ``g(theta, u*) = grad_u Pi = 0``:

    du*/dtheta = -H^{-1} dg/dtheta,   H = grad^2_u Pi (SPD at a stable equilibrium)

If ``g(u*)`` is not (numerically) zero, that identity is simply false, and the formula
still returns a plausible-looking gradient -- there is no forward test that can catch it.
A VBD sweep runs a *fixed* iteration schedule and stops wherever it stops; its output is
not a stationary point and must never feed this formula. (NVIDIA's Newton differentiates
VBD by unrolling instead, which is exact for the computation performed but yields the
gradient of the truncated iteration, not of the physics.) Hence the two structural rules
here: ``solve_static_equilibrium`` terminates on a **residual tolerance**, not an
iteration count, and the adjoint **raises** when asked to differentiate a state whose
residual exceeds that tolerance. VBD sweeps are welcome as a warm start; they are never
differentiated.

Scope, stated plainly rather than discovered by crash:

* friction is refused -- lagged Coulomb friction is not the gradient of any potential, so
  a state held up by friction is not a stationary point of ``Pi`` and the adjoint identity
  does not hold;
* rigid regions are refused -- the rigid projection is a constraint applied outside the
  energy, so stationarity of ``Pi`` is the wrong optimality condition for them;
* mesh-mesh contact is refused for now -- a static Newton step is not bounded by the
  detection band the pair set was built under, so the energy would be incomplete in
  exactly the configurations where it matters. Analytic colliders have no pair set and no
  band, so they are exact here.

Dynamic per-step and trajectory adjoints are out of scope on this branch. They inherit
the same precondition in a harsher form: every *step* of the trajectory must be solved to
stationarity of its own objective ``G`` before an implicit-function argument applies, and
the fixed-schedule sweep does not do that either.

There are **two drivers over the same kernels**, and a caller should pick deliberately:

* ``solve_static_equilibrium`` / ``StaticAdjoint`` -- **host-side**. The Newton loop is a
  Python ``while`` on concrete residuals: call it eagerly, never under ``jit``. Right for
  one large mesh and few solves, where the sync latency is negligible against the solve
  and buys loud diagnostics -- an unconverged adjoint is a Python exception with the
  failure named, not a flag to check.
* ``solve_static_equilibrium_traced`` / ``StaticAdjointTraced`` -- **traced**. The same
  iteration as a ``lax.while_loop``: pure, jit-able, and -- the reason it exists --
  ``vmap``-able, so a batch of independent solves (Monte Carlo over loads, design sweeps)
  runs as one device program instead of a Python loop of sequential solves. Failure is
  data, not control flow: an unconverged element reports ``converged=False`` and its
  adjoint is **NaN**, never an exception mid-batch; ``assert_converged`` restores the
  loud version at a batch boundary.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from diff_vbd.model import SimulationProblem
from diff_vbd.pytree import pytree_dataclass
from diff_vbd.setup.boundary_conditions import evaluate_dirichlet_targets
from diff_vbd.solver.contact.ccd import vertex_time_of_impact
from diff_vbd.solver.potential import potential_energy


@jax.jit
def body_force_potential(
    mass: jnp.ndarray,
    external_acceleration: jnp.ndarray,
    positions: jnp.ndarray,
) -> jnp.ndarray:
    """Return the potential of a constant body force: ``-sum_v m_v a_v . x_v``.

    The dynamic solver never needs this: gravity is baked into the inertial target ``y``,
    which exists only inside a timestep. A static problem has no ``y``, so the body force
    has to be stated as what it is -- a linear potential whose negative gradient is
    ``m a``. ``potential_energy + body_force_potential`` is then the complete static
    ``Pi``, with no timestep anywhere in it.
    """
    return -jnp.sum(mass[:, None] * external_acceleration * positions)


@jax.jit
def static_potential(
    problem: SimulationProblem, positions: jnp.ndarray
) -> jnp.ndarray:
    """Return the complete static energy ``Pi(x)``: elastic + contact + body force.

    Reuses ``potential_energy`` -- the same kernels the dynamic solver descends -- so the
    static and dynamic paths cannot disagree about what the energy is. ``positions`` is
    passed as its own lagged reference, which is safe here and only here: the lagged
    array feeds *friction* alone, and ``solve_static_equilibrium`` refuses problems with
    friction enabled (see the module docstring for why), so the term is identically zero
    with a well-defined zero gradient.
    """
    return potential_energy(problem, positions, positions) + body_force_potential(
        problem.topology.mass,
        problem.solver.external_acceleration,
        positions,
    )


@pytree_dataclass
class StaticParams:
    """The differentiable inputs of a static solve. Every field is optional.

    A field left as ``None`` keeps the template problem's value. This is deliberately a
    closed list rather than "any pytree": each entry is a parameter class the adjoint is
    *tested* against, and ``apply_static_params`` knows the one non-obvious coupling --
    rest positions and density change the lumped masses, which change the body force --
    so a caller cannot forget it.
    """

    mu: jnp.ndarray | None = None
    lam: jnp.ndarray | None = None
    density: jnp.ndarray | None = None
    rest_positions: jnp.ndarray | None = None
    external_acceleration: jnp.ndarray | None = None  # (3,) or (num_vertices, 3)
    collider_normal: jnp.ndarray | None = None
    collider_offset: jnp.ndarray | None = None
    collider_center: jnp.ndarray | None = None
    collider_radius: jnp.ndarray | None = None


def apply_static_params(
    problem: SimulationProblem, params: StaticParams
) -> SimulationProblem:
    """Return ``problem`` with the given parameters substituted, traceably.

    This is the ``theta -> problem`` map the adjoint differentiates through, so it must
    be pure JAX: no validation, no host round-trips. ``assemble_problem`` is neither and
    must not be called here. The one derived quantity is the lumped mass: it is a
    *function* of the rest positions and density, so when either changes the masses are
    rebuilt through ``build_lumped_masses`` (which is JAX code for exactly this reason)
    rather than left stale -- a stale mass would silently zero the body-force term of
    every rest-shape sensitivity.
    """
    # Imported here, not at module top: `setup.topology` imports `solver.kinematics`,
    # whose package __init__ imports this module -- a top-level import would close that
    # cycle while `setup.topology` is still half-initialised.
    from diff_vbd.setup.topology import build_lumped_masses

    material = problem.material
    if params.mu is not None:
        material = dataclasses.replace(material, mu=jnp.asarray(params.mu))
    if params.lam is not None:
        material = dataclasses.replace(material, lam=jnp.asarray(params.lam))
    if params.density is not None:
        material = dataclasses.replace(material, density=jnp.asarray(params.density))

    mesh = problem.mesh
    topology = problem.topology
    if params.rest_positions is not None or params.density is not None:
        rest = (
            mesh.rest_positions
            if params.rest_positions is None
            else jnp.asarray(params.rest_positions)
        )
        mesh = dataclasses.replace(mesh, rest_positions=rest)
        topology = dataclasses.replace(
            topology,
            mass=build_lumped_masses(rest, mesh.tets) * material.density,
        )

    solver = problem.solver
    if params.external_acceleration is not None:
        acceleration = jnp.asarray(params.external_acceleration)
        if acceleration.ndim == 1:
            acceleration = jnp.broadcast_to(
                acceleration[None, :], solver.external_acceleration.shape
            )
        solver = dataclasses.replace(solver, external_acceleration=acceleration)

    colliders = problem.contact.colliders
    for name in ("normal", "offset", "center", "radius"):
        value = getattr(params, f"collider_{name}")
        if value is not None:
            colliders = dataclasses.replace(colliders, **{name: jnp.asarray(value)})
    contact = dataclasses.replace(problem.contact, colliders=colliders)

    return dataclasses.replace(
        problem,
        material=material,
        mesh=mesh,
        topology=topology,
        solver=solver,
        contact=contact,
    )


@pytree_dataclass
class StaticSolveResult:
    position: jnp.ndarray  # (N, 3), Dirichlet rows at their prescribed targets
    residual_norm: jnp.ndarray  # ||grad_u Pi||_2 over the free vertices
    iterations: jnp.ndarray  # Newton iterations consumed
    converged: jnp.ndarray  # bool: residual_norm <= tol


# --------------------------------------------------------------------------------------
# Jitted kernels. Module-level functions of explicit pytree arguments, deliberately:
# a per-solve closure over the problem or the current iterate would bake those arrays in
# as constants and recompile every Newton iteration -- and a finite-difference probe of
# the adjoint re-solves dozens of times, so the cache has to hit across solves too.
# --------------------------------------------------------------------------------------


@jax.jit
def _objective(problem, pinned, free, u):
    return static_potential(problem, pinned.at[free].set(u))


_gradient = jax.jit(jax.grad(_objective, argnums=3))


def _hvp_at(problem, pinned, free, u, v):
    return jax.jvp(lambda w: _gradient(problem, pinned, free, w), (u,), (v,))[1]


@partial(jax.jit, static_argnames=("maxiter",))
def _newton_direction(problem, pinned, free, u, gradient, cg_tol, damping, maxiter):
    """Damped inexact-Newton direction: CG on matrix-free HVPs, H never assembled.

    ``damping`` is a Levenberg shift, in the Hessian's own units. It exists because a
    log-barrier force grows like 1/g and the quadratic model under it is wildly
    optimistic: near contact, an undamped Newton step overshoots the gap by orders of
    magnitude, gets clipped to the barrier floor, reverses, and oscillates -- each
    Armijo-accepted zigzag shaving a sliver off the energy while the residual goes
    nowhere. The caller adapts the shift: large while steps are being rejected or cut,
    zero once the quadratic model is trustworthy, so terminal convergence is undamped
    Newton.
    """
    direction, _ = jax.scipy.sparse.linalg.cg(
        lambda v: _hvp_at(problem, pinned, free, u, v) + damping * v,
        -gradient,
        tol=cg_tol,
        maxiter=maxiter,
    )
    return direction


@partial(jax.jit, static_argnames=("maxiter",))
def _adjoint_cg(problem, pinned, free, u_star, w, maxiter):
    """Solve H lambda = w at the equilibrium, matrix-free."""
    lam, _ = jax.scipy.sparse.linalg.cg(
        lambda v: _hvp_at(problem, pinned, free, u_star, v),
        w,
        maxiter=maxiter,
    )
    return lam


@jax.jit
def _clip_direction_to_colliders(problem, pinned, free, u, direction):
    """Clip each vertex's displacement to its *own* collider time of impact.

    Two things at once. First, the safeguard: a raw Newton step from a barely-touching
    state happily lands on the far side of a collider, where the barrier's linear
    continuation is finite and the line search alone would accept it. Second, the
    granularity: the clip is per vertex, **not** a global minimum over the mesh. A
    single vertex creeping toward a collider (whose own safe fraction is ~gap/|step|,
    tiny) must not scale the whole direction down and freeze the far field's elastic
    relaxation with it -- with a global cap this solver measurably stalled at ~1%
    progress per iteration. Per-vertex clipping is sound here for the same reason it is
    unsound in the dynamic mesh-mesh filter: these colliders are *static*, so each
    vertex's certificate involves no other moving party and composes trivially.

    The clipped vector is no longer the Newton direction, so the caller re-checks
    descent against the gradient and falls back to the clipped gradient, which is a
    descent direction whenever any vertex can move: each row is a non-negative multiple
    of its own negative gradient row.
    """
    x = pinned.at[free].set(u)
    d = jnp.zeros_like(x).at[free].set(direction)
    per_vertex = jax.vmap(
        lambda a, b: vertex_time_of_impact(problem.contact.colliders, a, b)
    )(x, x + d)
    return direction * per_vertex[free][:, None]


def _validate_static_problem(problem: SimulationProblem) -> None:
    """Refuse the problem classes whose static answer would be quietly wrong.

    Same register as ``vbd._audit_step``: name the failure, say why the number would be
    wrong, say what to change.
    """
    if float(problem.contact.params.friction_mu) != 0.0:
        raise ValueError(
            "static equilibrium with contact_friction_mu != 0 was requested, but lagged "
            "Coulomb friction is not the gradient of any potential: a configuration held "
            "in place by friction is not a stationary point of Pi, so both the solve's "
            "convergence test and the adjoint's implicit-function identity would be "
            "built on a false premise and return plausible nonsense. Set "
            "contact_friction_mu=0, or treat the frictional problem dynamically."
        )
    if problem.boundary_conditions.rigid_region_specs:
        raise ValueError(
            "static equilibrium with rigid regions was requested, but the rigid "
            "projection is a constraint applied outside the energy: stationarity of Pi "
            "is the wrong optimality condition for the projected vertices, so the "
            "residual test would certify a state that the projection then moves. "
            "Replace the rigid region with Dirichlet constraints for static solves."
        )
    if problem.contact.surface_triangles.shape[0] > 0:
        raise ValueError(
            "static equilibrium with self-collision was requested, but the mesh-mesh "
            "pair set is only complete within a detection band sized for one *dynamic* "
            "step, and a static Newton step respects no such band: vertices can leave "
            "the band mid-solve, pairs outside it are invisible to the energy, and the "
            "'equilibrium' found can rest on missing contacts. Analytic colliders have "
            "no pair set and are exact here; use those, or solve dynamically."
        )


def _free_indices(problem: SimulationProblem) -> np.ndarray:
    free = ~np.asarray(problem.boundary_conditions.dirichlet_mask, dtype=bool)
    return np.flatnonzero(free)


def solve_static_equilibrium(
    problem: SimulationProblem,
    *,
    tol: float,
    initial_position: jnp.ndarray | None = None,
    prescribed_position: jnp.ndarray | None = None,
    max_iterations: int = 100,
    cg_max_iterations: int = 250,
    warm_start_steps: int = 0,
) -> StaticSolveResult:
    """Minimise the static ``Pi`` to a residual tolerance. Never an iteration count.

    Newton-CG with matrix-free Hessian-vector products (``jax.jvp`` of the gradient --
    ``H`` is never assembled at any size), an inexact-Newton forcing sequence on the CG
    tolerance, backtracking line search on ``Pi``, and one contact-specific safeguard:
    the Newton direction is first clipped to each vertex's collider time of impact, since
    a raw Newton step from a barely-touching state happily lands on the far side of a
    collider where the barrier's linear continuation is finite and the line search alone
    would accept it.

    ``tol`` is required, deliberately: it is in force units, so no default can be
    meaningful across unit systems, and it is the *precondition of the adjoint* -- the
    number below which ``StaticAdjoint`` will consent to differentiate this state.

    ``warm_start_steps`` runs velocity-zeroed VBD steps first (dynamic relaxation): each
    is a proximal step on ``Pi`` and cheap, so they carry the state from a cold start
    into Newton's basin. They are host-side forward iterations only; nothing about them
    is differentiated.
    """
    _validate_static_problem(problem)

    if prescribed_position is None:
        prescribed_position, _ = evaluate_dirichlet_targets(
            problem.boundary_conditions, 0.0, float(problem.solver.dt)
        )
    if initial_position is None:
        initial_position = problem.mesh.rest_positions

    if warm_start_steps > 0:
        from diff_vbd.model import SimulationState
        from diff_vbd.solver import vbd

        state = SimulationState(
            position=initial_position,
            velocity=jnp.zeros_like(initial_position),
            time=jnp.asarray(0.0, dtype=initial_position.dtype),
        )
        for _ in range(warm_start_steps):
            state = vbd.step(
                problem,
                dataclasses.replace(
                    state, velocity=jnp.zeros_like(state.velocity)
                ),
            )
        initial_position = state.position

    dirichlet = np.flatnonzero(
        np.asarray(problem.boundary_conditions.dirichlet_mask, dtype=bool)
    )
    free = jnp.asarray(_free_indices(problem), dtype=jnp.int32)
    pinned = (
        jnp.asarray(initial_position)
        .at[dirichlet]
        .set(jnp.asarray(prescribed_position)[dirichlet])
    )

    u = pinned[free]
    gradient = _gradient(problem, pinned, free, u)
    residual = float(jnp.linalg.norm(gradient))
    initial_residual = max(residual, 1e-300)

    # Levenberg shift, relative to a per-iterate curvature scale (|H g| / |g|, one extra
    # HVP). Starts at zero -- plain Newton -- and is raised only when the quadratic model
    # demonstrably lied (a rejected or heavily cut step), then decays on smooth progress,
    # so the terminal iterations are undamped and quadratic.
    damping_rel = 0.0
    iterations = 0
    while residual > tol and iterations < max_iterations:
        # Inexact Newton: solve H dx = -g only as tightly as the residual deserves.
        forcing = min(0.5, float(np.sqrt(residual / initial_residual)))
        curvature_scale = float(
            jnp.linalg.norm(_hvp_at(problem, pinned, free, u, gradient))
        ) / max(residual, 1e-300)
        direction = _newton_direction(
            problem,
            pinned,
            free,
            u,
            gradient,
            jnp.asarray(forcing),
            jnp.asarray(damping_rel * curvature_scale),
            maxiter=cg_max_iterations,
        )
        # Truncated CG on an indefinite Hessian can hand back an ascent direction;
        # steepest descent is always available and always descends.
        if float(jnp.sum(direction * gradient)) >= 0.0:
            direction = -gradient

        direction = _clip_direction_to_colliders(problem, pinned, free, u, direction)
        # Clipping can spoil the descent property of a CG direction (it cannot spoil
        # the gradient's -- see _clip_direction_to_colliders).
        if float(jnp.sum(direction * gradient)) >= 0.0:
            direction = _clip_direction_to_colliders(
                problem, pinned, free, u, -gradient
            )

        current = float(_objective(problem, pinned, free, u))
        slope = float(jnp.sum(direction * gradient))
        if slope >= 0.0:
            break  # every movable direction is blocked: stagnated, say so
        alpha = 1.0
        accepted = False
        for _ in range(40):
            candidate = u + alpha * direction
            if float(_objective(problem, pinned, free, candidate)) <= (
                current + 1.0e-4 * alpha * slope
            ):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            if damping_rel < 1.0e6:
                damping_rel = max(10.0 * damping_rel, 1.0e-4)
                continue  # retry from the same iterate with a heavier shift
            break  # stagnated even fully damped: report the residual honestly
        if alpha >= 0.5:
            damping_rel /= 3.0
            if damping_rel < 1.0e-7:
                damping_rel = 0.0
        else:
            damping_rel = max(10.0 * damping_rel, 1.0e-4)
        u = candidate
        gradient = _gradient(problem, pinned, free, u)
        residual = float(jnp.linalg.norm(gradient))
        iterations += 1

    return StaticSolveResult(
        position=pinned.at[free].set(u),
        residual_norm=jnp.asarray(residual),
        iterations=jnp.asarray(iterations),
        converged=jnp.asarray(residual <= tol),
    )


class StaticAdjoint:
    """A differentiable static solve: ``theta -> equilibrium positions``.

    Wraps ``solve_static_equilibrium`` in ``jax.custom_vjp``. The forward pass is the
    host-side Newton solve (so call this eagerly, not under ``jit`` -- the solver's
    convergence loop needs concrete values); the backward pass is one matrix-free CG
    solve of ``H lambda = w`` at the equilibrium plus a VJP of the residual function
    with respect to ``theta``:

        dL/dtheta = -(dg/dtheta)^T H^{-1} (dL/du*)

    ``H`` is never assembled. All solver options are fixed at construction; ``theta`` is
    the only traced input.

    **The backward pass refuses to be wrong.** If the state it is asked to differentiate
    has a residual above the construction-time tolerance -- because the forward solve
    stalled, or because someone fed it a fixed-count VBD sweep's output -- it raises
    rather than returning the plausible-but-false implicit gradient. That check is the
    entire reason the adjoint is attached to a residual-tolerance solver and not to the
    sweep loop.
    """

    def __init__(
        self,
        problem: SimulationProblem,
        *,
        tol: float,
        max_iterations: int = 100,
        cg_max_iterations: int = 250,
        warm_start_steps: int = 0,
        initial_position: jnp.ndarray | None = None,
    ):
        _validate_static_problem(problem)
        self._template = problem
        self._tol = float(tol)
        self._free = jnp.asarray(_free_indices(problem), dtype=jnp.int32)
        self._solve_options = dict(
            tol=float(tol),
            max_iterations=max_iterations,
            cg_max_iterations=cg_max_iterations,
            warm_start_steps=warm_start_steps,
            initial_position=initial_position,
        )
        self._cg_max_iterations = cg_max_iterations
        prescribed, _ = evaluate_dirichlet_targets(
            problem.boundary_conditions, 0.0, float(problem.solver.dt)
        )
        self._pinned = jnp.asarray(prescribed)

        solve = jax.custom_vjp(self._primal)
        solve.defvjp(self._forward, self._backward)
        self._solve = solve

    def __call__(self, params: StaticParams) -> jnp.ndarray:
        return self._solve(params)

    # -- forward ---------------------------------------------------------------------

    def _primal(self, params: StaticParams) -> jnp.ndarray:
        problem = apply_static_params(self._template, params)
        return solve_static_equilibrium(problem, **self._solve_options).position

    def _forward(self, params: StaticParams):
        problem = apply_static_params(self._template, params)
        result = solve_static_equilibrium(problem, **self._solve_options)
        return result.position, (params, result.position, result.residual_norm)

    # -- backward --------------------------------------------------------------------

    def _backward(self, saved, cotangent):
        params, position, residual_norm = saved

        if float(residual_norm) > self._tol:
            raise ValueError(
                f"an adjoint was requested at a state whose static residual is "
                f"{float(residual_norm):g}, above the tolerance {self._tol:g} the "
                f"adjoint was constructed with. Implicit differentiation is the "
                f"derivative of the optimality condition grad Pi = 0; at this state "
                f"that condition does not hold, so the formula would return a "
                f"plausible-looking gradient of nothing, and no forward test can catch "
                f"it. Loosen nothing: re-run the solve with more iterations (or a warm "
                f"start) until it converges, or raise the constructor tolerance only "
                f"if the downstream use genuinely tolerates that residual."
            )

        u_star = position[self._free]
        w = cotangent[self._free]

        # H lambda = w at the equilibrium, matrix-free. H is SPD at a stable
        # equilibrium, which the residual check above has just certified numerically.
        problem = apply_static_params(self._template, params)
        lam = _adjoint_cg(
            problem,
            self._pinned,
            self._free,
            u_star,
            w,
            maxiter=self._cg_max_iterations,
        )

        # dL/dtheta = -lambda^T dg/dtheta, as the VJP of g w.r.t. theta at fixed u*.
        template, pinned, free = self._template, self._pinned, self._free

        def residual_of_theta(theta):
            return _gradient(
                apply_static_params(template, theta), pinned, free, u_star
            )

        _, vjp_fn = jax.vjp(residual_of_theta, params)
        (params_bar,) = vjp_fn(lam)
        params_bar = jax.tree_util.tree_map(
            lambda leaf: -leaf if leaf is not None else None, params_bar
        )
        return (params_bar,)


# --------------------------------------------------------------------------------------
# The traced driver. Same kernels, same math, same constants as the host loop above --
# the only thing that moves is who drives the iteration. A Python `while` on concrete
# residuals cannot be vmapped, so a caller who needs a *batch* of independent solves
# (Monte Carlo over loads, a design sweep) is forced to run them sequentially; this
# driver keeps every value on device, so `jax.vmap` over a batch of StaticParams runs B
# solves as one program. The constants below mirror the host loop's literals verbatim;
# if one side changes, change both, or the agreement gate in test_static_traced.py fails.
# --------------------------------------------------------------------------------------

_ARMIJO_C1 = 1.0e-4
_MAX_LINE_SEARCH_HALVINGS = 40
_DAMPING_FLOOR = 1.0e-4  # first nonzero Levenberg shift after a rejection
_DAMPING_CEILING = 1.0e6  # a rejection at this shift means stagnation, not retry
_DAMPING_SNAP_TO_ZERO = 1.0e-7  # decayed below this, the shift is plain-Newton again


def _traced_newton_step(problem, pinned, free, initial_residual, cg_max_iterations, carry):
    """One pass of the traced Newton loop: the host loop's body, as pure data flow.

    Module-level and driven one pass at a time by the transcription test in
    ``test_static_traced.py``, which replays it decision-for-decision against a
    line-for-line replica of the host loop -- the only way to gate the rejection and
    retry paths, which healthy fixtures never visit. ``_traced_newton_solve`` wraps it
    in the ``lax.while_loop``.

    Every predicate here is written against the host loop's *break* conditions, not
    their complements, because the two differ on NaN. The host tests ``slope >= 0.0``
    and a NaN slope (a CG breakdown handing back a NaN direction -- the exact pathology
    the Levenberg shift exists for) does NOT break: it runs the Armijo ladder, fails
    every trial (any comparison with NaN is False), and lands in the rejection path,
    where the raised shift typically recovers the next pass. Writing ``slope < 0.0``
    here instead would silently reclassify that recoverable state as terminal
    stagnation -- caught by adversarial review, pinned by the NaN-direction
    transcription test.
    """
    u, gradient, residual, damping_rel, newton_iters, body_passes, _ = carry
    dtype = u.dtype

    # Inexact Newton: solve H dx = -g only as tightly as the residual deserves.
    forcing = jnp.minimum(0.5, jnp.sqrt(residual / initial_residual))
    curvature_scale = jnp.linalg.norm(
        _hvp_at(problem, pinned, free, u, gradient)
    ) / jnp.maximum(residual, 1e-300)
    direction = _newton_direction(
        problem,
        pinned,
        free,
        u,
        gradient,
        forcing,
        damping_rel * curvature_scale,
        maxiter=cg_max_iterations,
    )
    # Truncated CG on an indefinite Hessian can hand back an ascent direction;
    # steepest descent is always available and always descends.
    direction = jnp.where(
        jnp.sum(direction * gradient) >= 0.0, -gradient, direction
    )
    direction = _clip_direction_to_colliders(problem, pinned, free, u, direction)
    # The host driver computes the clipped-gradient fallback only when the clipped
    # CG direction fails the descent re-check; a traced body has no branch, so both
    # candidates are computed every pass and a select does the choosing. The cost is
    # one extra per-vertex TOI sweep, small next to the CG solve.
    fallback = _clip_direction_to_colliders(problem, pinned, free, u, -gradient)
    direction = jnp.where(
        jnp.sum(direction * gradient) >= 0.0, fallback, direction
    )

    current = _objective(problem, pinned, free, u)
    slope = jnp.sum(direction * gradient)
    # The host loop's first `break`: every movable direction is blocked. NaN slope is
    # deliberately NOT blocked (see the docstring); it must fall through to the ladder
    # and the rejection path exactly as it does on the host.
    blocked = slope >= 0.0

    def line_search_cond(ls):
        _, trials, done = ls
        return jnp.logical_not(done) & (trials < _MAX_LINE_SEARCH_HALVINGS)

    def line_search_body(ls):
        alpha, trials, _ = ls
        trial = _objective(problem, pinned, free, u + alpha * direction)
        ok = trial <= current + _ARMIJO_C1 * alpha * slope
        return jnp.where(ok, alpha, 0.5 * alpha), trials + 1, ok

    # Seeded as already-done for a blocked element, so it burns no objective
    # evaluations it would discard (the host breaks before the ladder).
    alpha, _, ls_done = jax.lax.while_loop(
        line_search_cond,
        line_search_body,
        (
            jnp.ones((), dtype=dtype),
            jnp.zeros((), dtype=jnp.int32),
            blocked,
        ),
    )
    accepted = ls_done & jnp.logical_not(blocked)

    # The host loop's shift schedule, as selects: decay on a confident step
    # (alpha >= 1/2, snapping to plain Newton below the floor), raise on a timid or
    # rejected one -- including a NaN-slope rejection, whose raise is what lets the
    # next pass recover. A rejection with the shift already at the ceiling is the host
    # loop's second `break`; a *blocked* element freezes its shift, because the host
    # breaks before touching it.
    raised = jnp.maximum(10.0 * damping_rel, _DAMPING_FLOOR)
    decayed = damping_rel / 3.0
    decayed = jnp.where(
        decayed < _DAMPING_SNAP_TO_ZERO, jnp.zeros((), dtype=dtype), decayed
    )
    can_retry = damping_rel < _DAMPING_CEILING
    damping_next = jnp.where(
        accepted,
        jnp.where(alpha >= 0.5, decayed, raised),
        jnp.where(jnp.logical_not(blocked) & can_retry, raised, damping_rel),
    )
    stagnated = blocked | (
        jnp.logical_not(accepted) & jnp.logical_not(can_retry)
    )

    u_next = jnp.where(accepted, u + alpha * direction, u)
    # On a rejected pass this recomputes the gradient at an unchanged iterate --
    # the same value, deterministically. Selecting it away would not save the work
    # (both sides of a traced select are computed), so recompute unconditionally.
    gradient_next = _gradient(problem, pinned, free, u_next)
    residual_next = jnp.linalg.norm(gradient_next)

    return (
        u_next,
        gradient_next,
        residual_next,
        damping_next,
        newton_iters + accepted.astype(jnp.int32),
        body_passes + 1,
        stagnated,
    )


@partial(jax.jit, static_argnames=("max_iterations", "cg_max_iterations"))
def _traced_newton_solve(
    problem, pinned, free, u0, tol, max_iterations, cg_max_iterations
):
    """The host Newton loop as a ``lax.while_loop``: pure, jit-able, vmappable.

    The host loop has two pieces of control flow that do not survive tracing literally,
    and both become *data* in ``_traced_newton_step``:

    * **The retry (`continue`).** A rejected line search raises the Levenberg shift and
      retries the *same* iterate without counting a Newton iteration. A while_loop body
      always runs to completion, so the retry is a masked non-update of ``u`` with the
      shift raised -- which splits the iteration count in two: ``newton_iters`` (accepted
      steps, what the host loop calls ``iterations``) and ``body_passes`` (loop trips,
      accepted or not). The predicate bounds both. ``newton_iters`` alone would not
      terminate a pathological element that rejects forever; the pass cap is sized so it
      can never bind first on a healthy solve: a rejection ladder from zero shift to the
      ceiling is at most 12 passes (0 -> 1e-4 -> ... -> 1e6 is 11 raises, and the next
      rejection at the ceiling sets ``stagnated``), so 13 passes per accepted step, plus
      one final ladder, bounds every trajectory the host loop could take.
    * **The two `break`s** (a genuinely non-negative slope after both clip fallbacks; a
      rejection with the shift already at the ceiling) become a ``stagnated`` flag in
      the carry -- the predicate can only see carried state, so "stop this element"
      must be a value.

    Under ``vmap`` the predicate becomes ``any(...)`` and JAX masks the carry per
    element, so a converged element idles bit-exactly (its ``u``, shift and counters
    frozen) while the batch pays the max over its elements. That is the expected cost
    model, not a bug.

    The nested line-search ``while_loop`` in the step preserves the host's acceptance
    rule exactly: first Armijo-acceptable step of a halving ladder, 1-5 objective
    evaluations in the common case. It is deliberately *not*
    ``vbd._apply_line_search``'s evaluate-all-rungs ``argmin``: that is a different
    acceptance rule, and swapping it would change which equilibrium basin a marginal
    step lands in -- a solver behaviour change, not a refactor.
    """
    gradient0 = _gradient(problem, pinned, free, u0)
    residual0 = jnp.linalg.norm(gradient0)
    initial_residual = jnp.maximum(residual0, 1e-300)
    dtype = u0.dtype

    body_pass_cap = 13 * max_iterations + 13

    def newton_cond(carry):
        _, _, residual, _, newton_iters, body_passes, stagnated = carry
        return (
            (residual > tol)
            & (newton_iters < max_iterations)
            & (body_passes < body_pass_cap)
            & jnp.logical_not(stagnated)
        )

    def newton_body(carry):
        return _traced_newton_step(
            problem, pinned, free, initial_residual, cg_max_iterations, carry
        )

    u, _, residual, _, newton_iters, _, _ = jax.lax.while_loop(
        newton_cond,
        newton_body,
        (
            u0,
            gradient0,
            residual0,
            jnp.zeros((), dtype=dtype),  # damping_rel: start at plain Newton
            jnp.zeros((), dtype=jnp.int32),  # newton_iters
            jnp.zeros((), dtype=jnp.int32),  # body_passes
            jnp.zeros((), dtype=bool),  # stagnated
        ),
    )
    return StaticSolveResult(
        position=pinned.at[free].set(u),
        residual_norm=residual,
        iterations=newton_iters,
        converged=residual <= tol,
    )


def solve_static_equilibrium_traced(
    problem: SimulationProblem,
    *,
    tol: float,
    initial_position: jnp.ndarray | None = None,
    prescribed_position: jnp.ndarray | None = None,
    max_iterations: int = 100,
    cg_max_iterations: int = 250,
) -> StaticSolveResult:
    """``solve_static_equilibrium``, but pure and traceable: jit it, ``vmap`` it.

    Same kernels, same acceptance rules, same constants -- see ``_traced_newton_solve``
    for the two places host control flow had to become carried data. What this driver
    gives up in exchange is the host loop's concreteness: no Python exception can name a
    mid-solve failure, so the *result* carries the verdict (``converged``), and the
    adjoint built on it (``StaticAdjointTraced``) poisons rather than raises.

    The tracing contract, stated plainly: trace ``StaticParams`` (through
    ``apply_static_params``) and close over everything else. The Dirichlet mask, ``dt``
    and the friction/rigid/self-collision refusals are read *concretely* at trace time --
    they are the problem's structure, not its parameters -- so a caller who wraps this in
    ``jit``/``vmap`` must keep the template problem a trace constant. Batching the mask
    would change which rows are unknowns, i.e. the *shape* of the problem, which no
    single compiled program can represent.

    Two host-path options are deliberately absent. ``warm_start_steps`` runs host-side
    VBD steps (``vbd.step`` reads concrete state) and belongs to the eager path; a traced
    caller passes ``initial_position`` instead. And there is no traced analogue of the
    host loop's per-iteration diagnostics: a batch has no meaningful single narrative.

    Feasibility is still the caller's job, per element: the barrier is infinite at zero
    gap, so every element of a batch needs a non-penetrating ``initial_position`` --
    one infeasible element does not poison its neighbours (the loop is masked
    per element), but it will burn the whole batch's wall-clock to ``max_iterations``.
    """
    _validate_static_problem(problem)

    if prescribed_position is None:
        prescribed_position, _ = evaluate_dirichlet_targets(
            problem.boundary_conditions, 0.0, float(problem.solver.dt)
        )
    if initial_position is None:
        initial_position = problem.mesh.rest_positions

    dirichlet = np.flatnonzero(
        np.asarray(problem.boundary_conditions.dirichlet_mask, dtype=bool)
    )
    free = jnp.asarray(_free_indices(problem), dtype=jnp.int32)
    pinned = (
        jnp.asarray(initial_position)
        .at[dirichlet]
        .set(jnp.asarray(prescribed_position)[dirichlet])
    )
    return _traced_newton_solve(
        problem,
        pinned,
        free,
        pinned[free],
        jnp.asarray(tol, dtype=pinned.dtype),
        max_iterations=max_iterations,
        cg_max_iterations=cg_max_iterations,
    )


def assert_converged(result: StaticSolveResult, *, tol: float | None = None) -> None:
    """Host-side loudness for the traced path: raise if any element is unconverged.

    The traced adjoint cannot raise per sample (an exception is control flow, and
    ``vmap`` has none to give a single element), so it poisons unconverged gradients
    with NaN instead. This is the batch-boundary complement: call it eagerly on a
    (possibly vmapped) ``StaticSolveResult`` to get the host adjoint's error, with the
    failing batch elements named. It blocks on a device transfer -- that is its job.
    """
    # Raveled, not atleast_1d: flatnonzero returns *flat* indices, and a nested-vmap
    # result has batch rank >= 2 -- flat indices into an unraveled residual array would
    # select wrong rows or raise IndexError, losing the diagnostic this function exists
    # to give (caught by adversarial review).
    converged = np.asarray(result.converged).ravel()
    if bool(np.all(converged)):
        return
    residuals = np.asarray(result.residual_norm).ravel()
    failing = np.flatnonzero(~converged)
    worst = float(np.max(residuals[failing]))
    against = f", above the tolerance {tol:g} the solve was given" if tol is not None else ""
    raise ValueError(
        f"a static solve left {failing.size} of {converged.size} batch element(s) "
        f"unconverged (elements {failing.tolist()}, worst residual {worst:g}{against}). "
        f"Implicit differentiation is the derivative of the optimality condition "
        f"grad Pi = 0; at this state that condition does not hold, so the formula "
        f"would return a plausible-looking gradient of nothing, and no forward test "
        f"can catch it -- which is why StaticAdjointTraced has already poisoned these "
        f"elements' gradients with NaN. Loosen nothing: re-run the solve with more "
        f"iterations (or a better initial position) until it converges, or raise the "
        f"tolerance only if the downstream use genuinely tolerates that residual."
    )


class StaticAdjointTraced:
    """``StaticAdjoint``'s traced sibling: same implicit gradient, batchable.

    Wraps ``solve_static_equilibrium_traced`` in ``jax.custom_vjp``: the forward pass is
    the ``lax.while_loop`` Newton solve, the backward pass the same matrix-free
    ``H lambda = w`` CG plus residual VJP as the host adjoint. Because both passes are
    pure JAX, the whole object composes with ``jit``, ``grad`` and -- the reason it
    exists -- ``vmap`` over a batch of ``StaticParams``. A sibling class rather than a
    flag on ``StaticAdjoint``, deliberately: the two have different failure contracts,
    and a flag that silently swaps "raises on bad input" for "returns NaN on bad input"
    is exactly the kind of behaviour change that should be visible at the call site.

    **The refusal survives batching by becoming data.** The host adjoint raises when
    asked to differentiate an unconverged state; per-sample control flow does not exist
    under ``vmap``, so here the convergence certificate rides through the custom_vjp
    residuals and the backward pass multiplies each unconverged sample's cotangent by
    NaN. NaN is the traced analogue of the exception: it propagates through every
    downstream reduction, cannot be silently consumed, and stays confined to its own
    batch element -- a converged neighbour's gradient is untouched. Callers who want
    the loud version call ``assert_converged(adjoint.solve_result(params))`` at a batch
    boundary. What is *not* on offer is the third option: silently returning the
    plausible-but-false implicit gradient of a non-stationary state.
    """

    def __init__(
        self,
        problem: SimulationProblem,
        *,
        tol: float,
        max_iterations: int = 100,
        cg_max_iterations: int = 250,
        initial_position: jnp.ndarray | None = None,
    ):
        _validate_static_problem(problem)
        self._template = problem
        self._tol = float(tol)
        self._free = jnp.asarray(_free_indices(problem), dtype=jnp.int32)
        self._max_iterations = max_iterations
        self._cg_max_iterations = cg_max_iterations
        self._initial_position = initial_position
        prescribed, _ = evaluate_dirichlet_targets(
            problem.boundary_conditions, 0.0, float(problem.solver.dt)
        )
        self._pinned = jnp.asarray(prescribed)

        solve = jax.custom_vjp(self._primal)
        solve.defvjp(self._forward, self._backward)
        self._solve = solve

    def __call__(self, params: StaticParams) -> jnp.ndarray:
        return self._solve(params)

    def solve_result(self, params: StaticParams) -> StaticSolveResult:
        """The full result, certificate included, for ``assert_converged`` at a batch
        boundary. Not differentiable -- gradients go through ``__call__``; this is a
        second forward solve, priced accordingly."""
        problem = apply_static_params(self._template, params)
        return solve_static_equilibrium_traced(
            problem,
            tol=self._tol,
            initial_position=self._initial_position,
            prescribed_position=self._pinned,
            max_iterations=self._max_iterations,
            cg_max_iterations=self._cg_max_iterations,
        )

    # -- forward ---------------------------------------------------------------------

    def _primal(self, params: StaticParams) -> jnp.ndarray:
        return self.solve_result(params).position

    def _forward(self, params: StaticParams):
        result = self.solve_result(params)
        # `converged` rides the residuals: it is the per-sample certificate the
        # backward pass turns into a poison factor, and carrying it as data is what
        # lets the refusal survive vmap where the host adjoint's `raise` cannot.
        return result.position, (params, result.position, result.converged)

    # -- backward --------------------------------------------------------------------

    def _backward(self, saved, cotangent):
        # Mirrors StaticAdjoint._backward's math exactly (one adjoint CG, one residual
        # VJP, negate); shared deliberately by transcription rather than refactor so the
        # host path stays byte-identical. The one semantic difference is the guard:
        # where the host adjoint raises on an unconverged state, this multiplies the
        # sample's cotangent by NaN -- per-sample control flow does not exist under
        # vmap, but per-sample data does.
        params, position, converged = saved

        u_star = position[self._free]
        w = cotangent[self._free]

        problem = apply_static_params(self._template, params)
        lam = _adjoint_cg(
            problem,
            self._pinned,
            self._free,
            u_star,
            w,
            maxiter=self._cg_max_iterations,
        )

        template, pinned, free = self._template, self._pinned, self._free

        def residual_of_theta(theta):
            return _gradient(
                apply_static_params(template, theta), pinned, free, u_star
            )

        _, vjp_fn = jax.vjp(residual_of_theta, params)
        (params_bar,) = vjp_fn(lam)
        poison = jnp.where(converged, 1.0, jnp.nan)
        params_bar = jax.tree_util.tree_map(
            lambda leaf: -poison * leaf if leaf is not None else None, params_bar
        )
        return (params_bar,)
