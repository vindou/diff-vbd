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
