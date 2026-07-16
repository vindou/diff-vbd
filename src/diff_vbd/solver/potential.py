"""The whole-mesh energies, defined once.

``contact.potential`` already states the contact energy a single time so that its two
consumers -- ``vbd.vertex_local_objective`` and any future tangent-stiffness or adjoint
path -- cannot silently disagree about what the energy *is*. This module completes the
arrangement for the rest of the physics. Elasticity and inertia get the same treatment,
and the two sums the VBD paper is written in terms of get names:

* ``potential_energy`` is E(x): elastic plus contact, the energy of a *configuration*;
* ``variational_energy`` is G(x): inertia plus E(x), the objective of one implicit-Euler
  step (Eq. 2 of the paper) and the quantity a VBD sweep is claimed to decrease.

**The counting is the entire point**, exactly as it is for ``contact_potential``.
``vertex_local_objective`` sums the tets *incident to* one vertex, so summing it over
every vertex counts each tet four times -- a tet has four vertices and sits in each one's
incidence list. ``elastic_potential`` sums over tets, never over incidences:

    sum_v [masked incident-tet energies]  ==  4 * elastic_potential

and the factor is exactly 4, never anything else, because the incidence tables are built
from the tet array itself. That identity is what makes the local objective and this global
statement provably the same energy, and it is tested rather than assumed.

The solver calls none of this, deliberately: VBD's premise is a 3x3 solve against a frozen
block, and a global scalar is an object it never forms. The dependency is strictly one-way
-- this module knows the energies, not the solver -- which is also why
``variational_energy`` takes the inertial target as an argument instead of importing
``predict_inertial_target`` from ``vbd``.
"""

import jax
import jax.numpy as jnp

from diff_vbd.model import MaterialParams, SimulationProblem
from diff_vbd.solver.contact.potential import contact_potential
from diff_vbd.solver.materials import tet_energy


@jax.jit
def elastic_potential(
    material: MaterialParams,
    tets: jnp.ndarray,
    rest_positions: jnp.ndarray,
    positions: jnp.ndarray,
) -> jnp.ndarray:
    """Return the mesh's total elastic energy, counting each TET exactly once.

    ``rest_positions`` is a bare array rather than a ``MeshData``, mirroring both
    ``tet_energy`` and ``contact_potential``: the rest shape is an *input to the energy*,
    not a fixed property of the mesh, and a caller that varies it -- shape optimisation,
    an adjoint differentiating w.r.t. the rest state -- needs it exposed as one.
    """

    def one_tet(tet_vertices):
        return tet_energy(
            material, rest_positions[tet_vertices], positions[tet_vertices]
        )

    return jnp.sum(jax.vmap(one_tet)(tets))


@jax.jit
def inertia_potential(
    mass: jnp.ndarray,
    dt: jnp.ndarray,
    positions: jnp.ndarray,
    inertial_target: jnp.ndarray,
) -> jnp.ndarray:
    """Return the inertia term of the paper's Eq. 2: ``1/(2 h^2) ||x - y||_M^2``.

    This term is separable per vertex, so here -- unlike the elastic and contact terms --
    the whole-mesh sum really is the sum of the local objectives' inertia contributions
    with no counting factor at all. It is split out from ``variational_energy`` so a test
    or an adjoint can weigh the inertia and potential halves separately.
    """
    residual = positions - inertial_target
    return jnp.sum(mass * jnp.sum(residual * residual, axis=-1)) / (2.0 * dt**2)


@jax.jit
def potential_energy(
    problem: SimulationProblem,
    positions: jnp.ndarray,
    previous_positions: jnp.ndarray,
) -> jnp.ndarray:
    """Return E(x): the mesh's elastic plus contact energy at ``positions``.

    ``previous_positions`` is the lagged reference the friction terms are measured from,
    with the same caveat as ``contact_potential``: pass a *distinct* array, because
    passing ``positions`` itself puts the live positions on both sides of a
    ``stop_gradient`` and the gradient then legitimately disagrees with a finite
    difference. The analytic-collider-only case (no pair buffers at all) is handled
    inside ``contact_potential`` by a static shape branch, so this composes cleanly
    whether or not self-collision is on.
    """
    elastic = elastic_potential(
        problem.material,
        problem.mesh.tets,
        problem.mesh.rest_positions,
        positions,
    )
    contact = contact_potential(
        problem.contact.params,
        problem.contact.colliders,
        problem.contact.state,
        problem.mesh.rest_positions,
        positions,
        previous_positions,
        problem.solver.dt,
    )
    return elastic + contact


@jax.jit
def variational_energy(
    problem: SimulationProblem,
    positions: jnp.ndarray,
    inertial_target: jnp.ndarray,
    previous_positions: jnp.ndarray,
) -> jnp.ndarray:
    """Return G(x): the objective one implicit-Euler step minimises (paper Eq. 2).

    This is the scalar behind the paper's Section 3.1 descent argument -- with the line
    search on and Chebyshev acceleration off, every local solve decreases its G_i against
    a frozen block, and colours partition the mesh into non-interacting vertices, so a
    sweep decreases G. Neither accelerator preserves that: Chebyshev extrapolation is not
    a descent step, and the contact sweep filter rescales the whole update. Test the
    monotone claim only in the regime that makes it.

    ``inertial_target`` is an argument, not recomputed here from the state, and that is a
    boundary rather than an inconvenience: ``vbd.predict_inertial_target`` lives in the
    solver, the solver must never need this module, and a caller doing sensitivity
    analysis may well want a target that is itself being differentiated.
    """
    inertia = inertia_potential(
        problem.topology.mass, problem.solver.dt, positions, inertial_target
    )
    return inertia + potential_energy(problem, positions, previous_positions)
