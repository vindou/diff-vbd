"""Whole-mesh energy tests.

These are the gates the new global potentials exist for, and each one closes a hole that
would otherwise fail *silently*:

* a rest state that is not actually stress-free (the alpha remap in the material drifted);
* a local objective and a global potential that quietly disagree -- the forward solve and
  a future adjoint path would then be optimising different energies while both look fine;
* a counting error, the same 4x trap ``contact_potential``'s docstring documents for
  pairs, now for tets;
* a VBD sweep that does not actually descend the variational energy G, which is the
  paper's central claim and, until now, was not testable globally at all.

The monotone-descent test runs the exact regime the Section 3.1 argument covers: no
contact, line search on (its alpha grid includes 0, so a vertex may decline to move),
Chebyshev acceleration off. Neither accelerator preserves monotonicity -- extrapolation is
not a descent step and the contact sweep filter rescales the whole update -- so they are
excluded rather than tolerated with a looser threshold.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from diff_vbd import (
    assemble_problem,
    elastic_potential,
    inertia_potential,
    initial_state,
    potential_energy,
    variational_energy,
)
from diff_vbd.setup.boundary_conditions import evaluate_dirichlet_targets
from diff_vbd.solver import vbd
from diff_vbd.solver.materials import tet_energy


def _beam(nx=5, ny=2, nz=2, spacing=0.5):
    """A Kuhn-split tetrahedral beam with binary-exact coordinates.

    The spacing is a power of two on purpose: every rest coordinate, edge matrix entry and
    edge-matrix inverse is then exact in binary, so the rest deformation gradient comes out
    as *exactly* the identity and the zero-energy rest test can demand equality rather than
    an "almost". The Kuhn 6-tet split is the same one the contact tests use -- it conforms
    unconditionally, and every tet winds positively.
    """
    grid = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1
    )
    positions = grid.reshape(-1, 3).astype(np.float64) * spacing
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


def _cantilever_problem(**kwargs):
    """The x = 0 face clamped, gravity on: the paper's canonical no-contact problem."""
    positions, tets = _beam()
    free_mask = (np.asarray(positions)[:, 0] > 0.0).astype(np.float64)
    defaults = dict(
        dt=0.01,
        external_acceleration=(0.0, 0.0, -9.81),
        num_iterations=8,
        mu=50.0,
        lam=50.0,
        eps=1.0e-8,
        line_search_enabled=True,
        acceleration_enabled=False,
    )
    defaults.update(kwargs)
    problem = assemble_problem(
        positions, tets, jnp.asarray(free_mask, dtype=jnp.float64), **defaults
    )
    return problem, initial_state(problem)


def _elastic(problem, positions):
    return elastic_potential(
        problem.material, problem.mesh.tets, problem.mesh.rest_positions, positions
    )


def _jittered_positions(problem, scale=0.05, seed=0):
    """A generic deformed configuration: bent enough that every tet carries energy."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(scale=scale, size=problem.mesh.rest_positions.shape)
    return problem.mesh.rest_positions + jnp.asarray(jitter, dtype=jnp.float64)


class ElasticPotentialTests(unittest.TestCase):
    def test_rest_state_has_exactly_zero_energy(self):
        """Zero by construction, not by subtraction.

        The stable Neo-Hookean density is written so the rest energy is zero with no
        constant subtracted afterwards (alpha only zeroes the rest *stress*), and the
        beam's binary-exact coordinates make F exactly the identity -- so the whole-mesh
        sum must be exactly 0.0, not merely small.
        """
        problem, _ = _cantilever_problem()
        rest = problem.mesh.rest_positions
        self.assertEqual(float(_elastic(problem, rest)), 0.0)

    def test_rest_state_is_stress_free(self):
        """The global statement of what the alpha remap buys.

        ``alpha = 1 + 0.75 mu/lam`` in the material exists precisely so the rest state
        carries no stress; the per-density test checks one F, this checks the assembled
        mesh gradient, where a counting or gather bug would also surface.
        """
        problem, _ = _cantilever_problem()
        rest = problem.mesh.rest_positions
        gradient = jax.grad(lambda x: _elastic(problem, x))(rest)
        self.assertLess(float(jnp.max(jnp.abs(gradient))), 1.0e-12)

    def test_incidence_sum_is_four_times_the_potential(self):
        """The counting gate, with the factor stated: the elastic twin of the 4x argument
        in ``contact_potential``'s docstring.

        Summing each vertex's incident-tet energies over all vertices counts every tet
        four times -- a tet has four vertices and appears in each one's incidence list.
        The per-vertex sum below is built from ``topology.incident_tets`` and
        ``incident_mask``, mirroring ``vertex_local_objective``'s accumulation exactly,
        so this is the test that proves the local objective and the global potential
        agree about the elastic energy.
        """
        problem, _ = _cantilever_problem()
        positions = _jittered_positions(problem)

        def vertex_incident_elastic(vertex_index):
            incident = problem.topology.incident_tets[vertex_index]
            mask = problem.topology.incident_mask[vertex_index]

            def one_tet(tet_index):
                tet_vertices = problem.mesh.tets[tet_index]
                return tet_energy(
                    problem.material,
                    problem.mesh.rest_positions[tet_vertices],
                    positions[tet_vertices],
                )

            energies = jax.vmap(one_tet)(incident)
            return jnp.sum(energies * mask.astype(energies.dtype))

        num_vertices = positions.shape[0]
        per_incidence = float(
            jnp.sum(
                jax.vmap(vertex_incident_elastic)(
                    jnp.arange(num_vertices, dtype=jnp.int32)
                )
            )
        )
        per_tet = float(_elastic(problem, positions))
        self.assertGreater(per_tet, 0.0)
        self.assertAlmostEqual(
            per_incidence, 4.0 * per_tet, delta=1.0e-10 * abs(per_incidence)
        )

    def test_summed_local_objectives_equal_inertia_plus_four_elastic(self):
        """The same identity, asserted against ``vertex_local_objective`` itself.

        Inertia is separable per vertex (factor 1), elastic is incident-summed
        (factor 4), and contact is absent here -- so the sum of the actual local
        objectives the solver descends must equal ``inertia_potential + 4 * elastic``.
        If the local and global definitions ever drift, this is the line that goes red.
        """
        problem, state = _cantilever_problem()
        positions = _jittered_positions(problem)
        inertial_target = vbd.predict_inertial_target(problem, state)

        summed = float(
            sum(
                vbd.vertex_local_objective(
                    problem,
                    positions,
                    inertial_target,
                    state.position,
                    jnp.int32(i),
                    positions[i],
                )
                for i in range(positions.shape[0])
            )
        )
        expected = float(
            inertia_potential(
                problem.topology.mass, problem.solver.dt, positions, inertial_target
            )
            + 4.0 * _elastic(problem, positions)
        )
        self.assertAlmostEqual(summed, expected, delta=1.0e-10 * abs(summed))

    def test_gradient_matches_central_differences(self):
        problem, _ = _cantilever_problem()
        positions = _jittered_positions(problem)

        energy = lambda x: _elastic(problem, x)
        gradient = np.asarray(jax.grad(energy)(positions))
        self.assertTrue(np.all(np.isfinite(gradient)))
        scale = float(np.abs(gradient).max())
        self.assertGreater(scale, 0.0)

        rng = np.random.default_rng(1)
        h = 1.0e-6
        probes = rng.choice(positions.shape[0], size=6, replace=False)
        for vertex in probes:
            for axis in range(3):
                shift = np.zeros_like(np.asarray(positions))
                shift[vertex, axis] = h
                plus = float(energy(positions + shift))
                minus = float(energy(positions - shift))
                numeric = (plus - minus) / (2.0 * h)
                self.assertAlmostEqual(
                    gradient[vertex, axis], numeric, delta=1.0e-6 * scale
                )


class VariationalEnergyTests(unittest.TestCase):
    def test_vbd_sweeps_decrease_the_variational_energy_monotonically(self):
        """The paper's Section 3.1 claim, in exactly the regime that makes it.

        Line search on (the alpha grid includes 0, so no vertex is ever forced uphill),
        acceleration off, no contact: each local solve then decreases its G_i against a
        frozen block, colours partition the mesh into non-interacting vertices, so every
        sweep decreases G. The tolerance below covers float summation-order noise only;
        if this test flakes, the descent property is broken -- fix the solver, do not
        widen the tolerance.
        """
        problem, state = _cantilever_problem()
        prescribed_position, _ = evaluate_dirichlet_targets(
            problem.boundary_conditions, float(state.time), float(problem.solver.dt)
        )
        inertial_target = vbd.predict_inertial_target(problem, state)

        def g(positions):
            return float(
                variational_energy(problem, positions, inertial_target, state.position)
            )

        position = state.position
        energies = [g(position)]
        for _ in range(8):
            position = vbd._raw_vbd_iteration(
                problem, position, inertial_target, state.position, prescribed_position
            )
            energies.append(g(position))

        for before, after in zip(energies, energies[1:]):
            self.assertLessEqual(after, before + 1.0e-10 * abs(before))
        # Gravity guarantees the start is far from the minimiser, so the aggregate
        # descent must be strict, not a chain of ties.
        self.assertLess(energies[-1], 0.99 * energies[0])

    def test_variational_energy_is_finite_and_stable_with_contact(self):
        """With contact in play only finiteness is asserted, deliberately.

        The sweep filter rescales whole updates and Chebyshev-adjacent machinery lives in
        the step path, so monotone G is *not* guaranteed here and is not claimed. What
        must hold is that the global energy of a resting-contact configuration is a real
        number: a NaN would mean the potential evaluated a barrier at or through zero gap.
        """
        positions, tets = _beam(nx=2, ny=2, nz=2)
        positions = positions + jnp.asarray([0.0, 0.0, 0.05], dtype=jnp.float64)
        free_mask = jnp.ones((positions.shape[0],), dtype=jnp.float64)
        problem = assemble_problem(
            positions,
            tets,
            free_mask,
            dt=0.01,
            external_acceleration=(0.0, 0.0, -9.81),
            num_iterations=6,
            mu=50.0,
            lam=50.0,
            eps=1.0e-8,
            colliders=[{"kind": "plane", "normal": (0.0, 0.0, 1.0), "offset": 0.0}],
            contact_d_hat=1.0e-3,
        )
        state = initial_state(problem)

        for _ in range(3):
            inertial_target = vbd.predict_inertial_target(problem, state)
            next_state = vbd.step(problem, state)
            total = float(
                variational_energy(
                    problem, next_state.position, inertial_target, state.position
                )
            )
            self.assertTrue(np.isfinite(total))
            self.assertTrue(
                float(
                    potential_energy(problem, next_state.position, state.position)
                )
                >= 0.0
            )
            state = next_state


if __name__ == "__main__":
    unittest.main()
