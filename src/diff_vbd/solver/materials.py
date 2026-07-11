"""Constitutive model helpers."""

import jax
import jax.numpy as jnp

from diff_vbd.model import MaterialParams
from diff_vbd.solver.kinematics import deformation_gradient, tet_volume


@jax.jit
def stable_neo_hookean_energy_density(
    material: MaterialParams, f: jnp.ndarray
) -> jnp.ndarray:
    """Return the stable Neo-Hookean energy density (Smith et al. 2018).

    ``material.mu`` and ``material.lam`` are the physical Lame parameters. They are
    remapped to the model parameters so the small-strain response still matches
    linear elasticity, and the energy is finite and smooth for every ``f``, including
    inverted elements where ``det(f) <= 0``.
    """
    mu = (4.0 / 3.0) * material.mu
    lam = material.lam + (5.0 / 6.0) * material.mu
    alpha = 1.0 + 0.75 * mu / lam

    ic = jnp.sum(f * f)
    f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]
    j = jnp.dot(f0, jnp.cross(f1, f2))

    # alpha zeroes the rest-state stress, not the rest-state energy; subtract the
    # leftover constant so an undeformed element reports zero energy.
    rest_density = 0.5 * lam * (1.0 - alpha) ** 2 - 0.5 * mu * jnp.log(4.0)

    return (
        0.5 * mu * (ic - 3.0)
        + 0.5 * lam * (j - alpha) ** 2
        - 0.5 * mu * jnp.log(ic + 1.0)
        - rest_density
    )


@jax.jit
def tet_energy(
    material: MaterialParams,
    rest_tet_positions: jnp.ndarray,
    deformed_tet_positions: jnp.ndarray,
) -> jnp.ndarray:
    """Return total elastic energy of one tetrahedron."""
    f = deformation_gradient(rest_tet_positions, deformed_tet_positions)
    return tet_volume(rest_tet_positions) * stable_neo_hookean_energy_density(
        material, f
    )
