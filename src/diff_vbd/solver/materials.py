"""Constitutive model helpers."""

import jax
import jax.numpy as jnp

from diff_vbd.model import MaterialParams
from diff_vbd.solver.kinematics import deformation_gradient, tet_volume


@jax.jit
def neo_hookean_energy_density(material: MaterialParams, f: jnp.ndarray) -> jnp.ndarray:
    """Return textbook compressible Neo-Hookean energy density."""
    j = jnp.linalg.det(f)
    ic = jnp.trace(f.T @ f)

    def valid_density():
        log_j = jnp.log(j)
        return 0.5 * material.mu * (ic - 3.0) - material.mu * log_j + 0.5 * (
            material.lam * (log_j**2)
        )

    def invalid_density():
        return jnp.array(1.0e20, dtype=f.dtype)

    return jax.lax.cond(j > 0.0, valid_density, invalid_density)


@jax.jit
def tet_energy(
    material: MaterialParams,
    rest_tet_positions: jnp.ndarray,
    deformed_tet_positions: jnp.ndarray,
) -> jnp.ndarray:
    """Return total elastic energy of one tetrahedron."""
    f = deformation_gradient(rest_tet_positions, deformed_tet_positions)
    return tet_volume(rest_tet_positions) * neo_hookean_energy_density(material, f)
