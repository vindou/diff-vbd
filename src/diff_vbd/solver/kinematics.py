"""Tet kinematics primitives."""

import jax
import jax.numpy as jnp


@jax.jit
def tet_edges_matrix(positions: jnp.ndarray) -> jnp.ndarray:
    """Return the edge matrix [x1-x0, x2-x0, x3-x0]."""
    return jnp.stack(
        [
            positions[1] - positions[0],
            positions[2] - positions[0],
            positions[3] - positions[0],
        ],
        axis=1,
    )


@jax.jit
def tet_volume(rest_positions: jnp.ndarray) -> jnp.ndarray:
    """Return tetrahedron rest volume."""
    dm = tet_edges_matrix(rest_positions)
    return jnp.abs(jnp.linalg.det(dm)) / 6.0


@jax.jit
def deformation_gradient(
    rest_positions: jnp.ndarray, deformed_positions: jnp.ndarray
) -> jnp.ndarray:
    """Return the deformation gradient F = Ds @ inv(Dm)."""
    dm = tet_edges_matrix(rest_positions)
    ds = tet_edges_matrix(deformed_positions)
    return ds @ jnp.linalg.inv(dm)
