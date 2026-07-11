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

    Every small quantity is formed without cancellation, which matters because the
    solver spends its life near ``f = I``. Writing the deviatoric term as the textbook
    ``sum(f * f) - 3`` subtracts two numbers that are both close to 3, and once that is
    scaled by ``mu`` the float32 rounding error swamps the real energy difference
    between nearby candidate positions. The forms below are algebraically identical and
    agree to machine precision in float64, but they resolve the energy near rest about
    two orders of magnitude more finely in float32 -- enough for the line search to rank
    candidate step sizes by signal rather than by noise. The rest-state energy is zero
    by construction here (alpha only zeroes the rest *stress*), so no constant is
    subtracted afterwards.
    """
    mu = (4.0 / 3.0) * material.mu
    lam = material.lam + (5.0 / 6.0) * material.mu
    alpha = 1.0 + 0.75 * mu / lam

    identity = jnp.eye(3, dtype=f.dtype)
    ic_minus_3 = jnp.sum((f - identity) * (f + identity))  # == sum(f * f) - 3

    f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]
    j = jnp.dot(f0, jnp.cross(f1, f2))

    # (j - alpha)**2 - (1 - alpha)**2, expanded to keep the small factor (j - 1) exact.
    volumetric = (j - 1.0) * (j + 1.0 - 2.0 * alpha)
    # (ic - 3) - [log(ic + 1) - log(4)], with the bracket as log1p to avoid cancelling.
    deviatoric = ic_minus_3 - jnp.log1p(ic_minus_3 / 4.0)

    return 0.5 * mu * deviatoric + 0.5 * lam * volumetric


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
