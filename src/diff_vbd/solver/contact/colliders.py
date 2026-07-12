"""Analytic colliders: closed-form signed distance from a vertex to a rigid obstacle.

These are the cheap path and the general-convenience path at once. A plane or a sphere has
a closed-form distance per vertex, so there is no pair list, no broad phase, and no
classification: the combinatorial layer collapses to a constant mask and the whole contact
element is a pure function of one vertex position. That makes them trivially jit-able and
trivially differentiable.

The distance is **signed**: negative means the vertex is on the wrong side. The barrier
consumes the sign directly, so a vertex that does get pushed through is driven back to the
correct side rather than being expelled further out the far side (which is what a squared
distance would do).

Collider *kinds* are ``int32`` codes rather than strings, because every field of a
``pytree_dataclass`` is a traced leaf and a string leaf is a hard jit failure.
"""

import jax
import jax.numpy as jnp

COLLIDER_KINDS = {
    "PLANE": 0,
    "SPHERE": 1,
}


@jax.jit
def plane_signed_distance(
    position: jnp.ndarray, normal: jnp.ndarray, offset: jnp.ndarray
) -> jnp.ndarray:
    """Return the signed distance from ``position`` to a plane.

    The plane is ``{x : dot(normal, x) == offset}`` with ``normal`` pointing into the
    free half-space, so the result is positive on the allowed side.
    """
    return jnp.dot(normal, position) - offset


@jax.jit
def sphere_signed_distance(
    position: jnp.ndarray,
    center: jnp.ndarray,
    radius: jnp.ndarray,
    outside: jnp.ndarray,
) -> jnp.ndarray:
    """Return the signed distance from ``position`` to a sphere's surface.

    ``outside`` true means the vertex is constrained to stay *outside* the sphere (a ball
    obstacle); false means it is constrained to stay *inside* (a spherical container).
    """
    offset = position - center
    # Softened so the derivative is finite at the centre, where the direction is undefined.
    distance = jnp.sqrt(jnp.dot(offset, offset) + 1.0e-24)
    return jnp.where(outside, distance - radius, radius - distance)


@jax.jit
def collider_normal(
    position: jnp.ndarray,
    kind: jnp.ndarray,
    normal: jnp.ndarray,
    center: jnp.ndarray,
    outside: jnp.ndarray,
) -> jnp.ndarray:
    """Return the unit contact normal at ``position``, pointing into the free space.

    This is the gradient of the signed distance. It is evaluated at the lagged (step-start)
    position and frozen, so it is never differentiated -- which is what makes the friction
    potential below a well-posed, smooth function of the current position.
    """

    def plane():
        return normal

    def sphere():
        offset = position - center
        direction = offset / jnp.sqrt(jnp.dot(offset, offset) + 1.0e-24)
        return jnp.where(outside, direction, -direction)

    return jax.lax.switch(kind, (plane, sphere))


@jax.jit
def collider_signed_distance(
    position: jnp.ndarray,
    kind: jnp.ndarray,
    normal: jnp.ndarray,
    offset: jnp.ndarray,
    center: jnp.ndarray,
    radius: jnp.ndarray,
    outside: jnp.ndarray,
) -> jnp.ndarray:
    """Dispatch on the collider kind and return the signed distance.

    All collider parameters are passed for every kind and the unused ones are ignored;
    that keeps the buffers rectangular and the shapes static, which is what a padded,
    fixed-capacity contact set requires.
    """
    return jax.lax.switch(
        kind,
        (
            lambda: plane_signed_distance(position, normal, offset),
            lambda: sphere_signed_distance(position, center, radius, outside),
        ),
    )
