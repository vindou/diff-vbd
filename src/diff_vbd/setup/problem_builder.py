"""Validated problem assembly for solver-ready inputs."""

import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import jax.numpy as jnp

from diff_vbd.model import (
    CcdParams,
    AccelerationOptions,
    BoundaryConditions,
    ColliderData,
    ContactData,
    ContactParams,
    ContactState,
    LineSearchOptions,
    MaterialParams,
    MeshData,
    SimulationProblem,
    SimulationState,
    SolverOptions,
    TopologyData,
)
from diff_vbd.setup.boundary_conditions import build_fixed_boundary_conditions
from diff_vbd.setup.topology import (
    build_incidence,
    build_lumped_masses,
    build_surface_topology,
    build_vertex_coloring,
)
from diff_vbd.solver.contact.barrier import ACTIVATION_KINDS, barrier_stiffness
from diff_vbd.solver.contact.colliders import COLLIDER_KINDS


def _as_vector3(value, dtype):
    vector = jnp.asarray(value, dtype=dtype)
    if vector.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {vector.shape}")
    return vector


def _validate_mesh(rest_positions: jnp.ndarray, tets: jnp.ndarray):
    if rest_positions.ndim != 2 or rest_positions.shape[1] != 3:
        raise ValueError(
            f"rest_positions must have shape (num_vertices, 3), got {rest_positions.shape}"
        )
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tets must have shape (num_tets, 4), got {tets.shape}")
    if rest_positions.shape[0] == 0 or tets.shape[0] == 0:
        raise ValueError("Mesh must contain at least one vertex and one tet")
    if not jnp.issubdtype(tets.dtype, jnp.integer):
        raise ValueError(f"tets must have an integer dtype, got {tets.dtype}")

    min_index = int(jnp.min(tets))
    max_index = int(jnp.max(tets))
    if min_index < 0 or max_index >= rest_positions.shape[0]:
        raise ValueError(
            f"tet indices must be in [0, {rest_positions.shape[0] - 1}], got [{min_index}, {max_index}]"
        )

    # The elastic energy is orientation sensitive: a tet whose vertices wind the wrong
    # way has det(F) < 0 in its own rest pose, so it is deformed before the sim starts.
    tet_positions = rest_positions[tets]
    edges = jnp.stack(
        [
            tet_positions[:, 1] - tet_positions[:, 0],
            tet_positions[:, 2] - tet_positions[:, 0],
            tet_positions[:, 3] - tet_positions[:, 0],
        ],
        axis=-1,
    )
    signed_volume = jnp.linalg.det(edges) / 6.0
    bad_tets = jnp.nonzero(signed_volume <= 0.0)[0]
    if bad_tets.shape[0] > 0:
        preview = [int(index) for index in bad_tets[:10]]
        raise ValueError(
            f"{bad_tets.shape[0]} tet(s) have non-positive rest volume; reorder their "
            f"vertices so each tet winds positively (first offenders: {preview})"
        )


def _expand_external_acceleration(external_acceleration, num_vertices: int, dtype):
    acceleration = jnp.asarray(external_acceleration, dtype=dtype)
    if acceleration.shape == (3,):
        return jnp.tile(acceleration[None, :], (num_vertices, 1))
    if acceleration.shape == (num_vertices, 3):
        return acceleration
    raise ValueError(
        "external_acceleration must have shape (3,) or (num_vertices, 3), "
        f"got {acceleration.shape}"
    )


def _validate_boundary_conditions(
    boundary_conditions: BoundaryConditions,
    num_vertices: int,
    has_colliders: bool = False,
):
    dirichlet_mask = jnp.asarray(boundary_conditions.dirichlet_mask)
    if dirichlet_mask.ndim != 1 or dirichlet_mask.shape[0] != num_vertices:
        raise ValueError(
            "boundary_conditions.dirichlet_mask must have shape "
            f"({num_vertices},), got {dirichlet_mask.shape}"
        )
    if boundary_conditions.dirichlet_group_indices.shape != (num_vertices,):
        raise ValueError(
            "boundary_conditions.dirichlet_group_indices must have shape "
            f"({num_vertices},), got {boundary_conditions.dirichlet_group_indices.shape}"
        )
    if boundary_conditions.rigid_region_indices.shape != (num_vertices,):
        raise ValueError(
            "boundary_conditions.rigid_region_indices must have shape "
            f"({num_vertices},), got {boundary_conditions.rigid_region_indices.shape}"
        )
    if boundary_conditions.reference_positions.shape != (num_vertices, 3):
        raise ValueError(
            "boundary_conditions.reference_positions must have shape "
            f"({num_vertices}, 3), got {boundary_conditions.reference_positions.shape}"
        )
    for rigid_region in boundary_conditions.rigid_region_specs:
        if rigid_region.vertex_indices.ndim != 1:
            raise ValueError("rigid_region.vertex_indices must be rank 1")
        if rigid_region.reference_local_positions.shape != (
            rigid_region.vertex_indices.shape[0],
            3,
        ):
            raise ValueError(
                "rigid_region.reference_local_positions must have shape "
                f"({rigid_region.vertex_indices.shape[0]}, 3), got "
                f"{rigid_region.reference_local_positions.shape}"
            )
        if rigid_region.reference_com.shape != (3,):
            raise ValueError(
                f"rigid_region.reference_com must have shape (3,), got {rigid_region.reference_com.shape}"
            )

    restricted_mask = jnp.logical_or(
        boundary_conditions.dirichlet_mask,
        boundary_conditions.rigid_mask,
    )
    restricted_count = int(jnp.sum(restricted_mask))
    free_count = int(jnp.sum(~restricted_mask))
    # A body held up only by a collider needs no Dirichlet vertex at all -- resting on the
    # ground is a perfectly well-posed problem -- so the requirement is waived once contact
    # can supply the support.
    if restricted_count == 0 and not has_colliders:
        raise ValueError(
            "Boundary conditions must constrain at least one vertex or rigid region "
            "(or the problem must define a collider to support the body)"
        )
    if free_count == 0:
        raise ValueError("Boundary conditions must leave at least one vertex free")


def _validate_topology(
    incident_tets: jnp.ndarray,
    incident_mask: jnp.ndarray,
    color_groups: jnp.ndarray,
    color_group_mask: jnp.ndarray,
    mass: jnp.ndarray,
    num_vertices: int,
):
    if incident_tets.shape != incident_mask.shape:
        raise ValueError("incident_tets and incident_mask must have the same shape")
    if incident_tets.ndim != 2 or incident_tets.shape[0] != num_vertices:
        raise ValueError("incident_tets must have shape (num_vertices, max_incident)")
    if color_groups.shape != color_group_mask.shape:
        raise ValueError("color_groups and color_group_mask must have the same shape")
    if color_groups.ndim != 2:
        raise ValueError("color_groups must be rank 2")
    if mass.ndim != 1 or mass.shape[0] != num_vertices:
        raise ValueError(f"mass must have shape ({num_vertices},), got {mass.shape}")


def _validate_chebyshev_options(acceleration_enabled: bool, chebyshev_rho: float):
    if acceleration_enabled and not (0.0 < chebyshev_rho < 1.0):
        raise ValueError(
            "chebyshev_rho must satisfy 0 < chebyshev_rho < 1 when acceleration is enabled"
        )


DEFAULT_LINE_SEARCH_NUM_ALPHAS = 9


def _resolve_line_search_alphas(line_search_alphas, line_search_num_alphas):
    """Return the alpha grid, generating a linear one from ``num_alphas`` if needed.

    An explicit alpha list always wins. The generated grid runs from 1.0 down to 0.0
    inclusive: alpha=0 is what lets a vertex decline to move when every positive step
    would increase its objective, which is what makes this a descent safeguard rather
    than a step-size lottery. The grid is linear rather than geometric because a
    geometric grid's extra points collapse toward zero (0.5**16 is already 1.5e-5) and
    stop adding resolution where it matters.
    """
    if line_search_alphas is not None and line_search_num_alphas is not None:
        raise ValueError(
            "specify line_search_alphas or line_search_num_alphas, not both"
        )
    if line_search_alphas is not None:
        return line_search_alphas

    num_alphas = (
        DEFAULT_LINE_SEARCH_NUM_ALPHAS
        if line_search_num_alphas is None
        else line_search_num_alphas
    )
    if int(num_alphas) < 2:
        raise ValueError(
            f"line_search_num_alphas must be at least 2, got {num_alphas}"
        )
    return jnp.linspace(1.0, 0.0, int(num_alphas))


def _validate_line_search_options(line_search_alphas):
    alphas = jnp.asarray(line_search_alphas)
    if alphas.ndim != 1:
        raise ValueError(f"line_search_alphas must be rank 1, got {alphas.shape}")
    if alphas.shape[0] == 0:
        raise ValueError("line_search_alphas must contain at least one alpha")
    if not bool(jnp.all((alphas >= 0.0) & (alphas <= 1.0))):
        raise ValueError("line_search_alphas must satisfy 0 <= alpha <= 1")
    if not bool(jnp.any(alphas > 0.0)):
        raise ValueError(
            "line_search_alphas must contain at least one positive alpha"
        )


def _validate_contact_conditioning(rest_positions: jnp.ndarray, d_hat: float):
    """Reject a d_hat that the float type cannot resolve at this mesh's coordinates.

    The barrier resolves a gap that is orders of magnitude smaller than the mesh
    coordinates, and a distance is computed by *subtracting* those coordinates. In float32
    at coordinate 100 the absolute resolution is ~1.2e-5, so a true gap of 1e-5 comes out
    23% wrong and a small positive gap can round to zero or below -- at which point
    log(g/d_hat) is NaN and intersection-freedom is lost to rounding alone, before any
    solver logic runs. Catch that at assembly rather than as a NaN a thousand steps in.
    """
    if d_hat <= 0.0:
        raise ValueError(f"contact d_hat must be positive, got {d_hat}")

    dtype = rest_positions.dtype
    resolution = float(np.finfo(np.dtype(dtype)).eps)
    max_coordinate = float(jnp.max(jnp.abs(rest_positions)))
    representable = resolution * max(max_coordinate, 1.0)

    # 100x headroom: the gap must survive not just being represented but being subtracted,
    # squared, and pushed through a 1/g gradient.
    if d_hat < 100.0 * representable:
        raise ValueError(
            f"contact d_hat={d_hat:g} is below what {jnp.dtype(dtype).name} can resolve at "
            f"this mesh's coordinates (max |coord| = {max_coordinate:g}, resolution = "
            f"{representable:g}). The barrier would see rounding noise instead of a gap, and "
            f"a positive gap could round negative and produce NaN. Use precision='float64' "
            f"(runtime_config), enlarge d_hat, or rescale the mesh."
        )


def empty_contact_state(num_vertices: int, capacity: int, max_per_vertex: int):
    """Return a zeroed, fully-invalid contact state of the given fixed capacity."""
    return ContactState(
        pair_vertices=jnp.zeros((capacity, 4), dtype=jnp.int32),
        pair_type=jnp.zeros((capacity,), dtype=jnp.int32),
        pair_valid=jnp.zeros((capacity,), dtype=jnp.bool_),
        incident_contacts=jnp.zeros((num_vertices, max_per_vertex), dtype=jnp.int32),
        incident_contact_mask=jnp.zeros(
            (num_vertices, max_per_vertex), dtype=jnp.bool_
        ),
    )


def _resolve_contact_activation(
    activation: str | None, use_barrier: bool | None
) -> int:
    """Resolve the activation name (and the legacy ``use_barrier`` bool) to an int code.

    ``use_barrier`` predates the activation enum and is kept working so existing configs
    and call sites do not silently change behaviour; specifying both is an error rather
    than a precedence rule, because a config that says two different things about the
    contact model should not quietly mean one of them.
    """
    if use_barrier is not None:
        if activation is not None:
            raise ValueError(
                "specify contact_activation or the legacy contact_use_barrier, not both"
            )
        activation = "barrier" if use_barrier else "penalty"
    if activation is None:
        activation = "barrier"
    key = str(activation).upper()
    if key not in ACTIVATION_KINDS:
        raise ValueError(
            f"unknown contact activation {activation!r}; expected one of "
            f"{sorted(name.lower() for name in ACTIVATION_KINDS)}"
        )
    return ACTIVATION_KINDS[key]


def _build_contact_data(
    colliders: Sequence[Mapping[str, object]] | None,
    *,
    d_hat: float,
    kappa: float,
    friction_mu: float,
    eps_v: float,
    activation: int,
    contact_enabled: bool,
    dtype,
    tets=None,
    num_vertices: int = 0,
    self_collision: bool = False,
    self_collision_ccd: bool = True,
    ccd_slack: float = 0.9,
    capacity: int = 0,
    max_per_vertex: int = 0,
) -> ContactData:
    """Build the (fixed-shape) collider buffers.

    An empty collider list yields zero-length arrays, not ``None``: the shapes stay static
    and the vmap over colliders simply iterates zero times, so the contact term costs
    nothing and the pytree structure is identical whether contact is on or off.
    """
    specs = list(colliders or ())

    kinds, normals, offsets, centers, radii, outside, enabled = [], [], [], [], [], [], []
    for index, spec in enumerate(specs):
        kind_name = str(spec.get("kind", "plane")).upper()
        if kind_name not in COLLIDER_KINDS:
            raise ValueError(
                f"collider[{index}] has unknown kind {kind_name!r}; "
                f"expected one of {sorted(COLLIDER_KINDS)}"
            )
        kinds.append(COLLIDER_KINDS[kind_name])

        if kind_name == "PLANE":
            normal = np.asarray(spec.get("normal", (0.0, 0.0, 1.0)), dtype=np.float64)
            norm = float(np.linalg.norm(normal))
            if norm == 0.0:
                raise ValueError(f"collider[{index}] plane normal must be non-zero")
            normals.append(normal / norm)
            offsets.append(float(spec.get("offset", 0.0)))
            centers.append(np.zeros(3))
            radii.append(0.0)
            outside.append(True)
        else:  # SPHERE
            normals.append(np.array([0.0, 0.0, 1.0]))
            offsets.append(0.0)
            centers.append(np.asarray(spec.get("center", (0.0, 0.0, 0.0)), dtype=np.float64))
            radius = float(spec.get("radius", 1.0))
            if radius <= 0.0:
                raise ValueError(f"collider[{index}] sphere radius must be positive")
            radii.append(radius)
            outside.append(bool(spec.get("outside", True)))
        enabled.append(bool(spec.get("enabled", True)))

    def stack(values, shape, value_dtype):
        if not values:
            return jnp.zeros((0, *shape), dtype=value_dtype)
        return jnp.asarray(np.asarray(values), dtype=value_dtype)

    collider_data = ColliderData(
        kind=stack(kinds, (), jnp.int32),
        normal=stack(normals, (3,), dtype),
        offset=stack(offsets, (), dtype),
        center=stack(centers, (3,), dtype),
        radius=stack(radii, (), dtype),
        outside=stack(outside, (), jnp.bool_),
        enabled=stack(enabled, (), jnp.bool_),
    )
    params = ContactParams(
        d_hat=jnp.asarray(d_hat, dtype=dtype),
        kappa=jnp.asarray(kappa, dtype=dtype),
        friction_mu=jnp.asarray(friction_mu, dtype=dtype),
        eps_v=jnp.asarray(eps_v, dtype=dtype),
        activation=jnp.asarray(activation, dtype=jnp.int32),
        enabled=jnp.asarray(
            contact_enabled and (bool(specs) or self_collision), dtype=jnp.bool_
        ),
    )

    if self_collision:
        triangles, edges, surface_vertices = build_surface_topology(tets)
        _validate_closed_surface(triangles, edges, surface_vertices)
    else:
        triangles = jnp.zeros((0, 3), dtype=jnp.int32)
        edges = jnp.zeros((0, 2), dtype=jnp.int32)

    ccd = CcdParams(
        slack=jnp.asarray(ccd_slack, dtype=dtype),
        # Both are refreshed from the mesh's actual motion at every detection. These are the
        # at-rest values: a body that is not moving may still travel d_hat, and the band is
        # then the usual 2 * d_hat.
        max_displacement=jnp.asarray(d_hat, dtype=dtype),
        detection_band=jnp.asarray(2.0 * d_hat, dtype=dtype),
        enabled=jnp.asarray(self_collision and self_collision_ccd, dtype=jnp.bool_),
    )

    return ContactData(
        params=params,
        colliders=collider_data,
        state=empty_contact_state(num_vertices, capacity, max_per_vertex),
        ccd=ccd,
        surface_triangles=triangles,
        surface_edges=edges,
    )


def _validate_closed_surface(triangles, edges, surface_vertices) -> None:
    """Reject a self-collision mesh whose surface is not a closed manifold.

    The test is that **every surface edge is shared by exactly two triangles**. That is the
    defining local property of a closed manifold, and unlike the Euler characteristic it does
    not care how many separate bodies the mesh contains -- two blocks are two closed surfaces,
    and ``V - E + F`` is then 4 rather than 2.

    This is not a formality. Boundary faces are found by counting how many tets own each face,
    so a *non-conforming* tet split -- one where neighbouring cells disagree about how to cut a
    shared face -- leaves interior faces unpaired, and they are then reported as *surface*
    faces. Self-collision duly finds contacts on them: phantom pairs deep inside a solid body,
    at rest, pushing it apart from within. The 5-tet cube split does exactly this unless
    adjacent cells alternate on a checkerboard (the Kuhn 6-tet split conforms unconditionally).

    The failure is silent and its symptom is remote from its cause, which is what makes it
    worth an assembly-time error: a raw triangle count looks entirely reasonable while being
    wrong.
    """
    faces = np.asarray(triangles)
    if faces.shape[0] == 0:
        return

    edge_pairs = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    canonical = np.sort(edge_pairs, axis=1)
    _, counts = np.unique(canonical, axis=0, return_counts=True)

    if not np.all(counts == 2):
        bad = int(np.sum(counts != 2))
        raise ValueError(
            f"self-collision needs a closed surface, but {bad} of "
            f"{counts.size} surface edges are not shared by exactly two triangles "
            f"(degrees seen: {sorted(set(counts.tolist()))}). The mesh is not a conforming "
            f"tetrahedralisation: neighbouring cells disagree about how to cut a shared face, "
            f"so interior faces go unpaired and get reported as *surface* faces. "
            f"Self-collision would then find phantom contacts inside the solid at rest, "
            f"pushing the body apart from within. If this is a structured grid, use the Kuhn "
            f"(Freudenthal) 6-tet split, which conforms unconditionally, rather than the "
            f"5-tet split (which only conforms if adjacent cells alternate on a "
            f"checkerboard)."
        )


def assemble_problem(
    rest_positions: jnp.ndarray,
    tets: jnp.ndarray,
    boundary_data: jnp.ndarray | BoundaryConditions,
    *,
    dt=0.02,
    external_acceleration: Iterable[float] | jnp.ndarray = (0.0, 0.0, -9.81),
    mu=4.638e5,
    lam=4.174e6,
    density=1.0,
    eps=1.0e-6,
    num_iterations: int = 10,
    acceleration_enabled: bool = False,
    chebyshev_rho: float = 0.95,
    line_search_enabled: bool = True,
    line_search_alphas: Iterable[float] | jnp.ndarray | None = None,
    line_search_num_alphas: int | None = None,
    colliders: Sequence[Mapping[str, object]] | None = None,
    contact_d_hat: float = 1.0e-3,
    contact_kappa: float | None = None,
    contact_friction_mu: float = 0.0,
    contact_eps_v: float = 1.0e-3,
    contact_activation: str | None = None,
    contact_use_barrier: bool | None = None,
    contact_enabled: bool = True,
    self_collision: bool = False,
    self_collision_ccd: bool = True,
    ccd_slack: float = 0.9,
    contact_capacity: int = 4096,
    contact_max_per_vertex: int = 192,
) -> SimulationProblem:
    """Assemble a validated solver-ready problem from mesh and setup arrays."""
    rest_positions = jnp.asarray(rest_positions)
    dtype = rest_positions.dtype
    tets = jnp.asarray(tets, dtype=jnp.int32)

    _validate_mesh(rest_positions, tets)
    _validate_chebyshev_options(acceleration_enabled, chebyshev_rho)
    line_search_alphas = _resolve_line_search_alphas(
        line_search_alphas, line_search_num_alphas
    )
    _validate_line_search_options(line_search_alphas)
    if isinstance(boundary_data, BoundaryConditions):
        boundary_conditions = boundary_data
    else:
        free_mask = jnp.asarray(boundary_data, dtype=dtype)
        if free_mask.ndim != 1 or free_mask.shape[0] != rest_positions.shape[0]:
            raise ValueError(
                f"free_mask must have shape ({rest_positions.shape[0]},), got {free_mask.shape}"
            )
        constrained_mask = free_mask == 0
        boundary_conditions = build_fixed_boundary_conditions(
            rest_positions, constrained_mask
        )
    _validate_boundary_conditions(
        boundary_conditions,
        rest_positions.shape[0],
        has_colliders=bool(colliders) or self_collision,
    )

    incident_tets, incident_mask = build_incidence(tets, rest_positions.shape[0])
    color_groups, color_group_mask, _ = build_vertex_coloring(
        tets, rest_positions.shape[0]
    )
    mass = build_lumped_masses(rest_positions, tets) * jnp.asarray(density, dtype=dtype)
    _validate_topology(
        incident_tets,
        incident_mask,
        color_groups,
        color_group_mask,
        mass,
        rest_positions.shape[0],
    )

    mesh = MeshData(rest_positions=rest_positions, tets=tets)
    topology = TopologyData(
        incident_tets=incident_tets,
        incident_mask=incident_mask,
        color_groups=color_groups,
        color_group_mask=color_group_mask,
        mass=mass,
    )
    material = MaterialParams(
        mu=jnp.asarray(mu, dtype=dtype),
        lam=jnp.asarray(lam, dtype=dtype),
        density=jnp.asarray(density, dtype=dtype),
    )
    solver = SolverOptions(
        dt=jnp.asarray(dt, dtype=dtype),
        external_acceleration=_expand_external_acceleration(
            external_acceleration, rest_positions.shape[0], dtype
        ),
        eps=jnp.asarray(eps, dtype=dtype),
        iteration_schedule=jnp.arange(num_iterations, dtype=jnp.int32),
        acceleration=AccelerationOptions(
            enabled=jnp.asarray(acceleration_enabled, dtype=jnp.bool_),
            chebyshev_rho=jnp.asarray(chebyshev_rho, dtype=dtype),
        ),
        line_search=LineSearchOptions(
            enabled=jnp.asarray(line_search_enabled, dtype=jnp.bool_),
            alphas=jnp.asarray(line_search_alphas, dtype=dtype),
        ),
    )

    if colliders or self_collision:
        _validate_contact_conditioning(rest_positions, contact_d_hat)

    if not 0.0 < ccd_slack < 1.0:
        raise ValueError(
            f"ccd_slack must lie strictly in (0, 1), got {ccd_slack}. At 1 a step may close "
            f"the whole gap, which puts the barrier's argument at zero; at 0 nothing moves."
        )

    if self_collision and not self_collision_ccd:
        warnings.warn(
            "self_collision is on but self_collision_ccd is off: mesh-mesh contact is a "
            "barrier penalty only, with NO intersection-free guarantee. Two surfaces driven "
            "together hard enough will pass through each other. This is a deliberate escape "
            "hatch (the CCD's detection band grows with speed, and can be too expensive); it "
            "is not a safe default.",
            RuntimeWarning,
            stacklevel=2,
        )

    if contact_kappa is None:
        # Match the barrier's stiffness to the inertia term's, so it resists without
        # swamping the local solve. See contact.barrier.barrier_stiffness.
        average_mass = float(jnp.mean(mass))
        contact_kappa = barrier_stiffness(average_mass, float(dt), contact_d_hat)

    contact = _build_contact_data(
        colliders,
        d_hat=contact_d_hat,
        kappa=contact_kappa,
        friction_mu=contact_friction_mu,
        eps_v=contact_eps_v,
        activation=_resolve_contact_activation(
            contact_activation, contact_use_barrier
        ),
        contact_enabled=contact_enabled,
        dtype=dtype,
        tets=tets,
        num_vertices=rest_positions.shape[0],
        self_collision=self_collision,
        self_collision_ccd=self_collision_ccd,
        ccd_slack=ccd_slack,
        capacity=contact_capacity if self_collision else 0,
        max_per_vertex=contact_max_per_vertex if self_collision else 0,
    )

    return SimulationProblem(
        mesh=mesh,
        topology=topology,
        boundary_conditions=boundary_conditions,
        material=material,
        solver=solver,
        contact=contact,
    )


def initial_state(
    problem: SimulationProblem,
    position: jnp.ndarray | None = None,
    velocity: jnp.ndarray | None = None,
    time: jnp.ndarray | float | None = None,
) -> SimulationState:
    """Create a simulation state compatible with a problem definition."""
    if position is None:
        position = problem.mesh.rest_positions
    if velocity is None:
        velocity = jnp.zeros_like(problem.mesh.rest_positions)
    if time is None:
        time = jnp.array(0.0, dtype=problem.mesh.rest_positions.dtype)

    position = jnp.asarray(position, dtype=problem.mesh.rest_positions.dtype)
    velocity = jnp.asarray(velocity, dtype=problem.mesh.rest_positions.dtype)
    time = jnp.asarray(time, dtype=problem.mesh.rest_positions.dtype)

    expected_shape = problem.mesh.rest_positions.shape
    if position.shape != expected_shape:
        raise ValueError(
            f"position must have shape {expected_shape}, got {position.shape}"
        )
    if velocity.shape != expected_shape:
        raise ValueError(
            f"velocity must have shape {expected_shape}, got {velocity.shape}"
        )
    if time.shape != ():
        raise ValueError(f"time must be a scalar, got {time.shape}")

    return SimulationState(position=position, velocity=velocity, time=time)
