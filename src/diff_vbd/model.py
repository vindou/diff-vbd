"""Core immutable data models for the VBD solver."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from diff_vbd.pytree import pytree_dataclass


@pytree_dataclass
class MeshData:
    rest_positions: jnp.ndarray
    tets: jnp.ndarray


@pytree_dataclass
class TopologyData:
    incident_tets: jnp.ndarray
    incident_mask: jnp.ndarray
    color_groups: jnp.ndarray
    color_group_mask: jnp.ndarray
    mass: jnp.ndarray


@dataclass(frozen=True)
class DirichletSpec:
    selector_name: str
    mode: str
    components: tuple[str, str, str]


@dataclass(frozen=True)
class RigidRegionSpec:
    selector_name: str
    vertex_indices: jnp.ndarray
    reference_local_positions: jnp.ndarray
    reference_com: jnp.ndarray


@jax.tree_util.register_pytree_node_class
class BoundaryConditions:
    """Mixed Dirichlet and rigid-region boundary conditions."""

    def __init__(
        self,
        dirichlet_mask: jnp.ndarray,
        dirichlet_group_indices: jnp.ndarray,
        reference_positions: jnp.ndarray,
        dirichlet_specs: tuple[DirichletSpec, ...] = (),
        rigid_region_indices: jnp.ndarray | None = None,
        rigid_region_specs: tuple[RigidRegionSpec, ...] = (),
    ):
        reference_positions = jnp.asarray(reference_positions)
        self.dirichlet_mask = jnp.asarray(dirichlet_mask).astype(bool)
        self.dirichlet_group_indices = jnp.asarray(
            dirichlet_group_indices, dtype=jnp.int32
        )
        self.reference_positions = reference_positions
        self.dirichlet_specs = tuple(dirichlet_specs)
        if rigid_region_indices is None:
            rigid_region_indices = -jnp.ones(
                (reference_positions.shape[0],), dtype=jnp.int32
            )
        self.rigid_region_indices = jnp.asarray(rigid_region_indices, dtype=jnp.int32)
        self.rigid_region_specs = tuple(rigid_region_specs)

    def tree_flatten(self):
        children = [
            self.dirichlet_mask,
            self.dirichlet_group_indices,
            self.reference_positions,
            self.rigid_region_indices,
        ]
        for spec in self.rigid_region_specs:
            children.extend(
                (
                    spec.vertex_indices,
                    spec.reference_local_positions,
                    spec.reference_com,
                )
            )
        aux_data = (
            self.dirichlet_specs,
            tuple(spec.selector_name for spec in self.rigid_region_specs),
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dirichlet_specs, rigid_selector_names = aux_data
        dirichlet_mask, dirichlet_group_indices, reference_positions, rigid_region_indices = (
            children[:4]
        )
        rigid_region_specs = []
        child_index = 4
        for selector_name in rigid_selector_names:
            rigid_region_specs.append(
                RigidRegionSpec(
                    selector_name=selector_name,
                    vertex_indices=children[child_index],
                    reference_local_positions=children[child_index + 1],
                    reference_com=children[child_index + 2],
                )
            )
            child_index += 3
        return cls(
            dirichlet_mask=dirichlet_mask,
            dirichlet_group_indices=dirichlet_group_indices,
            reference_positions=reference_positions,
            dirichlet_specs=dirichlet_specs,
            rigid_region_indices=rigid_region_indices,
            rigid_region_specs=tuple(rigid_region_specs),
        )

    @property
    def constrained_mask(self) -> jnp.ndarray:
        return self.dirichlet_mask

    @property
    def group_indices(self) -> jnp.ndarray:
        return self.dirichlet_group_indices

    @property
    def rigid_mask(self) -> jnp.ndarray:
        return self.rigid_region_indices >= 0

    @property
    def free_mask(self) -> jnp.ndarray:
        restricted_mask = jnp.logical_or(self.dirichlet_mask, self.rigid_mask)
        return jnp.where(restricted_mask, 0.0, 1.0).astype(
            self.reference_positions.dtype
        )


@pytree_dataclass
class MaterialParams:
    mu: jnp.ndarray
    lam: jnp.ndarray
    density: jnp.ndarray


@pytree_dataclass
class ColliderData:
    """Analytic rigid colliders, one row per collider.

    Every collider carries every parameter and the unused ones are ignored, which keeps
    the buffers rectangular and the shapes static. ``kind`` is an int32 code rather than a
    string: every field of a ``pytree_dataclass`` is a traced leaf, and a string leaf is a
    hard jit failure the moment the problem reaches a jitted function.
    """

    kind: jnp.ndarray  # (K,) int32, see contact.colliders.COLLIDER_KINDS
    normal: jnp.ndarray  # (K, 3) plane normal, pointing into the free half-space
    offset: jnp.ndarray  # (K,) plane offset
    center: jnp.ndarray  # (K, 3) sphere centre
    radius: jnp.ndarray  # (K,) sphere radius
    outside: jnp.ndarray  # (K,) bool: stay outside the sphere, rather than inside
    enabled: jnp.ndarray  # (K,) bool


@pytree_dataclass
class ContactParams:
    d_hat: jnp.ndarray  # activation distance, in length units
    kappa: jnp.ndarray  # activation stiffness (k_c in OGC's notation)
    friction_mu: jnp.ndarray  # Coulomb coefficient
    eps_v: jnp.ndarray  # friction static/dynamic transition velocity
    activation: jnp.ndarray  # (,) int32, see contact.barrier.ACTIVATION_KINDS
    enabled: jnp.ndarray  # bool


@pytree_dataclass
class ContactState:
    """The frozen output of the combinatorial layer: rebuilt each detection.

    Every array is fixed-shape and padded. Rebuilding this each step with fresh contents of
    the same shape is a jit cache hit; a changing capacity is a full recompile of the solver
    every step, so the capacity is chosen once and overflow is an error.

    All of it is integer- or mask-valued and none of it is ever differentiated. Capacity is
    read from ``.shape``, never stored as an int field -- an int field on a pytree_dataclass
    becomes a *tracer* and cannot size an array.
    """

    pair_vertices: jnp.ndarray  # (C, 4) int32; vertex-triangle uses [v, t0, t1, t2]
    pair_type: jnp.ndarray  # (C,) int32, see contact.detection.PAIR_TYPES
    pair_valid: jnp.ndarray  # (C,) bool
    incident_contacts: jnp.ndarray  # (N, K) int32, pair indices touching each vertex
    incident_contact_mask: jnp.ndarray  # (N, K) bool


@pytree_dataclass
class CcdParams:
    """The parameters of the mesh-mesh intersection-free guarantee.

    ``max_displacement`` and ``detection_band`` are **refreshed every step** (they depend on
    how fast the mesh is currently moving) and they are a matched pair: the band is only large
    enough to make the pair set complete *because* the clamp stops any vertex leaving it. They
    are derived together, in one place, so they cannot drift apart -- and they live here, as
    array leaves, so rebuilding them each step is a jit cache hit rather than a recompile.
    """

    slack: jnp.ndarray  # fraction of the gap a step may consume, in (0, 1)
    max_displacement: jnp.ndarray  # Δmax: how far a vertex may stray from the step start
    detection_band: jnp.ndarray  # the radius the pair set was built with
    enabled: jnp.ndarray  # bool: the mesh-mesh guarantee (colliders are always filtered)


@pytree_dataclass
class ContactData:
    params: ContactParams
    colliders: ColliderData
    state: ContactState
    ccd: CcdParams
    surface_triangles: jnp.ndarray  # (F, 3) int32, global indices, outward wound
    surface_edges: jnp.ndarray  # (E, 2) int32, global indices


@pytree_dataclass
class AccelerationOptions:
    enabled: jnp.ndarray
    chebyshev_rho: jnp.ndarray


@pytree_dataclass
class LineSearchOptions:
    enabled: jnp.ndarray
    alphas: jnp.ndarray


@pytree_dataclass
class SolverOptions:
    dt: jnp.ndarray
    external_acceleration: jnp.ndarray
    eps: jnp.ndarray
    iteration_schedule: jnp.ndarray
    acceleration: AccelerationOptions
    line_search: LineSearchOptions


@pytree_dataclass
class SimulationProblem:
    mesh: MeshData
    topology: TopologyData
    boundary_conditions: BoundaryConditions
    material: MaterialParams
    solver: SolverOptions
    contact: ContactData


@pytree_dataclass
class SimulationState:
    position: jnp.ndarray
    velocity: jnp.ndarray
    time: jnp.ndarray
