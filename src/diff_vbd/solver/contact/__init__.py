"""Contact modelling: activation energies, distances, colliders, bounds, friction.

The package is split along one line, and the split is load bearing:

* The **combinatorial layer** (detection, active set, distance-type classification,
  conservative bounds, time-of-impact) is integer- or bound-valued, runs on the host or
  under ``stop_gradient``, and carries no gradient. It emits frozen fixed-shape buffers.
* The **smooth layer** (distances given a frozen classification, activation energies,
  friction) is a differentiable function of vertex positions and is the only part ever
  differentiated.

Keeping the line clean is what makes the whole thing jit-able (static shapes), general
(one force-element abstraction) and differentiable at once.
"""

from diff_vbd.solver.contact.barrier import (
    ACTIVATION_KINDS,
    activation_energy,
    barrier_energy,
    barrier_stiffness,
    penalty_energy,
    two_stage_energy,
)
from diff_vbd.solver.contact.bounds import (
    build_vertex_bounds,
    derive_bounds_band,
    redetection_threshold,
    truncate_to_bounds,
)
from diff_vbd.solver.contact.ccd import vertex_time_of_impact
from diff_vbd.solver.contact.colliders import (
    COLLIDER_KINDS,
    collider_normal,
    collider_signed_distance,
)
from diff_vbd.solver.contact.detection import (
    PAIR_TYPES,
    build_contact_incidence,
    detect_contact_pairs,
)
from diff_vbd.solver.contact.distances import (
    EDGE_EDGE_TYPES,
    POINT_TRIANGLE_TYPES,
    classify_edge_edge,
    classify_point_triangle,
    distance_from_squared,
    edge_edge_distance_sq,
    edge_edge_mollifier,
    edge_edge_mollifier_threshold,
    pair_distance_sq,
    pair_gap_and_mollifier,
    point_edge_distance_sq,
    point_point_distance_sq,
    point_triangle_distance_sq,
)
from diff_vbd.solver.contact.friction import (
    collider_friction_energy,
    contact_normal_force,
    pair_friction_energy,
    smooth_friction_f0,
    smooth_friction_f1,
    tangent_basis,
)
from diff_vbd.solver.contact.potential import (
    collider_contact_energy,
    colliding_vertex_mask,
    contact_potential,
    incident_pair_energy,
    incident_pair_min_gap,
    pair_contact_energy,
)

__all__ = [
    "ACTIVATION_KINDS",
    "COLLIDER_KINDS",
    "EDGE_EDGE_TYPES",
    "PAIR_TYPES",
    "POINT_TRIANGLE_TYPES",
    "activation_energy",
    "barrier_energy",
    "barrier_stiffness",
    "build_contact_incidence",
    "build_vertex_bounds",
    "classify_edge_edge",
    "classify_point_triangle",
    "collider_contact_energy",
    "collider_friction_energy",
    "collider_normal",
    "collider_signed_distance",
    "colliding_vertex_mask",
    "contact_normal_force",
    "contact_potential",
    "derive_bounds_band",
    "detect_contact_pairs",
    "distance_from_squared",
    "edge_edge_distance_sq",
    "edge_edge_mollifier",
    "edge_edge_mollifier_threshold",
    "incident_pair_energy",
    "incident_pair_min_gap",
    "pair_contact_energy",
    "pair_distance_sq",
    "pair_friction_energy",
    "pair_gap_and_mollifier",
    "penalty_energy",
    "point_edge_distance_sq",
    "point_point_distance_sq",
    "point_triangle_distance_sq",
    "redetection_threshold",
    "smooth_friction_f0",
    "smooth_friction_f1",
    "tangent_basis",
    "truncate_to_bounds",
    "two_stage_energy",
]
