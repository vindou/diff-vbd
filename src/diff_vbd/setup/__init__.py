"""Problem setup and precompute helpers."""

from diff_vbd.setup.boundary_conditions import (
    assemble_boundary_conditions,
    assemble_dirichlet_boundary_conditions,
    build_fixed_boundary_conditions,
    evaluate_dirichlet_targets,
)
from diff_vbd.setup.mesh_io import parse_gmsh22_binary_tets
from diff_vbd.setup.problem_builder import (
    assemble_problem,
    empty_contact_state,
    initial_state,
)
from diff_vbd.setup.selector_io import (
    SelectorClassificationOptions,
    SelectorClassificationStats,
    SelectorMesh,
    SelectorVertexMembership,
    classify_selector_vertices,
    parse_binary_stl_aabb,
    parse_binary_stl_mesh,
)
from diff_vbd.setup.topology import (
    build_incidence,
    build_lumped_masses,
    build_surface_topology,
    build_vertex_adjacency,
    build_vertex_coloring,
)

__all__ = [
    "assemble_problem",
    "assemble_boundary_conditions",
    "assemble_dirichlet_boundary_conditions",
    "build_incidence",
    "build_fixed_boundary_conditions",
    "build_lumped_masses",
    "build_surface_topology",
    "build_vertex_adjacency",
    "build_vertex_coloring",
    "classify_selector_vertices",
    "empty_contact_state",
    "evaluate_dirichlet_targets",
    "initial_state",
    "parse_binary_stl_aabb",
    "parse_binary_stl_mesh",
    "parse_gmsh22_binary_tets",
    "SelectorClassificationOptions",
    "SelectorClassificationStats",
    "SelectorMesh",
    "SelectorVertexMembership",
]
