"""Library-first API for the differentiable VBD solver."""

from diff_vbd.config import load_config, load_problem_from_yaml
from diff_vbd.export import (
    build_export_metadata,
    export_simulation_npz,
    extract_surface_mesh,
    surface_trajectory_from_history,
)
from diff_vbd.model import (
    AccelerationOptions,
    BoundaryConditions,
    DirichletSpec,
    LineSearchOptions,
    MaterialParams,
    MeshData,
    RigidRegionSpec,
    SimulationProblem,
    SimulationState,
    SolverOptions,
    TopologyData,
)
from diff_vbd.problems import build_cantilever_problem
from diff_vbd.setup.problem_builder import assemble_problem, initial_state
from diff_vbd.setup.boundary_conditions import (
    assemble_boundary_conditions,
    assemble_dirichlet_boundary_conditions,
    evaluate_dirichlet_targets,
)
from diff_vbd.solver import (
    elastic_potential,
    inertia_potential,
    potential_energy,
    simulate,
    step,
    variational_energy,
)

__all__ = [
    "BoundaryConditions",
    "AccelerationOptions",
    "DirichletSpec",
    "LineSearchOptions",
    "MaterialParams",
    "MeshData",
    "RigidRegionSpec",
    "SimulationProblem",
    "SimulationState",
    "SolverOptions",
    "TopologyData",
    "assemble_problem",
    "assemble_boundary_conditions",
    "assemble_dirichlet_boundary_conditions",
    "build_export_metadata",
    "build_cantilever_problem",
    "elastic_potential",
    "evaluate_dirichlet_targets",
    "export_simulation_npz",
    "extract_surface_mesh",
    "inertia_potential",
    "initial_state",
    "load_config",
    "load_problem_from_yaml",
    "potential_energy",
    "simulate",
    "step",
    "surface_trajectory_from_history",
    "variational_energy",
]
