"""Cantilever beam problem assembly."""

from pathlib import Path

import jax.numpy as jnp

from diff_vbd.model import DirichletSpec
from diff_vbd.setup import (
    SelectorClassificationOptions,
    assemble_problem,
    assemble_dirichlet_boundary_conditions,
    classify_selector_vertices,
    parse_binary_stl_mesh,
    parse_gmsh22_binary_tets,
)


def build_cantilever_problem(
    mesh_path: str | Path = "beam.msh",
    selector_path: str | Path = "bc_selector.stl",
    *,
    dtype=jnp.float32,
    num_iterations: int = 10,
    dt=0.02,
    external_acceleration=(0.0, 0.0, -9.81),
    mu=100000000.0,
    lam=10.0,
    eps=1.0e-6,
    acceleration_enabled: bool = False,
    chebyshev_rho: float = 0.95,
    selector_classification: SelectorClassificationOptions | None = None,
):
    """Build the cantilever beam problem from mesh and selector files."""
    rest_positions, tets = parse_gmsh22_binary_tets(mesh_path, dtype=dtype)
    selector = parse_binary_stl_mesh(selector_path)
    membership = classify_selector_vertices(
        rest_positions, selector, options=selector_classification
    )
    boundary_conditions = assemble_dirichlet_boundary_conditions(
        rest_positions,
        [membership],
        [
            DirichletSpec(
                selector_name=membership.selector_name,
                mode="position",
                components=("0.0", "0.0", "0.0"),
            )
        ],
    )
    return assemble_problem(
        rest_positions,
        tets,
        boundary_conditions,
        dt=dt,
        external_acceleration=external_acceleration,
        mu=mu,
        lam=lam,
        eps=eps,
        num_iterations=num_iterations,
        acceleration_enabled=acceleration_enabled,
        chebyshev_rho=chebyshev_rho,
    )
