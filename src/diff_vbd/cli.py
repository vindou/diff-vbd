#!/usr/bin/env python3
"""Minimal YAML-driven simulation runner and NPZ exporter."""

from __future__ import annotations

import argparse
from pathlib import Path

from diff_vbd.runtime_config import apply_runtime_config, collect_runtime_report


def _log_stage(stage_index: int, total_stages: int, message: str):
    print(f"[{stage_index}/{total_stages}] {message}")


def _log_stage_detail(stage_index: int, total_stages: int, message: str):
    print(f"[{stage_index}/{total_stages}]   {message}")


def _load_yaml_output_path(config_path: Path) -> Path | None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read YAML output paths from the simulation config."
        ) from exc

    raw_config = yaml.safe_load(config_path.read_text())
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Simulation config must be a mapping, got {type(raw_config).__name__}"
        )

    raw_output = raw_config.get("output")
    if raw_output is None:
        return None
    if not isinstance(raw_output, str) or not raw_output:
        raise ValueError("config output must be a non-empty string path")

    output_path = Path(raw_output)
    if not output_path.is_absolute():
        output_path = (config_path.parent / output_path).resolve()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a diff_vbd simulation from YAML and export an NPZ trajectory."
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default="gpu",
        help="JAX platform to use. Default: gpu.",
    )
    parser.add_argument(
        "--gpu-preallocate",
        choices=("true", "false"),
        default=None,
        help="Whether to enable XLA GPU preallocation. Default for GPU: false.",
    )
    parser.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=None,
        help="Optional XLA GPU memory fraction override.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the YAML simulation config.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional override for the number of timesteps to simulate. Default: use simulation.steps from the YAML config.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Path to the output NPZ archive. Overrides the YAML output field when set. "
            "Default if unset: YAML output or simulation_export.npz."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_stages = 6
    _log_stage(1, total_stages, f"Applying runtime configuration (platform={args.platform})")
    apply_runtime_config(
        platform=args.platform,
        gpu_preallocate=(
            None if args.gpu_preallocate is None else args.gpu_preallocate == "true"
        ),
        gpu_mem_fraction=args.gpu_mem_fraction,
    )
    from diff_vbd import (
        DirichletSpec,
        export_simulation_npz,
        initial_state,
        load_config,
        simulate,
    )
    from diff_vbd.setup import (
        SelectorClassificationOptions,
        SelectorVertexMembership,
        assemble_boundary_conditions,
        assemble_problem,
        classify_selector_vertices,
        parse_binary_stl_mesh,
        parse_gmsh22_binary_tets,
    )

    _log_stage(2, total_stages, "Collecting runtime backend report")
    runtime_report = collect_runtime_report()
    _log_stage(3, total_stages, f"Loading YAML config from {args.config}")
    config_output_path = _load_yaml_output_path(args.config)
    config = load_config(args.config)
    selector_options = SelectorClassificationOptions(
        mode=config.selector_classification.mode,
        grid_resolution=config.selector_classification.grid_resolution,
        atol=config.selector_classification.atol,
    )
    output_path = (
        args.out
        if args.out is not None
        else config_output_path
        if config_output_path is not None
        else Path("simulation_export.npz")
    )
    if args.out is not None:
        _log_stage_detail(3, total_stages, f"Using CLI output path {output_path}")
    elif config_output_path is not None:
        _log_stage_detail(3, total_stages, f"Using YAML output path {output_path}")
    else:
        _log_stage_detail(
            3, total_stages, f"Using default output path {output_path}"
        )
    _log_stage(4, total_stages, "Building simulation problem and initial state")
    _log_stage_detail(4, total_stages, f"Loading tet mesh from {config.mesh.path}")
    rest_positions, tets = parse_gmsh22_binary_tets(config.mesh.path)
    _log_stage_detail(
        4,
        total_stages,
        f"Loaded mesh with {rest_positions.shape[0]} vertices and {tets.shape[0]} tetrahedra",
    )

    selector_memberships = []
    total_selectors = len(config.selectors)
    for selector_index, (selector_name, selector_path) in enumerate(
        config.selectors.items(), start=1
    ):
        _log_stage_detail(
            4,
            total_stages,
            f"Selector {selector_index}/{total_selectors}: parsing {selector_name} from {selector_path}",
        )
        selector_mesh = parse_binary_stl_mesh(selector_path)
        _log_stage_detail(
            4,
            total_stages,
            f"Selector {selector_name}: {selector_mesh.triangles.shape[0]} triangles, classifying mesh vertices",
        )
        membership = classify_selector_vertices(
            rest_positions, selector_mesh, options=selector_options
        )
        if membership.stats is not None and membership.stats.mode == "exact":
            _log_stage_detail(
                4,
                total_stages,
                (
                    f"Selector {selector_name}: {membership.stats.candidate_count} AABB candidates, "
                    f"grid {membership.stats.grid_resolution}^3 with "
                    f"{membership.stats.occupied_cells}/{membership.stats.total_cells} occupied cells"
                ),
            )
        elif membership.stats is not None:
            _log_stage_detail(
                4,
                total_stages,
                (
                    f"Selector {selector_name}: approximate AABB mode, "
                    f"{membership.stats.candidate_count} candidate vertices"
                ),
            )
        selected_count = int(membership.vertex_indices.shape[0])
        _log_stage_detail(
            4,
            total_stages,
            f"Selector {selector_name}: selected {selected_count} / {rest_positions.shape[0]} vertices",
        )
        selector_memberships.append(
            SelectorVertexMembership(
                selector_name=selector_name,
                vertex_indices=membership.vertex_indices,
                vertex_mask=membership.vertex_mask,
            )
        )

    _log_stage_detail(
        4,
        total_stages,
        "Assembling boundary conditions from selector memberships",
    )
    boundary_conditions = assemble_boundary_conditions(
        rest_positions,
        selector_memberships,
        dirichlet_specs=tuple(
            DirichletSpec(
                selector_name=entry.selector,
                mode=entry.mode,
                components=entry.components,
            )
            for entry in config.dirichlet
        ),
        rigid_selector_names=config.rigid,
    )
    dirichlet_count = int(boundary_conditions.dirichlet_mask.sum())
    rigid_count = int(boundary_conditions.rigid_mask.sum())
    _log_stage_detail(
        4,
        total_stages,
        f"Boundary conditions assembled: {dirichlet_count} Dirichlet vertices, {rigid_count} rigid vertices",
    )

    _log_stage_detail(4, total_stages, "Assembling solver-ready topology and material data")
    problem = assemble_problem(
        rest_positions,
        tets,
        boundary_conditions,
        dt=config.solver.dt,
        external_acceleration=config.body_force,
        mu=config.material.mu,
        lam=config.material.lam,
        density=config.material.density,
        eps=config.solver.eps,
        num_iterations=config.solver.num_iterations,
        acceleration_enabled=config.solver.acceleration.enabled,
        chebyshev_rho=(
            config.solver.acceleration.rho
            if config.solver.acceleration.rho is not None
            else 0.95
        ),
        line_search_enabled=config.solver.line_search.enabled,
        # Both None is fine: assemble_problem then builds its default linear grid.
        line_search_alphas=config.solver.line_search.alphas,
        line_search_num_alphas=config.solver.line_search.num_alphas,
    )
    _log_stage_detail(
        4,
        total_stages,
        f"Problem ready: {problem.topology.color_groups.shape[0]} color groups, {problem.topology.incident_tets.shape[1]} max incident tets",
    )
    _log_stage_detail(4, total_stages, "Initializing simulation state")
    state = initial_state(problem)
    num_steps = args.steps if args.steps is not None else config.simulation.steps
    _log_stage(5, total_stages, f"Running simulation for {num_steps} steps")
    final_state, history = simulate(problem, state, num_steps=num_steps, show_progress=True)
    _log_stage(6, total_stages, f"Exporting trajectory to {output_path}")
    output_path = export_simulation_npz(output_path, problem, history)

    print(f"backend: {runtime_report['default_backend']}")
    print(f"devices: {runtime_report['devices']}")
    print(f"config: {args.config}")
    print(f"steps: {num_steps}")
    print(f"final_time: {float(final_state.time):.6f}")
    print(f"output: {output_path}")
    print(f"frames: {history.time.shape[0]}")
    print(f"vertices: {problem.mesh.rest_positions.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
