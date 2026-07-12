import struct
import tempfile
import unittest
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from diff_vbd import (
    DirichletSpec,
    assemble_dirichlet_boundary_conditions,
    assemble_problem,
    build_export_metadata,
    export_simulation_npz,
    extract_surface_mesh,
    initial_state,
    load_config,
    load_problem_from_yaml,
    simulate,
    step,
    surface_trajectory_from_history,
)
from diff_vbd.problems import build_cantilever_problem
from diff_vbd.setup import (
    SelectorClassificationOptions,
    build_incidence,
    build_lumped_masses,
    build_vertex_coloring,
    classify_selector_vertices,
    parse_binary_stl_aabb,
    parse_binary_stl_mesh,
    parse_gmsh22_binary_tets,
)
from diff_vbd.model import MaterialParams, SimulationState
from diff_vbd.solver.materials import stable_neo_hookean_energy_density, tet_energy
from diff_vbd.solver.vbd import (
    chebyshev_weight,
    clamped_hessian,
    predict_inertial_target,
    solve_local_vertex_step,
    sweep_positions,
    vertex_local_gradient,
    vertex_local_hessian,
    vertex_local_objective,
)
from diff_vbd.runtime_config import apply_runtime_config


def _make_single_tet_problem(
    *,
    dt=0.01,
    num_iterations=2,
    line_search_enabled: bool = False,
    line_search_alphas=(1.0, 0.5, 0.25, 0.125),
):
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    problem = assemble_problem(
        positions,
        tets,
        free_mask,
        dt=dt,
        external_acceleration=(0.0, 0.0, -9.81),
        num_iterations=num_iterations,
        mu=10.0,
        lam=10.0,
        eps=1.0e-5,
        line_search_enabled=line_search_enabled,
        line_search_alphas=line_search_alphas,
    )
    state = initial_state(problem)
    return problem, state


def _write_binary_gmsh22(path: Path):
    nodes = [
        (10, (0.0, 0.0, 0.0)),
        (20, (1.0, 0.0, 0.0)),
        (30, (0.0, 1.0, 0.0)),
        (40, (0.0, 0.0, 1.0)),
    ]
    tet_record = struct.pack("<iiiiii", 1, 7, 9, 10, 20, 30) + struct.pack("<i", 40)
    data = bytearray()
    data.extend(b"$Nodes\n")
    data.extend(f"{len(nodes)}\n".encode("ascii"))
    for tag, coords in nodes:
        data.extend(struct.pack("<i", tag))
        data.extend(struct.pack("<ddd", *coords))
    data.extend(b"$EndNodes\n")
    data.extend(b"$Elements\n1\n")
    data.extend(struct.pack("<iii", 4, 1, 2))
    data.extend(tet_record)
    data.extend(b"$EndElements\n")
    path.write_bytes(bytes(data))


def _write_binary_stl(path: Path, *, vmax=(1.0, 2.0, 3.0)):
    header = b"selector".ljust(80, b"\0")
    triangle_count = 1
    normal = (0.0, 0.0, 1.0)
    v0 = (0.0, 0.0, 0.0)
    v1 = (vmax[0], 0.0, 0.0)
    v2 = (0.0, vmax[1], vmax[2])
    body = struct.pack("<I", triangle_count)
    body += struct.pack("<12fH", *(normal + v0 + v1 + v2), 0)
    path.write_bytes(header + body)


def _write_closed_tetra_selector(path: Path):
    vertices = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    ]
    faces = [
        (0, 2, 1),
        (0, 1, 3),
        (0, 3, 2),
        (1, 2, 3),
    ]
    header = b"closed-tetra".ljust(80, b"\0")
    body = struct.pack("<I", len(faces))
    for face in faces:
        v0 = np.array(vertices[face[0]], dtype=np.float32)
        v1 = np.array(vertices[face[1]], dtype=np.float32)
        v2 = np.array(vertices[face[2]], dtype=np.float32)
        normal = np.cross(v1 - v0, v2 - v0)
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 0.0:
            normal = normal / normal_norm
        else:
            normal = np.zeros((3,), dtype=np.float32)
        body += struct.pack(
            "<12fH",
            *(
                tuple(float(x) for x in normal)
                + tuple(float(x) for x in v0)
                + tuple(float(x) for x in v1)
                + tuple(float(x) for x in v2)
            ),
            0,
        )
    path.write_bytes(header + body)


def _write_hex_prism_stl(path: Path, *, radius=1.0, y_min=-1.0, y_max=1.0):
    angles = [2.0 * np.pi * index / 6.0 for index in range(6)]
    bottom = [
        np.array([radius * np.cos(angle), y_min, radius * np.sin(angle)], dtype=np.float32)
        for angle in angles
    ]
    top = [
        np.array([radius * np.cos(angle), y_max, radius * np.sin(angle)], dtype=np.float32)
        for angle in angles
    ]
    center_bottom = np.array([0.0, y_min, 0.0], dtype=np.float32)
    center_top = np.array([0.0, y_max, 0.0], dtype=np.float32)

    triangles = []
    for index in range(6):
        next_index = (index + 1) % 6
        triangles.append((bottom[index], bottom[next_index], top[next_index]))
        triangles.append((bottom[index], top[next_index], top[index]))
    for index in range(1, 5):
        triangles.append((center_bottom, bottom[index], bottom[index + 1]))
        triangles.append((center_top, top[index + 1], top[index]))

    header = b"hex-prism".ljust(80, b"\0")
    body = struct.pack("<I", len(triangles))
    for triangle in triangles:
        v0, v1, v2 = triangle
        normal = np.cross(v1 - v0, v2 - v0)
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 0.0:
            normal = normal / normal_norm
        else:
            normal = np.zeros((3,), dtype=np.float32)
        body += struct.pack(
            "<12fH",
            *(
                tuple(float(x) for x in normal)
                + tuple(float(x) for x in v0)
                + tuple(float(x) for x in v1)
                + tuple(float(x) for x in v2)
            ),
            0,
        )
    path.write_bytes(header + body)


def _write_yaml_config(
    path: Path,
    mesh_name: str,
    selector_name: str,
    *,
    steps: int = 12,
    acceleration_enabled: bool = False,
    acceleration_rho: float | None = None,
    line_search_enabled: bool = False,
    line_search_alphas: tuple[float, ...] | None = None,
    include_line_search: bool = True,
    selector_classification_mode: str | None = None,
    selector_grid_resolution: int | None = None,
    selector_atol: float | None = None,
):
    lines = [
        "mesh:",
        f"  path: {mesh_name}",
        "selectors:",
        "  fixed_selector:",
        f"    path: {selector_name}",
        "material:",
        "  mu: 123.0",
        "  lam: 45.0",
        "  density: 2.5",
        "simulation:",
        f"  steps: {steps}",
        "solver:",
        "  dt: 0.05",
        "  num_iterations: 4",
        "  eps: 1.0e-5",
        "  acceleration:",
        f"    enabled: {'true' if acceleration_enabled else 'false'}",
    ]
    if acceleration_rho is not None:
        lines.append(f"    rho: {acceleration_rho}")
    if include_line_search:
        lines.extend(
            [
                "  line_search:",
                f"    enabled: {'true' if line_search_enabled else 'false'}",
            ]
        )
        if line_search_alphas is not None:
            if len(line_search_alphas) == 0:
                lines.append("    alphas: []")
            else:
                lines.append("    alphas:")
                for alpha in line_search_alphas:
                    lines.append(f"      - {alpha}")
    if (
        selector_classification_mode is not None
        or selector_grid_resolution is not None
        or selector_atol is not None
    ):
        lines.append("selector_classification:")
        if selector_classification_mode is not None:
            lines.append(f"  mode: {selector_classification_mode}")
        if selector_grid_resolution is not None:
            lines.append(f"  grid_resolution: {selector_grid_resolution}")
        if selector_atol is not None:
            lines.append(f"  atol: {selector_atol}")
    lines.extend(
        [
            "body_force:",
            "  - 0.0",
            "  - 0.0",
            "  - -1.25",
            "dirichlet:",
            "  - selector: fixed_selector",
            "    mode: position",
            "    components:",
            "      - '0.0'",
            "      - '0.0'",
            "      - '0.0'",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


class DiffVbdTests(unittest.TestCase):
    def test_parse_gmsh22_binary_tets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_path = Path(tmpdir) / "beam.msh"
            _write_binary_gmsh22(mesh_path)
            positions, tets = parse_gmsh22_binary_tets(mesh_path)

        self.assertEqual(positions.shape, (4, 3))
        self.assertEqual(tets.shape, (1, 4))

    def test_parse_binary_stl_aabb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = Path(tmpdir) / "selector.stl"
            _write_binary_stl(stl_path)
            mins, maxs = parse_binary_stl_aabb(stl_path)

        self.assertTrue(jnp.allclose(mins, jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)))
        self.assertTrue(jnp.allclose(maxs, jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)))

    def test_parse_binary_stl_mesh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = Path(tmpdir) / "selector.stl"
            _write_closed_tetra_selector(stl_path)
            selector = parse_binary_stl_mesh(stl_path)

        self.assertEqual(selector.triangles.shape, (4, 3, 3))

    def test_classify_selector_vertices_uses_closed_volume_not_aabb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = Path(tmpdir) / "hex_selector.stl"
            _write_hex_prism_stl(stl_path, radius=1.0, y_min=-1.0, y_max=1.0)
            selector = parse_binary_stl_mesh(stl_path)

        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.9, 0.0, 0.8],
                [0.0, 1.5, 0.0],
            ],
            dtype=jnp.float32,
        )
        membership = classify_selector_vertices(positions, selector)
        self.assertTrue(
            jnp.array_equal(
                membership.vertex_mask, jnp.array([True, True, False, False])
            )
        )

    def test_classify_selector_vertices_aabb_mode_matches_selector_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = Path(tmpdir) / "hex_selector.stl"
            _write_hex_prism_stl(stl_path, radius=1.0, y_min=-1.0, y_max=1.0)
            selector = parse_binary_stl_mesh(stl_path)

        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.9, 0.0, 0.8],
                [0.0, 1.5, 0.0],
            ],
            dtype=jnp.float32,
        )
        membership = classify_selector_vertices(
            positions,
            selector,
            options=SelectorClassificationOptions(mode="aabb"),
        )
        self.assertTrue(
            jnp.array_equal(
                membership.vertex_mask, jnp.array([True, True, True, False])
            )
        )
        self.assertEqual(membership.stats.mode, "aabb")

    def test_classify_selector_vertices_rejects_invalid_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = Path(tmpdir) / "selector.stl"
            _write_closed_tetra_selector(stl_path)
            selector = parse_binary_stl_mesh(stl_path)

        positions = jnp.zeros((1, 3), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "Unsupported selector classification mode"):
            classify_selector_vertices(
                positions,
                selector,
                options=SelectorClassificationOptions(mode="bogus"),
            )

    def test_topology_and_mass_builders(self):
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        incident_tets, incident_mask = build_incidence(tets, positions.shape[0])
        color_groups, color_group_mask, colors = build_vertex_coloring(
            tets, positions.shape[0]
        )
        mass = build_lumped_masses(positions, tets)

        self.assertEqual(incident_tets.shape, incident_mask.shape)
        self.assertEqual(color_groups.shape, color_group_mask.shape)
        self.assertEqual(jnp.unique(colors).shape[0], 4)
        self.assertAlmostEqual(float(jnp.sum(mass)), 1.0 / 6.0, places=6)

    def test_assemble_problem_rejects_unconstrained_mesh(self):
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        free_mask = jnp.ones((4,), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "constrain at least one vertex"):
            assemble_problem(positions, tets, free_mask)

    def test_solver_step_preserves_fixed_vertices(self):
        problem, state = _make_single_tet_problem(dt=0.01, num_iterations=2)
        next_state = step(problem, state)
        self.assertTrue(jnp.allclose(next_state.position[0], problem.mesh.rest_positions[0]))
        self.assertAlmostEqual(float(next_state.time), 0.01, places=6)

    def test_local_vertex_step_matches_manual_vbd(self):
        problem, state = _make_single_tet_problem(dt=0.02, num_iterations=1)
        inertial_target = predict_inertial_target(problem, state)
        vertex_index = jnp.array(1, dtype=jnp.int32)
        x_i_iter = state.position[vertex_index]
        gradient = vertex_local_gradient(
            problem,
            state.position,
            inertial_target,
            state.position,
            vertex_index,
            x_i_iter,
        )
        hessian = vertex_local_hessian(
            problem,
            state.position,
            inertial_target,
            state.position,
            vertex_index,
            x_i_iter,
        )
        regularized_hessian = hessian + problem.solver.eps * jnp.eye(
            x_i_iter.shape[0], dtype=x_i_iter.dtype
        )
        expected = x_i_iter + jnp.linalg.solve(regularized_hessian, -gradient)
        actual = solve_local_vertex_step(
            problem,
            state.position,
            inertial_target,
            state.position,
            vertex_index,
        )
        self.assertTrue(jnp.allclose(actual, expected))

    def test_local_vertex_step_with_single_line_search_alpha_matches_full_step(self):
        problem, state = _make_single_tet_problem(
            dt=0.02,
            num_iterations=1,
            line_search_enabled=True,
            line_search_alphas=(1.0,),
        )
        inertial_target = predict_inertial_target(problem, state)
        vertex_index = jnp.array(1, dtype=jnp.int32)
        x_i_iter = state.position[vertex_index]
        gradient = vertex_local_gradient(
            problem,
            state.position,
            inertial_target,
            state.position,
            vertex_index,
            x_i_iter,
        )
        hessian = vertex_local_hessian(
            problem,
            state.position,
            inertial_target,
            state.position,
            vertex_index,
            x_i_iter,
        )
        regularized_hessian = hessian + problem.solver.eps * jnp.eye(
            x_i_iter.shape[0], dtype=x_i_iter.dtype
        )
        expected = x_i_iter + jnp.linalg.solve(regularized_hessian, -gradient)
        actual = solve_local_vertex_step(
            problem,
            state.position,
            inertial_target,
            state.position,
            vertex_index,
        )
        self.assertTrue(jnp.allclose(actual, expected))

    def test_local_vertex_step_selects_lowest_objective_line_search_alpha(self):
        problem, state = _make_single_tet_problem(
            dt=2.0,
            num_iterations=1,
            line_search_enabled=True,
            line_search_alphas=(1.0, 0.5, 0.25, 0.125),
        )
        inertial_target = predict_inertial_target(problem, state)
        vertex_index = jnp.array(1, dtype=jnp.int32)
        x_i_iter = state.position[vertex_index]
        gradient = vertex_local_gradient(
            problem,
            state.position,
            inertial_target,
            state.position,
            vertex_index,
            x_i_iter,
        )
        hessian = vertex_local_hessian(
            problem,
            state.position,
            inertial_target,
            state.position,
            vertex_index,
            x_i_iter,
        )
        regularized_hessian = hessian + problem.solver.eps * jnp.eye(
            x_i_iter.shape[0], dtype=x_i_iter.dtype
        )
        delta_x = jnp.linalg.solve(regularized_hessian, -gradient)
        alphas = problem.solver.line_search.alphas
        candidate_positions = x_i_iter[None, :] + alphas[:, None] * delta_x[None, :]
        candidate_objectives = jax.vmap(
            lambda candidate: vertex_local_objective(
                problem,
                state.position,
                inertial_target,
                state.position,
                vertex_index,
                candidate,
            )
        )(candidate_positions)
        expected = candidate_positions[jnp.argmin(candidate_objectives)]
        actual = solve_local_vertex_step(
            problem,
            state.position,
            inertial_target,
            state.position,
            vertex_index,
        )

        self.assertTrue(jnp.allclose(actual, expected))

    def test_sweep_positions_returns_finite_positions(self):
        problem, state = _make_single_tet_problem(dt=0.02, num_iterations=2)
        prescribed_position = problem.mesh.rest_positions
        swept = sweep_positions(problem, state, prescribed_position)
        self.assertEqual(swept.shape, state.position.shape)
        self.assertTrue(jnp.all(jnp.isfinite(swept)))

    def test_dirichlet_position_motion_translates_selected_vertices(self):
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        membership = type("Membership", (), {
            "selector_name": "clamp",
            "vertex_indices": jnp.array([0], dtype=jnp.int32),
            "vertex_mask": jnp.array([True, False, False, False]),
        })()
        boundary_conditions = assemble_dirichlet_boundary_conditions(
            positions,
            [membership],
            [DirichletSpec("clamp", "position", ("t", "0.0", "0.0"))],
        )
        problem = assemble_problem(positions, tets, boundary_conditions, dt=0.1, num_iterations=1)
        state = initial_state(problem)
        next_state = step(problem, state)
        self.assertTrue(
            jnp.allclose(next_state.position[0], jnp.array([0.1, 0.0, 0.0], dtype=jnp.float32))
        )

    def test_dirichlet_velocity_motion_integrates_to_target_position(self):
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        membership = type("Membership", (), {
            "selector_name": "drive",
            "vertex_indices": jnp.array([0], dtype=jnp.int32),
            "vertex_mask": jnp.array([True, False, False, False]),
        })()
        boundary_conditions = assemble_dirichlet_boundary_conditions(
            positions,
            [membership],
            [DirichletSpec("drive", "velocity", ("2.0", "0.0", "0.0"))],
        )
        problem = assemble_problem(positions, tets, boundary_conditions, dt=0.1, num_iterations=1)
        state = initial_state(problem)
        next_state = step(problem, state)
        self.assertTrue(
            jnp.allclose(next_state.position[0], jnp.array([0.2, 0.0, 0.0], dtype=jnp.float32))
        )

    def test_overlapping_dirichlet_selectors_raise_error(self):
        positions = jnp.zeros((4, 3), dtype=jnp.float32)
        memberships = [
            type("Membership", (), {
                "selector_name": "a",
                "vertex_indices": jnp.array([0], dtype=jnp.int32),
                "vertex_mask": jnp.array([True, False, False, False]),
            })(),
            type("Membership", (), {
                "selector_name": "b",
                "vertex_indices": jnp.array([0], dtype=jnp.int32),
                "vertex_mask": jnp.array([True, False, False, False]),
            })(),
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            assemble_dirichlet_boundary_conditions(
                positions,
                memberships,
                [
                    DirichletSpec("a", "position", ("0.0", "0.0", "0.0")),
                    DirichletSpec("b", "position", ("0.0", "0.0", "0.0")),
                ],
            )

    def test_load_config_resolves_paths_and_validates_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(config_path, "beam.msh", "selector.stl")
            config = load_config(config_path)

        self.assertEqual(config.mesh.path.name, "beam.msh")
        self.assertEqual(config.material.density, 2.5)
        self.assertEqual(config.solver.num_iterations, 4)
        self.assertEqual(config.simulation.steps, 12)
        self.assertFalse(config.solver.acceleration.enabled)
        self.assertIsNone(config.solver.acceleration.rho)
        self.assertFalse(config.solver.line_search.enabled)
        self.assertIsNone(config.solver.line_search.alphas)
        self.assertEqual(config.selector_classification.mode, "exact")
        self.assertEqual(config.selector_classification.grid_resolution, 16)
        self.assertAlmostEqual(config.selector_classification.atol, 1.0e-6, places=12)

    def test_load_problem_from_yaml_assembles_simulation_problem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(config_path, "beam.msh", "selector.stl")
            problem = load_problem_from_yaml(config_path)

        self.assertAlmostEqual(float(problem.material.mu), 123.0, places=6)
        self.assertAlmostEqual(float(problem.solver.dt), 0.05, places=6)
        self.assertFalse(bool(problem.solver.acceleration.enabled))
        self.assertFalse(bool(problem.solver.line_search.enabled))

    def test_load_config_parses_selector_classification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                selector_classification_mode="aabb",
                selector_grid_resolution=8,
                selector_atol=1.0e-5,
            )
            config = load_config(config_path)

        self.assertEqual(config.selector_classification.mode, "aabb")
        self.assertEqual(config.selector_classification.grid_resolution, 8)
        self.assertAlmostEqual(config.selector_classification.atol, 1.0e-5, places=12)

    def test_load_problem_from_yaml_honors_aabb_selector_classification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_hex_prism_stl(selector_path, radius=1.0, y_min=-1.0, y_max=1.0)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                selector_classification_mode="aabb",
            )
            problem = load_problem_from_yaml(config_path)

        expected = jnp.array([True, True, True, False], dtype=jnp.bool_)
        self.assertTrue(jnp.array_equal(problem.boundary_conditions.dirichlet_mask, expected))

    def test_load_config_rejects_invalid_selector_classification_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                selector_classification_mode="bad",
            )

            with self.assertRaisesRegex(ValueError, "must be one of: exact, aabb"):
                load_config(config_path)

    def test_load_config_rejects_invalid_selector_grid_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                selector_grid_resolution=0,
            )

            with self.assertRaisesRegex(ValueError, "positive integer"):
                load_config(config_path)

    def test_load_config_rejects_invalid_selector_atol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                selector_atol=0.0,
            )

            with self.assertRaisesRegex(ValueError, "must be positive"):
                load_config(config_path)

    def test_load_config_parses_enabled_chebyshev_acceleration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                acceleration_enabled=True,
                acceleration_rho=0.95,
            )
            config = load_config(config_path)

        self.assertTrue(config.solver.acceleration.enabled)
        self.assertAlmostEqual(config.solver.acceleration.rho, 0.95, places=6)

    def test_load_config_rejects_missing_acceleration_rho_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                acceleration_enabled=True,
            )

            with self.assertRaisesRegex(ValueError, "rho is required"):
                load_config(config_path)

    def test_load_config_rejects_invalid_acceleration_rho(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                acceleration_enabled=True,
                acceleration_rho=1.1,
            )

            with self.assertRaisesRegex(ValueError, "0 < rho < 1"):
                load_config(config_path)

    def test_load_config_parses_enabled_line_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                line_search_enabled=True,
                line_search_alphas=(1.0, 0.5, 0.25),
            )
            config = load_config(config_path)

        self.assertTrue(config.solver.line_search.enabled)
        self.assertEqual(config.solver.line_search.alphas, (1.0, 0.5, 0.25))

    def test_load_config_rejects_missing_line_search_alphas_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                line_search_enabled=True,
            )

            with self.assertRaisesRegex(ValueError, "alphas is required"):
                load_config(config_path)

    def test_load_config_rejects_empty_line_search_alpha_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                line_search_enabled=False,
                line_search_alphas=(),
            )

            with self.assertRaisesRegex(ValueError, "at least one alpha"):
                load_config(config_path)

    def test_load_config_rejects_invalid_line_search_alpha(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                line_search_enabled=True,
                line_search_alphas=(1.0, 1.5),
            )

            with self.assertRaisesRegex(ValueError, "0 <= alpha <= 1"):
                load_config(config_path)

    def test_extract_surface_mesh_returns_boundary_faces(self):
        problem, _ = _make_single_tet_problem(dt=0.1, num_iterations=1)
        surface_vertex_indices, faces, rest_positions = extract_surface_mesh(problem)
        self.assertEqual(surface_vertex_indices.shape, (4,))
        self.assertEqual(faces.shape, (4, 3))
        self.assertEqual(rest_positions.shape, (4, 3))

    def test_surface_trajectory_from_history_projects_surface_vertices(self):
        problem, state = _make_single_tet_problem(dt=0.1, num_iterations=1)
        _, history = simulate(problem, state, num_steps=2, show_progress=False)
        _, faces, _, surface_positions = surface_trajectory_from_history(problem, history)
        self.assertEqual(faces.shape, (4, 3))
        self.assertEqual(surface_positions.shape, (2, 4, 3))

    def test_export_simulation_npz_writes_expected_payload(self):
        problem, state = _make_single_tet_problem(dt=0.1, num_iterations=1)
        _, history = simulate(problem, state, num_steps=2, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "sim_export.npz"
            result_path = export_simulation_npz(export_path, problem, history)
            archive = np.load(result_path, allow_pickle=False)
            metadata = json.loads(str(archive["metadata_json"]))

        self.assertEqual(archive["positions"].shape, (2, 4, 3))
        self.assertEqual(archive["faces"].shape, (4, 3))
        self.assertEqual(metadata["schema_version"], 1)

    def test_build_export_metadata_reports_problem_summary(self):
        problem, state = _make_single_tet_problem(dt=0.1, num_iterations=1)
        _, history = simulate(problem, state, num_steps=2, show_progress=False)
        metadata = build_export_metadata(problem, history)
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["num_frames"], 2)
        self.assertEqual(metadata["num_vertices"], 4)

    def test_load_config_rejects_unknown_dirichlet_selector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            config_path = tmpdir_path / "problem.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "mesh:",
                        "  path: beam.msh",
                        "selectors:",
                        "  fixed_selector:",
                        "    path: selector.stl",
                        "material:",
                        "  mu: 1.0",
                        "  lam: 1.0",
                        "  density: 1.0",
                        "simulation:",
                        "  steps: 2",
                        "solver:",
                        "  dt: 0.01",
                        "  num_iterations: 2",
                        "dirichlet:",
                        "  - selector: missing_selector",
                        "    mode: position",
                        "    components: ['0.0', '0.0', '0.0']",
                    ]
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "unknown selector"):
                load_config(config_path)

    def test_load_config_rejects_missing_simulation_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            lines = [
                "mesh:",
                "  path: beam.msh",
                "selectors:",
                "  fixed_selector:",
                "    path: selector.stl",
                "material:",
                "  mu: 123.0",
                "  lam: 45.0",
                "  density: 2.5",
                "solver:",
                "  dt: 0.05",
                "  num_iterations: 4",
                "  eps: 1.0e-5",
                "dirichlet:",
                "  - selector: fixed_selector",
                "    mode: position",
                "    components:",
                "      - '0.0'",
                "      - '0.0'",
                "      - '0.0'",
            ]
            config_path.write_text("\n".join(lines) + "\n")

            with self.assertRaisesRegex(ValueError, "simulation must be a mapping"):
                load_config(config_path)

    def test_load_config_rejects_non_positive_simulation_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path,
                "beam.msh",
                "selector.stl",
                steps=0,
            )

            with self.assertRaisesRegex(ValueError, "positive integer"):
                load_config(config_path)

    def test_cantilever_problem_builder_matches_selector_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_path = Path(tmpdir) / "beam.msh"
            selector_path = Path(tmpdir) / "selector.stl"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            problem = build_cantilever_problem(
                mesh_path=mesh_path,
                selector_path=selector_path,
                num_iterations=3,
            )

        self.assertTrue(
            jnp.array_equal(
                problem.boundary_conditions.free_mask,
                jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
            )
        )
        self.assertEqual(problem.solver.iteration_schedule.shape, (3,))

    def test_assemble_problem_stores_acceleration_options(self):
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        problem = assemble_problem(
            positions,
            tets,
            free_mask,
            acceleration_enabled=True,
            chebyshev_rho=0.9,
        )

        self.assertTrue(bool(problem.solver.acceleration.enabled))
        self.assertAlmostEqual(
            float(problem.solver.acceleration.chebyshev_rho), 0.9, places=6
        )

    def test_assemble_problem_stores_line_search_options(self):
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        problem = assemble_problem(
            positions,
            tets,
            free_mask,
            line_search_enabled=True,
            line_search_alphas=(1.0, 0.5, 0.25),
        )

        self.assertTrue(bool(problem.solver.line_search.enabled))
        self.assertTrue(
            jnp.allclose(
                problem.solver.line_search.alphas,
                jnp.array([1.0, 0.5, 0.25], dtype=jnp.float32),
            )
        )

    def test_assemble_problem_rejects_invalid_line_search_alphas(self):
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "0 <= alpha <= 1"):
            assemble_problem(
                positions, tets, free_mask, line_search_alphas=(1.0, 1.5)
            )
        with self.assertRaisesRegex(ValueError, "0 <= alpha <= 1"):
            assemble_problem(
                positions, tets, free_mask, line_search_alphas=(1.0, -0.25)
            )

    def test_assemble_problem_accepts_zero_line_search_alpha(self):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

        # alpha=0 is the "decline to move" option; it must be a legal grid point.
        problem = assemble_problem(
            positions, tets, free_mask, line_search_alphas=(1.0, 0.5, 0.0)
        )
        self.assertAlmostEqual(float(problem.solver.line_search.alphas[-1]), 0.0)

    def test_assemble_problem_rejects_all_zero_line_search_alphas(self):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

        # An all-zero grid would freeze every vertex forever.
        with self.assertRaisesRegex(ValueError, "at least one positive alpha"):
            assemble_problem(
                positions, tets, free_mask, line_search_alphas=(0.0, 0.0)
            )

    def test_assemble_problem_generates_linear_grid_from_num_alphas(self):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

        problem = assemble_problem(
            positions, tets, free_mask, line_search_num_alphas=9
        )
        self.assertTrue(
            jnp.allclose(
                problem.solver.line_search.alphas, jnp.linspace(1.0, 0.0, 9)
            )
        )

        with self.assertRaisesRegex(ValueError, "at least 2"):
            assemble_problem(
                positions, tets, free_mask, line_search_num_alphas=1
            )
        with self.assertRaisesRegex(ValueError, "not both"):
            assemble_problem(
                positions,
                tets,
                free_mask,
                line_search_alphas=(1.0, 0.0),
                line_search_num_alphas=9,
            )

    def test_assemble_problem_rejects_invalid_chebyshev_rho(self):
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "0 < chebyshev_rho < 1"):
            assemble_problem(
                positions,
                tets,
                free_mask,
                acceleration_enabled=True,
                chebyshev_rho=1.0,
            )

    def test_chebyshev_weight_matches_closed_form_recurrence(self):
        rho = jnp.array(0.95, dtype=jnp.float32)
        omega_1 = chebyshev_weight(jnp.array(0, dtype=jnp.int32), rho, jnp.array(1.0))
        omega_2 = chebyshev_weight(jnp.array(1, dtype=jnp.int32), rho, omega_1)
        omega_3 = chebyshev_weight(jnp.array(2, dtype=jnp.int32), rho, omega_2)

        expected_omega_2 = 2.0 / (2.0 - 0.95**2)
        expected_omega_3 = 4.0 / (4.0 - 0.95**2 * expected_omega_2)

        self.assertAlmostEqual(float(omega_1), 1.0, places=6)
        self.assertAlmostEqual(float(omega_2), expected_omega_2, places=6)
        self.assertAlmostEqual(float(omega_3), expected_omega_3, places=6)

    def test_sweep_positions_with_disabled_acceleration_matches_default_behavior(self):
        problem_default, state_default = _make_single_tet_problem(dt=0.02, num_iterations=3)
        prescribed_position = problem_default.mesh.rest_positions
        swept_default = sweep_positions(problem_default, state_default, prescribed_position)

        accelerated_off_problem = assemble_problem(
            problem_default.mesh.rest_positions,
            problem_default.mesh.tets,
            problem_default.boundary_conditions,
            dt=0.02,
            external_acceleration=(0.0, 0.0, -9.81),
            num_iterations=3,
            mu=10.0,
            lam=10.0,
            eps=1.0e-5,
            acceleration_enabled=False,
            chebyshev_rho=0.95,
            line_search_enabled=False,
        )
        state_off = initial_state(accelerated_off_problem)
        swept_off = sweep_positions(
            accelerated_off_problem, state_off, prescribed_position
        )

        self.assertTrue(jnp.allclose(swept_default, swept_off))

    def test_sweep_positions_with_chebyshev_acceleration_preserves_constraints(self):
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        membership = type("Membership", (), {
            "selector_name": "drive",
            "vertex_indices": jnp.array([0], dtype=jnp.int32),
            "vertex_mask": jnp.array([True, False, False, False]),
        })()
        boundary_conditions = assemble_dirichlet_boundary_conditions(
            positions,
            [membership],
            [DirichletSpec("drive", "position", ("t", "0.0", "0.0"))],
        )
        problem = assemble_problem(
            positions,
            tets,
            boundary_conditions,
            dt=0.1,
            num_iterations=3,
            acceleration_enabled=True,
            chebyshev_rho=0.95,
        )
        state = initial_state(problem)
        next_state = step(problem, state)

        self.assertTrue(jnp.all(jnp.isfinite(next_state.position)))
        self.assertTrue(
            jnp.allclose(next_state.position[0], jnp.array([0.1, 0.0, 0.0], dtype=jnp.float32))
        )

    def test_apply_runtime_config_disables_gpu_preallocation_by_default(self):
        runtime_config = apply_runtime_config(platform="gpu")
        self.assertEqual(runtime_config["platform"], "gpu")
        self.assertFalse(runtime_config["gpu_preallocate"])

    def test_load_config_enables_line_search_when_section_is_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mesh_path = tmpdir_path / "beam.msh"
            selector_path = tmpdir_path / "selector.stl"
            config_path = tmpdir_path / "problem.yaml"
            _write_binary_gmsh22(mesh_path)
            _write_closed_tetra_selector(selector_path)
            _write_yaml_config(
                config_path, "beam.msh", "selector.stl", include_line_search=False
            )
            config = load_config(config_path)
            problem = load_problem_from_yaml(config_path)

        self.assertTrue(config.solver.line_search.enabled)
        self.assertTrue(bool(problem.solver.line_search.enabled))
        self.assertTrue(
            jnp.allclose(
                problem.solver.line_search.alphas, jnp.linspace(1.0, 0.0, 9)
            )
        )


def _material(mu=10.0, lam=7.0):
    return MaterialParams(
        mu=jnp.asarray(mu, dtype=jnp.float32),
        lam=jnp.asarray(lam, dtype=jnp.float32),
        density=jnp.asarray(1.0, dtype=jnp.float32),
    )


def _analytic_first_piola(material, f):
    """Closed-form dPsi/dF, derived independently of the autodiff path under test."""
    mu = (4.0 / 3.0) * material.mu
    lam = material.lam + (5.0 / 6.0) * material.mu
    alpha = 1.0 + 0.75 * mu / lam

    ic = jnp.sum(f * f)
    f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]
    j = jnp.dot(f0, jnp.cross(f1, f2))
    dj_df = jnp.stack(
        [jnp.cross(f1, f2), jnp.cross(f2, f0), jnp.cross(f0, f1)], axis=1
    )
    return mu * f + lam * (j - alpha) * dj_df - mu * f / (ic + 1.0)


class StableNeoHookeanTests(unittest.TestCase):
    def test_rest_state_has_zero_energy(self):
        density = stable_neo_hookean_energy_density(_material(), jnp.eye(3))
        self.assertAlmostEqual(float(density), 0.0, places=5)

    def test_rest_state_is_stress_free(self):
        stress = jax.grad(stable_neo_hookean_energy_density, argnums=1)(
            _material(), jnp.eye(3)
        )
        self.assertTrue(jnp.allclose(stress, jnp.zeros((3, 3)), atol=1.0e-5))

    def test_autodiff_stress_matches_analytic_first_piola(self):
        material = _material()
        deformations = [
            jnp.eye(3),
            jnp.array(
                [[1.2, 0.1, -0.3], [0.0, 0.9, 0.2], [0.15, -0.05, 1.1]],
                dtype=jnp.float32,
            ),
            jnp.diag(jnp.array([1.0, 1.0, -0.5], dtype=jnp.float32)),
        ]
        for f in deformations:
            with self.subTest(determinant=float(jnp.linalg.det(f))):
                autodiff = jax.grad(stable_neo_hookean_energy_density, argnums=1)(
                    material, f
                )
                self.assertTrue(
                    jnp.allclose(
                        autodiff, _analytic_first_piola(material, f), rtol=1.0e-4,
                        atol=1.0e-4,
                    )
                )

    def test_inverted_element_has_finite_energy_and_restoring_force(self):
        material = _material()
        f = jnp.diag(jnp.array([1.0, 1.0, -0.5], dtype=jnp.float32))

        density = stable_neo_hookean_energy_density(material, f)
        stress = jax.grad(stable_neo_hookean_energy_density, argnums=1)(material, f)

        self.assertTrue(jnp.isfinite(density))
        # The old log(J) formulation returned a constant barrier here, so its gradient
        # was exactly zero and an inverted element could never recover.
        self.assertGreater(float(jnp.max(jnp.abs(stress))), 1.0e-3)

    def test_energy_and_stress_stay_finite_through_the_inversion_boundary(self):
        material = _material()
        for scale in [-1.5, -1.0, -0.25, 0.0, 0.25, 1.0, 1.5]:
            with self.subTest(scale=scale):
                f = jnp.diag(jnp.array([1.0, 1.0, scale], dtype=jnp.float32))
                density = stable_neo_hookean_energy_density(material, f)
                stress = jax.grad(stable_neo_hookean_energy_density, argnums=1)(
                    material, f
                )
                self.assertTrue(jnp.isfinite(density))
                self.assertTrue(jnp.all(jnp.isfinite(stress)))

    def test_energy_is_rotation_invariant(self):
        material = _material()
        f = jnp.array(
            [[1.2, 0.1, -0.3], [0.0, 0.9, 0.2], [0.15, -0.05, 1.1]], dtype=jnp.float32
        )
        rotation, _ = jnp.linalg.qr(
            jax.random.normal(jax.random.PRNGKey(0), (3, 3), dtype=jnp.float32)
        )
        rotation = rotation * jnp.sign(jnp.linalg.det(rotation))

        self.assertAlmostEqual(
            float(stable_neo_hookean_energy_density(material, rotation @ f)),
            float(stable_neo_hookean_energy_density(material, f)),
            places=4,
        )

    def test_small_strain_response_matches_linear_elasticity(self):
        mu, lam = 10.0, 7.0
        hessian = jax.hessian(stable_neo_hookean_energy_density, argnums=1)(
            _material(mu, lam), jnp.eye(3)
        )

        # The 4/3 and 5/6 remap exists precisely so these recover the physical Lame
        # parameters; if they drift, every solver tuning constant silently changes.
        self.assertAlmostEqual(float(hessian[0, 0, 0, 0]), lam + 2.0 * mu, places=3)
        self.assertAlmostEqual(float(hessian[0, 0, 1, 1]), lam, places=3)
        self.assertAlmostEqual(float(hessian[0, 1, 0, 1]), mu, places=3)

    def test_energy_resolves_small_strains_without_cancellation(self):
        """The energy must not quantize to noise near rest, or the line search guesses.

        The textbook form computes ``sum(f*f) - 3`` with ``sum(f*f) ~ 3``; scaled by a
        stiff ``mu`` its float32 rounding error swamps the true energy. Here the energy
        at a 1e-5 strain must stay far below the energy at a 10x larger strain, which
        the cancelling form cannot do (it returns the same quantized value for both).
        """
        material = _material(mu=4.638e5, lam=4.174e6)
        direction = jnp.array(
            [[1.0, 0.3, 0.0], [0.0, -0.5, 0.2], [0.1, 0.0, 0.7]], dtype=jnp.float32
        )

        def energy(strain):
            return float(
                stable_neo_hookean_energy_density(
                    material, jnp.eye(3) + strain * direction
                )
            )

        small, large = energy(1.0e-5), energy(1.0e-4)
        # Energy is quadratic in strain, so a 10x strain is ~100x the energy.
        self.assertGreater(large, 0.0)
        self.assertLess(small, large / 10.0)

    def test_tet_energy_is_zero_in_the_rest_pose(self):
        rest = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        energy = tet_energy(_material(), rest, rest)
        self.assertAlmostEqual(float(energy), 0.0, places=5)


class ClampedHessianTests(unittest.TestCase):
    def _indefinite_hessian(self):
        material = _material()
        f = jnp.diag(jnp.array([1.0, 1.0, -0.5], dtype=jnp.float32))
        hessian = jax.hessian(stable_neo_hookean_energy_density, argnums=1)(material, f)
        return hessian.reshape(9, 9)

    def test_energy_hessian_is_indefinite_under_inversion(self):
        # Guards the premise of the clamp: without it, solve() returns an ascent
        # direction along these negative eigenvectors.
        eigenvalues = jnp.linalg.eigvalsh(self._indefinite_hessian())
        self.assertLess(float(jnp.min(eigenvalues)), 0.0)

    def test_clamped_hessian_floors_eigenvalues_at_eps(self):
        # eps is kept well above the float32 reconstruction noise of a matrix whose
        # eigenvalues run to ~100, so this measures the clamp and not round-off.
        eps = jnp.asarray(1.0, dtype=jnp.float32)
        clamped = clamped_hessian(self._indefinite_hessian(), eps)
        eigenvalues = jnp.linalg.eigvalsh(clamped)

        self.assertGreaterEqual(float(jnp.min(eigenvalues)), float(eps) - 1.0e-3)

    def test_clamped_hessian_preserves_eigenvectors(self):
        eps = jnp.asarray(1.0, dtype=jnp.float32)
        hessian = self._indefinite_hessian()
        eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
        clamped = clamped_hessian(hessian, eps)

        # Each original eigenvector must survive as an eigenvector of the clamped
        # matrix, with only its eigenvalue raised to the floor. Comparing eigh output
        # column-by-column would not show this: clamping reorders the spectrum.
        self.assertLess(float(jnp.min(eigenvalues)), 0.0)
        for index in range(9):
            with self.subTest(eigenvalue=float(eigenvalues[index])):
                eigenvector = eigenvectors[:, index]
                expected = jnp.maximum(eigenvalues[index], eps) * eigenvector
                self.assertTrue(
                    jnp.allclose(clamped @ eigenvector, expected, atol=1.0e-3)
                )

    def test_clamped_hessian_leaves_a_positive_definite_matrix_alone(self):
        eps = jnp.asarray(1.0e-6, dtype=jnp.float32)
        hessian = jnp.diag(jnp.array([3.0, 5.0, 7.0], dtype=jnp.float32))
        self.assertTrue(jnp.allclose(clamped_hessian(hessian, eps), hessian, atol=1e-5))


class InvertedMeshValidationTests(unittest.TestCase):
    def test_assemble_problem_rejects_negatively_oriented_tet(self):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        # Swapping two vertices flips the winding, so det(F) < 0 in the rest pose.
        tets = jnp.array([[0, 2, 1, 3]], dtype=jnp.int32)
        free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

        with self.assertRaisesRegex(ValueError, "non-positive rest volume"):
            assemble_problem(positions, tets, free_mask)

    def test_assemble_problem_accepts_positively_oriented_tet(self):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        free_mask = jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

        problem = assemble_problem(positions, tets, free_mask)
        self.assertEqual(problem.mesh.tets.shape, (1, 4))


def _beam_problem(*, num_iterations, line_search_alphas=None, line_search_num_alphas=None):
    """A small clamped beam: enough vertices for the line search to matter."""
    nx, ny, nz = 4, 2, 2
    grid = jnp.stack(
        jnp.meshgrid(
            jnp.arange(nx, dtype=jnp.float32),
            jnp.arange(ny, dtype=jnp.float32),
            jnp.arange(nz, dtype=jnp.float32),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    index = lambda i, j, k: (i * ny + j) * nz + k
    # Each unit cube is split into 5 tets, all positively wound.
    tets = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c = [index(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                v000, v001, v010, v011, v100, v101, v110, v111 = c
                tets += [
                    [v000, v100, v010, v001],
                    [v110, v010, v100, v111],
                    [v101, v100, v001, v111],
                    [v011, v001, v010, v111],
                    [v100, v010, v001, v111],
                ]
    tets = jnp.array(tets, dtype=jnp.int32)
    free_mask = jnp.where(grid[:, 0] == 0.0, 0.0, 1.0).astype(jnp.float32)
    return assemble_problem(
        grid,
        tets,
        free_mask,
        dt=0.02,
        external_acceleration=(0.0, 0.0, -9.81),
        num_iterations=num_iterations,
        mu=1.0e4,
        lam=1.0e4,
        eps=1.0e-6,
        line_search_enabled=True,
        line_search_alphas=line_search_alphas,
        line_search_num_alphas=line_search_num_alphas,
    )


class LineSearchDescentTests(unittest.TestCase):
    def test_line_search_never_increases_the_local_objective(self):
        """The property that is false without alpha=0, and the point of the change."""
        problem = _beam_problem(num_iterations=1)
        state = initial_state(problem)
        target = predict_inertial_target(problem, state)
        free = np.where(
            ~np.asarray(problem.boundary_conditions.dirichlet_mask)
        )[0]

        def objective(vertex_index, x_i):
            return vertex_local_objective(
                problem, state.position, target, state.position, vertex_index, x_i
            )

        increased = 0
        for vertex in free:
            vertex_index = jnp.asarray(vertex, dtype=jnp.int32)
            before = objective(vertex_index, state.position[vertex_index])
            after = objective(
                vertex_index,
                solve_local_vertex_step(
                    problem, state.position, target, state.position, vertex_index
                ),
            )
            if float(after) > float(before) + 1.0e-4 * abs(float(before)) + 1.0e-6:
                increased += 1

        self.assertEqual(
            increased,
            0,
            f"{increased}/{len(free)} vertices took an objective-increasing step",
        )

    def _simulate_beam(self, num_iterations, alphas=None):
        problem = _beam_problem(num_iterations=num_iterations, line_search_alphas=alphas)
        final, _ = simulate(
            problem, initial_state(problem), num_steps=10, show_progress=False
        )
        return np.asarray(final.position)

    def test_solution_is_invariant_to_num_iterations(self):
        """Once converged, extra sweeps must not move the answer."""
        drift = np.abs(self._simulate_beam(20) - self._simulate_beam(60)).max()
        self.assertLess(drift, 1.0e-5, f"solution moved by {drift:.3e} when sweeps grew")

    def test_line_search_without_zero_alpha_diverges_with_num_iterations(self):
        """Pins the bug this change fixes, so it cannot silently come back.

        With no alpha=0 on the grid the line search cannot decline to move, so it is
        forced to return objective-increasing steps. Those pump energy into the mesh and
        the deflection keeps growing as sweeps are added instead of settling.
        """
        no_zero = (1.0, 0.5, 0.25, 0.125)
        drift = np.abs(
            self._simulate_beam(20, no_zero) - self._simulate_beam(60, no_zero)
        ).max()
        self.assertGreater(drift, 1.0e-3)


class InvertedElementRecoveryTests(unittest.TestCase):
    def test_inverted_tet_recovers_to_positive_volume(self):
        """The behavioural payoff: an inverted element must be able to un-invert.

        Under the old log(J) formulation the elastic gradient here is identically zero,
        so the tet stays inverted forever.
        """
        rest = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        tets = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        # Pin the base triangle; only the apex is free, pushed through the base plane.
        free_mask = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        problem = assemble_problem(
            rest,
            tets,
            free_mask,
            dt=0.005,
            external_acceleration=(0.0, 0.0, 0.0),
            num_iterations=20,
            mu=10.0,
            lam=10.0,
            eps=1.0e-6,
        )

        inverted = rest.at[3].set(jnp.array([0.2, 0.2, -0.4], dtype=jnp.float32))
        state = SimulationState(
            position=inverted,
            velocity=jnp.zeros_like(inverted),
            time=jnp.asarray(0.0, dtype=jnp.float32),
        )
        self.assertLess(float(_signed_volume(inverted)), 0.0)

        final_state, _ = simulate(problem, state, num_steps=60, show_progress=False)

        self.assertTrue(jnp.all(jnp.isfinite(final_state.position)))
        self.assertGreater(float(_signed_volume(final_state.position)), 0.0)


def _signed_volume(positions):
    edges = jnp.stack(
        [
            positions[1] - positions[0],
            positions[2] - positions[0],
            positions[3] - positions[0],
        ],
        axis=-1,
    )
    return jnp.linalg.det(edges) / 6.0


if __name__ == "__main__":
    unittest.main()
