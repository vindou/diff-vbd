"""Compact NPZ export for downstream rendering."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from diff_vbd.model import SimulationProblem, SimulationState


def _to_numpy(value):
    return np.asarray(jax.device_get(value))


def extract_surface_mesh(
    problem: SimulationProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the triangle boundary of the tet mesh."""
    tets = _to_numpy(problem.mesh.tets).astype(np.int32)
    face_table: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    face_counts: dict[tuple[int, int, int], int] = {}

    for tet in tets:
        a, b, c, d = [int(index) for index in tet]
        oriented_faces = (
            (a, c, b),
            (a, b, d),
            (a, d, c),
            (b, c, d),
        )
        for face in oriented_faces:
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1
            if key not in face_table:
                face_table[key] = face

    boundary_faces = [
        face_table[key] for key, count in face_counts.items() if count == 1
    ]
    if not boundary_faces:
        raise ValueError("Tet mesh does not have a boundary surface to export")

    surface_vertex_indices = np.array(
        sorted({vertex for face in boundary_faces for vertex in face}), dtype=np.int32
    )
    surface_index_lookup = {
        int(vertex): index for index, vertex in enumerate(surface_vertex_indices.tolist())
    }
    surface_faces = np.array(
        [
            [surface_index_lookup[int(vertex)] for vertex in face]
            for face in boundary_faces
        ],
        dtype=np.int32,
    )
    surface_rest_positions = _to_numpy(problem.mesh.rest_positions)[surface_vertex_indices]
    return surface_vertex_indices, surface_faces, surface_rest_positions


def surface_trajectory_from_history(
    problem: SimulationProblem, history: SimulationState
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project the full tet trajectory onto the extracted boundary surface."""
    surface_vertex_indices, surface_faces, surface_rest_positions = extract_surface_mesh(
        problem
    )
    history_positions = _to_numpy(history.position)
    surface_positions = history_positions[:, surface_vertex_indices, :]
    return (
        surface_vertex_indices,
        surface_faces,
        surface_rest_positions,
        surface_positions,
    )


def build_export_metadata(
    problem: SimulationProblem, history: SimulationState
) -> dict[str, object]:
    """Build compact metadata for downstream consumers."""
    dirichlet_mask = _to_numpy(problem.boundary_conditions.dirichlet_mask).astype(bool)
    rigid_mask = _to_numpy(problem.boundary_conditions.rigid_mask).astype(bool)
    constrained_mask = dirichlet_mask | rigid_mask
    return {
        "schema_version": 1,
        "num_frames": int(_to_numpy(history.time).shape[0]),
        "num_vertices": int(_to_numpy(problem.mesh.rest_positions).shape[0]),
        "num_tets": int(_to_numpy(problem.mesh.tets).shape[0]),
        "dt": float(_to_numpy(problem.solver.dt)),
        "mu": float(_to_numpy(problem.material.mu)),
        "lam": float(_to_numpy(problem.material.lam)),
        "density": float(_to_numpy(problem.material.density)),
        "constrained_vertex_count": int(constrained_mask.sum()),
        "dirichlet_vertex_count": int(dirichlet_mask.sum()),
        "rigid_vertex_count": int(rigid_mask.sum()),
    }


def export_simulation_npz(
    path: str | Path,
    problem: SimulationProblem,
    history: SimulationState,
) -> Path:
    """Write a Blender-oriented NPZ archive for a simulated trajectory."""
    output_path = Path(path)
    (
        surface_vertex_indices,
        surface_faces,
        surface_rest_positions,
        surface_positions,
    ) = surface_trajectory_from_history(problem, history)
    metadata = build_export_metadata(problem, history)

    np.savez(
        output_path,
        positions=surface_positions,
        time=_to_numpy(history.time),
        faces=surface_faces,
        rest_positions=surface_rest_positions,
        surface_vertex_indices=surface_vertex_indices,
        tet_positions=_to_numpy(problem.mesh.rest_positions),
        tets=_to_numpy(problem.mesh.tets),
        metadata_json=np.array(json.dumps(metadata)),
    )
    return output_path
