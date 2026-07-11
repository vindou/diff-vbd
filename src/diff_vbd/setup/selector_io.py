"""Selector geometry ingestion and mesh classification helpers."""

from dataclasses import dataclass
from pathlib import Path
import struct

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SelectorMesh:
    """Closed STL selector geometry with cached bounds."""

    name: str
    triangles: jnp.ndarray
    aabb_min: jnp.ndarray
    aabb_max: jnp.ndarray


@dataclass(frozen=True)
class SelectorVertexMembership:
    """Tet-mesh vertices selected by one boundary-condition selector."""

    selector_name: str
    vertex_indices: jnp.ndarray
    vertex_mask: jnp.ndarray
    stats: "SelectorClassificationStats | None" = None


@dataclass(frozen=True)
class SelectorClassificationOptions:
    mode: str = "exact"
    grid_resolution: int = 16
    atol: float = 1.0e-6


@dataclass(frozen=True)
class SelectorClassificationStats:
    mode: str
    candidate_count: int
    selected_count: int
    grid_resolution: int | None = None
    occupied_cells: int | None = None
    total_cells: int | None = None


def parse_binary_stl_mesh(path: str | Path) -> SelectorMesh:
    """Parse a binary STL and return triangle geometry plus cached bounds."""
    mesh_path = Path(path)
    data = mesh_path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Binary STL {mesh_path} is too short to contain a header")

    tri_count = struct.unpack("<I", data[80:84])[0]
    if tri_count <= 0:
        raise ValueError(f"Binary STL {mesh_path} must contain at least one triangle")

    expected_size = 84 + 50 * tri_count
    if len(data) < expected_size:
        raise ValueError(
            f"Binary STL {mesh_path} is truncated: expected {expected_size} bytes, got {len(data)}"
        )

    triangles = np.empty((tri_count, 3, 3), dtype=np.float64)
    pos = 84
    for tri_index in range(tri_count):
        values = struct.unpack("<12fH", data[pos : pos + 50])
        triangles[tri_index, 0, :] = values[3:6]
        triangles[tri_index, 1, :] = values[6:9]
        triangles[tri_index, 2, :] = values[9:12]
        pos += 50

    aabb_min = triangles.reshape(-1, 3).min(axis=0)
    aabb_max = triangles.reshape(-1, 3).max(axis=0)
    return SelectorMesh(
        name=mesh_path.stem,
        triangles=jnp.asarray(triangles, dtype=jnp.float32),
        aabb_min=jnp.asarray(aabb_min, dtype=jnp.float32),
        aabb_max=jnp.asarray(aabb_max, dtype=jnp.float32),
    )


def parse_binary_stl_aabb(path: str | Path):
    """Backward-compatible STL AABB helper."""
    selector = parse_binary_stl_mesh(path)
    return selector.aabb_min, selector.aabb_max


def _build_selector_grid(
    triangles: np.ndarray,
    selector_min: np.ndarray,
    selector_max: np.ndarray,
    grid_resolution: int,
) -> tuple[dict[tuple[int, int, int], np.ndarray], int]:
    selector_extent = np.maximum(selector_max - selector_min, 1.0e-12)
    triangle_min = triangles.min(axis=1)
    triangle_max = triangles.max(axis=1)
    normalized_min = (triangle_min - selector_min) / selector_extent
    normalized_max = (triangle_max - selector_min) / selector_extent
    cell_min = np.clip(
        np.floor(normalized_min * grid_resolution).astype(np.int32),
        0,
        grid_resolution - 1,
    )
    cell_max = np.clip(
        np.floor(normalized_max * grid_resolution).astype(np.int32),
        0,
        grid_resolution - 1,
    )

    cell_map: dict[tuple[int, int, int], list[int]] = {}
    for triangle_index in range(triangles.shape[0]):
        for ix in range(cell_min[triangle_index, 0], cell_max[triangle_index, 0] + 1):
            for iy in range(cell_min[triangle_index, 1], cell_max[triangle_index, 1] + 1):
                for iz in range(cell_min[triangle_index, 2], cell_max[triangle_index, 2] + 1):
                    key = (ix, iy, iz)
                    bucket = cell_map.get(key)
                    if bucket is None:
                        cell_map[key] = [triangle_index]
                    else:
                        bucket.append(triangle_index)
    compact_cell_map = {
        key: np.asarray(indices, dtype=np.int32) for key, indices in cell_map.items()
    }
    return compact_cell_map, grid_resolution**3


def _point_on_any_triangle(
    point: np.ndarray, triangles: np.ndarray, atol: float
) -> bool:
    if triangles.shape[0] == 0:
        return False

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    edge0 = v1 - v0
    edge1 = v2 - v0
    normals = np.cross(edge0, edge1)
    normal_norm = np.linalg.norm(normals, axis=1)
    valid = normal_norm > atol
    if not np.any(valid):
        return False

    point_offset = point[None, :] - v0
    signed_distance = np.einsum("ij,ij->i", point_offset, normals) / np.where(
        valid, normal_norm, 1.0
    )
    near_plane = np.abs(signed_distance) <= atol

    dot00 = np.einsum("ij,ij->i", edge1, edge1)
    dot01 = np.einsum("ij,ij->i", edge1, edge0)
    dot11 = np.einsum("ij,ij->i", edge0, edge0)
    dot02 = np.einsum("ij,ij->i", edge1, point_offset)
    dot12 = np.einsum("ij,ij->i", edge0, point_offset)
    denominator = dot00 * dot11 - dot01 * dot01
    stable = np.abs(denominator) > atol
    inv_denominator = np.where(stable, 1.0 / denominator, 0.0)

    u = (dot11 * dot02 - dot01 * dot12) * inv_denominator
    v = (dot00 * dot12 - dot01 * dot02) * inv_denominator
    w = 1.0 - u - v
    inside = (u >= -atol) & (v >= -atol) & (w >= -atol)
    return bool(np.any(valid & near_plane & stable & inside))


def _count_ray_intersections(
    origin: np.ndarray, direction: np.ndarray, triangles: np.ndarray, atol: float
) -> int:
    if triangles.shape[0] == 0:
        return 0

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_term = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, cross_term)
    nonparallel = np.abs(determinant) > atol
    inv_determinant = np.where(nonparallel, 1.0 / determinant, 0.0)

    offset = origin[None, :] - v0
    bary_u = inv_determinant * np.einsum("ij,ij->i", offset, cross_term)
    q_vec = np.cross(offset, edge1)
    bary_v = inv_determinant * np.einsum(
        "j,ij->i", direction, q_vec
    )
    ray_t = inv_determinant * np.einsum("ij,ij->i", edge2, q_vec)
    intersects = (
        nonparallel
        & (bary_u >= -atol)
        & (bary_u <= 1.0 + atol)
        & (bary_v >= -atol)
        & (bary_u + bary_v <= 1.0 + atol)
        & (ray_t > atol)
    )
    return int(np.count_nonzero(intersects))


def _ray_intersects_triangle_aabbs(
    origin: np.ndarray,
    direction: np.ndarray,
    triangle_min: np.ndarray,
    triangle_max: np.ndarray,
) -> np.ndarray:
    safe_direction = np.where(np.abs(direction) > 0.0, direction, 1.0)
    t0 = (triangle_min - origin[None, :]) / safe_direction[None, :]
    t1 = (triangle_max - origin[None, :]) / safe_direction[None, :]
    t_near = np.minimum(t0, t1)
    t_far = np.maximum(t0, t1)
    enter = np.maximum.reduce(t_near, axis=1)
    exit = np.minimum.reduce(t_far, axis=1)
    return exit >= np.maximum(enter, 0.0)


def _point_inside_selector_mesh(
    point: np.ndarray,
    triangles: np.ndarray,
    direction: np.ndarray,
    atol: float,
) -> bool:
    if _point_on_any_triangle(point, triangles, atol):
        return True
    intersections = _count_ray_intersections(point, direction, triangles, atol)
    return (intersections % 2) == 1


def classify_selector_vertices(
    mesh_positions: jnp.ndarray,
    selector: SelectorMesh,
    *,
    options: SelectorClassificationOptions | None = None,
) -> SelectorVertexMembership:
    """Classify tet-mesh vertices against a closed selector volume."""
    if options is None:
        options = SelectorClassificationOptions()
    if options.mode not in {"exact", "aabb"}:
        raise ValueError(f"Unsupported selector classification mode {options.mode!r}")
    if options.grid_resolution <= 0:
        raise ValueError("grid_resolution must be a positive integer")
    if options.atol <= 0.0:
        raise ValueError("atol must be positive")

    positions = np.asarray(mesh_positions, dtype=np.float64)
    triangles = np.asarray(selector.triangles, dtype=np.float64)
    triangle_min = triangles.min(axis=1)
    triangle_max = triangles.max(axis=1)
    selector_min = np.asarray(selector.aabb_min, dtype=np.float64) - options.atol
    selector_max = np.asarray(selector.aabb_max, dtype=np.float64) + options.atol
    direction = np.asarray([1.0, 0.3713906763541037, 0.18293714059226867])
    direction /= np.linalg.norm(direction)

    inside_aabb = np.all((positions >= selector_min) & (positions <= selector_max), axis=1)
    candidate_indices = np.nonzero(inside_aabb)[0]
    candidate_count = int(candidate_indices.shape[0])
    selected = np.zeros((positions.shape[0],), dtype=bool)

    if options.mode == "aabb":
        selected = inside_aabb.astype(bool)
        vertex_indices = np.nonzero(selected)[0].astype(np.int32)
        return SelectorVertexMembership(
            selector_name=selector.name,
            vertex_indices=jnp.asarray(vertex_indices, dtype=jnp.int32),
            vertex_mask=jnp.asarray(selected),
            stats=SelectorClassificationStats(
                mode="aabb",
                candidate_count=candidate_count,
                selected_count=int(vertex_indices.shape[0]),
            ),
        )

    selector_grid, total_cells = _build_selector_grid(
        triangles,
        selector_min,
        selector_max,
        options.grid_resolution,
    )
    selector_extent = np.maximum(selector_max - selector_min, 1.0e-12)
    occupied_cells = len(selector_grid)
    for vertex_index in candidate_indices:
        normalized = (positions[vertex_index] - selector_min) / selector_extent
        cell_index = tuple(
            np.clip(
                np.floor(normalized * options.grid_resolution).astype(np.int32),
                0,
                options.grid_resolution - 1,
            ).tolist()
        )
        triangle_indices = selector_grid.get(cell_index)
        point = positions[vertex_index]
        if triangle_indices is not None and _point_on_any_triangle(
            point, triangles[triangle_indices], options.atol
        ):
            selected[vertex_index] = True
            continue

        ray_mask = _ray_intersects_triangle_aabbs(
            point, direction, triangle_min, triangle_max
        )
        if not np.any(ray_mask):
            continue
        selected[vertex_index] = (
            _count_ray_intersections(
                point,
                direction,
                triangles[ray_mask],
                options.atol,
            )
            % 2
        ) == 1

    vertex_indices = np.nonzero(selected)[0].astype(np.int32)
    return SelectorVertexMembership(
        selector_name=selector.name,
        vertex_indices=jnp.asarray(vertex_indices, dtype=jnp.int32),
        vertex_mask=jnp.asarray(selected),
        stats=SelectorClassificationStats(
            mode="exact",
            candidate_count=candidate_count,
            selected_count=int(vertex_indices.shape[0]),
            grid_resolution=options.grid_resolution,
            occupied_cells=occupied_cells,
            total_cells=total_cells,
        ),
    )
