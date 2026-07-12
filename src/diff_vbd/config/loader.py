"""Load full simulation problems from YAML configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from diff_vbd.model import DirichletSpec, SimulationProblem
from diff_vbd.setup import (
    SelectorClassificationOptions,
    assemble_boundary_conditions,
    SelectorVertexMembership,
    assemble_problem,
    classify_selector_vertices,
    parse_binary_stl_mesh,
    parse_gmsh22_binary_tets,
)


@dataclass(frozen=True)
class MeshConfig:
    path: Path


@dataclass(frozen=True)
class MaterialConfig:
    mu: float
    lam: float
    density: float


@dataclass(frozen=True)
class AccelerationConfig:
    enabled: bool
    rho: float | None


@dataclass(frozen=True)
class LineSearchConfig:
    enabled: bool
    alphas: tuple[float, ...] | None
    num_alphas: int | None = None


@dataclass(frozen=True)
class SolverConfig:
    dt: float
    num_iterations: int
    eps: float
    acceleration: AccelerationConfig
    line_search: LineSearchConfig


@dataclass(frozen=True)
class SelectorClassificationConfig:
    mode: str
    grid_resolution: int
    atol: float


@dataclass(frozen=True)
class SimulationConfig:
    steps: int


@dataclass(frozen=True)
class DirichletConfig:
    selector: str
    mode: str
    components: tuple[str, str, str]


@dataclass(frozen=True)
class ContactConfig:
    enabled: bool
    d_hat: float
    kappa: float | None
    friction_mu: float
    eps_v: float
    use_barrier: bool
    self_collision: bool
    self_collision_ccd: bool
    ccd_slack: float
    capacity: int
    max_per_vertex: int
    colliders: tuple[dict, ...]


@dataclass(frozen=True)
class ProblemConfig:
    mesh: MeshConfig
    selectors: dict[str, Path]
    selector_classification: SelectorClassificationConfig
    material: MaterialConfig
    solver: SolverConfig
    simulation: SimulationConfig
    body_force: tuple[float, float, float]
    dirichlet: tuple[DirichletConfig, ...]
    rigid: tuple[str, ...]
    contact: ContactConfig


def _require_mapping(data: Any, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping, got {type(data).__name__}")
    return data


def _require_string(data: Any, context: str) -> str:
    if not isinstance(data, str) or not data:
        raise ValueError(f"{context} must be a non-empty string")
    return data


def _require_float(data: Any, context: str) -> float:
    if isinstance(data, bool):
        raise ValueError(f"{context} must be numeric, got bool")
    try:
        return float(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric") from exc


def _require_int(data: Any, context: str) -> int:
    if isinstance(data, bool):
        raise ValueError(f"{context} must be an integer, got bool")
    try:
        value = int(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if isinstance(data, float) and not data.is_integer():
        raise ValueError(f"{context} must be an integer")
    return value


def _require_string_triplet(data: Any, context: str) -> tuple[str, str, str]:
    if not isinstance(data, (list, tuple)) or len(data) != 3:
        raise ValueError(f"{context} must have exactly 3 entries")
    return tuple(
        _require_string(value, f"{context}[{index}]") for index, value in enumerate(data)
    )


def _require_numeric_triplet(data: Any, context: str) -> tuple[float, float, float]:
    if not isinstance(data, (list, tuple)) or len(data) != 3:
        raise ValueError(f"{context} must have exactly 3 entries")
    return tuple(
        _require_float(value, f"{context}[{index}]") for index, value in enumerate(data)
    )


def _resolve_path(base_dir: Path, raw_path: Any, context: str) -> Path:
    path = Path(_require_string(raw_path, context))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_mesh_config(data: Any, base_dir: Path) -> MeshConfig:
    section = _require_mapping(data, "mesh")
    return MeshConfig(path=_resolve_path(base_dir, section.get("path"), "mesh.path"))


def _parse_selectors_config(data: Any, base_dir: Path) -> dict[str, Path]:
    # No selectors at all is legitimate now that contact exists: a body held up by a collider
    # needs no Dirichlet region, and therefore no geometry to pick one out with.
    if data is None:
        return {}

    section = _require_mapping(data, "selectors")
    if not section:
        raise ValueError("selectors must define at least one selector")

    selectors: dict[str, Path] = {}
    for selector_name, raw_selector in section.items():
        selector_id = _require_string(selector_name, "selectors key")
        if isinstance(raw_selector, str):
            selectors[selector_id] = _resolve_path(
                base_dir, raw_selector, f"selectors.{selector_id}"
            )
            continue
        selector_section = _require_mapping(raw_selector, f"selectors.{selector_id}")
        selectors[selector_id] = _resolve_path(
            base_dir, selector_section.get("path"), f"selectors.{selector_id}.path"
        )
    return selectors


def _parse_material_config(data: Any) -> MaterialConfig:
    section = _require_mapping(data, "material")
    return MaterialConfig(
        mu=_require_float(section.get("mu"), "material.mu"),
        lam=_require_float(section.get("lam"), "material.lam"),
        density=_require_float(section.get("density"), "material.density"),
    )


def _parse_selector_classification_config(data: Any) -> SelectorClassificationConfig:
    if data is None:
        return SelectorClassificationConfig(mode="exact", grid_resolution=16, atol=1.0e-6)

    section = _require_mapping(data, "selector_classification")
    mode = _require_string(section.get("mode", "exact"), "selector_classification.mode")
    if mode not in {"exact", "aabb"}:
        raise ValueError("selector_classification.mode must be one of: exact, aabb")
    grid_resolution = _require_int(
        section.get("grid_resolution", 16),
        "selector_classification.grid_resolution",
    )
    if grid_resolution <= 0:
        raise ValueError("selector_classification.grid_resolution must be a positive integer")
    atol = _require_float(section.get("atol", 1.0e-6), "selector_classification.atol")
    if atol <= 0.0:
        raise ValueError("selector_classification.atol must be positive")
    return SelectorClassificationConfig(
        mode=mode,
        grid_resolution=grid_resolution,
        atol=atol,
    )


def _parse_solver_config(data: Any) -> SolverConfig:
    section = _require_mapping(data, "solver")
    acceleration = _parse_acceleration_config(section.get("acceleration"))
    line_search = _parse_line_search_config(section.get("line_search"))
    return SolverConfig(
        dt=_require_float(section.get("dt"), "solver.dt"),
        num_iterations=_require_int(
            section.get("num_iterations"), "solver.num_iterations"
        ),
        eps=_require_float(section.get("eps", 1.0e-6), "solver.eps"),
        acceleration=acceleration,
        line_search=line_search,
    )


def _parse_acceleration_config(data: Any) -> AccelerationConfig:
    if data is None:
        return AccelerationConfig(enabled=False, rho=None)

    section = _require_mapping(data, "solver.acceleration")
    enabled_raw = section.get("enabled", False)
    if not isinstance(enabled_raw, bool):
        raise ValueError("solver.acceleration.enabled must be a boolean")

    rho_raw = section.get("rho")
    rho = None if rho_raw is None else _require_float(rho_raw, "solver.acceleration.rho")
    if enabled_raw:
        if rho is None:
            raise ValueError("solver.acceleration.rho is required when acceleration is enabled")
        if not (0.0 < rho < 1.0):
            raise ValueError("solver.acceleration.rho must satisfy 0 < rho < 1")

    return AccelerationConfig(enabled=enabled_raw, rho=rho)


def _parse_line_search_config(data: Any) -> LineSearchConfig:
    if data is None:
        # Line search is the only guard against an inverting step, so it is on unless
        # a config opts out explicitly. alphas=None picks up the caller's defaults.
        return LineSearchConfig(enabled=True, alphas=None)

    section = _require_mapping(data, "solver.line_search")
    enabled_raw = section.get("enabled", False)
    if not isinstance(enabled_raw, bool):
        raise ValueError("solver.line_search.enabled must be a boolean")

    alphas_raw = section.get("alphas")
    num_alphas_raw = section.get("num_alphas")
    if alphas_raw is not None and num_alphas_raw is not None:
        raise ValueError(
            "set solver.line_search.alphas or solver.line_search.num_alphas, not both"
        )

    if num_alphas_raw is not None:
        num_alphas = _require_int(num_alphas_raw, "solver.line_search.num_alphas")
        if num_alphas < 2:
            raise ValueError("solver.line_search.num_alphas must be at least 2")
        return LineSearchConfig(
            enabled=enabled_raw, alphas=None, num_alphas=num_alphas
        )

    if alphas_raw is None:
        if enabled_raw:
            raise ValueError(
                "solver.line_search.alphas is required when line search is enabled "
                "(or set solver.line_search.num_alphas instead)"
            )
        return LineSearchConfig(enabled=False, alphas=None)

    if not isinstance(alphas_raw, (list, tuple)):
        raise ValueError("solver.line_search.alphas must be a list")
    if len(alphas_raw) == 0:
        raise ValueError("solver.line_search.alphas must contain at least one alpha")

    alphas = tuple(
        _require_float(value, f"solver.line_search.alphas[{index}]")
        for index, value in enumerate(alphas_raw)
    )
    # alpha=0 is allowed: it is the "decline to move" option that makes the line search
    # a descent safeguard rather than a forced step.
    if not all(0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("solver.line_search.alphas must satisfy 0 <= alpha <= 1")
    if not any(alpha > 0.0 for alpha in alphas):
        raise ValueError(
            "solver.line_search.alphas must contain at least one positive alpha"
        )

    return LineSearchConfig(enabled=enabled_raw, alphas=alphas)


def _parse_contact_config(data: Any) -> ContactConfig:
    """Parse the optional `contact` block. Absent means no contact at all."""
    if data is None:
        return ContactConfig(
            enabled=False,
            d_hat=1.0e-3,
            kappa=None,
            friction_mu=0.0,
            eps_v=1.0e-3,
            use_barrier=True,
            self_collision=False,
            self_collision_ccd=True,
            ccd_slack=0.9,
            capacity=4096,
            max_per_vertex=192,
            colliders=(),
        )

    section = _require_mapping(data, "contact")
    colliders = []
    for index, spec in enumerate(section.get("colliders", ()) or ()):
        collider = _require_mapping(spec, f"contact.colliders[{index}]")
        colliders.append(dict(collider))

    kappa = section.get("kappa")
    return ContactConfig(
        enabled=bool(section.get("enabled", True)),
        d_hat=_require_float(section.get("d_hat", 1.0e-3), "contact.d_hat"),
        kappa=None if kappa is None else _require_float(kappa, "contact.kappa"),
        friction_mu=_require_float(
            section.get("friction_mu", 0.0), "contact.friction_mu"
        ),
        eps_v=_require_float(section.get("eps_v", 1.0e-3), "contact.eps_v"),
        use_barrier=bool(section.get("use_barrier", True)),
        self_collision=bool(section.get("self_collision", False)),
        self_collision_ccd=bool(section.get("self_collision_ccd", True)),
        ccd_slack=_require_float(
            section.get("ccd_slack", 0.9), "contact.ccd_slack"
        ),
        capacity=_require_int(section.get("capacity", 4096), "contact.capacity"),
        max_per_vertex=_require_int(
            section.get("max_per_vertex", 192), "contact.max_per_vertex"
        ),
        colliders=tuple(colliders),
    )


def _parse_simulation_config(data: Any) -> SimulationConfig:
    section = _require_mapping(data, "simulation")
    steps = _require_int(section.get("steps"), "simulation.steps")
    if steps <= 0:
        raise ValueError("simulation.steps must be a positive integer")
    return SimulationConfig(steps=steps)


def _parse_dirichlet_config(
    data: Any, selector_ids: set[str]
) -> tuple[DirichletConfig, ...]:
    if data is None:
        return ()
    if not isinstance(data, list):
        raise ValueError("dirichlet must be a list")

    seen_selectors = set()
    dirichlet_entries = []
    for index, raw_entry in enumerate(data):
        entry = _require_mapping(raw_entry, f"dirichlet[{index}]")
        selector_name = _require_string(
            entry.get("selector"), f"dirichlet[{index}].selector"
        )
        if selector_name not in selector_ids:
            raise ValueError(
                f"dirichlet[{index}] references unknown selector {selector_name!r}"
            )
        if selector_name in seen_selectors:
            raise ValueError(
                f"Duplicate Dirichlet entry for selector {selector_name!r}"
            )
        seen_selectors.add(selector_name)
        dirichlet_entries.append(
            DirichletConfig(
                selector=selector_name,
                mode=_require_string(entry.get("mode"), f"dirichlet[{index}].mode"),
                components=_require_string_triplet(
                    entry.get("components"), f"dirichlet[{index}].components"
                ),
            )
        )
    return tuple(dirichlet_entries)


def _parse_rigid_config(data: Any, selector_ids: set[str]) -> tuple[str, ...]:
    if data is None:
        return ()
    if not isinstance(data, list):
        raise ValueError("rigid must be a list")

    seen_selectors = set()
    rigid_entries = []
    for index, raw_selector in enumerate(data):
        selector_name = _require_string(raw_selector, f"rigid[{index}]")
        if selector_name not in selector_ids:
            raise ValueError(f"rigid[{index}] references unknown selector {selector_name!r}")
        if selector_name in seen_selectors:
            raise ValueError(f"Duplicate rigid entry for selector {selector_name!r}")
        seen_selectors.add(selector_name)
        rigid_entries.append(selector_name)
    return tuple(rigid_entries)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load YAML problem files. Install it in the project virtual environment."
        ) from exc

    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        raise ValueError(f"YAML file {path} is empty")
    return _require_mapping(loaded, "config")


def load_config(path: str | Path) -> ProblemConfig:
    """Parse and validate a YAML problem definition."""
    config_path = Path(path).resolve()
    raw_config = _load_yaml_mapping(config_path)
    base_dir = config_path.parent

    mesh = _parse_mesh_config(raw_config.get("mesh"), base_dir)
    selectors = _parse_selectors_config(raw_config.get("selectors"), base_dir)
    selector_classification = _parse_selector_classification_config(
        raw_config.get("selector_classification")
    )
    material = _parse_material_config(raw_config.get("material"))
    solver = _parse_solver_config(raw_config.get("solver"))
    simulation = _parse_simulation_config(raw_config.get("simulation"))
    body_force = _require_numeric_triplet(
        raw_config.get("body_force", (0.0, 0.0, -9.81)), "body_force"
    )
    dirichlet = _parse_dirichlet_config(raw_config.get("dirichlet"), set(selectors))
    rigid = _parse_rigid_config(raw_config.get("rigid"), set(selectors))
    contact = _parse_contact_config(raw_config.get("contact"))
    # A body supported by a collider needs no kinematic constraint at all.
    supported_by_contact = contact.enabled and bool(contact.colliders)
    if not dirichlet and not rigid and not supported_by_contact:
        raise ValueError(
            "config must define at least one dirichlet or rigid constraint "
            "(or a contact collider to support the body)"
        )

    return ProblemConfig(
        mesh=mesh,
        selectors=selectors,
        selector_classification=selector_classification,
        material=material,
        solver=solver,
        simulation=simulation,
        body_force=body_force,
        dirichlet=dirichlet,
        rigid=rigid,
        contact=contact,
    )


def load_problem_from_yaml(
    path: str | Path, *, dtype=jnp.float32
) -> SimulationProblem:
    """Load a full simulation problem from a YAML file."""
    config = load_config(path)
    rest_positions, tets = parse_gmsh22_binary_tets(config.mesh.path, dtype=dtype)
    selector_options = SelectorClassificationOptions(
        mode=config.selector_classification.mode,
        grid_resolution=config.selector_classification.grid_resolution,
        atol=config.selector_classification.atol,
    )

    selector_memberships = []
    for selector_name, selector_path in config.selectors.items():
        selector_mesh = parse_binary_stl_mesh(selector_path)
        membership = classify_selector_vertices(
            rest_positions, selector_mesh, options=selector_options
        )
        selector_memberships.append(
            SelectorVertexMembership(
                selector_name=selector_name,
                vertex_indices=membership.vertex_indices,
                vertex_mask=membership.vertex_mask,
            )
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
    return assemble_problem(
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
        colliders=list(config.contact.colliders) or None,
        contact_d_hat=config.contact.d_hat,
        contact_kappa=config.contact.kappa,
        contact_friction_mu=config.contact.friction_mu,
        contact_eps_v=config.contact.eps_v,
        contact_use_barrier=config.contact.use_barrier,
        contact_enabled=config.contact.enabled,
        self_collision=config.contact.self_collision,
        self_collision_ccd=config.contact.self_collision_ccd,
        ccd_slack=config.contact.ccd_slack,
        contact_capacity=config.contact.capacity,
        contact_max_per_vertex=config.contact.max_per_vertex,
    )
