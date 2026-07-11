# diff-vbd

A differentiable **Vertex Block Descent (VBD)** solver for soft-body simulation, built on
[JAX](https://github.com/jax-ml/jax). It loads tetrahedral meshes, assembles boundary
conditions from selector geometry, and time-integrates a stable Neo-Hookean elastic model
(Smith et al. 2018) with optional Chebyshev acceleration and line search.

## Installation

```bash
pip install -e .          # CPU
pip install -e ".[cuda]"  # CUDA 12 GPU wheels
pip install -e ".[dev]"   # + test dependencies
```

Requires Python 3.10+.

## Quickstart (library API)

```python
import diff_vbd as dv

# Build a problem from a YAML config (see examples/ for the schema)
problem = dv.load_problem_from_yaml("examples/cantilever_beam.yaml")

state = dv.initial_state(problem)
final_state, history = dv.simulate(problem, state, num_steps=100)

dv.export_simulation_npz("out.npz", problem, history)
```

Everything in the public API is re-exported from the top-level package — see
[`src/diff_vbd/__init__.py`](src/diff_vbd/__init__.py) for the full surface
(`assemble_problem`, `step`, `simulate`, boundary-condition builders, exporters, etc.).

## Quickstart (CLI)

Installing the package registers a `diff-vbd` console script:

```bash
diff-vbd --config examples/cantilever_beam.yaml --platform cpu --out out.npz
```

Key flags: `--platform {cpu,gpu}` (default `gpu`), `--steps N` (override the config),
`--out PATH` (NPZ trajectory output), and GPU memory controls
(`--gpu-preallocate`, `--gpu-mem-fraction`).

## Configuration

Simulations are described by a YAML config. Minimal cantilever example
([`examples/cantilever_beam.yaml`](examples/cantilever_beam.yaml)):

```yaml
mesh:
  path: beam.msh                 # Gmsh 2.2 binary tetrahedral mesh
selectors:
  clamp:
    path: bc_selector.stl        # binary STL used to select vertices
material:
  mu: 4.638e5                    # Lamé parameters + density
  lam: 4.174e6
  density: 1.04
simulation:
  steps: 100
solver:
  dt: 0.02
  num_iterations: 3
  eps: 1.0e-6
  acceleration: { enabled: false, rho: 0.95 }
  line_search: { enabled: true, alphas: [1.0, 0.5, 0.25, 0.125] }  # on by default
body_force: [0.0, 0.0, -9.81]
dirichlet:
  - selector: clamp
    mode: position
    components: ["0.0", "0.0", "0.0"]
```

The cantilever example ships with its mesh assets (`examples/beam.msh`,
`examples/bc_selector.stl`), so it runs as-is. For your own simulations, supply tetrahedral
meshes and selector geometry and point the config `path` fields at them (relative paths
resolve against the config file's directory).

## Testing

```bash
pytest
```

The test suite is self-contained: it synthesizes small meshes in temp directories, so no
external data files are required.

## Repository layout

```
src/diff_vbd/        # the installable package
  config/            #   YAML config loading
  setup/             #   mesh & selector I/O, topology, boundary conditions
  problems/          #   ready-made problem builders (e.g. cantilever)
  solver/            #   VBD kinematics, stable Neo-Hookean materials, time integration
  export/            #   NPZ trajectory export
  cli.py             #   `diff-vbd` command-line runner
  runtime_config.py  #   JAX/XLA backend configuration
examples/            # example YAML configs
tests/               # unit tests
```
