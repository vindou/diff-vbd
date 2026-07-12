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
  line_search: { enabled: true, num_alphas: 9 }   # on by default
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

### Contact

Intersection-free contact via the IPC log barrier: continuous collision detection, lagged
Coulomb friction, analytic colliders (plane / sphere), and mesh self-collision. Colliders and
mesh-mesh pairs go through the same barrier, the same friction model and the same
time-of-impact filter — they are one code path, not two.

```yaml
contact:
  enabled: true
  d_hat: 1.0e-3          # activation distance: the barrier turns on inside this gap
  # kappa: 1.0e3         # omit to derive a stiffness matched to the inertia term
  friction_mu: 0.3       # Coulomb coefficient; 0 for frictionless
  eps_v: 1.0e-4          # sliding speed below which a contact is treated as stuck
  self_collision: false  # also detect the mesh against itself
  colliders:
    - kind: plane
      normal: [0.0, 0.0, 1.0]   # points into the free half-space
      offset: 0.0
    - kind: sphere
      center: [0.0, 0.0, -2.0]
      radius: 1.0
      outside: true             # false to be contained *inside* the sphere
```

See [`examples/block_on_plane.yaml`](examples/block_on_plane.yaml). A body supported by a
collider needs **no Dirichlet constraint** — resting on the ground is well-posed on its own,
so that example has no `dirichlet` block at all.

**Contact needs float64.** Pass `--precision float64`. The barrier resolves a gap orders of
magnitude smaller than the mesh coordinates, and it obtains that gap by *subtracting* them:
in float32, at a coordinate of 100, the absolute resolution is `1.2e-5`, so a small positive
gap can round to zero or below — and `log(gap / d_hat)` is then NaN, losing
intersection-freedom to rounding before any solver logic runs. `assemble_problem` rejects a
`d_hat` the float type cannot resolve rather than letting it fail later as a NaN.

#### What "intersection-free" actually guarantees

The guarantee is enforced by a **time-of-impact filter on the whole sweep**, not per vertex.
A per-vertex bound certifies one vertex moving against frozen geometry, but VBD solves a
whole colour in parallel from one snapshot — if a contact pair's endpoints share a colour,
both move and neither certificate covers their combined motion. Worse, Chebyshev acceleration
extrapolates *after* every local solve has finished, where no per-vertex check can see it.
Scaling the entire update by a single conservative factor bounds where the mesh actually ended
up, whatever the interior did.

For analytic colliders the bound is closed-form and holds unconditionally. For mesh-mesh it
rests on the pair set being complete, and that needs two things that work only as a pair:

* the detection band is derived from how far the mesh is about to move, not from `d_hat`, so
  every pair that could reach contact this step is in the set; and
* the sweep clamps each vertex's **cumulative** displacement, so none can leave the band it
  was detected under.

The cost is real and it is the honest limit: the band grows with speed, and past roughly one
body-length of travel per step no once-per-step candidate set can be both bounded and
complete. At that point the pair capacity overflows and says so. **That is by design** — the
alternative is a band that quietly under-covers and lets a surface tunnel unseen. Reduce
`solver.dt`, or set `self_collision_ccd: false` to trade the guarantee for speed (it warns).

Two limits worth stating plainly:

* *"No **non-adjacent** surface primitive pair intersects."* Primitives that share a vertex are
  excluded from detection — a vertex always touches its own triangles — so a triangle inverting
  through its own neighbour is a tet-inversion problem, not a contact one.
* Grazing motion at speed is **over-throttled**, not unsafe. The conservative advance is
  proportional to the gap, so two surfaces sliding past each other fast at a tiny separation
  exhaust the iteration budget and the step gets shorter than it needed to be. The solver
  reports this rather than silently freezing.

`self_collision` also requires a **conforming** tetrahedralisation, and checks it: a
non-conforming split leaves interior faces unpaired, they get reported as *surface* faces, and
self-collision then finds phantom contacts deep inside a solid body at rest. Every surface
edge must be shared by exactly two triangles, or assembly fails. (For a structured grid, use
the Kuhn 6-tet split, which conforms unconditionally, rather than the 5-tet one.)

`self_collision` allocates fixed-capacity pair buffers (`capacity`, `max_per_vertex`).
They are fixed because rebuilding them each step at the *same* shape is a jit cache hit,
whereas a capacity that grew with the contact count would recompile the solver every step.
Overflow is a hard error, never a silent drop of whichever pairs happened to be last.

**`max_per_vertex` is a direct linear multiplier on solve cost**, not just a memory bound —
every vertex evaluates all of its slots on every local solve, of which there are
(colours × sweeps) per step, and the whole thing sits inside a Hessian. Measured on
[`examples/two_blocks.yaml`](examples/two_blocks.yaml): 128 → 9 s/step, 256 → 19 s/step. Size
it tightly; the overflow error names the exact number you need. `capacity` is much cheaper (it
only feeds the time-of-impact filter, a shallow kernel), so give that one room.

Flat-on-flat contact between two coincident faces is the case that stresses `max_per_vertex`:
it generates a large edge-edge fan at a single vertex.

See [`examples/two_blocks.yaml`](examples/two_blocks.yaml) for the mesh-mesh path end to end —
a block dropped onto a fixed block, with no analytic colliders at all. Its assets are generated
by `python examples/make_two_blocks.py`.

### Line search

`num_alphas: N` builds a linear step-size grid from `1.0` down to `0.0` inclusive; raise it to
refine the search. Refining is cheap — the per-vertex gradient, Hessian and solve dominate, so
9 alphas cost only ~9% more than 4, and every alpha is evaluated in parallel under `jax.vmap`.

The `0.0` endpoint matters: it lets a vertex decline to move when every positive step would
increase its local objective. Without it the search is forced to return an objective-*increasing*
step for such vertices, which pumps energy into the mesh and makes the solve diverge as
`num_iterations` grows. Set `alphas: [...]` instead to pin the grid explicitly.

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
