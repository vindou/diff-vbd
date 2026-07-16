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

Intersection-free contact: IPC-style activation energies (log barrier, quadratic penalty,
or OGC's two-stage function), lagged Coulomb friction, analytic colliders (plane / sphere),
and mesh self-collision guaranteed intersection-free by per-vertex conservative bounds
[Wu et al. 2020]. Colliders and mesh-mesh pairs go through the same activation energy, the
same friction model and the same per-vertex motion clipping — one code path, not two.

```yaml
contact:
  enabled: true
  d_hat: 1.0e-3          # activation distance: contact forces turn on inside this gap
  # kappa: 1.0e3         # omit to derive a stiffness matched to the inertia term
  friction_mu: 0.3       # Coulomb coefficient; 0 for frictionless
  eps_v: 1.0e-4          # sliding speed below which a contact is treated as stuck
  activation: barrier    # barrier (IPC log) | penalty (quadratic) | two_stage (OGC)
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

#### Contact activation energies

Three activation energies are selectable (all published; none of this is novel to this
repo): the **IPC log barrier** [Li et al. 2020] — C2 at `d_hat`, infinite force at zero
gap, curvature growing like `1/gap^2`; the **quadratic penalty** (the VBD paper's) —
cheap, bounded, and *not* intersection-free; and OGC's **two-stage activation**
[Chen et al. 2025, Eq. 18] — a quadratic near `d_hat` stitched C2-continuously at
`tau = d_hat/2` onto a pure log, keeping the barrier's infinite force while the quadratic
stage's curvature stays exactly `kappa`. Two honest notes: the two-stage energy is only C1
*at* `d_hat` (its curvature steps from `kappa` to zero there, as any quadratic
activation's must — the log barrier is C2 there), and the stitch coefficient printed in
the OGC paper (its Eq. 19) is dimensionally inconsistent; the implementation derives the
correct one and the tests would catch the printed form. The legacy `use_barrier` boolean
still works and maps onto the enum; specifying both is an error.

**Contact needs float64.** Pass `--precision float64`. The barrier resolves a gap orders of
magnitude smaller than the mesh coordinates, and it obtains that gap by *subtracting* them:
in float32, at a coordinate of 100, the absolute resolution is `1.2e-5`, so a small positive
gap can round to zero or below — and `log(gap / d_hat)` is then NaN, losing
intersection-freedom to rounding before any solver logic runs. `assemble_problem` rejects a
`d_hat` the float type cannot resolve rather than letting it fail later as a NaN.

#### What "intersection-free" actually guarantees

The guarantee is enforced by **per-vertex conservative bounds** (Wu et al. 2020, as used
by Offset Geometric Contact [Chen et al., SIGGRAPH 2025]). At every contact detection,
each vertex gets a budget

```
b_v = 0.45 * min(distance to everything v's primitives could first touch, detection band)
```

and every displacement the solver produces — the initial guess, each iteration's output,
Chebyshev extrapolation included — is truncated so no vertex strays further than `b_v`
from where detection saw it. If the state at detection was intersection-free, it stays
intersection-free: both sides of a pair at distance `d` have budgets below `d/2`, and a
primitive distance is 1-Lipschitz in each side, so the pair cannot close. Both endpoints
moving simultaneously is priced in — that factor of one half is what lets single-vertex
certificates compose where they naively could not, and it needs **no CCD and no global
reduction at all**. (An earlier revision of this solver scaled the whole sweep by a single
time-of-impact scalar for exactly that composability worry; the bound above is the
literature's answer to it, and the global filter — along with its documented failure mode,
a time of impact collapsing to zero and silently freezing the entire mesh — is gone.)

Bounds are refreshed by detection, and detection re-runs mid-step once ~1% of the
vertices have consumed their budgets, re-anchoring everyone with fresh distances. The
detection band is now a **performance knob, not a soundness parameter**: a pair beyond
the band is further apart than any two budgets can close, so an undersized band costs
re-detections, never an intersection. The band still grows with speed so that a fast
free-flying body fits its step inside one detection interval.

Three limits worth stating plainly:

* *"No **non-adjacent** surface primitive pair intersects."* Primitives that share a vertex
  are excluded from detection — a vertex always touches its own triangles — so a triangle
  inverting through its own neighbour is a tet-inversion problem, not a contact one.
* **Near-contact stepping is still slow — but now only near the contact.** A vertex resting
  at gap `~d_hat` has a budget of `~0.45 * d_hat` per detection interval; OGC states plainly
  that these bounds only pay off fully with a *large* contact radius, which needs its offset
  geometry (not implemented here) to be artifact-free. So this scheme converts **global**
  throttling into **local** throttling: only vertices actually near contact are slowed,
  instead of the whole mesh being scaled back by its worst offender. It does not make
  near-contact stepping cheap.
* Prescribed (Dirichlet) and rigid-region motion cannot be truncated — one is a boundary
  condition, the other is rewritten by the rigid projection after every filter. Prescribed
  targets are therefore applied in per-iteration increments (re-detecting on demand to
  refresh budgets), which keeps scripted presses at ordinary speeds working; a single
  increment that exceeds even a freshly detected budget, or a rigid row that outruns its
  bound, is a hard error naming the vertex and the fix (smaller `dt`, more iterations, or
  slow the motion).

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
only feeds the per-vertex bound computation, a shallow host-side pass), so give that one room.

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

### Energy API

The solver never forms a global energy — VBD's premise is a 3×3 solve per vertex against a
frozen block — but the energies it descends are also stated once, whole-mesh, in
`solver/potential.py`:

```python
import diff_vbd as dv

E = dv.potential_energy(problem, positions, previous_positions)          # elastic + contact
G = dv.variational_energy(problem, positions, inertial_target, previous_positions)
```

`elastic_potential` and `inertia_potential` are the individual terms; `variational_energy`
is G(x), the implicit-Euler objective a VBD sweep decreases. One energy definition, two
consumers: the local vertex objective and this global view share the same kernels
(`tet_energy`, the contact energies), so the forward solve and a future tangent-stiffness
or adjoint path cannot silently disagree about what the energy *is*. The local objective
sums the elements *incident to* a vertex, so it counts each tet (and each contact pair)
four times; the tests pin that factor exactly rather than assuming it.

Two limits worth stating: gradients of these potentials do not differentiate through
contact *detection* (the active set and classifications are frozen inputs, refreshed by
`redetect_contacts`), and monotone descent of G is only guaranteed in the regime the VBD
paper's argument covers — line search on, Chebyshev acceleration off, no per-vertex
bound truncation rescaling an update.

### Static equilibrium and the adjoint

`potential_energy` used to describe itself as "the hook a future adjoint path attaches
to"; the hook now has something attached:

```python
result = dv.solve_static_equilibrium(problem, tol=1e-9)   # residual tolerance, never a count
adjoint = dv.StaticAdjoint(problem, tol=1e-9)
gradients = jax.grad(lambda p: loss(adjoint(p)))(dv.StaticParams(mu=mu0))
```

`body_force_potential` completes the static energy (gravity was previously baked into the
per-step inertial target, which only exists inside a timestep), and
`solve_static_equilibrium` minimises it with matrix-free Newton-CG — Hessian-vector
products via `jvp`-of-gradient, `H` never assembled. `StaticAdjoint` differentiates the
equilibrium implicitly (`du*/dp = -H^{-1} dg/dp`) for material parameters, density, rest
positions, body force and collider parameters, at the cost of one extra CG solve — never
by unrolling the solver.

The design constraint worth knowing: **implicit differentiation is only true at a
stationary point.** A fixed-count VBD sweep's output is not one, so the solver terminates
on a residual tolerance, VBD sweeps are allowed only as a (never-differentiated) warm
start, and asking `StaticAdjoint` for a gradient at a state whose residual exceeds its
tolerance raises rather than returning the plausible-looking gradient of nothing. Static
solves refuse friction (not the gradient of any potential), rigid regions (a constraint
outside the energy) and self-collision (the pair set is only complete within a band sized
for one dynamic step); each refusal names its reason and remedy. Like every interior-point
method, the solve needs a feasible (non-penetrating) initial state. Dynamic per-step and
trajectory adjoints are out of scope on this branch; they inherit the same precondition
in a harsher form — every step of the trajectory would have to be solved to stationarity
of its own objective before the implicit-function argument applies.

### Validation

The forward contact model is checked against the canonical closed-form benchmark: a rigid
sphere indenting an elastic slab (Hertz). `tests/test_hertz.py` fits `log P` vs
`log delta` over a factor-4 range of indentations and pins **both** the exponent (3/2 —
the contact model) and the coefficient (`(4/3) E* sqrt(R)` — that stable Neo-Hookean
really reduces to linear elasticity at small strain, which nothing else in the suite
verifies). Measured numbers, mesh-resolution caveats and the barrier-standoff sensitivity
are in `RESULTS.md`.

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
