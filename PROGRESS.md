# Progress — feature/infrastructure

| Milestone | Status | Evidence | Commit |
|---|---|---|---|
| M1 — Hertz validation | DONE | barrier: exponent 1.5267, per-delta P/P_theory geomean 1.164; two_stage: exponent 1.4971, geomean 1.145 (gates: exponent +-0.075, geomean within 25% — calibrated by a 3-resolution refinement study, RESULTS.md); KE/PE < 1e-4; load-reaction residual < 5%; d_hat doubling shifts P by 1.06% | this commit |
| M2 — OGC two-stage activation | DONE | C2 at tau / C1 at d_hat pinned by FD (catches the paper's Eq. 19 typo); Hertz re-run under it (M1); convergence/conditioning vs log barrier in RESULTS.md | this commit |
| M3 — Per-vertex conservative bounds | DONE | per-step intersection audits clean (fast impact, no line search, Chebyshev on); negative twin still tunnels; far body keeps >= 95% of its inertial step while a near-contact interface is bound-limited (old filter: ~5%); adversarial review's rigid-row and prescribed-speed findings fixed and pinned by tests; global filter, ACCD and _MIN_USEFUL_TOI deleted | this commit |
| M4 — Static equilibrium + adjoint | DONE | 7 sensitivity classes vs central differences, worst rel err 9.6e-5 (gate 1e-4), barrier active (0 < gap 6.7e-4 < d_hat 1e-3); unconverged adjoint raises | this commit |
| M5 — Setup performance | DONE | element-wise identical outputs (masses bitwise); 231x masses, 41x incidence, 7x surface, 3x coloring on 105K tets | this commit |
| M6 — OGC offset geometry (stretch) | NOT ATTEMPTED | budget went to making M1 honest and verifying M3; a partial contact model is worse than none (DECISIONS.md); what its absence costs is quantified: M2's conditioning win exists only at a large standoff, M3's bounds stay ~0.45 d_hat for resting vertices | — |

# Progress — feature/traced-static

| Milestone | Status | Evidence | Commit |
|---|---|---|---|
| traced-solve + traced-adjoint | DONE | agreement 9.6e-14 (contact, gap 6.7e-4 in band) / 7.1e-14 (no contact) vs 1e-10 gate; vmap B=15 matches host sequential to 5.4e-13 worst; adjoint-under-vmap vs FD: mu 3.9e-6/7.9e-6, radius 3.3e-5/5.0e-6, rest-dir 3.9e-6/4.3e-6 (gate 1e-4); asymmetric refusal: converged=[T,F] → grad[1] all-NaN, grad[0] = solo within 4e-5 (CG-tol class); transcription trace bitwise vs host replica incl. forced rejections + NaN direction; adversarial review: 2 defects found and fixed (FINDINGS.md) | this commit |
| batch scaling + full suite | IN PROGRESS | benchmark B=1,4,16,64 and full `pytest -q` pending | — |
