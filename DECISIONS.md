# Decisions — feature/infrastructure

Format: date · decision · rationale. Append-only.

- **2026-07-16 · Verified the OGC Eq. 19 correction before writing code.** The paper prints
  `k'_c = tau * k_c * (tau - r)^2` (energy·length — dimensionally wrong). C1 matching of
  `g1' = -k_c (r - d)` and `g2' = -k'_c / d` at `d = tau` gives `k'_c = tau * k_c * (r - tau)`,
  and only that form reproduces the paper's own C2 result: `g2''(tau) = k'_c / tau^2
  = k_c (r - tau) / tau`, equal to `g1'' = k_c` iff `tau = r/2`, where `k'_c = k_c r^2 / 4`.
  The paper's Eq. 20 for `b` is used as printed.

- **2026-07-16 · Implementation order: M5 before M1, everything else per brief (M1, M2, M4, M3, M6?).**
  The Hertz half-space mesh needs ~80K tets to satisfy the brief's own modelling requirements
  (extent >= 10a, cell ~ a_min/2.5), and the current `build_lumped_masses` dispatches one op per
  tet — setup alone would take minutes per test run. M5 is mechanical and gated on output
  equality, so it goes first. M1 is still the first milestone that touches *physics*, and it
  lands before any contact-model change (M2/M3), which is what "M1 first" is for.

- **2026-07-16 · M2 keeps `use_barrier` working as a legacy spelling.** The enum
  (`contact_activation: barrier | penalty | two_stage`) replaces the bool everywhere
  internally; the bool maps onto it at the `assemble_problem` boundary, and specifying
  both is an error rather than a precedence rule. Existing configs must not change
  behaviour underneath their owners.

- **2026-07-16 · M2's continuity gate asserts what is mathematically true, not what the
  brief's shorthand says.** The two-stage activation is C2 at the stitch `tau` and only
  C1 at `d_hat`: `g''` steps from `k_c` to 0 there, as it must for any quadratic
  activation (the paper's own C2 claim is about `tau`). The test pins the size of that
  step to exactly `k_c` instead of pretending it is zero — hiding it would be tuning a
  gate to pass.
