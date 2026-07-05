# REPORT — readout-thermal Phase 1

## M0 — scaffold (2026-07-05)
Built: src layout (constants, sensor, thermal, selfconsistent, noise, sweep,
calibrate — physics modules are placeholders), tests/, scripts/, data/raw/,
pyproject.toml with pytest config, REPORT.md, QUESTIONS.md. constants.py holds
kB and e (CODATA 2018 exact SI values). Smoke test imports every module and
pins the constant values.
Pytest: 9 passed.
Open issues: none. No physics implemented, so no physics-auditor run for M0.
Note: environment provides Python 3.12.3 via a project-local venv (.venv/,
gitignored).

## M1 — sensor transport (2026-07-05)
Built (src/readout_thermal/sensor.py, all docstrings cite SPEC sections):
- fermi (§1), current (§2.1), analytic conductance (§2.2),
- lockin_x (§2.3): first-harmonic quadrature, trapezoid on a uniform 257-point
  grid over one period (spectral accuracy for the smooth periodic integrand);
  result exactly independent of freq for the quasi-static current, freq kept
  as an explicit input to mirror §2.3,
- delta_x (§2.4), optimal_bias (§2.4 re-optimized mode: coarse scan +
  bounded refinement), default_bias (§2.4 DEFAULT mode: base condition with
  "Vpp small" realized as e*Vpp = 0.01*kB*Te0 — a documented quadrature-scale
  choice; the resulting eps_b is Vpp-independent in the linear-response limit).
Tests: T1 (dI/dV vs §2.2, rtol 1e-6), T2 (FWHM = 3.525 kB*Te within 0.5%),
T6 (SNR proportional to Vpp within 2% for e*Vpp < 0.1 kB*Te0), plus two
cross-checks (small-signal X -> G*Vpp/2; I odd in V).
Pytest: 14 passed (before audit fixes; 15 after).
Physics-auditor: no blockers; 2 major + 3 minor findings, addressed as follows.
1. (major) optimal_bias scan bracket did not scale with drive amplitude —
   at e*Vpp/2 > |delta_eps| + 8 kB*Te the true optimum fell outside the grid
   and was silently missed. Fixed: bracket now +/-(|delta_eps| + e*Vpp/2 +
   8 kB*Te).
2. (major) minimize_scalar bounded refinement used SciPy's default absolute
   xatol = 1e-5, larger than the whole bracket in SI energy units (~1e-25 J),
   so refinement was a no-op. Fixed: xatol = 1e-6 * bracket width.
3. (minor) fixed n_t = 257 quadrature loses accuracy for e*Vpp >> kB*Te.
   Fixed: quadrature_points() scales n_t with 16*e*Vpp/(kB*Te) (floor 257);
   used by lockin_x, delta_x, optimal_bias and §3.1 dissipated_power. New
   convergence test at e*Vpp = 200 kB*Te vs. 4x-oversampled reference.
4. (minor) T2 measured the FWHM of the analytic §2.2 formula (tautological).
   Fixed: T2 now locates the half-maximum of the finite-difference dI/dV of
   the §2.1 current.
5. (minor) freq=1.0 placeholder default in lockin_x for a SPEC-fixed input.
   Kept (result is exactly freq-independent for the quasi-static §2.1
   current) with a strengthened docstring warning that any future
   frequency-dependent physics must replace it by the paper's value.
Open issues: none.
