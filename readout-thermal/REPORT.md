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

## M2 — thermal fixed point (2026-07-05)
Built:
- thermal.py: ModelParams dataclass; dissipated_power (§3.1, mean of the two
  charge states, same adaptive trapezoidal quadrature as §2.3);
  cooling_power (§3.2, Sigma_eff*(Te^p - Tph^p) plus optional WF term as an
  additional cooling channel — reading documented, see QUESTIONS.md Q1);
  background_power (§3.2, P_bg from Te(Vpp->0) = Te0; raises for Te0 < Tph).
- selfconsistent.py: residual R(Te) (§3.3); solve_te with brentq on
  [Tph, 10 K], explicit sign-change check at both bracket ends, uniqueness
  check (exactly one sign change on a 200-point grid), hard ValueError on
  no-root/non-unique, brentq disp=True raises on non-convergence — never
  clips; solve_te_grid vectorized wrapper.
Tests: T3 (|Te - Te0| < 1e-6 K at Vpp -> 0), T4 (Te strictly increasing over
3 decades of Vpp), T5 (exactly one sign change for Vpp from 1e-3 to 100
kB*Te0/e), T7 (argmax DeltaX with self-heating strictly below the heating-off
argmax, both interior; argmax SNR == argmax DeltaX since sigma_I is constant,
§4), T8 (P_diss >= 0 on a (Te, eps_b, Vpp) grid including Vpp = 0), plus P_bg
consistency and out-of-bracket hard-failure tests.
Pytest: 22 passed (before audit fixes; 29 after, including M3 drafts).
Physics-auditor: no blockers; 2 major + 7 minor, addressed as follows.
1. (major) Uniqueness check was a 200-point sign count that could miss
   tangent/close double roots and never verified the §3.3 condition. Fixed:
   solve_te now asserts R strictly negative AND strictly monotone decreasing
   on a grid beyond the found root (_assert_monotone_beyond); a tangent root
   above would violate monotone decrease and raise.
2. (major) No test pinned the P_diss magnitude/amplitude convention (scale
   errors degenerate with fitted Sigma_eff). Fixed: new test pins the §3.1
   small-signal limit per state P -> G(eps)*Vpp^2/8 (from §2.2) and the
   mean-of-states, rtol 1e-5.
3. (minor) T3 tolerance loosened vs. solver tolerance. Fixed: e*Vpp =
   1e-4*kB*Te0 where the physical shift is ~1e-10 K; bound tightened to
   1e-9 K.
4. (minor) T5 was circular (tested the same helper solve_te uses). Fixed:
   test adds an independent 1500-point sign count computed directly from
   residual().
5. (minor) te0 >= tph restriction was smuggled. Now logged as QUESTIONS.md
   Q3 (P_bg >= 0 reading).
6. (minor) WF term's induced modification of the §3.3 residual and P_bg was
   not covered by Q1. Q1 extended.
7. (minor) R(Te) was stepwise-discontinuous inside brentq (te-dependent
   adaptive n_t). Fixed: n_t frozen once per solve at the worst case
   Te = Tph.
8. (minor) Zero-sample handling in count_sign_changes contradicted its
   comment; r_lo == 0 bypassed uniqueness. Fixed: zeros inherit the
   preceding sign; the monotone-beyond check now runs in the boundary-root
   corner too.
9. (minor) p in {4,6} and include_wf paths untested. Fixed: new sensitivity
   test covers P_bg closed form and Te(Vpp->0) = Te0 for p in {4,5,6} x
   include_wf in {False, True}.
Open issues: QUESTIONS.md Q1 (WF reading), Q3 (te0 >= tph constraint);
default configuration unaffected by both.

## M3 — sweep and figure (2026-07-05)
Built:
- noise.py (§4): sigma_i = sqrt(S_I*B) as the single fitted constant used
  directly; optional shot term sigma_shot = sqrt(2*e*Ibar*B) with Ibar as an
  explicit caller input (averaging convention not in SPEC — QUESTIONS.md Q2);
  total_sigma adds the optional term in quadrature; snr = DeltaX/sigma (§4).
- sweep.py: sweep() composes §3.3 Te (or Te0 when self_heating=False, the
  Sigma_eff -> infinity limit), §2.4 DeltaX (fixed or reoptimize_bias), §4
  SNR over a Vpp grid; µVpp/mK conversion helpers at the I/O boundary.
- scripts/fig1d.py: dual-axis figure (SNR left, Te [mK] right, log x in
  µVpp) with PLACEHOLDER constants; wrote figures/fig1d_placeholder.png.
  With placeholders (A = 1 nA, Te0 = 100 mK, Tph = 50 mK, delta_eps =
  2 kB*Te0, Sigma_eff = 5e-11 W/K^5, sigma_I = 0.2 pA) the model already
  produces the Fig. 1(d) phenomenology: SNR rollover (optimum ~58 µVpp) and
  superlinear Te rise — shapes emerge from the model, not the fit (§5).
Tests: §4 definitions (snr, shot term off by default, quadrature sum), unit
helpers roundtrip, sweep composition (Te matches §3.3 path, snr = dx/sigma,
heating-off pins Te0 and does not reduce the signal), reoptimized bias never
worse than fixed bias.
Pytest: 29 passed. PNG produced (figures/fig1d_placeholder.png).
Physics-auditor: 1 blocker + 2 major + 4 minor, all addressed before commit.
1. (BLOCKER) noise.total_sigma implemented a quadrature sum of sigma_I and
   the shot term — an equation absent from SPEC §4 — and implicitly redefined
   SNR's denominator. Removed entirely: noise.py now contains only the
   literal §4 expressions (sigma_shot = sqrt(2*e*Ibar*B); SNR =
   DeltaX/sigma_I); the combination rule and SNR-denominator question are
   logged in the extended Q2. sweep() divides by sigma_i directly.
2. (major) reoptimize_bias reading unlogged: the §3.3 heat balance uses the
   FIXED params.eps_b, with eps_b re-optimized afterwards at the resulting
   Te (no joint (Te, eps_b) fixed point). Logged as QUESTIONS.md Q4 and in
   the sweep() docstring.
3. (major) test_sweep_composition could not catch a sweep that computed the
   signal at the wrong Te. Fixed: every point is now pinned by independent
   recomputation (solve_te + delta_x + division) at rtol 1e-12, for both
   heating-on and heating-off paths.
4. (minor) shot_sigma silently applied abs(Ibar) despite Q2. Fixed: raises
   ValueError for Ibar < 0 pending Q2.
5. (minor) heating-off >= heating-on signal assertion was unproven physics
   not in SPEC. Removed; the sanctioned heating comparison remains T7.
6. (minor) TPH in fig1d.py presented 50 mK as the paper's value. Now marked
   as a placeholder to be replaced by the cited fridge base temperature in
   M4.
7. (minor) dead import E in fig1d.py removed.
Open issues: QUESTIONS.md Q2 (shot-noise conventions), Q4 (reoptimize_bias
coupling); both default-off / M4-only.

## M4 — calibration: BLOCKED (2026-07-05)
data/mills_fig1d_snr.csv and data/mills_fig1d_te.csv do not exist yet.
Digitizing Fig. 1(d) is designated a HUMAN task (see data/README.md for the
required format). Per the milestone definition M4 does not start until the
human provides these files; stopping after M3 as instructed.
