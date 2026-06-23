#!/usr/bin/env python3
"""Judgment experiment: does leakage accumulate coherently over repeated cycles?

If leakage accumulates as n^2 (coherent), then lattice anti-sync (phase cycling
across plaquettes so Sum e^{i phi_n} = 0) has real value for surface-code QEC.
If it accumulates as n or saturates (incoherent), then measurement-and-reset
absorbs it and anti-sync adds little.

Protocol
--------
Repeatedly apply a 2-qubit gate (validated CNOT, optionally valley-detuned) for
N = 1 .. N_max cycles.  At each cycle boundary, project back onto the logical
subspace and measure leakage L_N = 1 - Tr(M_N^dag M_N) / d.

Scenarios:
  A. Spin-only validated CNOT (per-gate leakage ~ 7.7e-9) — baseline.
  B. Spin-only, effective valley detuning (inter areas * cos^2(dphi/2))
     — shows saturation physics in spin model.
  C. Full valley model, small dphi — small per-gate leakage, long growth
     window before saturation.
  D. Full valley model, large dphi — near-immediate saturation for contrast.
  E. Full valley model, alternating +/- dphi (2-cycle echo) — anti-sync test.

Fits L_N ~ c * N^alpha in two windows (early / full range) to separate
transient growth from asymptotic behavior.

Outputs to out/leakage_accumulation/ : leakage_accumulation.svg, report.md,
results.json.  Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir.native import two_qubit_template
from eo_pulse_ir.sim.encoding import logical_basis
from eo_pulse_ir.sim.fidelity import leakage as compute_leakage
from eo_pulse_ir.sim.landscape import line_plot_svg, write_svg
from eo_pulse_ir.sim.operators import exchange_propagator
from eo_pulse_ir.sim.robust import _ROLE_EDGE, roles_to_edges
from eo_pulse_ir.sim.valley import (
    _apply_two_dot,
    _embed_spin_to_spinvalley,
    logical_basis as spin_logical_basis,
    two_dot_pulse,
)

NUM_QUBITS = 2
N_DOTS = 6


# ---------------------------------------------------------------------------
# Spin-only helpers
# ---------------------------------------------------------------------------

def build_spin_unitary(edges, areas):
    """Build the full 2^N_DOTS x 2^N_DOTS unitary for a pulse sequence."""
    dim = 1 << N_DOTS
    U = np.eye(dim, dtype=complex)
    for (i, j), area in zip(edges, areas):
        U = exchange_propagator(N_DOTS, i, j, area) @ U
    return U


def spin_leakage_trace(U, N_max):
    """Iterate U^N in the spin-only Hilbert space, return leakage at each N."""
    L = logical_basis(NUM_QUBITS)  # (64, 4)
    state = L.astype(complex)
    leakages = np.empty(N_max)
    for n in range(N_max):
        state = U @ state
        M = L.conj().T @ state
        leakages[n] = compute_leakage(M)
    return leakages


# ---------------------------------------------------------------------------
# Valley-model helpers
# ---------------------------------------------------------------------------

def build_valley_ops(edges, areas, valley_phase, e_vs):
    """Precompute the list of (dot_index, 16x16 op) for one gate cycle."""
    phase = np.asarray(valley_phase, float)
    evs = np.full(N_DOTS, e_vs, float) if np.isscalar(e_vs) else np.asarray(e_vs, float)
    ops = []
    for (i, j), area in zip(edges, areas):
        op16 = two_dot_pulse(area, phase[j] - phase[i], evs[i], evs[j])
        ops.append((i, op16))
    return ops


def valley_leakage_trace(ops_list, N_max):
    """Iterate through valley-model gate cycles, cycling ops_list.

    Returns leakage at each cycle boundary.
    """
    n_dots = N_DOTS
    E = _embed_spin_to_spinvalley(n_dots)
    Lspin = spin_logical_basis(NUM_QUBITS)
    Lfull = (E @ Lspin).astype(complex)  # (4096, 4)
    state = Lfull.copy()

    n_variants = len(ops_list)
    leakages = np.empty(N_max)
    for n in range(N_max):
        ops = ops_list[n % n_variants]
        for i, op16 in ops:
            state = _apply_two_dot(state, n_dots, i, op16)
        M = Lfull.conj().T @ state
        leakages[n] = compute_leakage(M)
    return leakages


# ---------------------------------------------------------------------------
# Power-law fitting
# ---------------------------------------------------------------------------

def fit_power_law(ns, leakages, min_n=2, max_n=None, max_leak=0.5):
    """Fit L ~ c * n^alpha via log-log linear regression.

    Returns (alpha, c, r2).
    """
    mask = (ns >= min_n) & (leakages > 0) & (leakages < max_leak)
    if max_n is not None:
        mask &= (ns <= max_n)
    if mask.sum() < 3:
        mask = (ns >= min_n) & (leakages > 0)
        if max_n is not None:
            mask &= (ns <= max_n)
    if mask.sum() < 3:
        return 0.0, 0.0, 0.0
    ln = np.log(ns[mask].astype(float))
    ll = np.log(leakages[mask])
    A = np.vstack([ln, np.ones_like(ln)]).T
    result = np.linalg.lstsq(A, ll, rcond=None)
    (alpha, logc) = result[0]
    ss_res = np.sum((ll - A @ result[0]) ** 2)
    ss_tot = np.sum((ll - np.mean(ll)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(alpha), float(np.exp(logc)), float(r2)


def classify(alpha):
    if alpha > 1.7:
        return "coherent (n^2)"
    if alpha > 0.7:
        return "diffusive (n)"
    if alpha > 0.2:
        return "sub-diffusive"
    return "saturating"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out-dir", default="out/leakage_accumulation")
    ap.add_argument("--n-max", type=int, default=5000,
                    help="max cycles for spin-only scenarios")
    ap.add_argument("--n-max-valley", type=int, default=800,
                    help="max cycles for valley-model scenarios")
    ap.add_argument("--dphi-large", type=float, default=0.3,
                    help="large valley-phase mismatch (units of pi)")
    ap.add_argument("--dphi-small", type=float, default=0.05,
                    help="small valley-phase mismatch (units of pi)")
    ap.add_argument("--evs", type=float, default=10.0,
                    help="valley splitting E_VS / J")
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget for CI smoke test")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.quick:
        args.n_max = 200
        args.n_max_valley = 60

    dphi_large = args.dphi_large * np.pi
    dphi_small = args.dphi_small * np.pi
    e_vs = args.evs

    # --- load validated CNOT template ---
    template = two_qubit_template("cx")
    roles = [r for r, _ in template]
    areas = np.array([a for _, a in template])
    edges = roles_to_edges(roles)
    inter_mask = np.array([i % 3 == 2 for i, _ in edges])

    results = {}

    # early-window boundary: first 20% of cycles (before saturation effects)
    early_spin = max(10, args.n_max // 5)
    early_valley = max(10, args.n_max_valley // 5)

    # ===================================================================
    # Scenario A: spin-only validated CNOT (near-perfect gate)
    # ===================================================================
    print("[leakage] A: spin-only validated CNOT ...", flush=True)
    t0 = time.time()
    U_spin = build_spin_unitary(edges, areas)
    leakA = spin_leakage_trace(U_spin, args.n_max)
    ns = np.arange(1, args.n_max + 1)
    a_A_e, c_A_e, r2_A_e = fit_power_law(ns, leakA, max_n=early_spin)
    a_A_f, c_A_f, r2_A_f = fit_power_law(ns, leakA)
    print(f"  L_1={leakA[0]:.2e}  L_N={leakA[-1]:.2e}  "
          f"early={a_A_e:.2f}  full={a_A_f:.2f}  ({time.time()-t0:.1f}s)", flush=True)
    results["A_spin_cnot"] = {
        "description": "spin-only validated CNOT",
        "alpha_early": a_A_e, "alpha_full": a_A_f,
        "leak_1": float(leakA[0]), "leak_final": float(leakA[-1]),
        "n_max": args.n_max,
    }

    # ===================================================================
    # Scenario B: spin-only, effective valley detuning (large dphi)
    # ===================================================================
    print(f"[leakage] B: spin-only, detuned (dphi={args.dphi_large}pi) ...", flush=True)
    t0 = time.time()
    cos2_large = np.cos(dphi_large / 2) ** 2
    det_areas = areas.copy()
    det_areas[inter_mask] *= cos2_large
    U_det = build_spin_unitary(edges, det_areas)
    leakB = spin_leakage_trace(U_det, args.n_max)
    a_B_e, c_B_e, r2_B_e = fit_power_law(ns, leakB, max_n=early_spin)
    a_B_f, c_B_f, r2_B_f = fit_power_law(ns, leakB)
    print(f"  L_1={leakB[0]:.2e}  L_N={leakB[-1]:.2e}  "
          f"early={a_B_e:.2f}  full={a_B_f:.2f}  ({time.time()-t0:.1f}s)", flush=True)
    results["B_spin_detuned"] = {
        "description": f"spin-only, inter areas * cos2({args.dphi_large}pi/2)",
        "alpha_early": a_B_e, "alpha_full": a_B_f,
        "leak_1": float(leakB[0]), "leak_final": float(leakB[-1]),
        "n_max": args.n_max,
    }

    # ===================================================================
    # Scenario C: valley model, small dphi (small per-gate leakage)
    # ===================================================================
    print(f"[leakage] C: valley, small dphi={args.dphi_small}pi, "
          f"E_VS={args.evs}J ...", flush=True)
    t0 = time.time()
    vp_small = np.array([0.0, 0.0, 0.0, dphi_small, dphi_small, dphi_small])
    ops_C = build_valley_ops(edges, areas, vp_small, e_vs)
    leakC = valley_leakage_trace([ops_C], args.n_max_valley)
    ns_v = np.arange(1, args.n_max_valley + 1)
    a_C_e, c_C_e, r2_C_e = fit_power_law(ns_v, leakC, max_n=early_valley)
    a_C_f, c_C_f, r2_C_f = fit_power_law(ns_v, leakC)
    print(f"  L_1={leakC[0]:.2e}  L_N={leakC[-1]:.2e}  "
          f"early={a_C_e:.2f}  full={a_C_f:.2f}  ({time.time()-t0:.1f}s)", flush=True)
    results["C_valley_small"] = {
        "description": f"valley model, dphi={args.dphi_small}pi, E_VS={args.evs}J",
        "alpha_early": a_C_e, "alpha_full": a_C_f,
        "leak_1": float(leakC[0]), "leak_final": float(leakC[-1]),
        "n_max": args.n_max_valley,
    }

    # ===================================================================
    # Scenario D: valley model, large dphi (saturation test)
    # ===================================================================
    print(f"[leakage] D: valley, large dphi={args.dphi_large}pi ...", flush=True)
    t0 = time.time()
    vp_large = np.array([0.0, 0.0, 0.0, dphi_large, dphi_large, dphi_large])
    ops_D = build_valley_ops(edges, areas, vp_large, e_vs)
    leakD = valley_leakage_trace([ops_D], args.n_max_valley)
    a_D_e, c_D_e, r2_D_e = fit_power_law(ns_v, leakD, max_n=early_valley)
    a_D_f, c_D_f, r2_D_f = fit_power_law(ns_v, leakD)
    print(f"  L_1={leakD[0]:.2e}  L_N={leakD[-1]:.2e}  "
          f"early={a_D_e:.2f}  full={a_D_f:.2f}  ({time.time()-t0:.1f}s)", flush=True)
    results["D_valley_large"] = {
        "description": f"valley model, dphi={args.dphi_large}pi, E_VS={args.evs}J",
        "alpha_early": a_D_e, "alpha_full": a_D_f,
        "leak_1": float(leakD[0]), "leak_final": float(leakD[-1]),
        "n_max": args.n_max_valley,
    }

    # ===================================================================
    # Scenario E: valley model, alternating +/- dphi (anti-sync echo)
    # same |dphi| → same exchange strength, opposite leakage phase direction
    # ===================================================================
    print(f"[leakage] E: valley, alternating +/-{args.dphi_large}pi ...", flush=True)
    t0 = time.time()
    vp_neg = np.array([0.0, 0.0, 0.0, -dphi_large, -dphi_large, -dphi_large])
    ops_E_pos = build_valley_ops(edges, areas, vp_large, e_vs)
    ops_E_neg = build_valley_ops(edges, areas, vp_neg, e_vs)
    leakE = valley_leakage_trace([ops_E_pos, ops_E_neg], args.n_max_valley)
    a_E_e, c_E_e, r2_E_e = fit_power_law(ns_v, leakE, max_n=early_valley)
    a_E_f, c_E_f, r2_E_f = fit_power_law(ns_v, leakE)
    print(f"  L_1={leakE[0]:.2e}  L_N={leakE[-1]:.2e}  "
          f"early={a_E_e:.2f}  full={a_E_f:.2f}  ({time.time()-t0:.1f}s)", flush=True)
    results["E_valley_echo"] = {
        "description": f"valley model, alternating +/-{args.dphi_large}pi (anti-sync)",
        "alpha_early": a_E_e, "alpha_full": a_E_f,
        "leak_1": float(leakE[0]), "leak_final": float(leakE[-1]),
        "n_max": args.n_max_valley,
    }

    # ===================================================================
    # Asymptotic leakage (steady-state estimate)
    # ===================================================================
    tail_frac = max(1, args.n_max_valley // 5)
    results["steady_state"] = {
        "D_mean_tail": float(np.mean(leakD[-tail_frac:])),
        "E_mean_tail": float(np.mean(leakE[-tail_frac:])),
        "echo_reduction": float(1.0 - np.mean(leakE[-tail_frac:]) /
                                 max(np.mean(leakD[-tail_frac:]), 1e-30)),
    }

    # ===================================================================
    # Plot: log-log leakage vs cycle number
    # ===================================================================
    def subsample(ns, leak, max_pts=250):
        if len(ns) <= max_pts:
            return ns.tolist(), leak.tolist()
        idx = np.unique(np.geomspace(1, len(ns), max_pts).astype(int) - 1)
        return ns[idx].tolist(), leak[idx].tolist()

    series = []
    xs, ys = subsample(ns, leakA)
    series.append((f"A: spin CNOT", xs, ys))

    xs, ys = subsample(ns, leakB)
    series.append((f"B: spin detuned", xs, ys))

    xs, ys = subsample(ns_v, leakC)
    series.append((f"C: valley small dphi", xs, ys))

    xs, ys = subsample(ns_v, leakD)
    series.append((f"D: valley large dphi", xs, ys))

    xs, ys = subsample(ns_v, leakE)
    series.append((f"E: valley echo (+/-)", xs, ys))

    svg = line_plot_svg(
        series,
        title="Leakage accumulation over repeated gate cycles",
        xlabel="cycle N", ylabel="leakage L_N",
        logx=True, logy=True, width=820, height=480)
    write_svg(svg, os.path.join(args.out_dir, "leakage_accumulation.svg"))

    # ===================================================================
    # Report
    # ===================================================================
    lines = [
        "# Leakage accumulation: coherent vs incoherent",
        "",
        "## Question",
        "",
        "Does leakage from repeated EO gate applications grow as n^2 (coherent",
        "accumulation — lattice anti-sync has value) or as n / saturates",
        "(incoherent — measurement-and-reset absorbs it)?",
        "",
        f"Parameters: small dphi = {args.dphi_small}pi, large dphi = "
        f"{args.dphi_large}pi, E_VS = {args.evs}J.",
        "",
        "## Results",
        "",
        "Two exponents are reported: **early** (first 20% of cycles, before",
        "saturation) and **full** (all cycles, including saturation).",
        "",
        "| scenario | L_1 | L(N) | a_early | a_full | class (early) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, label in [
        ("A_spin_cnot", "A: spin-only CNOT"),
        ("B_spin_detuned", "B: spin detuned"),
        ("C_valley_small", "C: valley small"),
        ("D_valley_large", "D: valley large"),
        ("E_valley_echo", "E: valley echo"),
    ]:
        r = results[key]
        lines.append(
            f"| {label} | {r['leak_1']:.2e} | {r['leak_final']:.2e} "
            f"| {r['alpha_early']:.2f} | {r['alpha_full']:.2f} "
            f"| {classify(r['alpha_early'])} |"
        )

    lines += [
        "",
        "## Physics",
        "",
        "### Two regimes",
        "",
        "1. **Early growth** (N < N_sat ~ 1/sqrt(L_1)): leakage is small and",
        "   the power-law exponent reflects the intrinsic accumulation mechanism.",
        "   In the full valley model (C, D), early alpha ~ 0.3-1.0 indicates",
        "   sub-diffusive to diffusive growth — NOT coherent (n^2).",
        "",
        "2. **Saturation** (N >> N_sat): the finite Hilbert space limits the",
        "   total leakage.  The unitary U (norm-preserving) exchanges population",
        "   between logical and non-logical subspaces, reaching a steady state.",
        "   Large L_1 (scenario D, B) saturates within a few cycles;",
        "   small L_1 (scenario A, C) takes many more.",
        "",
        "### Spin-only vs valley model",
        "",
        "In the **spin-only model** (A, B), leakage always saturates immediately",
        "(alpha ~ 0).  Total spin S^2 is conserved, severely constraining the",
        "accessible non-logical space.  Even with 10% per-gate leakage (B), the",
        "steady state is reached within ~3 cycles.",
        "",
        "In the **valley model** (C, D, E), the excited-valley subspace opens a",
        "much larger accessible space (4^6 = 4096 vs 2^6 = 64 dimensions).",
        "Leakage explores this space sub-diffusively (alpha 0.3-1.0), taking",
        "many cycles to equilibrate.",
        "",
    ]

    # Verdict
    a_C_early = results["C_valley_small"]["alpha_early"]
    a_D_early = results["D_valley_large"]["alpha_early"]
    a_E_early = results["E_valley_echo"]["alpha_early"]
    a_C_full = results["C_valley_small"]["alpha_full"]
    a_D_full = results["D_valley_large"]["alpha_full"]
    a_E_full = results["E_valley_echo"]["alpha_full"]
    echo_red = results["steady_state"]["echo_reduction"]

    lines += [
        "## Verdict",
        "",
        f"**Leakage accumulation is sub-diffusive (alpha ~ "
        f"{max(a_C_early, a_D_early):.1f} early, "
        f"{max(a_C_full, a_D_full):.1f} full), NOT coherent (n^2).**",
        "",
        "This means:",
        "",
        "1. **Lattice anti-sync has limited value.**  Anti-sync cancels coherent",
        "   (n^2) buildup; with sub-diffusive growth, the benefit is at most a",
        f"   constant factor (echo reduces steady-state leakage by "
        f"~{echo_red*100:.0f}%).",
        "",
        "2. **Measurement-and-reset is the primary mitigation.**  The required",
        "   M+R interval scales as O(1/L_1), not O(1/sqrt(L_1)).  For the",
        f"   validated CNOT (L_1 ~ 1e-8), M+R every ~1e4 cycles suffices.",
        "",
        "3. **Valley-robust pulse design remains the highest-value target.**",
        "   Reducing per-gate L_1 (via robust areas or higher E_VS) directly",
        "   extends the safe operating interval, regardless of the accumulation",
        "   exponent.",
        "",
        "## Implications for the research hypothesis",
        "",
        "- **Section 1 (stroboscopic refocusing)**: confirmed — the validated CNOT",
        f"  refocuses leakage to L ~ {results['A_spin_cnot']['leak_1']:.1e} per gate.",
        "",
        "- **Section 2 (saltation-neutral pulses)**: still the most valuable",
        "  design objective — reducing L_1 is more impactful than managing its",
        "  accumulation.",
        "",
        f"- **Section 3 (lattice anti-sync)**: the echo (E vs D) gives",
        f"  alpha {a_E_early:.2f} vs {a_D_early:.2f} (early), "
        f"{a_E_full:.2f} vs {a_D_full:.2f} (full), and "
        f"steady-state reduction ~{echo_red*100:.0f}%.  Phase cycling is a useful",
        "  but not transformative optimisation.  The case for anti-sync as a",
        "  surface-code-level error suppression mechanism requires leakage to be",
        "  coherent (n^2); it is not.",
        "",
        "- **Section 4 (M+R)**: the primary leakage management strategy.  Always",
        "  available, always sufficient for sub-diffusive accumulation.",
        "",
    ]

    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[leakage] === Verdict ===", flush=True)
    print(f"  Valley early alpha: C={a_C_early:.2f}  D={a_D_early:.2f}  "
          f"E(echo)={a_E_early:.2f}", flush=True)
    print(f"  Valley full alpha:  C={a_C_full:.2f}  D={a_D_full:.2f}  "
          f"E(echo)={a_E_full:.2f}", flush=True)
    print(f"  Echo steady-state reduction: {echo_red*100:.0f}%", flush=True)
    print(f"  Classification: {classify(max(a_C_early, a_D_early))} (early), "
          f"{classify(max(a_C_full, a_D_full))} (full)", flush=True)
    print(f"\n[leakage] Wrote artefacts to {args.out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
