#!/usr/bin/env python3
"""Gate adiabaticity vs valley leakage vs integrated power for a single-spin CZ.

Closes the loop on the session's two senses of "adiabatic":

  1. **Control-electronics adiabaticity** (eo_qsoc_budget.py): slow, energy-
     recovering switching cuts cold *power*.
  2. **Qubit-gate adiabaticity** (here): a slowly-ramped exchange pulse stays in
     the instantaneous ground-valley eigenbasis, suppressing diabatic leakage to
     the excited valley.

Both favour "not too fast", and silicon spin qubits gate slowly enough to afford
it.  This script quantifies sense (2) in the spin(x)valley model: a single-spin
two-qubit CZ is a finite-area exchange pulse on two 4-level (spin(x)valley) dots
(eo_pulse_ir.sim.valley.two_dot_pulse), shaped over a ramp time ``T``.  We:

  - sweep ``T`` and measure end-of-gate valley leakage for a smooth (sin^2) ramp
    vs a hard square pulse of the same duration -> adiabatic suppression;
  - add a decoherence error ~ T/T2 (slow gates dephase) to get a U-shaped total
    error with an **optimal ramp time**;
  - read off the leakage-limited reset cadence and tie it back to the cold-power
    budget (faster -> more leakage -> more measure+reset overhead; slower -> more
    dephasing and longer cycle).  The optimum is "as adiabatic as coherence
    allows", which is exactly the regime where energy-recovery control also wins.

Outputs to out/gate_adiabaticity/ : leakage_vs_ramp.svg, total_error.svg,
report.md, results.json.  Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir.sim.valley import two_dot_pulse


def _embed_two_singlespin() -> np.ndarray:
    """Isometry (16 x 4): two single-spin qubits -> spin(x)valley, both valleys ground.

    Each dot is 4-level with index = 2*spin + valley (valley 0 = ground).  The
    logical state |s0 s1> maps to ground-valley on both dots.
    """
    E = np.zeros((16, 4), dtype=complex)
    for s0 in (0, 1):
        for s1 in (0, 1):
            f = (2 * s0 + 0) * 4 + (2 * s1 + 0)   # both valleys ground
            col = s0 * 2 + s1
            E[f, col] = 1.0
    return E


def ramped_cz_leakage(area: float, T: float, steps: int, dphi: float, evs: float,
                      profile: str = "sin2") -> float:
    """End-of-gate valley leakage for an exchange pulse of fixed ``area`` over time ``T``.

    ``profile`` = "sin2" (smooth/adiabatic envelope) or "square" (hard pulse).
    ``dphi`` = valley-phase mismatch (the leakage channel), ``evs`` = valley
    splitting (the adiabatic gap).
    """
    E = _embed_two_singlespin()
    psi = E.copy()
    dt = T / steps
    if profile == "sin2":
        j_pk = 2.0 * area / T                      # area = integral of j_pk*sin^2 over [0,T]
    else:
        j_pk = area / T
    for k in range(steps):
        t = (k + 0.5) * dt
        j = j_pk * np.sin(np.pi * t / T) ** 2 if profile == "sin2" else j_pk
        if j <= 1e-12:
            continue
        op = two_dot_pulse(j * dt, dphi, evs, evs, j_max=j)
        psi = op @ psi
    M = E.conj().T @ psi
    return float(1.0 - np.real(np.trace(M.conj().T @ M)) / 4.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out-dir", default="out/gate_adiabaticity")
    ap.add_argument("--area", type=float, default=np.pi,
                    help="entangling exchange area of the CZ (rad)")
    ap.add_argument("--dphi", type=float, default=0.2, help="valley-phase mismatch (units of pi)")
    ap.add_argument("--evs", type=float, default=5.0, help="valley splitting E_VS (units of j scale)")
    ap.add_argument("--t2", type=float, default=2000.0, help="dephasing time (units of 1/j_max)")
    ap.add_argument("--steps", type=int, default=200, help="ramp discretisation")
    ap.add_argument("--n-ramps", type=int, default=40, help="ramp-time sweep points")
    ap.add_argument("--quick", action="store_true", help="tiny budget for CI smoke")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.quick:
        args.steps = 40
        args.n_ramps = 12

    dphi = args.dphi * np.pi
    evs = args.evs

    # ramp-time sweep (gate duration in units of 1/j_max)
    ramps = np.geomspace(0.5, 200.0, args.n_ramps)
    leak_sin2 = [ramped_cz_leakage(args.area, T, args.steps, dphi, evs, "sin2") for T in ramps]
    leak_sq = [ramped_cz_leakage(args.area, T, args.steps, dphi, evs, "square") for T in ramps]

    # total error model: diabatic leakage (down with slow ramp) + dephasing (up with slow ramp)
    dephase = [T / args.t2 for T in ramps]
    total_sin2 = [l + d for l, d in zip(leak_sin2, dephase)]
    i_opt = int(np.argmin(total_sin2))
    T_opt, err_opt, leak_opt = ramps[i_opt], total_sin2[i_opt], leak_sin2[i_opt]

    # adiabatic suppression factor: how much the smooth ramp beats the square pulse
    supp = [(sq / sn) if sn > 0 else float("inf") for sq, sn in zip(leak_sq, leak_sin2)]
    supp_mid = supp[len(supp) // 2]
    supp_max = max(s for s in supp if s != float("inf"))

    print(f"[adiab] dphi={args.dphi}pi evs={args.evs} T2={args.t2}", flush=True)
    print(f"[adiab] leakage: square {leak_sq[0]:.2e}->{leak_sq[-1]:.2e}, "
          f"sin2 {leak_sin2[0]:.2e}->{leak_sin2[-1]:.2e} over T={ramps[0]:.1f}..{ramps[-1]:.0f}",
          flush=True)
    print(f"[adiab] optimal ramp T*={T_opt:.1f} (1/j): total err {err_opt:.2e} "
          f"(leak {leak_opt:.2e} + dephase {dephase[i_opt]:.2e})", flush=True)

    # --- plots ---
    try:
        from eo_pulse_ir.sim.landscape import line_plot_svg, write_svg
        rr = ramps.tolist()
        write_svg(line_plot_svg(
            [("square pulse (diabatic)", rr, leak_sq),
             ("sin^2 ramp (adiabatic)", rr, leak_sin2)],
            title=f"Valley leakage vs gate ramp time (dphi={args.dphi}pi, E_VS={args.evs})",
            xlabel="gate duration T (1/j_max)", ylabel="valley leakage",
            logx=True, logy=True), os.path.join(args.out_dir, "leakage_vs_ramp.svg"))
        write_svg(line_plot_svg(
            [("diabatic leakage (sin^2)", rr, leak_sin2),
             ("dephasing ~ T/T2", rr, dephase),
             ("total error", rr, total_sin2)],
            title=f"Gate-time tradeoff: optimum at T*={T_opt:.1f}/j_max",
            xlabel="gate duration T (1/j_max)", ylabel="error",
            logx=True, logy=True), os.path.join(args.out_dir, "total_error.svg"))
    except Exception:
        pass

    # --- results.json ---
    results = {
        "params": {"area": args.area, "dphi_pi": args.dphi, "evs": args.evs,
                   "t2": args.t2, "steps": args.steps},
        "ramps": ramps.tolist(),
        "leakage_square": leak_sq, "leakage_sin2": leak_sin2,
        "dephasing": dephase, "total_error_sin2": total_sin2,
        "optimum": {"T_ramp": float(T_opt), "total_error": float(err_opt),
                    "leakage": float(leak_opt), "dephasing": float(dephase[i_opt])},
        "adiabatic_suppression_mid": float(supp_mid),
    }
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # --- report.md ---
    lines = [
        "# Gate adiabaticity vs valley leakage vs integrated power",
        "",
        "Closes the loop on the two senses of *adiabatic* in this work: slow,",
        "energy-recovering **control electronics** cut cold power",
        "(`eo_qsoc_budget.py`), and a slowly-ramped **qubit gate** suppresses",
        "diabatic valley leakage (here).  Both favour \"not too fast\", and silicon",
        "spin qubits gate slowly enough to afford both.",
        "",
        "## Model",
        "",
        "A single-spin two-qubit CZ is a finite-area exchange pulse on two",
        "4-level (spin⊗valley) dots (`sim.valley.two_dot_pulse`), shaped over a",
        f"ramp time `T`.  Valley-phase mismatch dphi = {args.dphi}pi opens the",
        f"leakage channel; valley splitting E_VS = {args.evs} (in units of the",
        "exchange scale) is the adiabatic gap.  Leakage = excited-valley population",
        "at the end of the gate.",
        "",
        "## Result 1 — adiabatic suppression",
        "",
        "| gate duration T (1/j_max) | square pulse | sin^2 ramp |",
        "|---:|---:|---:|",
    ]
    for idx in (0, len(ramps) // 3, 2 * len(ramps) // 3, len(ramps) - 1):
        lines.append(f"| {ramps[idx]:.1f} | {leak_sq[idx]:.2e} | {leak_sin2[idx]:.2e} |")
    lines += [
        "",
        "Leakage falls as the gate slows (lower peak J vs E_VS, slower change =",
        "more adiabatic).  A smooth sin^2 ramp beats a hard square pulse of the",
        f"same duration by a factor that grows with T — ~{supp_mid:.0f}x mid-sweep,",
        f"up to ~{supp_max:.0e}x at the slow end — so diabatic turn-on/off is a",
        "large, avoidable leakage source.",
        "",
        "## Result 2 — the optimal gate time",
        "",
        "Slower is not free: a gate of duration T dephases ~ T/T2.  Total error =",
        "diabatic leakage (down with T) + dephasing (up with T) is U-shaped:",
        "",
        f"- **optimal ramp T\\* = {T_opt:.1f}/j_max** (with T2 = {args.t2}/j_max)",
        f"- total error at optimum = **{err_opt:.2e}** "
        f"(leakage {leak_opt:.2e} + dephasing {dephase[i_opt]:.2e})",
        "",
        "The optimum is \"as adiabatic as coherence allows\".",
        "",
        "## Tie-back to the cold-power budget",
        "",
        "- **Faster than T\\*** → more valley leakage → more frequent measure+reset",
        "  → more cold switching activity and reset overhead.",
        "- **Slower than T\\*** → more dephasing *and* longer cycles → more",
        "  integrated static cold power per logical operation.",
        "- The leakage-limited reset cadence (~1/leakage gates between resets) sets",
        "  a floor on cold reset activity; pushing the gate to T\\* minimises it.",
        "- The *same* T\\* regime (slow-ish, smooth pulses) is where energy-recovery",
        "  control electronics (`eo_qsoc_budget.py`, adiabatic section) also win —",
        "  so gate adiabaticity and control-electronics adiabaticity point the same",
        "  way.  Silicon spin qubits' slow gates make both reachable at once.",
        "",
        "## Scope",
        "",
        "Effective spin⊗valley model (cos^2(dphi/2) exchange suppression + 4-level",
        "leakage space), two dots.  Times are in units of 1/j_max and T2 is a knob;",
        "the deliverable is the **scaling** (adiabatic suppression and the U-shaped",
        "optimum), not absolute nanoseconds.  Feed measured E_VS, dphi spread, and",
        "T2 to specialise.",
        "",
    ]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[adiab] wrote artefacts to {args.out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
