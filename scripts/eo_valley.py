#!/usr/bin/env python3
"""Valley physics in the EO simulator: leakage, E_VS >> J, and gate impact.

Uses the spin (x) valley 4-level extension (sim/valley.py) to reproduce the
silicon-specific constraints from docs/valley_splitting_research.md:

  (A) valley leakage of a boundary exchange vs valley splitting E_VS/J — the
      E_VS >> J requirement (leakage suppressed as E_VS grows);
  (B) leakage vs valley-phase mismatch Δφ, with and without large E_VS;
  (C) impact on the validated CNOT: fidelity / valley leakage vs E_VS at a fixed
      valley-phase mismatch — large E_VS fixes leakage but the phase still
      detunes the gate (you need alignment or re-calibration too).

Outputs to out/valley/ : valley_evs.svg, valley_phase.svg, cnot_vs_evs.svg,
report.md, results.json. Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir import parse_circuit, synthesize
from eo_pulse_ir.sim import gates
from eo_pulse_ir.sim.fidelity import leakage
from eo_pulse_ir.sim.landscape import line_plot_svg, write_svg
from eo_pulse_ir.sim.valley import simulate_valley, two_dot_pulse


def _boundary_leakage(area, dphi, evs):
    """Valley leakage of one boundary exchange pulse on two spin(x)valley dots."""
    U = two_dot_pulse(area, dphi, evs, evs, j_max=1.0)
    # logical = spin states with both valleys ground (valley index 0)
    L = np.zeros((16, 4))
    for k, (sa, sb) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        L[(2 * sa) * 4 + (2 * sb), k] = 1.0
    M = L.conj().T @ U @ L
    return leakage(M)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/valley")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    # (A) leakage vs E_VS/J for a sqrt-SWAP boundary exchange at fixed mismatch
    evs = np.linspace(0, 12, 25)
    for dphi, tag in [(np.pi / 2, "dphi=pi/2"), (np.pi / 4, "dphi=pi/4")]:
        results.setdefault("A_evs", {})[tag] = \
            [_boundary_leakage(np.pi / 2, dphi, e) for e in evs]
    write_svg(line_plot_svg(
        [(tag, evs.tolist(), results["A_evs"][tag]) for tag in results["A_evs"]],
        title="(A) Valley leakage of a boundary exchange vs valley splitting",
        xlabel="valley splitting E_VS / J", ylabel="valley leakage"),
        os.path.join(args.out_dir, "valley_evs.svg"))
    print(f"[valley] (A) leakage(dphi=pi/2): E_VS=0 -> {results['A_evs']['dphi=pi/2'][0]:.3f},"
          f" E_VS=12 -> {results['A_evs']['dphi=pi/2'][-1]:.3f}", flush=True)

    # (B) leakage vs valley-phase mismatch, with/without large E_VS
    dphis = np.linspace(0, np.pi, 25)
    results["B_phase"] = {
        "E_VS=0": [_boundary_leakage(np.pi / 2, d, 0.0) for d in dphis],
        "E_VS=10": [_boundary_leakage(np.pi / 2, d, 10.0) for d in dphis],
    }
    write_svg(line_plot_svg(
        [(k, (dphis / np.pi).tolist(), v) for k, v in results["B_phase"].items()],
        title="(B) Valley leakage vs valley-phase mismatch",
        xlabel="valley-phase mismatch Δφ / π", ylabel="valley leakage"),
        os.path.join(args.out_dir, "valley_phase.svg"))
    print(f"[valley] (B) leakage at Δφ=π: E_VS=0 -> {results['B_phase']['E_VS=0'][-1]:.3f},"
          f" E_VS=10 -> {results['B_phase']['E_VS=10'][-1]:.3f}", flush=True)

    # (C) validated CNOT (full 4-level model) vs E_VS at a fixed phase mismatch
    p, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
    phase = np.zeros(6); phase[3:] = np.pi / 6           # qubit B valley-rotated
    evs2 = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    fids, leaks = [], []
    for e in evs2:
        r = simulate_valley(p, 2, valley_phase=phase, e_vs=e, target=gates.CNOT)
        fids.append(r["fidelity"]); leaks.append(r["valley_leakage"])
    # reference: aligned valleys
    r0 = simulate_valley(p, 2, valley_phase=0.0, e_vs=5.0, target=gates.CNOT)
    results["C_cnot"] = {"e_vs": evs2, "fidelity": fids, "valley_leakage": leaks,
                         "aligned_fidelity": r0["fidelity"]}
    write_svg(line_plot_svg(
        [("CNOT fidelity (Δφ=π/6)", evs2, fids),
         ("valley leakage", evs2, leaks)],
        title="(C) Validated CNOT vs valley splitting at fixed phase mismatch",
        xlabel="valley splitting E_VS / J", ylabel="fidelity / leakage"),
        os.path.join(args.out_dir, "cnot_vs_evs.svg"))
    print(f"[valley] (C) CNOT Δφ=π/6: leakage {leaks[0]:.3f}->{leaks[-1]:.3f}, "
          f"fidelity {fids[0]:.3f}->{fids[-1]:.3f}; aligned F={r0['fidelity']:.6f}", flush=True)

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    a = results["A_evs"]["dphi=pi/2"]
    lines = [
        "# Valley physics in the EO simulator (spin (x) valley)",
        "",
        "The simulator now carries a per-dot valley pseudo-spin (4-level dots).",
        "Consistency: with aligned valley phases it reproduces the spin-only model",
        f"exactly (validated CNOT F = {r0['fidelity']:.8f}).",
        "",
        "## (A) E_VS >> J suppresses valley leakage",
        f"A boundary exchange with valley-phase mismatch Δφ=π/2 leaks "
        f"{a[0]:.3f} at E_VS=0, falling to {a[-1]:.3f} at E_VS/J=12 — the literature's",
        "E_VS >> J requirement, emergent from the model.",
        "",
        "## (B) Leakage grows with valley-phase mismatch",
        f"At E_VS=0 leakage rises with Δφ (to {results['B_phase']['E_VS=0'][-1]:.3f} at Δφ=π);",
        f"at E_VS/J=10 it is suppressed (to {results['B_phase']['E_VS=10'][-1]:.3f}).",
        "",
        "## (C) Validated CNOT under valley-phase mismatch",
        "| E_VS/J | CNOT fidelity | valley leakage |",
        "|---|---|---|",
    ]
    for e, f, l in zip(evs2, fids, leaks):
        lines.append(f"| {e} | {f:.4f} | {l:.4f} |")
    lines += [
        "",
        "Key insight: large E_VS suppresses the **leakage**, but a valley-phase",
        "mismatch also **detunes** the exchange, so the CNOT fidelity stays low until",
        "the valley phase is aligned (or the pulses re-calibrated for it). Both large",
        "and uniform E_VS *and* valley-phase control are needed — matching the",
        "research consensus that uniformity, not peak E_VS, is the bottleneck.",
        "",
    ]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[valley] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
