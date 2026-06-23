#!/usr/bin/env python3
"""Monte-Carlo robustness of the validated EO gates under area (charge) noise.

For each gate it synthesises the IR pulse sequence, then samples the gate
fidelity while perturbing pulse areas (multiplicative dJ/J noise, quasi-static
per edge by default).  Outputs to ``out/noise/``:

  - robustness_curve.svg   mean infidelity vs noise strength sigma (log-log)
  - hist_<gate>.svg        fidelity distribution at a representative sigma
  - correlation_<gate>.svg per-edge vs global vs independent noise comparison
  - noise_report.md        susceptibility coefficients + worst-case table
  - results.json           raw sweep data

Run:  python scripts/eo_noise_mc.py --samples 400 --rep-sigma 0.01
Requires numpy.
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
from eo_pulse_ir.sim.landscape import histogram_svg, line_plot_svg, write_svg
from eo_pulse_ir.sim.noise import (montecarlo_fidelity, robustness_sweep,
                                   susceptibility)

# (gate label, circuit, num_qubits, target)
GATES = [
    ("H", "qubits 1\nh 0\n", 1, gates.H),
    ("CNOT", "qubits 2\ncx 0 1\n", 2, gates.CNOT),
    ("SWAP", "qubits 2\nswap 0 1\n", 2, gates.SWAP),
    ("CXSWAP", "qubits 2\ncxswap 0 1\n", 2, gates.CXSWAP),
]


def gate_pulses(circuit_text):
    pulses, _ = synthesize(parse_circuit(circuit_text))
    edges = [tuple(p.edge) for p in pulses]
    areas = [p.area for p in pulses]
    return edges, areas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/noise")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--rep-sigma", type=float, default=0.01,
                    help="representative noise strength for histograms")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    sigmas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
    results = {}
    curve_series = []

    for label, circ, nq, target in GATES:
        edges, areas = gate_pulses(circ)
        sweep = robustness_sweep(edges, areas, nq, target, sigmas,
                                 n_samples=args.samples, relative=True,
                                 correlation="per_edge", seed=0)
        c = susceptibility(sweep)
        results[label] = {"num_pulses": len(edges), "susceptibility": c,
                          "sweep": sweep}
        # robustness curve (skip sigma=0 for log axis)
        xs = [r["sigma"] for r in sweep if r["sigma"] > 0]
        ys = [max(r["mean_infidelity"], 1e-12) for r in sweep if r["sigma"] > 0]
        curve_series.append((f"{label} (N={len(edges)})", xs, ys))
        print(f"[noise] {label}: N={len(edges)} susceptibility c={c:.4g}  "
              f"(infid≈c·σ²)", flush=True)

        # histogram at representative sigma
        rep = montecarlo_fidelity(edges, areas, nq, target, args.rep_sigma,
                                  n_samples=max(args.samples, 600),
                                  relative=True, correlation="per_edge", seed=7)
        write_svg(histogram_svg(rep["samples"], title=
                  f"{label} fidelity under {args.rep_sigma:.0%} per-edge area noise",
                  xlabel="gate fidelity"),
                  os.path.join(args.out_dir, f"hist_{label.lower()}.svg"))
        results[label]["hist_rep"] = {k: rep[k] for k in
            ("sigma", "mean_fidelity", "std_fidelity", "p10", "p90", "worst_fidelity")}

    # robustness curve (all gates)
    write_svg(line_plot_svg(curve_series,
              title="EO gate robustness: mean infidelity vs area noise σ",
              xlabel="relative area noise σ (dJ/J)", ylabel="mean infidelity",
              logx=True, logy=True),
              os.path.join(args.out_dir, "robustness_curve.svg"))

    # correlation-model comparison for the worst gate (CXSWAP)
    label, circ, nq, target = GATES[-1]
    edges, areas = gate_pulses(circ)
    corr_series = []
    for corr in ("per_edge", "global", "independent"):
        sweep = robustness_sweep(edges, areas, nq, target, sigmas,
                                 n_samples=args.samples, relative=True,
                                 correlation=corr, seed=0)
        xs = [r["sigma"] for r in sweep if r["sigma"] > 0]
        ys = [max(r["mean_infidelity"], 1e-12) for r in sweep if r["sigma"] > 0]
        corr_series.append((corr, xs, ys))
    write_svg(line_plot_svg(corr_series,
              title=f"{label}: noise-correlation model comparison",
              xlabel="relative area noise σ", ylabel="mean infidelity",
              logx=True, logy=True),
              os.path.join(args.out_dir, f"correlation_{label.lower()}.svg"))

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # report
    lines = [
        "# EO gate noise robustness (Monte-Carlo)",
        "",
        "Multiplicative area noise A → A·(1+ε), ε ~ N(0, σ), quasi-static per edge "
        "(realistic charge-noise model). Mean over "
        f"{args.samples} shots per σ.",
        "",
        "## Noise susceptibility  (mean infidelity ≈ c · σ²)",
        "",
        "| gate | pulses | susceptibility c | mean F @ σ=1% | 10th-pctile F @ 1% | worst F @ 1% |",
        "|---|---|---|---|---|---|",
    ]
    for label, _, _, _ in GATES:
        r = results[label]
        h = r["hist_rep"]
        lines.append(f"| {label} | {r['num_pulses']} | {r['susceptibility']:.4g} | "
                     f"{h['mean_fidelity']:.5f} | {h['p10']:.5f} | {h['worst_fidelity']:.5f} |")
    lines += [
        "",
        "Lower `c` = flatter response to charge noise. Infidelity grows ~quadratically "
        "with σ (the small-noise regime), consistent with a stationary optimum.",
        "",
        "See `robustness_curve.svg`, `hist_<gate>.svg`, and "
        "`correlation_cxswap.svg` (per-edge vs global vs independent noise).",
        "",
    ]
    with open(os.path.join(args.out_dir, "noise_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[noise] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
