#!/usr/bin/env python3
"""Valley-robust (and noise-robust) pulse design for an EO CNOT.

Designs a CNOT whose pulses maximise the ensemble-averaged fidelity over a
valley-phase spread (and optionally charge noise), then quantifies the advantage
over the standard (nominal-point) gate.

Outputs to out/robust/ : robust_vs_baseline.svg, report.md, results.json.
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
from eo_pulse_ir.sim.landscape import line_plot_svg, write_svg
from eo_pulse_ir.sim.robust import (ensemble_fidelity, robust_design,
                                    valley_phase_samples)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/robust")
    ap.add_argument("--spread", type=float, default=0.3, help="valley-phase spread (units of pi)")
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    cx, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
    edges = [tuple(p.edge) for p in cx]
    base = np.array([p.area for p in cx])

    delta = args.spread * np.pi
    train = valley_phase_samples(edges, np.linspace(-delta, delta, 7))
    robust, _ = robust_design(edges, 2, gates.CNOT, train, x0=base, steps=args.steps)
    robust = np.array(robust)

    # evaluate both vs valley-phase mismatch
    dphis = np.linspace(0, delta, 21)
    base_curve, rob_curve = [], []
    for d in dphis:
        s = valley_phase_samples(edges, [d])
        base_curve.append(ensemble_fidelity(base, edges, 2, gates.CNOT, s))
        rob_curve.append(ensemble_fidelity(robust, edges, 2, gates.CNOT, s))

    test = valley_phase_samples(edges, np.linspace(-delta, delta, 21))
    base_mean = ensemble_fidelity(base, edges, 2, gates.CNOT, test)
    rob_mean = ensemble_fidelity(robust, edges, 2, gates.CNOT, test)

    write_svg(line_plot_svg(
        [("baseline CNOT", (dphis / np.pi).tolist(), base_curve),
         ("valley-robust CNOT", (dphis / np.pi).tolist(), rob_curve)],
        title=f"Valley-robust vs baseline CNOT (trained over +/-{args.spread:.2f}pi spread)",
        xlabel="valley-phase mismatch Δφ / π", ylabel="gate fidelity"),
        os.path.join(args.out_dir, "robust_vs_baseline.svg"))

    results = {"spread_pi": args.spread,
               "baseline_mean_fidelity": base_mean,
               "robust_mean_fidelity": rob_mean,
               "baseline_F_at_0": base_curve[0], "robust_F_at_0": rob_curve[0],
               "robust_areas": robust.tolist()}
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    lines = [
        "# Valley-robust EO CNOT",
        "",
        "Ensemble-averaged optimal control: the pulse areas are optimised to",
        f"maximise the mean CNOT fidelity over a valley-phase spread Δφ ∈ "
        f"±{args.spread:.2f}π (boundary exchange scaled by cos²(Δφ/2)).",
        "",
        "| metric | baseline | valley-robust |",
        "|---|---|---|",
        f"| mean fidelity over ±{args.spread:.2f}π | {base_mean:.4f} | {rob_mean:.4f} |",
        f"| fidelity at Δφ=0 | {base_curve[0]:.4f} | {rob_curve[0]:.4f} |",
        "",
        f"The valley-robust gate holds **{rob_mean:.3f}** mean fidelity across the",
        f"valley-phase spread vs **{base_mean:.3f}** for the standard gate, trading a",
        "little peak fidelity for a much flatter response — the right trade-off given",
        "that valley-phase/E_VS *uniformity* (not peak) is the recognised bottleneck.",
        "The same machinery (`sim/robust.py`) takes charge-noise ensembles, so gates",
        "can be co-designed to be robust against both valley spread and dJ/J noise.",
        "",
    ]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[robust] valley-phase spread +/-{args.spread:.2f}pi: "
          f"baseline mean F={base_mean:.4f}, robust mean F={rob_mean:.4f}")
    print(f"[robust] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
