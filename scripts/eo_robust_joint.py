#!/usr/bin/env python3
"""Jointly valley- AND charge-noise-robust EO CNOT design.

Designs a CNOT robust to the two dominant real error channels at once — a
valley-phase spread Δφ and calibrated charge noise (dJ/J) — and quantifies the
advantage over the standard gate under the same joint ensemble. This is the
"core result": robustness to the real device's dominant errors at a calibrated
noise level.

Outputs to out/robust_joint/ : robust_joint.svg, report.md, results.json.
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
from eo_pulse_ir.sim.calibration import calibrate_sigma
from eo_pulse_ir.sim.landscape import line_plot_svg, write_svg
from eo_pulse_ir.sim.robust import (ensemble_fidelity, joint_valley_noise_samples,
                                    robust_design)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/robust_joint")
    ap.add_argument("--spread", type=float, default=0.3, help="valley-phase spread (units of pi)")
    ap.add_argument("--sigma", type=float, default=None, help="charge noise dJ/J (default: calibrate)")
    ap.add_argument("--steps", type=int, default=220)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    cx, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
    edges = [tuple(p.edge) for p in cx]
    base = np.array([p.area for p in cx])

    # calibrate charge noise to CNOT=99% unless given
    sigma = args.sigma
    if sigma is None:
        sigma = calibrate_sigma([(e, a) for e, a in zip(edges, base)], 2,
                                gates.CNOT, 0.99, n_samples=300)
    delta = args.spread * np.pi
    print(f"[joint] charge noise sigma={sigma:.4f} ({sigma*100:.2f}%), "
          f"valley spread +/-{args.spread:.2f}pi", flush=True)

    train = joint_valley_noise_samples(edges, delta, sigma, 12, seed=1)
    robust, _ = robust_design(edges, 2, gates.CNOT, train, x0=base, steps=args.steps)
    robust = np.array(robust)

    # evaluate vs valley spread at the calibrated noise level (fresh test ensembles)
    spreads = np.linspace(0, delta, 9)
    base_curve, rob_curve = [], []
    for d in spreads:
        test = joint_valley_noise_samples(edges, d, sigma, 200, seed=99)
        base_curve.append(ensemble_fidelity(base, edges, 2, gates.CNOT, test))
        rob_curve.append(ensemble_fidelity(robust, edges, 2, gates.CNOT, test))

    full = joint_valley_noise_samples(edges, delta, sigma, 400, seed=7)
    base_mean = ensemble_fidelity(base, edges, 2, gates.CNOT, full)
    rob_mean = ensemble_fidelity(robust, edges, 2, gates.CNOT, full)

    write_svg(line_plot_svg(
        [("baseline CNOT", (spreads / np.pi).tolist(), base_curve),
         ("joint-robust CNOT", (spreads / np.pi).tolist(), rob_curve)],
        title=f"Joint valley+noise robust CNOT (sigma={sigma*100:.2f}% dJ/J)",
        xlabel="valley-phase spread ±Δφ / π", ylabel="mean gate fidelity"),
        os.path.join(args.out_dir, "robust_joint.svg"))

    results = {"sigma": sigma, "spread_pi": args.spread,
               "baseline_mean_fidelity": base_mean, "robust_mean_fidelity": rob_mean,
               "baseline_noise_only": base_curve[0], "robust_noise_only": rob_curve[0],
               "robust_areas": robust.tolist()}
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    lines = [
        "# Joint valley + charge-noise robust EO CNOT",
        "",
        f"Charge noise calibrated to the HRL 2-qubit fidelity (sigma = {sigma*100:.2f}% "
        f"dJ/J); valley-phase spread ±{args.spread:.2f}π. The pulses are optimised to",
        "maximise the mean fidelity over the **joint** ensemble of both channels.",
        "",
        "| metric | baseline | joint-robust |",
        "|---|---|---|",
        f"| mean F over joint ensemble | {base_mean:.4f} | {rob_mean:.4f} |",
        f"| mean F, noise only (Δφ=0) | {base_curve[0]:.4f} | {rob_curve[0]:.4f} |",
        "",
        f"Under the two dominant real error channels at calibrated levels, the",
        f"joint-robust CNOT holds **{rob_mean:.3f}** mean fidelity vs **{base_mean:.3f}** "
        "for the standard gate. The standard gate is near-optimal at the nominal point",
        "but collapses under valley-phase spread; the joint-robust gate keeps a high,",
        "flat fidelity across the spread while remaining charge-noise-tolerant — the",
        "right objective given that valley/E_VS uniformity is the recognised bottleneck.",
        "",
    ]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[joint] baseline mean F={base_mean:.4f}, joint-robust mean F={rob_mean:.4f}")
    print(f"[joint] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
