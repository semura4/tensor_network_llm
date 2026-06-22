#!/usr/bin/env python3
"""Optimal pulse design for n-qubit exchange-only gates (control synthesis).

The pulse-control-engineering view: an n-qubit logical target is a boundary
condition for the switched bilinear EO control system, and we *design* the
control word (exchange-pulse areas) that steers the propagator onto it, with a
layered ansatz and the exact analytic-gradient optimiser (GRAPE-style).

By default this runs the fast, exact single- and two-qubit designs plus the
"3-for-1 => quaternion" algebra check.  Pass --ccz-layers L to also attempt a
3-qubit CCZ (slow); if out/.. has a cached design it is reported.

Outputs to out/synthesis/ : report.md, results.json.  Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir.sim import gates
from eo_pulse_ir.sim.simulator import logical_block
from eo_pulse_ir.sim.synthesis import design_gate


def quaternion_three_for_one() -> int:
    """Rank of span{G1, G2, [G1,G2]} for the single-qubit control algebra (=3)."""
    def gen(edge):
        eps = 1e-6
        M = logical_block([(edge, eps)], 1)
        G = (np.eye(2) - M) / (1j * eps)
        return G - np.trace(G) / 2 * np.eye(2)
    G1, G2 = gen((0, 1)), gen((1, 2))
    G3 = G1 @ G2 - G2 @ G1
    flat = lambda M: np.concatenate([M.real.ravel(), M.imag.ravel()])
    return int(np.linalg.matrix_rank(np.array([flat(G1), flat(G2), flat(G3)])))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/synthesis")
    ap.add_argument("--ccz-layers", type=int, default=0,
                    help="attempt a 3-qubit CCZ with this many layers (slow; 0 = skip)")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    rank = quaternion_three_for_one()
    print(f"[synthesis] 3-for-1 control algebra rank = {rank} "
          f"(su(2) = quaternion i,j,k)", flush=True)

    rows = []
    # 1-qubit arbitrary gate
    rng = np.random.default_rng(2)
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    U1, _ = np.linalg.qr(A)
    d1 = design_gate(U1, 1, n_layers=2, restarts=10, steps=900, seed=0)
    rows.append(("1q random unitary", 1, d1.num_pulses, d1.fidelity, d1.leakage))

    # 2-qubit gates
    for name, V in (("2q CNOT", gates.CNOT), ("2q CZ", gates.CZ)):
        t0 = time.time()
        d = design_gate(V, 2, n_layers=4, restarts=12, steps=1200, seed=1)
        rows.append((name, 2, d.num_pulses, d.fidelity, d.leakage))
        print(f"[synthesis] {name}: N={d.num_pulses} F={d.fidelity:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # 3-qubit CCZ (optional / cached)
    ccz = None
    if args.ccz_layers > 0:
        CCZ = np.eye(8, dtype=complex); CCZ[7, 7] = -1
        t0 = time.time()
        d = design_gate(CCZ, 3, n_layers=args.ccz_layers, restarts=20, steps=1500, seed=7)
        ccz = {"num_pulses": d.num_pulses, "fidelity": d.fidelity, "leakage": d.leakage}
        rows.append(("3q CCZ", 3, d.num_pulses, d.fidelity, d.leakage))
        print(f"[synthesis] 3q CCZ: N={d.num_pulses} F={d.fidelity:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    elif os.path.exists("/tmp/ccz.json"):
        with open("/tmp/ccz.json") as fh:
            c = json.load(fh)
        rows.append(("3q CCZ (cached)", 3, c["num_pulses"], c["fidelity"], c["leakage"]))
        ccz = c

    results = {"three_for_one_rank": rank,
               "designs": [{"name": n, "qubits": q, "pulses": p,
                            "fidelity": f, "leakage": l} for n, q, p, f, l in rows]}
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    lines = [
        "# n-qubit exchange-only gate design (optimal pulse control)",
        "",
        "An n-qubit logical target is a boundary condition for the switched bilinear",
        "EO control system; `design_gate` finds the control word (pulse areas) that",
        "steers the propagator onto it (layered ansatz + analytic-gradient GRAPE).",
        "",
        f"**3-for-1 ⇒ quaternion:** the single-qubit control algebra "
        f"span{{G1, G2, [G1,G2]}} has rank **{rank}** = su(2) = the quaternion units "
        "{i, j, k}. Three dots make one logical qubit whose control algebra is the "
        "quaternions.",
        "",
        "## Designed gates",
        "",
        "| target | qubits | pulses | fidelity | leakage |",
        "|---|---|---|---|---|",
    ]
    for n, q, p, f, l in rows:
        lines.append(f"| {n} | {q} | {p} | {f:.5f} | {l:.1e} |")
    lines += [
        "",
        "The control cost (pulses to reach a given fidelity) grows with the number",
        "of qubits and the entangling content of the target — single-qubit gates are",
        "exact with a handful of pulses, two-qubit gates need tens, and a 3-qubit",
        "CCZ needs a deep layered sequence. This is the optimal-control cost of",
        "n-qubit exchange-only gates.",
        "",
    ]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[synthesis] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
