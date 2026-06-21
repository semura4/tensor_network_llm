#!/usr/bin/env python3
"""Optimize a leakage-free exchange-only two-qubit gate and emit it as a template.

Uses a physically-motivated KAK-style ansatz on two encoded qubits (6 dots):
several boundary exchanges, each dressed by a full single-qubit-capable block on
the control and target triples.  The pulse areas are optimised (analytic-gradient
Adam) to maximise the leakage-aware average gate fidelity vs the chosen target.

The best sequence is written to JSON as a list of (role, area) pairs ready to
drop into ``eo_pulse_ir/native.py``, where roles resolve to:

    ctrl_low=(3c,3c+1)  ctrl_high=(3c+1,3c+2)  inter=(3c+2,3t)
    tgt_low=(3t,3t+1)   tgt_high=(3t+1,3t+2)

Run:  python scripts/eo_optimize_2q.py --target cnot  -o out/cnot.json
      python scripts/eo_optimize_2q.py --target swap  -o out/swap.json
      python scripts/eo_optimize_2q.py --target cxswap -o out/cxswap.json
Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir.sim import gates
from eo_pulse_ir.sim.fidelity import leakage
from eo_pulse_ir.sim.optimize import optimize_areas_grad
from eo_pulse_ir.sim.simulator import logical_block

CL, CH, IN, TL, TH = (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)
EDGE_ROLE = {CL: "ctrl_low", CH: "ctrl_high", IN: "inter", TL: "tgt_low", TH: "tgt_high"}

TARGETS = {"cnot": gates.CNOT, "cx": gates.CNOT, "swap": gates.SWAP,
           "cxswap": gates.CXSWAP, "cz": gates.CZ}


def build(local: int, n_boundary: int, final_local: bool):
    """KAK-style ansatz: n_boundary x [A-block, B-block, inter] then optional locals."""
    A = [CH, CL, CH, CL][:local]
    B = [TL, TH, TL, TH][:local]
    seq = []
    for _ in range(n_boundary):
        seq += A + B + [IN]
    if final_local:
        seq += A + B
    return seq


# candidate ansaetze, smallest N first; larger ones are more expressive
def candidates():
    return [
        ("loc3-bnd3", build(3, 3, True)),         # N=27
        ("loc3-bnd4", build(3, 4, True)),         # N=34
        ("loc3-bnd5", build(3, 5, True)),         # N=41
        ("loc3-bnd6", build(3, 6, True)),         # N=48
    ]


def optimize_one(edges, target, restarts, steps, seed):
    areas, F = optimize_areas_grad(edges, 2, target, steps=steps,
                                   restarts=restarts, seed=seed)
    M = logical_block(list(zip(edges, areas)), 2)
    return areas, F, leakage(M)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="cnot", choices=sorted(TARGETS),
                    help="two-qubit gate to synthesise")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--restarts", type=int, default=80)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.9999,
                    help="accept the smallest sequence reaching this fidelity")
    args = ap.parse_args(argv)
    out = args.out or f"out/{args.target}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    target = TARGETS[args.target]
    tag = args.target

    results = []
    chosen = None
    for name, edges in candidates():
        areas, F, leak = optimize_one(edges, target, args.restarts, args.steps, args.seed)
        roles = [EDGE_ROLE[e] for e in edges]
        entry = {"name": name, "num_pulses": len(edges), "fidelity": F,
                 "leakage": leak, "roles": roles, "areas": areas}
        results.append(entry)
        print(f"[{tag}] {name}: N={len(edges)} F={F:.6f} leak={leak:.2e}", flush=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"target": tag, "results": results, "chosen": chosen}, fh, indent=2)
        if chosen is None and F >= args.threshold:
            chosen = entry
            print(f"[{tag}] -> chosen {name} (N={len(edges)}, F={F:.6f})", flush=True)
            break

    if chosen is None:
        chosen = max(results, key=lambda r: r["fidelity"])
        print(f"[{tag}] no candidate hit threshold; best is {chosen['name']} "
              f"(F={chosen['fidelity']:.6f})", flush=True)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"target": tag, "results": results, "chosen": chosen}, fh, indent=2)

    print(f"\n[{tag}] template (role, area) for native.py:")
    for role, area in zip(chosen["roles"], chosen["areas"]):
        print(f'    ("{role}", {area:.10f}),')
    print(f"\n[{tag}] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
