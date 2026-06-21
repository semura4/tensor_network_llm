#!/usr/bin/env python3
"""Optimize a leakage-free exchange-only CNOT and emit it as a role/area template.

Uses a physically-motivated KAK-style ansatz on two encoded qubits (6 dots):
several boundary exchanges, each dressed by a full single-qubit-capable block on
the control and target triples.  The pulse areas are optimised (analytic-gradient
Adam) to maximise the leakage-aware average gate fidelity vs CNOT.

The best sequence is written to JSON as a list of (role, area) pairs ready to
drop into ``eo_pulse_ir/native.py``'s CNOT template, where roles resolve to:

    ctrl_low=(3c,3c+1)  ctrl_high=(3c+1,3c+2)  inter=(3c+2,3t)
    tgt_low=(3t,3t+1)   tgt_high=(3t+1,3t+2)

Run:  python scripts/eo_optimize_cnot.py -o out/cnot.json --restarts 120
Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir.sim import gates
from eo_pulse_ir.sim.fidelity import leakage
from eo_pulse_ir.sim.optimize import optimize_areas_grad
from eo_pulse_ir.sim.simulator import logical_block

CL, CH, IN, TL, TH = (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)
EDGE_ROLE = {CL: "ctrl_low", CH: "ctrl_high", IN: "inter", TL: "tgt_low", TH: "tgt_high"}


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


# candidate ansaetze, smallest N first; the last is a guaranteed F=1 fallback.
# (loc2-bnd3, N=19, is known to top out at F~0.78 and is omitted for runtime.)
CANDIDATES = [
    ("loc3-bnd3", build(3, 3, True)),    # N=27
    ("loc3-bnd4-nofinal", build(3, 4, False)),  # N=28
    ("loc3-bnd4", build(3, 4, True)),    # N=34  (known F=1)
]


def optimize_one(edges, restarts, steps, seed):
    areas, F = optimize_areas_grad(edges, 2, gates.CNOT, steps=steps,
                                   restarts=restarts, seed=seed)
    M = logical_block(list(zip(edges, areas)), 2)
    return areas, F, leakage(M)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="out/cnot.json")
    ap.add_argument("--restarts", type=int, default=120)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.9999,
                    help="accept the smallest sequence reaching this fidelity")
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    results = []
    chosen = None
    for name, edges in CANDIDATES:
        areas, F, leak = optimize_one(edges, args.restarts, args.steps, args.seed)
        roles = [EDGE_ROLE[e] for e in edges]
        entry = {"name": name, "num_pulses": len(edges), "fidelity": F,
                 "leakage": leak, "roles": roles, "areas": areas}
        results.append(entry)
        print(f"[cnot] {name}: N={len(edges)} F={F:.6f} leak={leak:.2e}", flush=True)
        # write incrementally so a long run always leaves a usable artefact
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"results": results, "chosen": chosen}, fh, indent=2)
        if chosen is None and F >= args.threshold:
            chosen = entry
            print(f"[cnot] -> chosen {name} (N={len(edges)}, F={F:.6f})", flush=True)
            # keep going only if we want the very smallest; we already took it
            break

    if chosen is None:
        chosen = max(results, key=lambda r: r["fidelity"])
        print(f"[cnot] no candidate hit threshold; best is {chosen['name']} "
              f"(F={chosen['fidelity']:.6f})", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "chosen": chosen}, fh, indent=2)

    print("\n[cnot] template (role, area) for native.py:")
    for role, area in zip(chosen["roles"], chosen["areas"]):
        print(f'    ("{role}", {area:.10f}),')
    print(f"\n[cnot] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
