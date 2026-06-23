#!/usr/bin/env python3
"""Demonstrate 2-D grid compilation for exchange-only qubits.

Compiles the same logical circuit on a 1-D linear chain and a 2-D square-lattice
patch, then compares pulse counts, makespan, and routing overhead.  Shows the
interaction-weighted initial-layout heuristic and BFS routing on the grid.

Requires no external dependencies (stdlib + numpy-free core IR).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir import (GridTopology, LinearTopology, compile_circuit,
                         emit_artifacts, parse_circuit)


def _build_circuit(n_qubits: int, pattern: str) -> str:
    """Generate a test circuit as QASM text."""
    lines = [f"qubits {n_qubits}"]
    if pattern == "ring":
        for i in range(n_qubits):
            lines.append(f"h {i}")
        for i in range(n_qubits):
            lines.append(f"cx {i} {(i + 1) % n_qubits}")
    elif pattern == "star":
        lines.append("h 0")
        for i in range(1, n_qubits):
            lines.append(f"cx 0 {i}")
    elif pattern == "ladder":
        for i in range(n_qubits):
            lines.append(f"h {i}")
        for i in range(0, n_qubits - 1, 2):
            lines.append(f"cx {i} {i + 1}")
        for i in range(1, n_qubits - 1, 2):
            lines.append(f"cx {i} {i + 1}")
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return "\n".join(lines) + "\n"


def _grid_dims(n: int):
    """Pick a roughly-square grid for ``n`` qubits."""
    import math
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--num-qubits", type=int, default=4)
    ap.add_argument("-p", "--pattern", default="ring",
                    choices=["ring", "star", "ladder"])
    ap.add_argument("-o", "--out-dir", default="out/grid_demo")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    n = args.num_qubits
    qasm = _build_circuit(n, args.pattern)
    circuit = parse_circuit(qasm)
    print(f"[grid] circuit: {n} qubits, {len(circuit.gates)} gates ({args.pattern})")
    print(circuit)
    print()

    # --- 1-D linear compilation ---
    r_lin = compile_circuit(circuit)
    print(f"[1-D linear] pulses={r_lin.metrics.pulse_count}  "
          f"makespan={r_lin.metrics.total_time:.2f}  "
          f"boundary={r_lin.metrics.boundary_pulse_count}  "
          f"critical={r_lin.metrics.critical_path_pulses}")

    # --- 2-D grid compilation ---
    rows, cols = _grid_dims(n)
    grid = GridTopology(rows, cols)
    grid.assign_initial_layout(list(range(n)), circuit_gates=circuit.gates)
    print(f"[2-D grid {rows}x{cols}] initial layout:")
    for q in range(n):
        slot = grid.slot_of[q]
        r, c = divmod(slot, cols)
        dots = grid.dots_of_qubit(q)
        print(f"  q{q} -> slot {slot} ({r},{c})  dots {dots}")

    r_grid = compile_circuit(circuit, topology=grid)
    print(f"[2-D grid] pulses={r_grid.metrics.pulse_count}  "
          f"makespan={r_grid.metrics.total_time:.2f}  "
          f"boundary={r_grid.metrics.boundary_pulse_count}  "
          f"critical={r_grid.metrics.critical_path_pulses}")

    # --- comparison ---
    print()
    delta_p = r_grid.metrics.pulse_count - r_lin.metrics.pulse_count
    delta_t = r_grid.metrics.total_time - r_lin.metrics.total_time
    print(f"[compare] pulse delta: {delta_p:+d}  makespan delta: {delta_t:+.2f}")
    if r_grid.metrics.pulse_count < r_lin.metrics.pulse_count:
        print("[compare] >>> 2-D grid WINS on pulse count")
    elif r_grid.metrics.pulse_count > r_lin.metrics.pulse_count:
        print("[compare] >>> 1-D linear wins on pulse count "
              "(grid may still win on makespan via parallelism)")
    else:
        print("[compare] >>> tied on pulse count")

    # --- emit artifacts ---
    lin_dir = os.path.join(args.out_dir, "linear")
    grid_dir = os.path.join(args.out_dir, "grid")
    emit_artifacts(r_lin, lin_dir, title=f"{args.pattern}-{n}q-linear")
    emit_artifacts(r_grid, grid_dir, title=f"{args.pattern}-{n}q-grid-{rows}x{cols}")

    summary = {
        "circuit": {"num_qubits": n, "pattern": args.pattern, "gates": len(circuit.gates)},
        "linear": r_lin.metrics.as_dict(),
        "grid": {"rows": rows, "cols": cols, **r_grid.metrics.as_dict()},
    }
    with open(os.path.join(args.out_dir, "comparison.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[grid] artifacts written to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
