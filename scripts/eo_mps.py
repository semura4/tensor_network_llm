#!/usr/bin/env python3
"""Tensor-network (MPS/TEBD) evolution of exchange-only pulse dynamics.

The piecewise-constant EO dynamics is a stream of nearest-neighbour two-site
gates, so an MPS evolves it with bond dimension set by the entanglement, not by
2**n.  This script (1) validates the MPS evolver against the dense simulator on
small systems and (2) demonstrates scaling to dot arrays far beyond exact
diagonalisation.

Outputs to out/mps/ : mps_scaling.svg, mps_report.md, results.json.  Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir import parse_circuit, synthesize
from eo_pulse_ir.sim import gates
from eo_pulse_ir.sim.encoding import logical_basis
from eo_pulse_ir.sim.fidelity import average_gate_fidelity
from eo_pulse_ir.sim.landscape import line_plot_svg, write_svg
from eo_pulse_ir.sim.mps import MPS, evolve_pulses
from eo_pulse_ir.sim.operators import apply_pulse
from eo_pulse_ir.sim.simulator import logical_block


def _dense_evolve(pulses, num_qubits, col):
    psi = logical_basis(num_qubits)[:, col].astype(complex)
    for x in pulses:
        psi = apply_pulse(psi, num_qubits * 3, x.edge[0], x.edge[1], x.area)
    return psi


def validate():
    """MPS vs dense state fidelity for a few gates (full bond dim => exact)."""
    out = {}
    for name in ("cx", "swap", "cxswap"):
        pulses, _ = synthesize(parse_circuit(f"qubits 2\n{name} 0 1\n"))
        dense = _dense_evolve(pulses, 2, 0)
        mE, chi, disc = evolve_pulses(MPS.logical_register([0, 0]), pulses, chi_max=64)
        mps = mE.to_dense()
        fid = abs(np.vdot(dense, mps)) ** 2 / (
            np.vdot(dense, dense).real * mE.overlap(mE).real)
        out[name] = {"fidelity_vs_dense": float(fid), "max_bond": chi,
                     "discarded": disc}
    return out


def scaling(nqs, chi_max):
    rows = []
    for nq in nqs:
        circ = f"qubits {nq}\nh 0\n" + "".join(f"cx {i} {i+1}\n" for i in range(nq - 1))
        pulses, _ = synthesize(parse_circuit(circ))
        m = MPS.logical_register([0] * nq)
        t0 = time.time()
        mE, chi, disc = evolve_pulses(m, pulses, chi_max=chi_max, tol=1e-10)
        rows.append({"num_qubits": nq, "dots": nq * 3, "dense_log2dim": nq * 3,
                     "pulses": len(pulses), "max_bond": chi,
                     "seconds": time.time() - t0, "discarded": disc})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/mps")
    ap.add_argument("--chi-max", type=int, default=32)
    ap.add_argument("--max-qubits", type=int, default=12)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    val = validate()
    for k, v in val.items():
        print(f"[mps] validate {k}: fidelity_vs_dense={v['fidelity_vs_dense']:.10f} "
              f"max_bond={v['max_bond']}", flush=True)

    nqs = [nq for nq in (2, 3, 4, 6, 8, 10, 12) if nq <= args.max_qubits]
    rows = scaling(nqs, args.chi_max)
    for r in rows:
        print(f"[mps] scale nq={r['num_qubits']:2d} dots={r['dots']:2d} "
              f"(dense 2^{r['dense_log2dim']}) pulses={r['pulses']} "
              f"max_bond={r['max_bond']} {r['seconds']:.2f}s", flush=True)

    write_svg(line_plot_svg(
        [("max bond dim", [r["num_qubits"] for r in rows], [r["max_bond"] for r in rows]),
         ("dense dim (2^3n)", [r["num_qubits"] for r in rows],
          [2 ** r["dense_log2dim"] for r in rows])],
        title="MPS vs dense: state size for a GHZ-style EO circuit",
        xlabel="logical qubits", ylabel="dimension (log)", logy=True),
        os.path.join(args.out_dir, "mps_scaling.svg"))

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"validation": val, "scaling": rows}, fh, indent=2)

    big = rows[-1]
    lines = [
        "# Tensor-network (MPS/TEBD) evolution of EO pulse dynamics",
        "",
        "EO control is a stream of nearest-neighbour two-site gates, so an MPS",
        "propagates it with bond dimension set by entanglement, not by 2^n. This",
        "reuses the repository's original tensor-network theme for spin dynamics.",
        "",
        "## Validation (full bond dim = exact)",
        "",
        "| gate | MPS-vs-dense fidelity | max bond |",
        "|---|---|---|",
    ]
    for k, v in val.items():
        lines.append(f"| {k} | {v['fidelity_vs_dense']:.10f} | {v['max_bond']} |")
    lines += [
        "",
        "## Scaling (GHZ-style CNOT chain, chi_max %d)" % args.chi_max,
        "",
        "| logical qubits | dots | dense dim | pulses | max bond | time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['num_qubits']} | {r['dots']} | 2^{r['dense_log2dim']} | "
                     f"{r['pulses']} | {r['max_bond']} | {r['seconds']:.3f} |")
    lines += [
        "",
        f"The largest case is {big['num_qubits']} logical qubits = {big['dots']} dots, "
        f"a dense state of 2^{big['dense_log2dim']} amplitudes (infeasible to store), "
        f"evolved by MPS in {big['seconds']:.2f}s with max bond dimension "
        f"{big['max_bond']}. Bond dimension stays bounded because the circuit's "
        "entanglement is limited — the regime where tensor networks win.",
        "",
    ]
    with open(os.path.join(args.out_dir, "mps_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[mps] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
