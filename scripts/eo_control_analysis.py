#!/usr/bin/env python3
"""Exchange-only pulse control as a piecewise-constant (switched bilinear) system.

Demonstrates the control-theoretic structure behind the earlier physics findings:

  - controllability via the generated Lie algebra (Lie-algebra rank condition):
      * 1 qubit, exchange only -> su(2) on the DFS, NO coupling to leakage  (Q1);
      * 2 qubits, exchange only -> boundary exchange already couples logical
        <-> leakage even with no gradient                                   (Q2);
      * adding a field gradient enlarges the algebra to the full su(d) and
        couples to leakage -> the DFS protection is lost                    (Q3).
  - the single-qubit logical control is exactly SU(2) ~= unit quaternions:
    each exchange pulse is a unit quaternion (a fixed-axis rotation), and a pulse
    sequence is a quaternion product (a literal piecewise-constant trajectory).

Outputs to out/control/ : bloch_trajectory.svg, control_report.md, results.json.
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
from eo_pulse_ir.sim.control import EOControlSystem
from eo_pulse_ir.sim.landscape import trajectory_svg, write_svg


def unitary_to_quaternion(U: np.ndarray):
    """SU(2) 2x2 unitary -> unit quaternion (w,x,y,z): U = wI - i(xX+yY+zZ)."""
    U = U / np.sqrt(np.linalg.det(U))          # project to SU(2) (drop global phase)
    w = np.real(U[0, 0] + U[1, 1]) / 2
    z = np.real(1j * (U[0, 0] - U[1, 1]) / 2)
    x = np.real(1j * (U[0, 1] + U[1, 0]) / 2)
    y = np.real((U[0, 1] - U[1, 0]) / 2)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def bloch(state):
    a, b = state[0], state[1]
    return (2 * np.real(np.conj(a) * b), 2 * np.imag(np.conj(a) * b),
            np.abs(a) ** 2 - np.abs(b) ** 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/control")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    # ---- controllability table ------------------------------------------------
    sys1 = EOControlSystem(1)
    sys1g = EOControlSystem(1, gradient=1.0)
    sys2 = EOControlSystem(2)
    ctrl = {
        "1q_exchange": {"sector_dim": 3, "lie_dim": sys1.lie_dimension(1),
                        "leak_coupling": sys1.leakage_coupling(1),
                        "su_full": 8},
        "1q_exchange_plus_gradient": {"sector_dim": 3, "lie_dim": sys1g.lie_dimension(1),
                                      "leak_coupling": sys1g.leakage_coupling(1),
                                      "su_full": 8},
        "2q_exchange": {"sector_dim": 15, "lie_dim": sys2.lie_dimension(2),
                        "leak_coupling": sys2.leakage_coupling(2),
                        "su_full": 224},
    }
    results["controllability"] = ctrl
    for k, v in ctrl.items():
        print(f"[control] {k}: Lie dim {v['lie_dim']}/su={v['su_full']}  "
              f"leak-coupling {v['leak_coupling']:.2e}", flush=True)

    # ---- single-qubit Bloch trajectory of a piecewise-constant control --------
    h_pulses = [(tuple(p.edge), p.area) for p in
                synthesize(parse_circuit("qubits 1\nh 0\n"))[0]]
    word = sys1.word_from_pulses(h_pulses)
    # trace the Bloch trajectory: completed segments accumulate; sub-step the
    # current segment to draw each geodesic arc.
    prefix = []
    xs, zs, switch = [], [], []
    for edge, tau in word:
        switch.append(len(xs))
        for s in range(1, 21):
            w = prefix + [(edge, tau * s / 20)]
            U = sys1.logical_block(w)
            st = U[:, 0]                       # U|0_L>
            bx, by, bz = bloch(st)
            xs.append(bx); zs.append(bz)
        prefix = prefix + [(edge, tau)]
    write_svg(trajectory_svg(xs, zs, switch,
              title="Single-qubit EO control: Bloch trajectory of H (piecewise-constant)",
              xlabel="<X>", ylabel="<Z>"),
              os.path.join(args.out_dir, "bloch_trajectory.svg"))

    # ---- quaternion view: each pulse is a unit quaternion ---------------------
    q_total = np.array([1.0, 0, 0, 0])
    quats = []
    for edge, tau in word:
        Uk = sys1.logical_block([(edge, tau)])
        qk = unitary_to_quaternion(Uk)
        quats.append({"edge": list(edge), "dwell": tau, "quaternion": qk.tolist()})
        q_total = quat_mul(q_total, qk)
    q_h = unitary_to_quaternion(sys1.logical_block(word))
    # agreement (quaternions equal up to overall sign)
    agree = min(np.linalg.norm(q_total - q_h), np.linalg.norm(q_total + q_h))
    results["quaternion"] = {"per_pulse": quats, "product": q_total.tolist(),
                             "H_quaternion": q_h.tolist(),
                             "product_vs_H_error": float(agree)}
    print(f"[control] single-qubit H as quaternion product: error {agree:.2e}")

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ---- report ---------------------------------------------------------------
    lines = [
        "# Exchange-only pulse control as a piecewise-constant control system",
        "",
        "EO control is a right-invariant *switched bilinear* system on SU(2^n):",
        "`dU/dt = -i (H_drift + J·G_m(t)) U`, modes = nearest-neighbour exchange",
        "generators `G_e = S_i·S_j`, drift = Zeeman field, control word = a sequence",
        "of (mode, dwell-time) segments. A pulse list is exactly such a word.",
        "",
        "## Controllability (Lie-algebra rank condition)",
        "",
        "Dimension of the Lie algebra generated by the mode Hamiltonians, in a fixed",
        "total-S_z sector, and whether it couples the logical subspace to leakage:",
        "",
        "| system | sector dim | Lie dim | full su(d) | logical↔leakage coupling |",
        "|---|---|---|---|---|",
        f"| 1 qubit, exchange only | 3 | {ctrl['1q_exchange']['lie_dim']} | 8 | "
        f"{ctrl['1q_exchange']['leak_coupling']:.1e} |",
        f"| 1 qubit, exchange + gradient | 3 | {ctrl['1q_exchange_plus_gradient']['lie_dim']} "
        f"| 8 | {ctrl['1q_exchange_plus_gradient']['leak_coupling']:.2f} |",
        f"| 2 qubits, exchange only | 15 | {ctrl['2q_exchange']['lie_dim']} | 224 | "
        f"{ctrl['2q_exchange']['leak_coupling']:.2f} |",
        "",
        "Reading:",
        "- **1 qubit, exchange only** reaches `su(2)` on the logical doublet with",
        "  **zero** coupling to the S=3/2 leakage state — the algebra is confined to",
        "  the DFS block. Universal single-qubit control, no leakage (Q1).",
        "- **2 qubits, exchange only** already has nonzero logical↔leakage coupling:",
        "  boundary exchange intrinsically leaves S=1/2⊗S=1/2, so a two-qubit gate",
        "  must *refocus* leakage by careful switching (Q2) — exactly why the",
        "  validated CNOT needs many segments to return to F≈1.",
        "- **A gradient** enlarges the 1-qubit algebra from the DFS subalgebra to the",
        "  full `su(3)` and turns on logical↔leakage coupling — the DFS protection is",
        "  a *control-theoretic* property (a subalgebra), and the gradient breaks it (Q3).",
        "",
        "## Single-qubit logical control = unit quaternions",
        "",
        "On the DFS the logical group is `SU(2) ≅` the unit quaternions. Each exchange",
        "pulse is a fixed-axis rotation = a unit quaternion; a pulse sequence is a",
        "quaternion product. The validated H sequence, multiplied as quaternions,",
        f"reproduces the H quaternion to error **{agree:.1e}** — a literal",
        "piecewise-constant trajectory of fixed-axis rotations (the two intra-triple",
        "exchanges are two rotation axes ~120° apart). See `bloch_trajectory.svg`.",
        "",
    ]
    with open(os.path.join(args.out_dir, "control_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[control] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
