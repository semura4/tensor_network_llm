#!/usr/bin/env python3
"""Magnetic-field-gradient study for exchange-only qubits.

Answers four questions with the physics simulator (numpy):

  Q1  Does intra-block exchange alone leak?            -> no (DFS protected)
  Q2  Does inter-block (boundary) exchange leak?       -> yes (raw); refocused by
                                                          a well-designed gate
  Q3  Does a magnetic-field gradient change behaviour? -> yes: it breaks S^2
        conservation, inducing leakage even for intra-block exchange, and is
        catastrophic for long two-qubit gates (a uniform field is harmless)
  Q4  Is pulse-parameter sensitivity visible?          -> yes: a 2-D
        (gradient x area-calibration) infidelity map

Outputs to out/gradient/ : gradient_leakage.svg, gradient_infidelity.svg,
sensitivity_map.svg, gradient_report.md, results.json.  Requires numpy.
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
from eo_pulse_ir.sim.field import logical_block_field, simulate_field, zeeman_energies
from eo_pulse_ir.sim.fidelity import average_gate_fidelity, leakage
from eo_pulse_ir.sim.landscape import heatmap_svg, line_plot_svg, write_svg


def pulses_of(circuit_text):
    p, _ = synthesize(parse_circuit(circuit_text))
    return [(tuple(x.edge), x.area) for x in p]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/gradient")
    ap.add_argument("--res", type=int, default=24, help="2-D sensitivity grid size")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    x_pulses = pulses_of("qubits 1\nx 0\n")
    cnot_pulses = pulses_of("qubits 2\ncx 0 1\n")

    # ----- Q1: intra-block exchange, no field -----------------------------------
    leak_intra = leakage(logical_block_field(x_pulses, 1, np.zeros(3)))
    # arbitrary intra-only sequence too
    arb = [((0, 1), 0.7), ((1, 2), 1.3), ((0, 1), 2.1)]
    leak_arb = leakage(logical_block_field(arb, 1, np.zeros(3)))
    results["Q1_intra_no_field"] = {"X_leakage": leak_intra, "arbitrary_leakage": leak_arb}

    # ----- Q2: raw inter-block exchange vs leakage-free gate --------------------
    raw_boundary = [((2, 3), np.pi / 2)]   # a single boundary sqrt-SWAP
    leak_raw = leakage(logical_block_field(raw_boundary, 2, np.zeros(6)))
    leak_cnot = leakage(logical_block_field(cnot_pulses, 2, np.zeros(6)))
    results["Q2_inter_block"] = {"raw_boundary_leakage": leak_raw,
                                 "validated_cnot_leakage": leak_cnot}

    # ----- Q3: field gradient sweep (and uniform-field control) -----------------
    grads = np.logspace(-3, -1, 11)        # 0.001 .. 0.1 (units of j_max)
    q3 = {"gradient": grads.tolist(), "X_leakage": [], "X_infidelity": [],
          "CNOT_leakage": [], "CNOT_infidelity": []}
    for g in grads:
        rx = simulate_field(x_pulses, 1, g, target=gates.X)
        rc = simulate_field(cnot_pulses, 2, g, target=gates.CNOT)
        q3["X_leakage"].append(rx["leakage"])
        q3["X_infidelity"].append(rx["infidelity"])
        q3["CNOT_leakage"].append(rc["leakage"])
        q3["CNOT_infidelity"].append(rc["infidelity"])
    # uniform-field control: gradient=0, large offset -> harmless
    b_uniform = zeeman_energies(3, 0.0, b0=0.5)
    Mu = logical_block_field(x_pulses, 1, b_uniform)
    q3["uniform_field_X_infidelity"] = 1.0 - average_gate_fidelity(Mu, gates.X)
    results["Q3_gradient"] = q3

    write_svg(line_plot_svg(
        [("X (1 qubit, intra-only)", grads.tolist(), [max(v, 1e-16) for v in q3["X_leakage"]]),
         ("CNOT (2 qubit)", grads.tolist(), [max(v, 1e-16) for v in q3["CNOT_leakage"]])],
        title="Q3: leakage induced by a field gradient",
        xlabel="Zeeman gradient / j_max", ylabel="leakage", logx=True, logy=True),
        os.path.join(args.out_dir, "gradient_leakage.svg"))
    write_svg(line_plot_svg(
        [("X (1 qubit)", grads.tolist(), [max(v, 1e-16) for v in q3["X_infidelity"]]),
         ("CNOT (2 qubit)", grads.tolist(), [max(v, 1e-16) for v in q3["CNOT_infidelity"]])],
        title="Q3: gate infidelity vs field gradient",
        xlabel="Zeeman gradient / j_max", ylabel="infidelity", logx=True, logy=True),
        os.path.join(args.out_dir, "gradient_infidelity.svg"))

    # ----- Q4: 2-D pulse-parameter sensitivity (gradient x area calibration) ----
    gx = np.linspace(0.0, 0.03, args.res)        # gradient
    sy = np.linspace(0.95, 1.05, args.res)       # global area scale (calibration)
    grid = np.empty((len(sy), len(gx)))
    for r, s in enumerate(sy):
        scaled = [(e, a * s) for (e, a) in cnot_pulses]
        for c, g in enumerate(gx):
            b = zeeman_energies(6, g)
            M = logical_block_field(scaled, 2, b)
            grid[r, c] = 1.0 - average_gate_fidelity(M, gates.CNOT)
    write_svg(heatmap_svg(grid, gx, sy,
              "Q4: CNOT infidelity vs gradient & area calibration",
              "Zeeman gradient / j_max", "pulse-area scale", "infidelity"),
              os.path.join(args.out_dir, "sensitivity_map.svg"))
    results["Q4_sensitivity"] = {"gradient": gx.tolist(), "area_scale": sy.tolist(),
                                 "infidelity_min": float(grid.min()),
                                 "infidelity_max": float(grid.max())}

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ----- report --------------------------------------------------------------
    g_at = lambda key, gval: q3[key][int(np.argmin(np.abs(grads - gval)))]
    lines = [
        "# Exchange-only field-gradient study",
        "",
        "Static Zeeman field H_Z = Σ b_i S_z^i added to the exchange dynamics "
        "(exact eigen-propagation). Gradient in units of the exchange amplitude j_max.",
        "",
        "## Q1 — Does intra-block exchange alone leak?  **No.**",
        f"Single-qubit X (intra-only), no field: leakage = {leak_intra:.2e}. "
        f"Arbitrary intra sequence: leakage = {leak_arb:.2e}. "
        "Exchange conserves total S² within a triple, so the S=1/2 doublet is a "
        "decoherence-free subsystem — zero leakage by symmetry.",
        "",
        "## Q2 — Does inter-block (boundary) exchange leak?  **Yes (raw).**",
        f"A single boundary √SWAP on edge (2,3): leakage = {leak_raw:.3f}. "
        "Boundary exchange couples a control-triple spin to a target-triple spin, "
        "driving population out of S=1/2 ⊗ S=1/2. A well-designed gate refocuses "
        f"it: the validated CNOT has leakage = {leak_cnot:.2e}.",
        "",
        "## Q3 — Does a magnetic-field gradient change behaviour?  **Yes, strongly.**",
        "A *uniform* field is harmless (it is b·S_z^total, a global phase): X "
        f"infidelity under b0=0.5, gradient 0 = {q3['uniform_field_X_infidelity']:.2e}.",
        "A *gradient* breaks S² conservation and the DFS:",
        "",
        "| gradient/j_max | X leakage | X infidelity | CNOT leakage | CNOT infidelity |",
        "|---|---|---|---|---|",
    ]
    for gv in (0.003, 0.01, 0.03, 0.1):
        i = int(np.argmin(np.abs(grads - gv)))
        lines.append(f"| {grads[i]:.3g} | {q3['X_leakage'][i]:.2e} | "
                     f"{q3['X_infidelity'][i]:.2e} | {q3['CNOT_leakage'][i]:.2e} | "
                     f"{q3['CNOT_infidelity'][i]:.2e} |")
    lines += [
        "",
        "Intra-block (single-qubit) leakage now rises from zero, ~quadratically in "
        "the gradient. The two-qubit CNOT is far more sensitive: its long duration "
        "means the relevant figure of merit is gradient × gate-time, so field "
        "homogeneity (or short gates) is essential for EO two-qubit gates.",
        "",
        "## Q4 — Is pulse-parameter sensitivity visible?  **Yes.**",
        "`sensitivity_map.svg` maps CNOT infidelity over (Zeeman gradient) × "
        "(global pulse-area calibration scale). Infidelity ranges "
        f"{results['Q4_sensitivity']['infidelity_min']:.2e} … "
        f"{results['Q4_sensitivity']['infidelity_max']:.2e} across the window, "
        "showing the gate's joint sensitivity to a control parameter (area "
        "calibration) and an environmental parameter (field gradient). See also "
        "the landscape and Monte-Carlo noise studies.",
        "",
    ]
    with open(os.path.join(args.out_dir, "gradient_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[gradient] Q1 intra leakage={leak_intra:.2e}  "
          f"Q2 raw boundary leakage={leak_raw:.3f}  "
          f"Q3 X infid@grad0.1={q3['X_infidelity'][-1]:.3f}")
    print(f"[gradient] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
