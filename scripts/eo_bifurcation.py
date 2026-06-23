#!/usr/bin/env python3
"""Bifurcation analysis of the exchange-only pulse-control landscape.

Demonstrates nonlinear-dynamics / bifurcation tools on the EO control problem:

  (A) Landscape bifurcations: a single-qubit 2-pulse control landscape F(a, b)
      under a field gradient g.  As g grows the landscape gains optima via
      saddle-node bifurcations — the control problem becomes more rugged/complex.
  (B) Controllability bifurcation: the reachable-set Lie-algebra dimension jumps
      (4 -> 8) the instant a gradient is switched on — a structural bifurcation
      of what gates are reachable (the DFS subalgebra opens up to full su(3)).

Outputs to out/bifurcation/ : landscape_<g>.svg, bifurcation_diagram.svg,
controllability.svg, report.md, results.json.  Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir.sim import gates
from eo_pulse_ir.sim.bifurcation import classify_critical, local_maxima, optima_sweep
from eo_pulse_ir.sim.control import EOControlSystem
from eo_pulse_ir.sim.field import logical_block_field, zeeman_energies
from eo_pulse_ir.sim.fidelity import average_gate_fidelity
from eo_pulse_ir.sim.landscape import heatmap_svg, line_plot_svg, write_svg


def landscape(g: float, target, n: int) -> np.ndarray:
    """Single-qubit 2-pulse F(a,b): pulse 1 = intra_high(a), pulse 2 = intra_low(b)."""
    b3 = zeeman_energies(3, g)
    grid = np.zeros((n, n))
    axis = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, bb in enumerate(axis):
        for j, aa in enumerate(axis):
            M = logical_block_field([((1, 2), aa), ((0, 1), bb)], 1, b3)
            grid[i, j] = average_gate_fidelity(M, target)
    return grid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/bifurcation")
    ap.add_argument("--res", type=int, default=72, help="landscape grid resolution")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    axis = np.linspace(0, 2 * np.pi, args.res, endpoint=False)
    target = gates.H

    # ---- (A) landscape bifurcation sweep over gradient g -----------------------
    gs = [round(x, 3) for x in np.linspace(0.0, 0.8, 17)]
    sweep = optima_sweep(lambda g: landscape(g, target, args.res), gs, axis, axis,
                         periodic=True)
    diagram = [{"gradient": bp.param, "num_maxima": bp.num_maxima,
                "global_max": bp.global_max} for bp in sweep]
    for d in diagram:
        print(f"[bif] g={d['gradient']:.3f}  #optima={d['num_maxima']}  "
              f"maxF={d['global_max']:.4f}", flush=True)

    # landscape heatmaps at a few representative g (show optima appearing)
    for g in (0.0, 0.2, 0.6):
        G = landscape(g, target, args.res)
        write_svg(heatmap_svg(G, axis, axis,
                  f"control landscape F(a,b), gradient={g} "
                  f"({len(local_maxima(G))} optima)",
                  "intra_high area a", "intra_low area b", "fidelity"),
                  os.path.join(args.out_dir, f"landscape_g{g}.svg"))

    write_svg(line_plot_svg(
        [("# local optima", [d["gradient"] for d in diagram],
          [d["num_maxima"] for d in diagram]),
         ("global max fidelity", [d["gradient"] for d in diagram],
          [d["global_max"] for d in diagram])],
        title="Bifurcation diagram: control-landscape optima vs field gradient",
        xlabel="Zeeman gradient g / j_max", ylabel="count  /  fidelity"),
        os.path.join(args.out_dir, "bifurcation_diagram.svg"))

    # ---- (B) controllability bifurcation: Lie dimension vs gradient ------------
    ctrl = []
    for g in (0.0, 0.05, 0.1, 0.2, 0.4):
        dim = EOControlSystem(1, gradient=g).lie_dimension(sector_down=1)
        leak = EOControlSystem(1, gradient=g).leakage_coupling(1)
        ctrl.append({"gradient": g, "lie_dim": dim, "leak_coupling": leak})
        print(f"[bif] controllability g={g}: Lie dim={dim} leak={leak:.2e}", flush=True)
    write_svg(line_plot_svg(
        [("Lie algebra dim", [c["gradient"] for c in ctrl], [c["lie_dim"] for c in ctrl])],
        title="Controllability bifurcation: reachable Lie dimension vs gradient",
        xlabel="Zeeman gradient g / j_max", ylabel="Lie algebra dimension"),
        os.path.join(args.out_dir, "controllability.svg"))

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"landscape_bifurcation": diagram, "controllability": ctrl}, fh, indent=2)

    counts = [d["num_maxima"] for d in diagram]
    lines = [
        "# Bifurcation analysis of the EO control landscape",
        "",
        "The pulse-control problem is a nonlinear map (pulse parameters -> fidelity).",
        "Varying a physical parameter moves the landscape's critical points; their",
        "creation/merging are bifurcations of the control problem.",
        "",
        "## (A) Landscape bifurcations vs field gradient",
        "",
        "Single-qubit 2-pulse landscape F(a,b), target H, on the periodic area torus.",
        f"As the gradient g increases from 0 to 0.8, the number of local optima grows",
        f"from {counts[0]} to {max(counts)} via saddle-node bifurcations while the global",
        f"maximum fidelity falls from {diagram[0]['global_max']:.3f} to "
        f"{diagram[-1]['global_max']:.3f} — the gradient makes the control landscape",
        "progressively more rugged (harder to optimise: more local traps).",
        "",
        "| gradient g | # local optima | global max F |",
        "|---|---|---|",
    ]
    for d in diagram:
        lines.append(f"| {d['gradient']:.3f} | {d['num_maxima']} | {d['global_max']:.4f} |")
    lines += [
        "",
        "## (B) Controllability bifurcation",
        "",
        "The reachable-set Lie-algebra dimension jumps the instant a gradient is",
        "switched on: at g=0 the exchange-only algebra is the DFS subalgebra (dim 4,",
        "no logical<->leakage coupling); for any g>0 it opens to the full su(3)",
        "(dim 8) and couples to leakage. A structural (controllability) bifurcation.",
        "",
        "| gradient g | Lie dim | logical<->leakage coupling |",
        "|---|---|---|",
    ]
    for c in ctrl:
        lines.append(f"| {c['gradient']} | {c['lie_dim']} | {c['leak_coupling']:.2e} |")
    lines += [
        "",
        "Takeaway: a field gradient (or, by the same token, valley-induced drift)",
        "both **opens leakage channels** (controllability bifurcation) and **roughens",
        "the control landscape** (saddle-node cascade) — so robust pulse design must",
        "target broad, gradient-insensitive optima, not the sharpest peak.",
        "",
    ]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[bif] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
