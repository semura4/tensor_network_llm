#!/usr/bin/env python3
"""State-space (dynamical-systems) view of single-qubit exchange-only control.

EO control is a switched linear dynamical system: d|psi>/dt = -iH(t)|psi>, i.e.
dx/dt = A_m x on the real state sphere, with a finite set of constant
skew-symmetric vector fields A_m and a piecewise-constant switching signal.  For
one logical qubit this is the Bloch sphere with db/dt = omega_m x b: two rotation
vector fields about axes 120 deg apart.  The logical subspace is an invariant
manifold of the exchange-only flow; a field gradient breaks its invariance.

Outputs to out/statespace/ : bloch_phase_portrait.svg (the two vector fields +
their fixed points + a switched trajectory), manifold_defect.svg (leakage as the
invariant-manifold defect vs time, exchange-only vs gradient), report + json.
Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir.sim.encoding import logical_basis
from eo_pulse_ir.sim.field import zeeman_diagonal, zeeman_energies
from eo_pulse_ir.sim.landscape import line_plot_svg, write_svg
from eo_pulse_ir.sim.operators import s_dot_s
from eo_pulse_ir.sim.statespace import StateSpaceSystem, bloch_vector, flow
from eo_pulse_ir.sim import quaternion as quat

N1 = quat.N1            # intra_low rotation axis (logical -Z)
N2 = quat.N2            # intra_high rotation axis (120 deg from N1)

# orthographic view direction
_AZ, _EL = np.radians(35), np.radians(22)


def _project(p):
    x, y, z = p
    sx = x * np.cos(_AZ) + y * np.sin(_AZ)
    depth = -x * np.sin(_AZ) + y * np.cos(_AZ)
    sy = z * np.cos(_EL) + depth * np.sin(_EL)
    return sx, sy


def _rot_about(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _orbit(axis, polar, m=80):
    # a closed flow line: points at fixed angle 'polar' from 'axis'
    # start vector at angle polar from axis
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    perp = np.cross(axis, ref); perp /= np.linalg.norm(perp)
    v0 = np.cos(polar) * axis + np.sin(polar) * perp
    return [_rot_about(axis, 2 * np.pi * k / m) @ v0 for k in range(m + 1)]


def _poly(points, color, w=1.0, dash=None):
    pts = " ".join(f"{120 + 90 * _project(p)[0]:.1f},{120 - 90 * _project(p)[1]:.1f}"
                   for p in points)
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{w}'{d}/>"


def _dot(p, color, r=4):
    sx, sy = _project(p)
    return f"<circle cx='{120 + 90 * sx:.1f}' cy='{120 - 90 * sy:.1f}' r='{r}' fill='{color}'/>"


def phase_portrait(traj):
    parts = ["<svg xmlns='http://www.w3.org/2000/svg' width='300' height='270' "
             "font-family='monospace' font-size='10'>",
             "<rect width='300' height='270' fill='white'/>",
             "<text x='10' y='16' font-size='12' font-weight='bold'>"
             "Bloch phase portrait: two rotation fields (120°) + switched trajectory</text>",
             "<circle cx='120' cy='120' r='90' fill='none' stroke='#ccc'/>"]
    # flow orbits (closed integral curves) for each generator
    for axis, col in [(N1, "#9ecae1"), (N2, "#fdae6b")]:
        for polar in (np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            parts.append(_poly(_orbit(axis, polar), col, 1.0))
        parts.append(_poly([-axis, axis], "#666", 1.0, dash="3 2"))   # axis line
        parts.append(_dot(axis, col, 3))                              # fixed point
        parts.append(_dot(-axis, col, 3))
    parts.append(_poly(traj, "#4C78A8", 2.2))                        # trajectory
    parts.append(_dot(traj[0], "#54A24B"))
    parts.append(_dot(traj[-1], "#E45756"))
    parts.append("<text x='10' y='258' fill='#54A24B'>● start</text>")
    parts.append("<text x='70' y='258' fill='#E45756'>● end</text>")
    parts.append("<text x='130' y='258' fill='#9ecae1'>○ intra_low field</text>")
    parts.append("<text x='235' y='258' fill='#fdae6b'>○ intra_high</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def verify():
    from eo_pulse_ir import parse_circuit, synthesize
    from eo_pulse_ir.sim.control import EOControlSystem
    from eo_pulse_ir.sim.statespace import real_generator, to_real, from_real
    n = 3
    H = s_dot_s(n, 0, 1).astype(complex)
    A = real_generator(H)
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    vf_err = float(np.max(np.abs(A @ to_real(psi) - to_real(-1j * H @ psi))))
    skew = float(np.max(np.abs(A + A.T)))
    p, _ = synthesize(parse_circuit("qubits 2\ncx 0 1\n"))
    ss, cs = StateSpaceSystem(2), EOControlSystem(2)
    x0 = to_real(logical_basis(2)[:, 0].astype(complex))
    xT = ss.integrate(x0, cs.word_from_pulses(p))
    op = cs.propagator(cs.word_from_pulses(p)) @ logical_basis(2)[:, 0]
    flow_err = float(np.max(np.abs(from_real(xT) - op)))
    return {"vector_field_error": vf_err, "generator_skew_symmetry": skew,
            "flow_vs_operator_error": flow_err}


def bloch_trajectory(role_seq):
    ss = StateSpaceSystem(1)
    edge = {"intra_low": (0, 1), "intra_high": (1, 2)}
    psi = logical_basis(1)[:, 0].astype(complex)
    traj = [bloch_vector(psi)]
    for role, area in role_seq:
        for s in range(1, 13):
            ps = flow(ss.hamiltonian(edge[role]), psi, area * s / 12)
            traj.append(bloch_vector(ps))
        psi = flow(ss.hamiltonian(edge[role]), psi, area)
    return traj


def manifold_defect_curves():
    """Leakage (manifold defect) vs time: exchange-only (invariant) vs gradient."""
    ss = StateSpaceSystem(1)
    Hx = ss.hamiltonian((1, 2))                  # an intra exchange (drives rotation)
    ts = np.linspace(0, 6.0, 40)
    curves = {}
    for grad in (0.0, 0.1, 0.3):
        Hz = np.diag(zeeman_diagonal(zeeman_energies(3, grad))).astype(complex)
        H = Hx + Hz
        psi0 = logical_basis(1)[:, 0].astype(complex)
        d = [ss.logical_manifold_defect(flow(H, psi0, t)) for t in ts]
        curves[grad] = d
    return ts, curves


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", default="out/statespace")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    v = verify()
    print(f"[statespace] linear vector field error={v['vector_field_error']:.1e} "
          f"skew={v['generator_skew_symmetry']:.1e} "
          f"flow-vs-operator={v['flow_vs_operator_error']:.1e}", flush=True)

    # phase portrait with the validated-H switched trajectory
    from eo_pulse_ir.native import one_qubit_template
    traj = bloch_trajectory(one_qubit_template("h"))
    write_svg(phase_portrait(traj), os.path.join(args.out_dir, "bloch_phase_portrait.svg"))

    ts, curves = manifold_defect_curves()
    write_svg(line_plot_svg(
        [(f"gradient={g}", ts.tolist(), [max(x, 1e-16) for x in d])
         for g, d in curves.items()],
        title="Invariant-manifold defect (leakage) vs time",
        xlabel="time", ylabel="manifold defect (leakage)", logy=True),
        os.path.join(args.out_dir, "manifold_defect.svg"))

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"verification": v,
                   "manifold_defect_final": {str(g): d[-1] for g, d in curves.items()}},
                  fh, indent=2)

    lines = [
        "# State-space (dynamical-systems) view",
        "",
        "EO control is a **switched linear dynamical system** on the state sphere:",
        "`dx/dt = A_m x`, `A_m = -iH_m` constant skew-symmetric vector fields, with a",
        "piecewise-constant switching signal. Verified:",
        "",
        f"- the real vector field is exact: `A x = -iH psi` to {v['vector_field_error']:.1e}, "
        f"`A` skew-symmetric to {v['generator_skew_symmetry']:.1e};",
        f"- the state-space flow equals the operator picture to {v['flow_vs_operator_error']:.1e}.",
        "",
        "## Single qubit = Bloch sphere S²",
        "",
        "`db/dt = omega_m x b`: two rotation vector fields about axes 120° apart",
        "(intra_low = -Z, intra_high = (√3/2,0,1/2)); their fixed points are the axis",
        "poles. `bloch_phase_portrait.svg` shows both fields' closed orbits and a",
        "switched trajectory (Hadamard).",
        "",
        "## DFS = invariant manifold",
        "",
        "The logical subspace is an invariant manifold of the exchange-only flow",
        "(defect stays 0). A field gradient breaks invariance — the defect (leakage)",
        "grows:",
        "",
        "| gradient | manifold defect at t=6 |",
        "|---|---|",
    ]
    for g, d in curves.items():
        lines.append(f"| {g} | {d[-1]:.3e} |")
    lines.append("")
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[statespace] wrote artefacts to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
