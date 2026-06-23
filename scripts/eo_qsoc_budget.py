#!/usr/bin/env python3
"""QSoC control-memory & wiring budget for a single-spin silicon qubit array.

Contribution-path item #1 from ``docs/blueqat_contribution_memo.md``: a control-
plane budget for the AIST/Blueqat-style single-electron-spin silicon processor
(1 dot = 1 qubit, EDSR single-qubit gates, baseband exchange CZ).  It scales
three costs with qubit count N and locates the room-temperature -> cryo-CMOS
crossover that Blueqat's roadmap puts around 2027–28:

1. **Control-line count** crossing into the cold stage — room-temp (one analog
   line per control) grows ~linearly and hits the fridge port limit; a crossbar
   softens it to ~sqrt(N) (at the cost of parallelism); a cryo-CMOS QSoC keeps it
   ~constant (shared digital bus).
2. **Cross-interface data rate** — room-temp streams every waveform sample cold;
   the QSoC replays cold-resident patterns locally, so only the (bit-rate)
   readout/trigger stream crosses the boundary.
3. **Cold memory** — the QSoC's new constraint: the shared waveform bank is
   constant in N (gate-shape reuse), while instruction words grow ~linearly,
   setting the reload cadence.  Cold power then becomes the ultimate limit.

The baseband CZ word counts are anchored to the real controller-memory model
(``eo_pulse_ir.hardware.build_memory``) via a cross-check.

Outputs to out/qsoc_budget/ : qsoc_lines.svg, qsoc_bandwidth.svg,
qsoc_cold_memory.svg, report.md, results.json.  Requires numpy (for SVG only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eo_pulse_ir.hardware import HardwareConfig
from eo_pulse_ir.singlespin import (
    SingleSpinControl, Workload, qsoc_budget, distinct_pattern_words,
    crossover_wire_limit, crossover_cold_power, cross_check_against_build_memory,
    max_qubits_cold_power, required_cold_power_w_per_qubit,
)


def _fmt(n: float) -> str:
    """Human-readable numbers across many orders of magnitude."""
    if n >= 1e9:
        return f"{n/1e9:.1f}G"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}k"
    if n >= 1 or n == 0:
        return f"{n:.0f}"
    if n >= 0.01:
        return f"{n:.2f}"
    return f"{n:.0e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out-dir", default="out/qsoc_budget")
    ap.add_argument("--n-max-exp", type=int, default=20,
                    help="sweep N up to 2**this (default 2**20 ~ 1e6 qubits)")
    ap.add_argument("--mw-shared", action="store_true",
                    help="frequency-multiplex microwave onto a few global lines")
    ap.add_argument("--quick", action="store_true", help="tiny budget for CI smoke")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.quick:
        args.n_max_exp = 12

    ctrl = SingleSpinControl(microwave_shared=args.mw_shared)
    work = Workload()
    hw = HardwareConfig()

    # --- anchor to the real controller-memory model ---
    check = cross_check_against_build_memory(ctrl, hw)
    print(f"[qsoc] CZ DAC code={check['cz_dac_code']} V={check['cz_voltage']} "
          f"pattern/output={check['pattern_words_per_output']} "
          f"dedup_ok={check['all_outputs_single_pattern']}", flush=True)

    # --- sweep N (powers of two) ---
    ns = [2 ** k for k in range(2, args.n_max_exp + 1)]
    rows = [qsoc_budget(n, ctrl, work, hw) for n in ns]

    # crossovers
    n_wire_rt = crossover_wire_limit(ctrl, "roomtemp")
    n_wire_xb = crossover_wire_limit(ctrl, "crossbar")
    n_power = crossover_cold_power(ctrl)
    budget_w = ctrl.cooling_budget_w

    print(f"[qsoc] operating T={ctrl.operating_temp_k} K -> cooling budget "
          f"{budget_w*1e3:.1f} mW", flush=True)
    print(f"[qsoc] room-temp wire limit N={n_wire_rt}, crossbar N={_fmt(n_wire_xb)}, "
          f"cold-power limit N={_fmt(n_power)} @ {ctrl.cold_power_uW_per_qubit:.0f} uW/qubit",
          flush=True)

    # per-qubit cold-power scenarios (today -> target) and the required power for scale
    power_scenarios_uw = [10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01]
    scenario_rows = [(p, max_qubits_cold_power(ctrl, p * 1e-6)) for p in power_scenarios_uw]
    req_1k = required_cold_power_w_per_qubit(1000, ctrl)
    req_1m = required_cold_power_w_per_qubit(1_000_000, ctrl)

    # --- plots (numpy SVG helper) ---
    try:
        from eo_pulse_ir.sim.landscape import line_plot_svg, write_svg
        have_plot = True
    except Exception:
        have_plot = False

    if have_plot:
        xs = [float(n) for n in ns]
        # 1. line count
        write_svg(line_plot_svg(
            [("room-temp (1 line/control)", xs, [r["roomtemp"]["lines"]["total"] for r in rows]),
             ("crossbar (~sqrt N)", xs, [r["crossbar"]["lines"]["total"] for r in rows]),
             ("cryo-CMOS QSoC", xs, [r["cryo"]["lines"]["total"] for r in rows]),
             ("fridge port limit", xs, [ctrl.fridge_port_limit for _ in ns])],
            title="Control lines crossing into the cold stage",
            xlabel="qubits N", ylabel="line count", logx=True, logy=True),
            os.path.join(args.out_dir, "qsoc_lines.svg"))

        # 2. cross-interface bandwidth
        write_svg(line_plot_svg(
            [("room-temp (stream samples)", xs, [r["roomtemp"]["interface_gbps"] for r in rows]),
             ("crossbar", xs, [r["crossbar"]["interface_gbps"] for r in rows]),
             ("cryo-CMOS (replay local)", xs, [r["cryo"]["interface_gbps"] for r in rows]),
             ("interface budget", xs, [ctrl.interface_bandwidth_limit_gbps for _ in ns])],
            title="300 K -> cold interface data rate",
            xlabel="qubits N", ylabel="Gb/s", logx=True, logy=True),
            os.path.join(args.out_dir, "qsoc_bandwidth.svg"))

        # 3. cold memory
        write_svg(line_plot_svg(
            [("pattern bank (shared, const)", xs,
              [r["cryo"]["cold_memory"]["pattern_bank_words"] for r in rows]),
             ("instruction words / cycle", xs,
              [r["cryo"]["cold_memory"]["instr_words_per_cycle"] for r in rows]),
             ("instruction capacity", xs,
              [hw.instruction_memory_words for _ in ns])],
            title="Cryo-CMOS cold memory vs N",
            xlabel="qubits N", ylabel="words", logx=True, logy=True),
            os.path.join(args.out_dir, "qsoc_cold_memory.svg"))

        # 4. cold power vs the sub-K cooling budget (the binding wall)
        write_svg(line_plot_svg(
            [(f"today ~10 mW/qubit", xs, [n * 10e-3 for n in ns]),
             (f"near-term 1 mW/qubit", xs, [n * 1e-3 for n in ns]),
             (f"target 10 nW/qubit", xs, [n * 10e-9 for n in ns]),
             (f"{ctrl.operating_temp_k} K cooling budget", xs, [budget_w for _ in ns])],
            title=f"Cold dissipation vs cooling budget at {ctrl.operating_temp_k} K",
            xlabel="qubits N", ylabel="cold power (W)", logx=True, logy=True),
            os.path.join(args.out_dir, "qsoc_cold_power.svg"))

    # --- results.json ---
    results = {
        "params": {
            "control": ctrl.__dict__, "workload": work.__dict__,
            "hardware": {k: getattr(hw, k) for k in
                         ("j_max", "clock_ghz", "num_sequencers",
                          "instruction_memory_words", "pattern_memory_words_per_output",
                          "dac_bits", "vc")},
        },
        "cross_check": check,
        "operating_temp_k": ctrl.operating_temp_k,
        "cooling_budget_w": budget_w,
        "crossovers": {"roomtemp_wire": n_wire_rt, "crossbar_wire": n_wire_xb,
                       "cryo_cold_power": n_power},
        "cold_power_scenarios": [
            {"per_qubit_uw": p, "max_qubits": n} for p, n in scenario_rows],
        "required_cold_power_w_per_qubit": {"N_1e3": req_1k, "N_1e6": req_1m},
        "pattern_bank": distinct_pattern_words(ctrl),
        "sweep": rows,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # --- report.md ---
    pick = [n for n in (16, 256, 4096, 65536, 1048576) if n <= ns[-1]]
    pick_rows = {r["n"]: r for r in rows if r["n"] in pick}

    lines = [
        "# QSoC control-plane budget — single-spin silicon array",
        "",
        "First-order control-plane budget for a single-electron-spin silicon",
        "processor (1 dot = 1 qubit, EDSR 1-qubit gates, baseband exchange CZ) —",
        "contribution item #1 from `docs/blueqat_contribution_memo.md`.  It scales",
        "the control plane with qubit count and finds that at the **sub-K operating",
        f"temperature spin qubits actually need (~{ctrl.operating_temp_k} K)**, the",
        "binding constraint is **cold power**, not the room-temp wiring wall behind",
        "Blueqat's 2027–28 interconnect roadmap.",
        "",
        "## Anchor to the controller-memory model",
        "",
        f"The CZ baseband channel uses the same `J(V)=j0·exp((V−v0)/vc)` DAC model",
        f"as `eo_pulse_ir.hardware`: CZ amplitude -> DAC code "
        f"**{check['cz_dac_code']}** at **{check['cz_voltage']} V**.  Identical CZs",
        f"deduplicate to one pattern word per output "
        f"(`build_memory` cross-check: {check['all_outputs_single_pattern']}), which",
        "is why the cold waveform bank is constant in N.",
        "",
        "## The binding constraint: cold power at the real operating temperature",
        "",
        f"High-fidelity spin qubits need **~{ctrl.operating_temp_k} K** (sub-K), not",
        "the 1–4 K \"hot qubit\" regime.  Dilution-fridge cooling power scales ~T^2",
        f"(~1 mW @ 100 mK), so at {ctrl.operating_temp_k} K the budget is only",
        f"**~{budget_w*1e3:.0f} mW** — and today's cryo-CMOS dissipates ~9–10 mW",
        "*per channel*.  Co-locating control at the qubit stage is therefore",
        "thermally bound long before wiring bites:",
        "",
        "| per-qubit cold power | max qubits @ "
        f"{ctrl.operating_temp_k} K | note |",
        "|---:|---:|---|",
    ]
    notes = {10000.0: "today's full cryo-CMOS controller (~10 mW/ch)",
             1000.0: "near-term target", 100.0: "aggressive", 10.0: "very aggressive",
             1.0: "research target", 0.1: "≈ thermal/Landauer floor regime",
             0.01: "≈ thermal/Landauer floor regime"}
    for p, nmax in scenario_rows:
        lines.append(f"| {_fmt(p*1e-6*1e6)} µW | {_fmt(nmax)} | {notes.get(p,'')} |")
    lines += [
        "",
        f"To reach **1k qubits** at {ctrl.operating_temp_k} K needs ≤ "
        f"**{req_1k*1e6:.2f} µW/qubit**; **1M qubits** needs ≤ "
        f"**{req_1m*1e9:.1f} nW/qubit** — a ~10^5–10^6× reduction from today's",
        "per-channel dissipation.  *This*, not wiring, is the dominant wall, and it",
        "is the case for moving as much switching activity off the cold stage as",
        "possible (shared pattern replay, minimal cold dynamic power).",
        "",
        "## Secondary: the wiring wall (room-temp electronics)",
        "",
        f"- **Room-temp (1 analog line / control)** exceeds the "
        f"{ctrl.fridge_port_limit}-port fridge limit at **N = {n_wire_rt}**.",
        f"- **Crossbar (~{ctrl.crossbar_line_coeff:.0f}·√N rails)** pushes that to "
        f"**N = {_fmt(n_wire_xb)}**, at the cost of shared-line operations (parallelism loss).",
        f"- **Cryo-CMOS QSoC** keeps the line count ~constant (shared digital bus),",
        "  converting the wiring problem into the cold-power problem above.",
        "",
        "## Scaling table",
        "",
        "| N | room-temp lines | crossbar lines | cryo lines | room-temp Gb/s | cryo Gb/s | cold pattern words | instr words/cycle | cold power |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in pick:
        r = pick_rows[n]
        cm = r["cryo"]["cold_memory"]
        lines.append(
            f"| {_fmt(n)} | {_fmt(r['roomtemp']['lines']['total'])} "
            f"| {_fmt(r['crossbar']['lines']['total'])} "
            f"| {_fmt(r['cryo']['lines']['total'])} "
            f"| {_fmt(r['roomtemp']['interface_gbps'])} "
            f"| {_fmt(r['cryo']['interface_gbps'])} "
            f"| {_fmt(cm['pattern_bank_words'])} "
            f"| {_fmt(cm['instr_words_per_cycle'])} "
            f"| {r['cryo']['cold_power_w']*1e3:.1f} mW |"
        )

    bank = distinct_pattern_words(ctrl)
    lines += [
        "",
        "## Reading",
        "",
        f"1. **Cold power at {ctrl.operating_temp_k} K is the dominant wall.** The",
        "   sub-K cooling budget is ~mW while today's cold control is ~mW *per",
        f"   channel*, so co-located control tops out at **N ≈ {_fmt(n_power)}** at",
        f"   {ctrl.cold_power_uW_per_qubit:.0f} µW/qubit.  Reaching useful scale",
        f"   demands **nW-class per-qubit cold dissipation** ({req_1m*1e9:.1f} nW for",
        "   1M qubits).  Either control runs ultra-low-power at the qubit stage, or",
        "   it splits to a warmer (4 K, ~1 W) stage and pays a 0.3 K↔4 K wiring and",
        "   heat-leak cost instead — the real architectural fork.",
        "",
        "2. **The wiring wall is the secondary, warmer-stage limit.** A",
        f"   one-line-per-control room-temp architecture saturates a "
        f"{ctrl.fridge_port_limit}-line fridge at **N ≈ {n_wire_rt}**.  Cryo-CMOS",
        "   removes it by replaying cold-resident patterns — but only by taking on",
        "   the cold-power constraint above.",
        "",
        "3. **The QSoC primitives directly target cold power.** Moving waveform",
        "   generation cold makes cross-interface traffic the readout bit-stream",
        "   (orders below sample streaming) and the line count a shared digital bus.",
        f"   The cold waveform bank stays **constant in N** ({bank['total']} words —",
        "   all qubits share identical gate shapes), and shared low-duty-cycle",
        "   replay minimises cold *dynamic* power — exactly the lever the thermal",
        "   wall calls for.",
        "",
        "4. **Instruction memory sets the reload cadence, not the qubit ceiling.**",
        "   Instruction words grow ~linearly with N; against a fixed cold",
        f"   instruction memory ({hw.instruction_memory_words} words) this caps how",
        "   many qubits×cycles run before a reload, a throughput knob rather than a",
        "   hard limit.",
        "",
        "## How this plugs into the eo_pulse_ir stack",
        "",
        "- The CZ channel reuses `HardwareConfig` + the `J(V)` DAC model verbatim,",
        "  so a calibrated CZ pulse from the simulator drops straight into this",
        "  budget (and into `build_memory`).",
        "- The microwave EDSR channel is the only single-spin-specific addition;",
        "  everything else (pattern dedup, instruction/sequencer accounting) is the",
        "  existing controller-memory abstraction.",
        "- Next items in the memo (#2 exchange-CZ robust design, #3 valley leakage)",
        "  feed *calibrated* CZ areas and valley-limited cycle times into this same",
        "  budget, tightening the cold-power and reload numbers with real physics.",
        "",
        "## Scope",
        "",
        "A first-order budget: line counts, sample rates and the cold-power figure",
        "are representative and fully parameterised in `SingleSpinControl` /",
        "`Workload`.  The deliverable is the **scaling** and the **crossover**, not",
        "absolute watts.  Every number is a knob; re-run with measured device",
        "parameters to specialise it.",
        "",
    ]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[qsoc] wrote artefacts to {args.out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
