"""Single-spin silicon control-resource and QSoC memory-budget model.

A first-order (phenomenological) budget for the *control plane* of a single-
electron-spin silicon spin-qubit array — the Loss–DiVincenzo scheme that the
AIST/Blueqat "Open Silicon Quantum" devices use: **1 quantum dot = 1 qubit,
microwave EDSR single-qubit gates, baseband exchange two-qubit CZ**.  (This is a
different encoding from the exchange-only, 3-dot logical qubit the rest of
``eo_pulse_ir`` synthesises; see ``docs/blueqat_contribution_memo.md``.)

It answers the *wiring-bottleneck* question on Blueqat's public roadmap
(room-temperature electronics in 2026, cryo-CMOS forced by interconnect limits
around 2027–28): how three control-plane costs scale with qubit count ``N`` and
where a room-temperature, one-analog-line-per-control architecture must give way
to a **cryo-CMOS QSoC** (cold pattern/instruction memory with local waveform
replay).

What it reuses from the rest of the package
-------------------------------------------
The baseband (CZ) channel is modelled with the **same** exchange–voltage law
``J(V) = j0·exp((V−v0)/vc)``, DAC quantisation, and pattern/instruction-memory
abstraction as :mod:`eo_pulse_ir.hardware`; :func:`cross_check_against_build_memory`
confirms the per-CZ word counts here match :func:`eo_pulse_ir.hardware.build_memory`
exactly.  The microwave (EDSR) channel is the single-spin-specific addition.

Scope and honesty
-----------------
This is a *budget* model, not a device simulation.  Line counts, sample rates,
and the cold-power figure are first-order and fully parameterised; the value is
the **scaling** and the **room-temp vs cryo-CMOS crossover**, not absolute watts.
Defaults are representative of the spin-qubit-interconnect literature
(Vandersypen et al., "Interfacing spin qubits … hot, dense, and coherent",
npj QI 3, 34 (2017); Boter et al., "Spiderweb array", PRApplied 18, 024053
(2022); cryo-CMOS controllers e.g. Intel "Horse Ridge").  Every number is a knob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hardware import HardwareConfig, dac_code_for_j, voltage_for_j


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class SingleSpinControl:
    """Per-qubit control wiring and per-gate waveform model for a 2-D Si array."""

    # --- control lines per qubit (2-D square array, shared barriers) ---
    plunger_lines_per_qubit: float = 1.0      # one plunger (chemical potential) per dot
    barrier_lines_per_qubit: float = 2.0      # ~2 exchange bonds per site in a 2-D array
    microwave_lines_per_qubit: float = 1.0    # dedicated EDSR drive (room-temp worst case)
    microwave_shared: bool = False            # if True, frequency-multiplex onto few lines
    microwave_shared_lines: int = 4           # global EDSR feed lines when shared
    readout_mux_factor: int = 16              # qubits per shared RF-reflectometry line

    # --- baseband (plunger / barrier / CZ) waveform ---
    baseband_sample_ghz: float = 1.0          # GSa/s for baseband AWG streaming
    dac_bits: int = 14

    # --- microwave EDSR envelope ---
    mw_envelope_sample_ghz: float = 0.5       # GSa/s of the IQ baseband envelope
    mw_iq_channels: int = 2                   # I and Q
    mw_pi_ns: float = 100.0                   # EDSR pi-pulse duration (Rabi ~ a few MHz)
    mw_segments_per_gate: int = 12            # piecewise-constant pattern segments / shaped pulse
    distinct_1q_gates: int = 3                # gate-shape variety in the basis (e.g. X/2, Y/2, idle)

    # --- exchange CZ (baseband) ---
    cz_area: float = math.pi                  # controlled-phase via ~pi exchange action
    cz_segments_per_gate: int = 1             # square exchange pulse = 1 pattern segment
    distinct_2q_gates: int = 1                # just CZ

    # --- room-temperature shared-control (crossbar) variant ---
    crossbar_line_coeff: float = 2.0          # lines ~ coeff * sqrt(N) (row + column rails)

    # --- cryo-CMOS QSoC interface ---
    cryo_digital_bus_lines: int = 40          # shared data+addr+ctrl bus across the cold I/O
    cryo_power_clock_lines: int = 20          # shared power rails + clock/reference lines

    # --- cold-stage thermal budget (the binding constraint at sub-K) ---
    # Spin qubits need ~0.3 K (sub-K) for high-fidelity operation; co-locating
    # control there means a *tiny* cooling budget.  Dilution-fridge cooling power
    # scales ~ T^2, anchored at ~1 mW @ 100 mK (commercial; ~2 mW best-in-class).
    operating_temp_k: float = 0.3             # the temperature qubits actually need
    cooling_ref_w: float = 1.0e-3             # cooling power at the reference temperature
    cooling_ref_temp_k: float = 0.1           # reference (100 mK)
    # Per-control-channel cold dissipation.  Today's cryo-CMOS controllers run
    # ~9-10 mW/channel (RF front end + logic); 1 mW/qubit is a near-term target.
    cold_power_uW_per_qubit: float = 1000.0

    # --- fridge / interface limits ---
    fridge_port_limit: int = 500              # practical analog-coax count at the cold plate
    interface_bandwidth_limit_gbps: float = 1000.0  # aggregate 300 K -> cold data budget

    @property
    def cooling_budget_w(self) -> float:
        """Available cooling at ``operating_temp_k`` (dilution-fridge T^2 scaling)."""
        return self.cooling_ref_w * (self.operating_temp_k / self.cooling_ref_temp_k) ** 2

    # --- readout ---
    readout_bits_per_qubit: float = 1.0       # classified syndrome bit per qubit per cycle


@dataclass
class Workload:
    """A representative syndrome-extraction (QEC) cycle on a 2-D lattice."""

    single_qubit_gates_per_cycle: float = 2.0  # EDSR gates per qubit per cycle
    cz_bonds_per_qubit: float = 2.0            # avg exchange bonds per site (2-D square)
    cz_layers_per_cycle: int = 4               # surface-code interaction layers
    cycle_time_ns: float = 1000.0             # one syndrome-extraction cycle


# ---------------------------------------------------------------------------
# Control-line wiring
# ---------------------------------------------------------------------------

def control_lines(n: int, ctrl: SingleSpinControl, mode: str) -> Dict[str, float]:
    """Analog/digital line count crossing into the cold stage for ``n`` qubits.

    ``mode`` ∈ {"roomtemp", "crossbar", "cryo"}.
    """
    readout = math.ceil(n / max(ctrl.readout_mux_factor, 1))

    if mode == "roomtemp":
        mw = (ctrl.microwave_shared_lines if ctrl.microwave_shared
              else n * ctrl.microwave_lines_per_qubit)
        plunger = n * ctrl.plunger_lines_per_qubit
        barrier = n * ctrl.barrier_lines_per_qubit
        total = plunger + barrier + mw + readout
        return {"plunger": plunger, "barrier": barrier, "microwave": mw,
                "readout": readout, "total": total}

    if mode == "crossbar":
        # Shared row/column rails: ~coeff*sqrt(N) gate lines, at the cost of only
        # being able to apply the *same* operation along a shared line at once.
        rails = ctrl.crossbar_line_coeff * math.sqrt(max(n, 1))
        mw = (ctrl.microwave_shared_lines if ctrl.microwave_shared
              else ctrl.crossbar_line_coeff * math.sqrt(max(n, 1)))
        total = rails + mw + readout
        return {"gate_rails": rails, "microwave": mw, "readout": readout,
                "total": total}

    if mode == "cryo":
        # Waveform generation moves cold; only a shared digital bus, power, clocks
        # and (multiplexed) readout return cross the 300 K -> cold boundary.
        total = ctrl.cryo_digital_bus_lines + ctrl.cryo_power_clock_lines + readout
        return {"digital_bus": float(ctrl.cryo_digital_bus_lines),
                "power_clock": float(ctrl.cryo_power_clock_lines),
                "readout": float(readout), "total": float(total)}

    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Cross-interface data rate
# ---------------------------------------------------------------------------

def interface_bandwidth_gbps(n: int, ctrl: SingleSpinControl, work: Workload,
                             mode: str) -> float:
    """Aggregate data rate (Gb/s) that must cross the 300 K -> cold interface."""
    if mode in ("roomtemp", "crossbar"):
        lines = control_lines(n, ctrl, mode)
        # baseband lines streamed at the baseband DAC rate
        if mode == "roomtemp":
            baseband = (lines["plunger"] + lines["barrier"])
        else:
            baseband = lines["gate_rails"]
        baseband_gbps = baseband * ctrl.baseband_sample_ghz * ctrl.dac_bits
        # microwave envelope (I and Q) streamed at the envelope rate
        mw_gbps = (lines["microwave"] * ctrl.mw_envelope_sample_ghz
                   * ctrl.dac_bits * ctrl.mw_iq_channels)
        return baseband_gbps + mw_gbps

    if mode == "cryo":
        # Waveforms are cold-resident and replayed locally; at steady state the
        # only runtime cross-interface traffic is the classified readout stream
        # out plus a sparse trigger/sync stream in — both ~ bits, not samples.
        readout_out_gbps = (n * ctrl.readout_bits_per_qubit
                            / work.cycle_time_ns)  # bits/ns = Gb/s
        # one sync token per interaction layer per cycle (tiny)
        sync_in_gbps = (work.cz_layers_per_cycle + 1) * 32 / work.cycle_time_ns
        return readout_out_gbps + sync_in_gbps

    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Cold memory (cryo-CMOS only)
# ---------------------------------------------------------------------------

def distinct_pattern_words(ctrl: SingleSpinControl) -> Dict[str, int]:
    """Size of the shared cold *waveform bank*, in pattern words.

    Because every qubit's X/2 (etc.) pulse is identical and every CZ is identical,
    the distinct waveform *shapes* — not the qubit count — set the pattern memory.
    A shared bank therefore stays **constant in N**: the central QSoC advantage,
    and exactly the deduplication :func:`eo_pulse_ir.hardware.build_memory` performs.
    """
    mw = ctrl.distinct_1q_gates * ctrl.mw_segments_per_gate * ctrl.mw_iq_channels
    cz = ctrl.distinct_2q_gates * ctrl.cz_segments_per_gate
    return {"microwave": mw, "exchange": cz, "total": mw + cz}


def cold_memory(n: int, ctrl: SingleSpinControl, work: Workload,
                hw: HardwareConfig) -> Dict[str, float]:
    """Cold pattern-bank size, instruction words per cycle, and sequencer demand."""
    patterns = distinct_pattern_words(ctrl)

    # operations in one syndrome-extraction cycle
    cz_ops = n * work.cz_bonds_per_qubit / 2.0          # each bond is one shared CZ
    oneq_ops = n * work.single_qubit_gates_per_cycle
    readout_ops = float(n)
    total_ops = cz_ops + oneq_ops + readout_ops

    instr_words_per_cycle = (total_ops * hw.words_per_pulse
                             + (work.cz_layers_per_cycle
                                + math.ceil(work.single_qubit_gates_per_cycle) + 1)
                             * hw.sync_words_per_layer)

    # full-parallelism sequencer demand = busiest layer's active channels.
    # the single-qubit layer drives (almost) every qubit at once.
    peak_parallel = max(n, cz_ops / max(work.cz_layers_per_cycle, 1))

    return {
        "pattern_bank_words": float(patterns["total"]),
        "instr_words_per_cycle": float(instr_words_per_cycle),
        "ops_per_cycle": float(total_ops),
        "sequencers_full_parallel": float(peak_parallel),
        "instr_cycles_per_load": (hw.instruction_memory_words
                                  / instr_words_per_cycle if instr_words_per_cycle else 0.0),
    }


def cold_power_w(n: int, ctrl: SingleSpinControl) -> float:
    """Cryo-CMOS dissipation at the cold control stage (W)."""
    return n * ctrl.cold_power_uW_per_qubit * 1e-6


# ---------------------------------------------------------------------------
# Adiabatic (energy-recovery) switching — the lever for the cold-power wall
# ---------------------------------------------------------------------------

@dataclass
class SwitchingModel:
    """Cold control-node switching energy: conventional CV^2 vs adiabatic recovery.

    Conventional CMOS dissipates the full ``C·V^2`` per charge/discharge of a
    control node, independent of speed.  **Adiabatic (energy-recovery) logic**
    ramps the node slowly (constant-current-like) over ``ramp_time_ns``, recovering
    most of the energy: dissipation ≈ ``C·V^2 · (τ_RC / T_ramp)`` for
    ``T_ramp ≫ τ_RC``, floored by non-idealities at ``adiabatic_floor_factor·C·V^2``.

    Spin qubits gate slowly (~100 ns–1 µs), so adiabatic ramps (≫ the few-ns
    τ_RC) are *naturally* compatible — unlike fast superconducting gates.  This is
    why energy-recovery control is a realistic lever for the sub-K cold-power wall.
    """

    node_capacitance_f: float = 0.5e-12       # control-node capacitance (gate line + driver)
    swing_v: float = 0.5                       # full-swing control voltage
    switch_resistance_ohm: float = 5.0e3       # cryo switch on-resistance
    switching_events_per_qubit_per_cycle: float = 8.0  # charge/discharge events / cycle
    ramp_time_ns: float = 200.0                # adiabatic ramp time (≫ τ_RC)
    adiabatic_floor_factor: float = 0.01       # best recoverable fraction (non-idealities)

    @property
    def cv2_j(self) -> float:
        """Conventional full-swing switching energy C·V^2 (J)."""
        return self.node_capacitance_f * self.swing_v ** 2

    @property
    def rc_time_ns(self) -> float:
        return self.switch_resistance_ohm * self.node_capacitance_f * 1e9


def op_energy_j(sw: SwitchingModel, mode: str, ramp_time_ns: Optional[float] = None) -> float:
    """Energy dissipated per switching event (J) for ``mode`` ∈ {conventional, adiabatic}."""
    if mode == "conventional":
        return sw.cv2_j
    if mode == "adiabatic":
        t = sw.ramp_time_ns if ramp_time_ns is None else ramp_time_ns
        factor = max(sw.rc_time_ns / t, sw.adiabatic_floor_factor) if t > 0 else 1.0
        return sw.cv2_j * factor
    raise ValueError(f"unknown mode {mode!r}")


def dynamic_cold_power_per_qubit_w(sw: SwitchingModel, work: Workload, mode: str,
                                   ramp_time_ns: Optional[float] = None) -> float:
    """Per-qubit *dynamic* (switching) cold power (W) for the workload's cycle rate."""
    e = op_energy_j(sw, mode, ramp_time_ns)
    cycle_s = work.cycle_time_ns * 1e-9
    return sw.switching_events_per_qubit_per_cycle * e / cycle_s if cycle_s else 0.0


def max_qubits_dynamic_power(sw: SwitchingModel, work: Workload,
                             ctrl: SingleSpinControl, mode: str,
                             ramp_time_ns: Optional[float] = None) -> int:
    """Max N at ``operating_temp_k`` limited by *dynamic* switching power alone."""
    p = dynamic_cold_power_per_qubit_w(sw, work, mode, ramp_time_ns)
    if p <= 0:
        return -1
    return int(ctrl.cooling_budget_w / p)


def landauer_floor_j(ctrl: SingleSpinControl) -> float:
    """Landauer limit kT·ln2 per irreversible bit operation at ``operating_temp_k`` (J)."""
    return 1.380649e-23 * ctrl.operating_temp_k * math.log(2)


# ---------------------------------------------------------------------------
# Aggregate budget + crossovers
# ---------------------------------------------------------------------------

def qsoc_budget(n: int, ctrl: SingleSpinControl, work: Workload,
                hw: HardwareConfig) -> Dict[str, dict]:
    """Full control-plane budget for ``n`` qubits across all three architectures."""
    out: Dict[str, dict] = {"n": n}
    for mode in ("roomtemp", "crossbar", "cryo"):
        out[mode] = {
            "lines": control_lines(n, ctrl, mode),
            "interface_gbps": interface_bandwidth_gbps(n, ctrl, work, mode),
        }
    out["cryo"]["cold_memory"] = cold_memory(n, ctrl, work, hw)
    out["cryo"]["cold_power_w"] = cold_power_w(n, ctrl)
    out["cryo"]["cooling_budget_w"] = ctrl.cooling_budget_w
    out["cryo"]["cold_power_fits"] = cold_power_w(n, ctrl) <= ctrl.cooling_budget_w
    return out


def crossover_wire_limit(ctrl: SingleSpinControl, mode: str = "roomtemp") -> int:
    """Smallest N at which ``mode``'s line count exceeds the fridge port limit."""
    n = 1
    while n < 10_000_000:
        if control_lines(n, ctrl, mode)["total"] > ctrl.fridge_port_limit:
            return n
        n = n * 2 if n >= 8 else n + 1
    return -1


def crossover_cold_power(ctrl: SingleSpinControl) -> int:
    """Largest N whose cold dissipation fits the cooling budget at ``operating_temp_k``."""
    per = ctrl.cold_power_uW_per_qubit * 1e-6
    if per <= 0:
        return -1
    return int(ctrl.cooling_budget_w / per)


def max_qubits_cold_power(ctrl: SingleSpinControl, per_qubit_w: float) -> int:
    """Max N supportable at ``operating_temp_k`` given per-qubit cold dissipation (W)."""
    if per_qubit_w <= 0:
        return -1
    return int(ctrl.cooling_budget_w / per_qubit_w)


def required_cold_power_w_per_qubit(n_target: int, ctrl: SingleSpinControl) -> float:
    """Per-qubit cold dissipation (W) needed to reach ``n_target`` at ``operating_temp_k``."""
    if n_target <= 0:
        return float("inf")
    return ctrl.cooling_budget_w / n_target


# ---------------------------------------------------------------------------
# Anchor to the real controller-memory model
# ---------------------------------------------------------------------------

def cross_check_against_build_memory(ctrl: SingleSpinControl,
                                     hw: Optional[HardwareConfig] = None) -> dict:
    """Confirm the baseband CZ word counts match :func:`hardware.build_memory`.

    Builds a tiny exchange schedule of identical CZ pulses on distinct barrier
    channels and checks that (a) the DAC code / voltage for the CZ amplitude come
    out of the same ``J(V)`` model, and (b) every identical CZ deduplicates to a
    single pattern word per output — i.e. the cold pattern bank is reused, exactly
    as this module's :func:`distinct_pattern_words` assumes.
    """
    from .schedule import Pulse, schedule_pulses
    from .hardware import build_memory

    hw = hw or HardwareConfig()
    # three CZ pulses on three separate nearest-neighbour barrier edges
    pulses = [
        Pulse(edge=(0, 1), area=ctrl.cz_area, gate="cz", logical_qubits=(0, 1), role="inter"),
        Pulse(edge=(2, 3), area=ctrl.cz_area, gate="cz", logical_qubits=(2, 3), role="inter"),
        Pulse(edge=(4, 5), area=ctrl.cz_area, gate="cz", logical_qubits=(4, 5), role="inter"),
    ]
    sched = schedule_pulses(pulses, num_dots=6, j_max=hw.j_max)
    mem = build_memory(sched, hw)
    return {
        "cz_dac_code": dac_code_for_j(hw.j_max, hw),
        "cz_voltage": round(voltage_for_j(hw.j_max, hw), 6),
        "pattern_words_per_output": dict(mem.pattern_words_per_output),
        "all_outputs_single_pattern": all(
            v == ctrl.cz_segments_per_gate for v in mem.pattern_words_per_output.values()),
        "instruction_words_used": mem.instruction_words_used,
    }
