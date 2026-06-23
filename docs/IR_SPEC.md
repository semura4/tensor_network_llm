# EO Pulse Control IR — specification (v0.3)

A small, stable contract so other tools (a circuit front end, an optimiser,
`eoqrid`, a Blueqat back end, a cryo-CMOS controller, a place-and-route layer)
can produce and consume the exchange-only pulse intermediate representation.
Everything here is plain JSON/CSV; the core library reads/writes it with the
standard library only.

## Levels

```
logical circuit  ──►  topology + layout  ──►  pulse IR (schedule)  ──►  controller memory
 (OpenQASM/Blueqat)    (1-D chain / 2-D grid)   (pulse_timeline.json)   (instruction/pattern CSV)
        ▲                      ▲                       ▲  ▲
        │                      │                       │  └── external optimiser / eoqrid (pulse records)
   adapters.qasm       GridTopology /             adapters.external / adapters.eoqrid
   adapters.blueqat    LinearTopology
```

## 1. Logical circuit (front end)

QASM-lite / OpenQASM-2 subset, plus the EO-native `cxswap` and arbitrary
`u(theta,phi,lam)`. Gates: `x y z h s sdg t tdg rz rx ry u` (1-qubit),
`cx/cnot swap cxswap` (2-qubit). Parsed by `eo_pulse_ir.parse_circuit`; emitted by
`eo_pulse_ir.adapters.qasm.to_openqasm`.

## 2. Topology and layout

Two physical topologies are supported:

- **`LinearTopology(num_qubits)`** — 1-D chain.  Each logical qubit occupies
  3 consecutive dots.  Routing uses nearest-neighbour bubble SWAPs.

- **`GridTopology(rows, cols)`** — 2-D square-lattice patch.  Each grid slot
  holds one encoded qubit (3 dots).  Adjacent slots on the grid share an
  endpoint-to-endpoint inter-group edge (dot `3*s_a+2` ↔ dot `3*s_b`).

  **Initial layout**: `grid.assign_initial_layout(qubits, circuit_gates=...)`
  uses an interaction-weighted greedy heuristic (most-connected-first placement,
  minimising weighted Manhattan distance) with pair-swap local-search
  refinement.  No external dependencies (stdlib only; no CP-SAT optimality
  proof — use the `adapters.external` seam to ingest an externally-optimised
  layout if needed).

  **Routing**: BFS shortest-path encoded-SWAP chains on the grid.

  **Scheduling**: the existing ASAP dot-availability scheduler handles 2-D
  layouts automatically (pulses on well-separated edges parallelise for free).

Pass the topology to `compile_circuit(circuit, topology=grid)` or
`synthesize(circuit, topology=grid)`.

## 3. Pulse IR — `pulse_timeline.json`

The canonical interchange object. A pulse is one square exchange pulse on one
nearest-neighbour edge.

```json
{
  "num_dots": 6,
  "makespan": 40.84,
  "j_max": 1.0,
  "pulse_count": 22,
  "pulses": [
    {"edge": [1, 2], "output": "J1_2", "gate": "h", "role": "intra",
     "logical_qubits": [0], "start": 0.0, "duration": 1.41,
     "area": 1.4155, "j": 1.0}
  ]
}
```

Field contract (per pulse): `edge` = `[low, high]` adjacent dots; `area` = exchange
action ∫J dt in radians (full SWAP = π); `role` ∈ {`intra`,`inter`,`route`};
`start`/`duration`/`j` filled by the scheduler. **`edge` and `area` are the only
required fields to ingest** (see §5).

## 4. Controller memory (back end) — HRL-style cryo-CMOS

`instruction_memory.csv`: `addr, sequencer, output, op, pattern_id,
t_start_samples, width_samples, gate`.
`pattern_memory.csv`: `pattern_id, output, sequencer, dac_code, voltage,
width_samples, exchange_area`.
Pattern words carry a DAC code (from `J(V)=j0·exp((V−v0)/vc)`) and a pulse width;
budgeted against `num_sequencers`, `instruction_memory_words`,
`pattern_memory_words_per_output` (see `HardwareConfig`).

## 5. External pulse records (optimiser / eoqrid seam)

To drive the IR with calibrated pulses, supply a list of records — each needs at
least `edge` and `area`; `gate`, `role`, `logical_qubits` are optional:

```json
{"num_dots": 6,
 "pulses": [{"edge": [2,3], "area": 3.14159, "role": "inter", "gate": "cx"}]}
```

Ingest with `eo_pulse_ir.compile_pulse_records(records, num_dots)` or
`adapters.external.pulse_records_to_result`. This is the integration point for an
external optimiser or `eoqrid`'s pulse output: produce this shape and the whole
cost/scheduling/hardware/visualisation pipeline runs on it unchanged.

## 6. eoqrid adapter

`adapters.eoqrid.from_eoqrid(qc_native)` converts an eoqrid-transpiled
Qiskit `QuantumCircuit` (containing `Ex` gates) to pulse records (§5 format).
`adapters.eoqrid.compile_eoqrid(qc_native)` does the conversion and full
compilation in one step.  The Qiskit import is guarded — the adapter loads
without third-party packages and only requires Qiskit at call time.

## 7. Robust-cost-driven place-and-route (`gate_library`)

The place-and-route runs on a **gate library** — a mapping
`{gate_name: [(role, area), ...]}` of per-gate pulse sequences.  By default it
uses the built-in simulator-validated templates (nominal areas).  Pass a
`gate_library` to `synthesize` / `compile_circuit` to swap in **device-calibrated,
valley-robust** areas instead:

```python
from eo_pulse_ir.sim.robust import robust_gate_library
lib = robust_gate_library(sigma=0.0091, spread=0.3, gate_names=("cx", "swap"))
result = compile_circuit(circuit, topology=grid, gate_library=lib)
```

The role sequence is unchanged, so cost/scheduling/memory are unaffected in
structure — only the areas (and hence the noise response) change.  The library's
`"swap"` entry is also used for routing SWAPs, so 2-D routing inherits the robust
cost automatically.  This is the differentiator vs a place-and-route that costs
gates by a fixed integer: the layout and the end-to-end fidelity estimate run on
real, device-calibrated, valley-robust pulse costs (`scripts/eo_grid_robust.py`).
Building `robust_gate_library` requires the numpy simulator; applying it does not.

## 8. Integration with external place-and-route tools

The IR also interoperates with external place-and-route layers such as
[exchange-pulse-optimizer](https://github.com/kaluza1/exchange-pulse-optimizer)
(CP-SAT exact optimisation on 2-D square lattices) and front ends such as
[eoqrid](https://github.com/samn33/eoqrid) (§6).  The recommended seam:

1. Use the external tool for layout + routing (it produces a macro-level pulse
   plan with fixed integer costs like `cx=28`).
2. Replace those fixed costs with our calibrated, valley-robust pulse areas
   by feeding the macro plan through `adapters.external.pulse_records_to_result`.
3. The rest of the IR pipeline (scheduling, hardware memory, metrics,
   visualisation) runs unchanged on the externally-optimised plan.

This keeps the IR dependency-light while allowing full CP-SAT optimality when
the external tool is available.

## Versioning

This document is `v0.3`. Backwards-incompatible field changes bump the minor
version; new optional fields do not. The Python API mirrors the spec
(`eo_pulse_ir.__version__`).
