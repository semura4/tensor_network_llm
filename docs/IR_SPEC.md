# EO Pulse Control IR — specification (v0.1)

A small, stable contract so other tools (a circuit front end, an optimiser,
`eoqrid`, a Blueqat back end, a cryo-CMOS controller) can produce and consume the
exchange-only pulse intermediate representation. Everything here is plain
JSON/CSV; the core library reads/writes it with the standard library only.

## Levels

```
logical circuit  ──►  pulse IR (schedule)  ──►  controller memory
 (OpenQASM/Blueqat)    (pulse_timeline.json)     (instruction/pattern CSV)
        ▲                      ▲  ▲
        │                      │  └── external optimiser / eoqrid (pulse records)
   adapters.qasm          adapters.external
```

## 1. Logical circuit (front end)

QASM-lite / OpenQASM-2 subset, plus the EO-native `cxswap` and arbitrary
`u(theta,phi,lam)`. Gates: `x y z h s sdg t tdg rz rx ry u` (1-qubit),
`cx/cnot swap cxswap` (2-qubit). Parsed by `eo_pulse_ir.parse_circuit`; emitted by
`eo_pulse_ir.adapters.qasm.to_openqasm`.

## 2. Pulse IR — `pulse_timeline.json`

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
required fields to ingest** (see §4).

## 3. Controller memory (back end) — HRL-style cryo-CMOS

`instruction_memory.csv`: `addr, sequencer, output, op, pattern_id,
t_start_samples, width_samples, gate`.
`pattern_memory.csv`: `pattern_id, output, sequencer, dac_code, voltage,
width_samples, exchange_area`.
Pattern words carry a DAC code (from `J(V)=j0·exp((V−v0)/vc)`) and a pulse width;
budgeted against `num_sequencers`, `instruction_memory_words`,
`pattern_memory_words_per_output` (see `HardwareConfig`).

## 4. External pulse records (optimiser / eoqrid seam)

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

## Versioning

This document is `v0.1`. Backwards-incompatible field changes bump the minor
version; new optional fields do not. The Python API mirrors the spec
(`eo_pulse_ir.__version__`).
