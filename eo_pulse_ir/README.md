# EO Pulse Control IR

A small, **dependency-free** intermediate representation (IR) and control-cost
evaluator for **exchange-only (EO) spin qubits**. It sits in the gap between EO
*gate synthesis* and an HRL-style *cryo-CMOS controller*, turning a logical
circuit into scheduled exchange pulses and scoring what that schedule costs in
time, parallelism, heuristic leakage/noise, and controller memory.

```
QASM-lite / OpenQASM-2 subset
        │  parse_circuit
        ▼
  logical circuit  ──► EO native gates ──► CX / SWAP / CXSWAP templates
        │  synthesize (+ nearest-neighbour routing SWAPs)
        ▼
  exchange-pulse list  ──► schedule_pulses (ASAP, dot-availability)
        │
        ├─► metrics:  pulse_count · total_time · parallelism · idle ·
        │             estimated_leakage_risk · noise_sensitivity ·
        │             instruction/pattern memory usage · sequencer_conflict
        │
        └─► artefacts: instruction_memory.csv · pattern_memory.csv ·
                       pulse_timeline.json · pulse_timeline.svg ·
                       control_cost_report.md · metrics.json
```

## Why this exists

EO two-qubit gate synthesis, leakage-controlled gates, CXSWAP, and 2-D topology
optimisation are already well studied. The under-served space — and the scope of
this MVP — is the **control / evaluation / visualisation glue** between an EO
gate synthesiser (or `eoqrid` / a pulse optimiser) and a real cryo-CMOS
instruction format. This tool is that glue: a control IR plus a cost evaluator
plus a timeline visualiser.

## Quick start

```bash
# from the repo root (no third-party packages required)
python -m eo_pulse_ir.cli examples/bell.qasm -o out/bell
python -m eo_pulse_ir.cli examples/ghz.qasm  -o out/ghz

# ingest external optimiser / eoqrid pulses instead of synthesising
python -m eo_pulse_ir.cli --pulses examples/external_pulses.json --num-dots 6 -o out/ext
```

Python API:

```python
from eo_pulse_ir import parse_circuit, compile_circuit, emit_artifacts

circ = parse_circuit("qubits 2\nh 0\ncx 0 1\n")
res = compile_circuit(circ)
print(res.metrics.as_dict())
emit_artifacts(res, "out/bell", title="bell")
```

## The physical model (in brief)

- **Encoding.** Each logical qubit is a triple of quantum dots on a 1-D line;
  logical qubit at position `p` owns dots `[3p, 3p+1, 3p+2]`.
- **Control knob.** The only operation is a nearest-neighbour exchange pulse on
  edge `J_{i,i+1}`, modelled as a square pulse of area `A = ∫J dt` (a full SWAP
  is `A = π`). With constant amplitude `j_max`, a pulse lasts `A / j_max`.
- **Scheduling.** ASAP by dot availability: a pulse occupies both its dots for
  its duration, so well-separated edges run in parallel and gates sharing a dot
  serialise. Parallelism in EO control comes out of this for free.
- **Routing.** Non-adjacent two-qubit gates insert logical SWAPs (themselves
  exchange-pulse sequences), so routing cost shows up in the metrics.
- **Controller.** A cryo-CMOS model with `num_sequencers` sequencers, a shared
  instruction memory, and per-output pattern memory whose words carry a DAC code
  (from `J(V) = j0·exp((V−v0)/vc)`) and a pulse width.

## Honesty about the numbers

- **Exact** given a pulse list: every count, time, parallelism, idle figure, and
  all memory-usage / sequencer-conflict numbers.
- **Simulator-validated:** the gate→pulse templates in `native.py`. Every gate's
  pulse areas were optimised against the physics simulator (`sim/`) to reproduce
  the target unitary at F ≥ 0.9999, verified end-to-end through the IR (single-
  qubit gates are exact; the two-qubit gates are listed below).
- **Heuristic proxies (core IR, numpy-free path):** `estimated_leakage_risk` and
  `noise_sensitivity` from `metrics.py` *rank* schedules cheaply; for real
  fidelity/leakage and noise distributions use the `sim/` layer.

## Integration seam

To drive the IR with real, calibrated pulses (from a pulse optimiser or
`eoqrid`), bypass the templates and feed pulse records straight in:

```python
from eo_pulse_ir import compile_pulse_records, emit_artifacts
records = [{"edge": [2, 3], "area": 3.14159, "role": "inter", "gate": "cx"}, ...]
res = compile_pulse_records(records, num_dots=6)
emit_artifacts(res, "out/ext")
```

Everything downstream (schedule, metrics, memory images, visualisation) is exact
for whatever pulse list it is given.

## Layout

| file | role |
|---|---|
| `circuit.py` | logical circuit + QASM-lite / OpenQASM-2 subset parser |
| `topology.py` | dot-line layout, logical↔dot mapping, NN router |
| `native.py` | gate → exchange-pulse templates (CX/SWAP/CXSWAP, 1q) |
| `compile.py` | synthesiser: circuit → pulse list (+ routing) |
| `schedule.py` | `Pulse` IR, ASAP scheduler, external-pulse ingestion |
| `metrics.py` | control-cost metrics + heuristic leakage/noise proxies |
| `hardware.py` | cryo-CMOS model, instruction/pattern memory back end |
| `visualize.py` | dependency-free SVG timeline + JSON export |
| `report.py` | CSV writers + Markdown cost report |
| `pipeline.py` | end-to-end glue + artefact emission |
| `cli.py` | `python -m eo_pulse_ir.cli` |
| `sim/` | **optional** physics simulator (requires numpy) — see below |

Run the tests with `python -m unittest tests.test_eo_pulse_ir`.

## Physics simulator (`sim/`, optional, needs numpy)

The core IR's `estimated_leakage_risk` / `noise_sensitivity` are heuristics. The
`sim/` subpackage replaces them with **real numbers** from a small dense
Heisenberg-exchange simulation, and maps the pulse-parameter landscape.

- **Model.** `n` spin-½ dots, exchange `H = Σ J_ij S_i·S_j`. A pulse of area `A`
  on edge `(i,j)` has the exact propagator `e^{iA/4}(cos(A/2)I − i sin(A/2)·SWAP_ij)`
  (full SWAP at `A=π`). The logical qubit is the 3-dot `S=1/2` doublet; leakage is
  population leaving `S=1/2 ⊗ S=1/2`.
- **Scoring.** Subspace-restricted average gate fidelity
  `F = (|Tr(V†M)|² + Tr(M†M)) / (d(d+1))` and leakage `L = 1 − Tr(M†M)/d`, where
  `M = L†UL` is the logical block. It consumes the **same `Pulse` objects** the IR
  produces, so synthesised or optimiser/eoqrid pulses get a real fidelity.

```bash
pip install -r requirements-sim.txt
python scripts/eo_landscape.py --res 28 -o out/landscape
python -m unittest tests.test_eo_sim
```

`scripts/eo_landscape.py` validates the single-qubit templates and sweeps a
two-qubit leakage/robustness landscape. Findings it reproduces:

- `intra_low(0,1)` of area `θ` realises `Rz(−θ)` exactly, leakage-free.
- **X and H reach fidelity 1 with 3 alternating exchange pulses; Y cannot
  (F≈0.833) and needs 4** — the two EO generators are ~120° apart. These validated
  areas are baked into `native.py`'s single-qubit templates (`rx`/`ry` are then
  composed exactly as `H·Rz·H` / `S·H·Rz·H·Sdg`); aligned gates use the exact
  `intra_low` area `(−θ) mod 2π`.
- Single-qubit operations are **leakage-free** (exchange conserves `S²` in a triple);
  leakage is driven **only by inter-qubit (boundary) exchange**, with
  **noise-robust plateaus** at boundary area `0, π, 2π` (stationary points of `L`).

### Validated two-qubit templates

`scripts/eo_optimize_2q.py` optimises each EO-native two-qubit gate against the
simulator (KAK-style ansatz: boundary exchanges dressed by full single-qubit
blocks) and the results are baked into `native.py`:

| gate | pulses | fidelity | leakage | constant |
|---|---|---|---|---|
| CNOT  | 34 | 0.99999999 | 7.7e-9 | `_CX_VALIDATED` |
| SWAP  | 27 | 0.99999999 | 3.2e-10 | `_SWAP_VALIDATED` |
| CXSWAP| 48 | 0.99992    | 6.3e-5 | `_CXSWAP_VALIDATED` |

All three are verified end-to-end: the IR-synthesised gate reproduces the listed
fidelity after role resolution and scheduling. Findings: the **minimum pulse
count tracks entangling content** — SWAP (3 boundary exchanges) < CNOT (4) <
CXSWAP (6); and a naive 19-pulse round-robin CNOT tops out at **F ≈ 0.78**, so a
high-fidelity gate needs full single-qubit dressing around each boundary
exchange. Re-optimise with:

```bash
python scripts/eo_optimize_2q.py --target cnot   -o out/cnot.json
python scripts/eo_optimize_2q.py --target swap   -o out/swap.json
python scripts/eo_optimize_2q.py --target cxswap -o out/cxswap.json
```

### Noise robustness (Monte-Carlo)

`scripts/eo_noise_mc.py` injects multiplicative pulse-area noise `A → A·(1+ε)`
(the charge-noise channel `dJ/J`, quasi-static per edge) and samples the gate
fidelity, giving a real distribution rather than the heuristic
`noise_sensitivity`. Infidelity grows quadratically with σ (`infidelity ≈ c·σ²`),
and the susceptibility `c` tracks pulse count:

| gate | pulses | susceptibility c | mean F @ σ=1% | worst F @ 1% |
|---|---|---|---|---|
| H      | 3  | 6.4   | 0.99938 | 0.99446 |
| SWAP   | 27 | 83.5  | 0.99041 | 0.92354 |
| CNOT   | 34 | 99.9  | 0.98834 | 0.93702 |
| CXSWAP | 48 | 112.1 | 0.98622 | 0.91791 |

So more pulses = more accumulated charge-noise error: single-qubit gates lose
~0.06% fidelity at 1% area noise, two-qubit gates ~1–1.4%. The script also
emits per-gate fidelity histograms and a noise-correlation-model comparison
(`per_edge` vs `global` vs `independent`).

```bash
python scripts/eo_noise_mc.py --samples 500 --rep-sigma 0.01 -o out/noise
```

