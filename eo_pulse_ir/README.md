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

### Magnetic-field-gradient study

`sim/field.py` adds a static Zeeman field `H_Z = Σ b_i S_z^i` (exact eigen-
propagation, since exchange and field act simultaneously). `scripts/eo_gradient_study.py`
answers four physics questions:

- **Intra-block exchange alone does not leak** (`leakage ≈ 1e-15`) — the S=1/2
  triple is a decoherence-free subsystem, since exchange conserves S².
- **Raw inter-block (boundary) exchange leaks** (`≈0.31` for a boundary √SWAP);
  a well-designed gate refocuses it (validated CNOT leakage `7.7e-9`).
- **A uniform field is harmless** (`b·S_z^total` = global phase), but a **field
  gradient breaks S² conservation and the DFS**: single-qubit leakage rises from
  zero ∝ gradient², and two-qubit gates are far more sensitive because the figure
  of merit is gradient × gate-time (a 1% gradient already costs the CNOT ~70%).
- **Pulse-parameter sensitivity is visible** as a 2-D (gradient × area-calibration)
  infidelity map — for the CNOT the field-gradient axis dominates ±5% area error.

```bash
python scripts/eo_gradient_study.py --res 24 -o out/gradient
```

### Pulse control as a piecewise-constant (switched bilinear) system

EO pulse control is a right-invariant switched bilinear system on SU(2ⁿ):
`dU/dt = -i(H_drift + J·G_m(t))U`, where the modes `G_e = S_i·S_j` are the
nearest-neighbour exchange generators, the drift is the Zeeman field, and a
*control word* (sequence of `(mode, dwell-time)` segments) is exactly a pulse
list. `sim/control.py` (`EOControlSystem`) and `sim/lie.py` make this first-class
and compute controllability via the generated Lie algebra.
`scripts/eo_control_analysis.py` shows:

| system | sector dim | Lie dim | full su(d) | logical↔leakage |
|---|---|---|---|---|
| 1 qubit, exchange only | 3 | 4 | 8 | ~0 (DFS) |
| 1 qubit, exchange + gradient | 3 | 8 | 8 | 0.71 |
| 2 qubits, exchange only | 15 | 105 | 224 | 0.33 |

This recasts the physics findings as control theory: single-qubit exchange is
confined to a DFS subalgebra (no leakage), boundary exchange already couples to
leakage (so gates must refocus it), and a gradient enlarges the algebra to the
full `su(d)`, breaking the DFS. The single-qubit logical group is `SU(2) ≅` unit
quaternions — each pulse is a fixed-axis rotation and a sequence is a quaternion
product (`bloch_trajectory.svg` draws the piecewise-constant path; the validated
H reproduces as a quaternion product to ~1e-16).

```bash
python scripts/eo_control_analysis.py -o out/control
```

### Quaternion single-qubit layer

`sim/quaternion.py` gives the single-qubit (DFS) layer its natural representation:
`SU(2) ≅` unit quaternions. The two native exchange axes, measured from the
simulator, are `intra_low n1 = (0,0,-1)` and `intra_high n2 = (√3/2,0,1/2)` —
exactly **120° apart**, with rotation angle = pulse area. Any single-qubit gate
compiles **analytically and exactly** (`compile_unitary`, ZXZ from native Rz +
validated H), exposed in the front-end as the OpenQASM `u(θ,φ,λ)` gate (verified
F=1 over 200 random gates). `scripts/eo_quaternion.py` draws the piecewise-constant
Bloch trajectory computed purely by quaternion rotations.

```bash
python scripts/eo_quaternion.py -o out/quaternion
```

### Tensor-network (MPS/TEBD) scaling

EO dynamics is a stream of nearest-neighbour two-site gates, so an MPS evolves it
with bond dimension set by entanglement, not by 2ⁿ — bringing this repository's
original tensor-network theme to spin dynamics. `sim/mps.py` builds the encoded
register as an MPS (entangled within each triple, product across triples) and
applies each exchange pulse by TEBD (merge → gate → truncated SVD).

- **Validation:** at full bond dimension the MPS reproduces the dense simulator
  exactly — CNOT/SWAP/CXSWAP state fidelity 1.0 (discarded weight ~1e-32).
- **Scaling:** a GHZ-style CNOT chain on 12 logical qubits = 36 dots = dense
  dimension 2³⁶ (~7×10¹⁰ amplitudes, infeasible to store) evolves in ~0.04 s with
  max bond dimension **16** (`mps_scaling.svg`: bond dim flat while dense dimension
  explodes). Bond dimension stays bounded whenever the circuit's entanglement is
  limited — the regime where tensor networks win.

```bash
python scripts/eo_mps.py --chi-max 32 --max-qubits 12 -o out/mps
```

### State-space dynamical-systems view

`sim/statespace.py` gives the Schrödinger-flow picture: writing `psi = u + iv`
and `x = [u; v]`, EO control is a **switched linear dynamical system**
`dx/dt = A_m x` with constant skew-symmetric vector fields `A_m = -iH_m` (verified
exact; the flow is a rotation on the state sphere). For one qubit this is the
Bloch sphere with `db/dt = omega_m × b` — two rotation vector fields about axes
120° apart, whose fixed points are the axis poles. The logical subspace is an
**invariant manifold** of the exchange-only flow (defect = leakage stays 0); a
field gradient breaks its invariance and the defect grows.
`scripts/eo_statespace.py` draws the Bloch phase portrait and the
manifold-defect-vs-time curves.

```bash
python scripts/eo_statespace.py -o out/statespace
```

### n-qubit gate design (optimal pulse control)

`sim/synthesis.py` (`design_gate`) treats an n-qubit logical target as a boundary
condition for the switched bilinear control system and *designs* the control word
— a GRAPE-style scheme: a layered ansatz over all nearest-neighbour edges with the
exact analytic-gradient optimiser. It works for any n (dense, so a few logical
qubits): single-qubit gates compile exactly (F=1, ≤9 pulses), two-qubit gates need
tens of pulses (CNOT F≈1 at 34), and the control cost grows with entangling
content — a 3-qubit CCZ needs a deep layered sequence.

The **"3-for-1 ⇒ quaternion"** structure is concrete here: the single-qubit
control algebra `span{G₁, G₂, [G₁,G₂]}` has rank 3 = su(2) = the quaternion units
{i, j, k}. Three dots make one logical qubit whose control algebra *is* the
quaternions.

```bash
python scripts/eo_synthesis.py -o out/synthesis            # fast: 1q/2q + algebra
python scripts/eo_synthesis.py --ccz-layers 8 -o out/synthesis   # also attempt 3q CCZ (slow)
```

### MPS-GRAPE: optimal control on a tensor network

Gate fidelity needs all 2ⁿ logical columns (exponential), but the *state-transfer*
objective `J = |⟨target|U(θ)|init⟩|²` needs only one evolved state — so GRAPE on an
MPS (`sim/mps_grape.py`) scales optimal pulse control to many logical qubits when
the target is low-entanglement. The gradient is the **MPS adjoint method** (one
forward + one backward sweep, O(N) per gradient; verified against finite
differences to ~1e-10 — finite differences would be O(N²)). This unites the
control and tensor-network layers: design optimal pulses for n-qubit operations
without ever forming the 2³ⁿ state.

Demonstrated on encoded-GHZ preparation (`scripts/eo_mps_grape.py`): the optimiser
finds pulses preparing an entangled GHZ state on logical-qubit counts where dense
GRAPE is infeasible, with bond dimension staying small.

```bash
python scripts/eo_mps_grape.py --max-qubits 6 -o out/mps_grape
```

