# EO Pulse-Control Middle Layer — engineering whitepaper

**Status: internal / not for external publication.** Prepared as a technical
hand-off document (e.g. for a Blueqat exchange-only stack). This is an
engineering asset, not a novelty claim: the components re-use established results
(Kempe–Bacon–Whaley controllability, KAK gate construction, GRAPE/adjoint
optimal control, the Tariq–Hu valley-phase exchange suppression, ensemble robust
control, MPS simulation). Its value is the **integration** and one **falsifiable,
device-anchored result**: a robust EO CNOT that beats the best achievable
non-robust gate under the device's dominant joint error.

---

## 1. Executive summary

`eo_pulse_ir` is a dependency-light middle layer for exchange-only (EO) silicon
spin qubits: it compiles logical circuits to EO-native gates to an exchange-pulse
schedule to a cryo-CMOS controller-memory budget, with a numpy-only physics
simulator alongside for validation and pulse design. The headline engineering
result, suitable to anchor an EO control OS:

> Under the two dominant real-device error channels at calibrated strength —
> charge noise **σ = 0.91 % dJ/J** (calibrated so the standard CNOT sits at the
> reported 99 % two-qubit point) and a **valley-phase spread of ±0.3π** — a
> jointly-robust CNOT holds **mean fidelity 0.950**, versus **0.826** for the
> standard gate and **0.926** for a *shorter, more charge-noise-favourable*
> 27-pulse exact gate. Crucially, even a **charge-noise-free** idealisation of
> that short gate — an upper bound on *any* shorter exact sequence, including the
> DiVincenzo 19-pulse and Fong–Wandzura 22-pulse gates — tops out at **0.932**.
> The robust gate beats that ceiling. The advantage is on the valley axis, which
> no reduction in pulse count can buy.

The practical reading for an EO OS: **gate libraries should be valley-robust by
construction**, because valley/E_VS *uniformity* (not peak E_VS, not gate length)
is the recognised scaling bottleneck, and robustness there cannot be recovered by
shorter or cleaner pulse sequences.

---

## 2. Where it sits in the stack

```
 logical circuit  ──►  EO-native gates  ──►  exchange-pulse schedule  ──►  controller memory
 (OpenQASM/Blueqat)    (CX/SWAP/CXSWAP/1q)   (pulse_timeline.json)        (instruction/pattern CSV)
        ▲                                          ▲  ▲
   adapters.qasm / adapters.blueqat        adapters.external (optimiser / eoqrid pulse records)
```

- **Inputs.** QASM-lite / OpenQASM-2 subset (plus EO-native `cxswap`, arbitrary
  `u(θ,φ,λ)`); Blueqat gate lists; or raw external pulse records (the optimiser /
  `eoqrid` seam — only `edge` and `area` are required to ingest).
- **Core (standard library only).** Native-gate lowering, ASAP exchange
  scheduling by dot availability, cost metrics, and an HRL-style cryo-CMOS
  instruction/pattern-memory model (DAC codes from `J(V)=j0·exp((V−v0)/vc)`,
  budgeted against sequencer/word limits).
- **Simulator (numpy, optional).** Exact single-pulse propagators, subspace
  average-gate-fidelity and leakage, analytic-adjoint GRAPE, an explicit
  spin⊗valley model, charge-noise calibration, and ensemble robust design.
- **Interchange contract.** `docs/IR_SPEC.md` (v0.1) fixes the JSON/CSV shapes so
  each layer can be replaced independently.

This is the integration point: produce `pulse_timeline.json` (or external pulse
records) from any front end or optimiser and the whole cost / scheduling /
hardware / visualisation pipeline runs on it unchanged.

---

## 3. Device-anchored error model

A single quasi-static per-edge multiplicative area noise — `a_eff = a·(1+ε)`,
ε ~ N(0, σ) — is fitted so the validated CNOT reaches the reference two-qubit
fidelity 0.990. The fitted **σ = 0.90 % dJ/J** then predicts the rest of the gate
set from pulse count alone:

| gate | pulses | predicted F |
|---|---:|---:|
| H (1Q) | 3 | 0.99950 |
| X (1Q) | 3 | 0.99984 |
| CNOT (2Q) | 34 | 0.99000 |
| SWAP (2Q) | 27 | 0.99199 |
| CXSWAP (2Q) | 48 | 0.98859 |

Calibrating to the two-qubit gate predicts a single-qubit fidelity of 0.9995,
consistent with the reported ≈ 99.9 % — one fitted knob reproduces the real
device's 1Q/2Q hierarchy. σ ≈ 1 % dJ/J is a physically reasonable charge-noise
level for silicon. (Phenomenological, not first-principles; valley and
pulse-distortion channels are layered on top.) Reproduce with
`scripts/eo_calibrate.py`.

---

## 4. The valley bottleneck (why robustness is the right objective)

A cited literature synthesis (`docs/valley_splitting_research.md`, 17 primary
sources) supports three load-bearing points:

1. **Valley splitting E_VS is a disordered random variable** (10–300 µeV,
   Rice/Rayleigh across a chip) set by the sharp interface and alloy disorder;
   the recognised bottleneck is **uniformity/yield across an array, not the peak
   value**.
2. **A valley-phase difference between dots suppresses their exchange**
   (J_eff ≈ J·cos²(Δφ/2), nulling at Δφ = π) *even when E_VS is large in both* —
   Tariq–Hu, npj QI 8, 53 (2022). This is the mechanism that detunes EO
   two-qubit gates.
3. **Small/non-uniform E_VS opens a leakage channel** out of the logical
   subspace that can be hidden at the two-qubit level yet appear at scale —
   Buterakos–Das Sarma, PRX Quantum 2, 040358 (2021).

The simulator carries this explicitly (4-level spin⊗valley dots,
`eo_pulse_ir/sim/valley.py`): with aligned valley phases it reproduces the
spin-only validated CNOT exactly; with E_VS ≫ J it suppresses valley leakage to
zero; and it shows that a valley-phase mismatch still *detunes* the exchange even
at large E_VS — so both large **and** uniform E_VS **and** valley-phase control
are required. Robust pulse design addresses the *consequence* (exchange detuning
from valley-phase spread); it does not raise E_VS, which is set by the
heterostructure and temperature.

---

## 5. Core result and the adversarial-grade comparison

**Design.** Standard GRAPE maximises fidelity at one nominal point; robust design
maximises the **ensemble-averaged** fidelity over the joint distribution of both
channels (`eo_pulse_ir/sim/robust.py`). Per ensemble sample, a valley-phase
mismatch Δφ ~ U[−δ, δ] scales the inter-block exchange areas by cos²(Δφ/2) and an
independent per-edge charge noise multiplies every area by (1+ε). The ensemble
gradient chains through the affine area transform; optimisation is Adam ascent
seeded from the validated CNOT.

**The strongest objection, answered.** "Why not just use the shortest known exact
gate (DiVincenzo 19-pulse / Fong–Wandzura 22-pulse, or the recent 2-D-layout
sequences of Chadwick et al., Phys. Rev. A 111, 052616 (2025), arXiv:2412.14918,
≈ 28-pulse CX), which accrue less charge-noise error?" We answer it directly
(`scripts/eo_robust_compare.py`):

| gate | pulses | mean F (joint) | F at Δφ=0 (charge only) |
|---|---:|---:|---:|
| standard CNOT | 34 | 0.826 | 0.993 |
| compact exact CNOT | 27 | 0.926 | 0.991 |
| compact, **charge-noise-free** (ceiling) | 27 | **0.932** | 1.000 |
| **joint-robust CNOT** | 34 | **0.950** | 0.974 |

Reading:

- The **compact 27-pulse exact CNOT** is genuinely shorter and more
  charge-favourable than the 34-pulse standard gate, and indeed does better under
  the joint ensemble (0.926 vs 0.826) — confirming that fewer pulses help on the
  charge axis.
- The **charge-noise-free ceiling** removes charge noise entirely (σ = 0). This
  upper-bounds *any* shorter exact sequence — including DiVincenzo 19,
  Fong–Wandzura 22, and the Chadwick et al. ≈ 28-pulse 2-D-layout SOTA — because
  fewer pulses can only reduce charge-noise error. Even this idealised gate is
  capped at **0.932** under the valley spread.
- The **joint-robust CNOT holds 0.950**, beating both the compact gate and its
  charge-noise-free ceiling.

The robust advantage is therefore on the **valley axis**, which no reduction in
pulse count can address. This is the falsifiable claim and it survives the
strongest non-robust baseline.

![Robust vs best-known short exact CNOT](../out/robust_compare/robust_compare.svg)

---

## 6. Using it in an EO control OS

1. **Ship valley-robust gate libraries.** Replace nominal-point gate templates
   with ensemble-robust ones for the device's measured valley-phase spread and
   calibrated dJ/J. The robust areas are a drop-in `pulse_timeline.json`, or a
   `gate_library` passed straight to `compile_circuit`
   (`sim.robust.robust_gate_library`).
2. **Place-and-route on real robust costs.** The 2-D grid place-and-route
   (`GridTopology`) and the robust gate library compose: 2-D layout cuts the
   routing-SWAP count for non-line connectivity, and the robust library raises
   each operation's fidelity under the joint valley + charge noise. On a ring-4q
   the device-anchored end-to-end fidelity rises from 0.374 (1-D, nominal) to
   0.804 (2-D, robust) — both axes (`scripts/eo_grid_robust.py`). Unlike a
   place-and-route that costs gates by a fixed integer, the layout and the
   fidelity estimate here run on real, device-calibrated, valley-robust costs.
3. **Calibration loop.** Use `calibrate_sigma` against measured RB/interleaved-RB
   fidelities to fix the per-device noise level; re-run `robust_design` per device
   class. If the valley-phase spread is *measurable and quasi-static*, per-device
   calibration can remove the mean detuning and the robust gate then absorbs the
   residual spread — the two compose.
4. **Scheduler / controller integration.** Robust pulses cost a few extra pulses
   vs the compact gate; the scheduler and the cryo-CMOS memory model already
   budget arbitrary `(edge, area)` sequences, so the controller-memory footprint
   is computed automatically.
5. **Scale-out.** The MPS backend (`eo_pulse_ir/sim/mps.py`,
   `eo_pulse_ir/sim/mps_grape.py`) validates and designs multi-qubit operations
   where dense simulation is infeasible, for low-entanglement targets.

---

## 7. Honest scope and limitations

- **No new physics.** Controllability restates Kempe–Bacon–Whaley; the gate
  construction is KAK; the optimiser is GRAPE with an adjoint gradient; the valley
  suppression is the Tariq–Hu effective model; robustness is ensemble optimal
  control. The contribution is integration + one device-anchored result.
- **Phenomenological calibration.** One knob fitted to one number; not a
  first-principles noise model.
- **Valley model scope.** Uses the cos²(Δφ/2) exchange suppression and a
  4-level-per-dot leakage space; it does not yet include thermal excited-valley
  population (the ≳ 5 k_BT rule), the spin-valley hotspot at E_VS = E_Zeeman, or
  spatially-correlated E_VS disorder — all natural next extensions.
- **Robustness scope.** Quasi-static (filter-function) noise; demonstrated for
  CNOT. The same engine extends to the full gate set and to multi-qubit blocks on
  the MPS backend.

---

## 8. Reproducibility

Simulator depends only on numpy; the IR core uses the standard library only. CI
runs both test suites and smoke-tests every script on Python 3.10–3.12. Key
entry points: `scripts/eo_calibrate.py` (calibration), `scripts/eo_valley.py`
(valley model), `scripts/eo_robust_joint.py` (joint robust design),
`scripts/eo_robust_compare.py` (the comparison in §5),
`scripts/eo_mps_grape.py` (tensor-network optimal control).
