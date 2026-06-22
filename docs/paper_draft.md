# Valley- and charge-noise-robust exchange-only gate design via a pulse-control intermediate representation

**Draft v0.1 — working paper.** Author: _(to be filled)_. Toolkit: `eo_pulse_ir`
(this repository). Status: pre-print draft; numbers are reproducible from the
scripts in `scripts/` and the test suite (`tests/`).

---

## Abstract

Exchange-only (EO) silicon spin qubits encode one logical qubit in three quantum
dots and realise all gates with nearest-neighbour Heisenberg exchange pulses,
avoiding microwave control. Their leading scaling obstacle is not the nominal
exchange but the **non-uniformity of two device parameters across an array**:
the interface-induced *valley splitting* (and the associated *valley phase*,
which detunes inter-dot exchange) and quasi-static *charge noise* on the
exchange coupling. We present a compact, dependency-light pulse-control
intermediate representation (IR) and physics simulator for EO qubits, and use it
to (i) produce simulator-validated EO-native gate sequences (CNOT, SWAP, CXSWAP)
at average gate fidelity ≥ 0.9999 in the decoherence-free subspace; (ii)
calibrate a single quasi-static charge-noise parameter, σ = 0.90 % dJ/J, that
reproduces the reported HRL one- and two-qubit fidelity hierarchy; (iii)
reproduce, as an emergent property of an explicit spin⊗valley model, the
literature requirement E_VS ≫ J for low leakage; and (iv) design, by
ensemble-averaged optimal control (analytic-adjoint GRAPE), a CNOT that is
**jointly robust** to a valley-phase spread of ±0.3π and charge noise at the
calibrated level. Under both error channels at calibrated strength the
joint-robust CNOT holds a mean gate fidelity of **0.950** against **0.826** for
the standard gate — recovered by trading ≈ 1.8 percentage points of peak
fidelity at the nominal operating point. We frame the underlying control problem
as a switched bilinear system, show by Lie-algebraic analysis that exchange
alone generates exactly the logical subalgebra (no leakage coupling) while a
field gradient opens the full su(3) and couples to leakage, and demonstrate that
the same optimal-control engine scales to many logical qubits on a
matrix-product-state (MPS) backend.

---

## 1. Introduction

Exchange-only operation is attractive because every gate is driven by the same
physical knob — the voltage-controlled nearest-neighbour exchange J — with no
on-chip microwave lines, which simplifies the cryo-CMOS control stack. HRL
Laboratories demonstrated universal encoded EO logic in a six-dot Si/SiGe
"SLEDGE" device using nearest-neighbour partial-SWAP pulses [Weinstein 2023],
with a measured per-gate leakage of 0.17 % attributed primarily to nuclear
spins [Andrews 2019] rather than valley physics in present devices.

The practical bottleneck for scaling EO arrays, however, is increasingly
understood to be **parameter uniformity** rather than peak performance
[Thayil 2024; foundry 2024]. Two channels dominate:

1. **Valley structure.** Bulk silicon has six conduction-band valleys; biaxial
   strain and (001) confinement leave the two z-valleys lowest, split only by
   the sharp interface into E_VS ≈ 10–300 µeV. Alloy disorder makes E_VS a
   random variable that varies dot-to-dot [Wuetz 2022; Losert 2023], and the
   **valley-phase difference between two dots suppresses their exchange**,
   nulling at Δφ = π, even when E_VS is large in both [Tariq 2022]. Small or
   non-uniform E_VS opens a leakage channel out of the logical subspace that can
   be hidden in two-qubit benchmarks yet appear at scale [Buterakos 2021].

2. **Charge noise** on the exchange coupling (dJ/J), the dominant gate-error
   source in current devices [arXiv:2110.11329], at the ≈ 0.9 µeV·Hz^(−1/2)
   level in co-designed heterostructures [Degli Esposti 2024].

We do not attempt new two-qubit gate-synthesis physics, a surface-code decoder,
or a re-implementation of fab or cryo-CMOS. Instead we build the **middle
layer**: a stable pulse IR plus a small physics simulator, and use them to ask a
concrete control-engineering question — *can a single pulse schedule be made
robust to the two dominant non-uniformities at once, and at what cost?* The
answer (Section 6) is the paper's core result.

### Contributions

- A standard-library-only EO pulse-control IR (circuit → EO-native gates →
  exchange schedule → cost + HRL-style controller memory) with a documented JSON
  interchange contract (`docs/IR_SPEC.md`), and adapters to OpenQASM, external
  optimiser/`eoqrid` pulse records, and Blueqat gate lists.
- Simulator-validated EO-native CNOT/SWAP/CXSWAP templates at F ≥ 0.9999.
- A one-parameter charge-noise calibration tying the model to reported device
  fidelities.
- An explicit spin⊗valley simulator that reproduces E_VS ≫ J from first
  principles of the model, and quantifies a validated CNOT under valley-phase
  mismatch.
- A control-theoretic analysis (switched bilinear system; Lie-algebra
  controllability; bifurcation of the control landscape) and a tensor-network
  (MPS) optimal-control backend that scales to ≥ 12 dots.
- **The joint valley + charge-noise robust CNOT design and its quantified
  advantage.**

---

## 2. Background: the EO model

Three dots in a row hold one logical qubit in the S = 1/2, S_z = +1/2
decoherence-free-subspace (DFS) doublet:
|0_L⟩ = (|↑↓↑⟩ − |↓↑↑⟩)/√2, |1_L⟩ from the orthogonal doublet vector. The
control Hamiltonian is the nearest-neighbour Heisenberg exchange

  H = Σ_⟨i,j⟩ J_ij(t) S_i·S_j ,

driven as square pulses of area A = ∫ J dt. A single pulse on edge (i,j)
implements the exact propagator

  U_ij(A) = e^{iA/4} ( cos(A/2) I − i sin(A/2) SWAP_ij ),

a full SWAP at A = π. All gates are products of such pulses on a 1-D chain of
3n dots for n logical qubits.

**Subspace metrics.** With L the isometry embedding the logical (or logical⊗·)
subspace into the full dot Hilbert space and U the simulated dot-level
propagator, M = L† U L is the logical action. We score the average gate fidelity

  F = (|Tr(V† M)|² + Tr(M† M)) / (d(d+1))

against the target V (dimension d), and the leakage L_leak = 1 − Tr(M† M)/d.

---

## 3. The pulse-control IR and simulator

**IR (standard library only).** The core (`eo_pulse_ir/`) compiles a QASM-lite /
OpenQASM-2 subset (plus the EO-native `cxswap` and arbitrary `u(θ,φ,λ)`) into
EO-native gates, then into an exchange schedule placed as-soon-as-possible by
dot availability, and finally into cost metrics and an HRL-style cryo-CMOS
instruction/pattern memory budget. The canonical interchange object is
`pulse_timeline.json`; the only required per-pulse fields to ingest are `edge`
and `area` (`docs/IR_SPEC.md`). Adapters bridge OpenQASM, external pulse records
(the `eoqrid`/optimiser seam), and Blueqat gate lists. Visualisation is pure SVG
(no matplotlib), and the package installs an `eo` console app
(`eo {compile,dashboard,demo}`).

**Simulator (numpy only, optional).** `eo_pulse_ir/sim/` builds the dot-level
propagators from the exact single-pulse form above, embeds the logical subspace,
and computes F and leakage. Optimisation uses an **analytic adjoint gradient**
(GRAPE) of the subspace fidelity with respect to pulse areas, verified against
finite differences to ≈ 1×10^(−10).

**Validated native gates.** Round-robin templates cannot exceed F ≈ 0.78 for a
CNOT; a KAK-style ansatz (boundary exchanges with full single-qubit dressing)
optimised by analytic-gradient Adam reaches the validated sequences below
(`eo_pulse_ir/native.py`):

| gate   | pulses | average gate fidelity | leakage |
|--------|-------:|----------------------:|--------:|
| CNOT   |     34 | 0.99999999            | 7.7e-9  |
| SWAP   |     27 | 0.99999999            | 3.2e-10 |
| CXSWAP |     48 | 0.99992               | 6.3e-5  |

The single-qubit layer (H, X, Y as 3–4 pulse sequences; Rx/Ry/u composed; Rz as
a frame area) is likewise validated to F = 1.

---

## 4. Control-theoretic structure

**Switched bilinear system.** With the schedule a piecewise-constant choice of
active edge, the dynamics ẋ = (Σ_k u_k(t) A_k) x form a switched bilinear
(right-invariant) control system on SU(d). In the state-space (real) form the
generator M = [[H_i, H_r], [−H_r, H_i]] is skew-symmetric, so trajectories live
on a sphere — the EO Bloch picture.

**Controllability (Lie algebra).** Closing the Lie algebra generated by the
available exchange directions (`eo_pulse_ir/sim/control.py`,
`eo_pulse_ir/sim/lie.py`) gives, for one logical qubit:

| generators                 | sector dim | Lie dim | logical↔leak coupling |
|----------------------------|-----------:|--------:|----------------------:|
| exchange only              |          3 |       4 | 8.3e-17 (zero)        |
| exchange + field gradient  |          3 |       8 | 0.707                 |
| 2-qubit exchange           |         15 |     105 | 0.333                 |

Exchange alone generates **exactly** the logical subalgebra (Lie dimension 4)
with no coupling to leakage — the DFS is, precisely, the reachable set. Turning
on a field gradient opens the full su(3) (dimension 8) and couples logical states
to leakage: a *structural* controllability bifurcation.

**Bifurcation of the control landscape.** Treating fidelity as a nonlinear map
of pulse parameters (`eo_pulse_ir/sim/bifurcation.py`), increasing a field
gradient g from 0 to 0.8 grows the number of local optima of a single-qubit
two-pulse landscape from 1 to 6 via saddle-node bifurcations, while the global
maximum falls from 0.955 to 0.780 — the gradient makes the optimisation
landscape progressively more rugged. This connects "valley/field non-uniformity"
to "harder optimal-control problem" quantitatively.

---

## 5. Device-anchored noise and valley models

**Charge-noise calibration.** A single quasi-static per-edge multiplicative area
noise, a_eff = a·(1 + ε), ε ~ N(0, σ), is fitted so the validated CNOT reaches
the HRL reference two-qubit fidelity 0.990 (`eo_pulse_ir/sim/calibration.py`).
The fitted **σ = 0.90 % dJ/J** then predicts the rest of the gate set from pulse
count alone:

| gate     | pulses | predicted F |
|----------|-------:|------------:|
| H (1Q)   |      3 | 0.99950     |
| X (1Q)   |      3 | 0.99984     |
| CNOT (2Q)|     34 | 0.99000     |
| SWAP (2Q)|     27 | 0.99199     |
| CXSWAP   |     48 | 0.98859     |

Calibrating to the two-qubit gate predicts a single-qubit fidelity of 0.9995,
consistent with the reported HRL ≈ 99.9 %. One fitted knob reproduces the real
device's 1Q/2Q error hierarchy. σ ≈ 1 % dJ/J is a physically reasonable charge-
noise level for silicon; this is a phenomenological calibration, not a
first-principles noise model.

**Explicit valley model.** Each dot is enlarged to four levels (spin⊗valley,
index 2·spin+valley) with a valley splitting E_VS and a per-dot valley phase φ;
inter-dot exchange is dressed by a valley twist so that the effective coupling
carries the J_eff ≈ J·cos²(Δφ/2) suppression (`eo_pulse_ir/sim/valley.py`). The
model reproduces, without being told to:

- **E_VS ≫ J suppresses valley leakage.** A boundary exchange with Δφ = π/2
  leaks 0.375 at E_VS = 0, falling to ≈ 0 by E_VS/J = 12.
- **Leakage grows with valley-phase mismatch** (to 0.5 at Δφ = π for E_VS = 0),
  and is suppressed for E_VS/J = 10.
- With aligned valley phases the simulator recovers the spin-only validated CNOT
  exactly (F = 0.99999999).

Crucially, scoring the *validated* CNOT against E_VS/J under a fixed valley-phase
mismatch shows that large E_VS suppresses the **leakage** but the mismatch still
**detunes** the exchange, so fidelity stays low (≈ 0.85–0.88) until the valley
phase is aligned or the pulses are re-calibrated for it. **Both** large *and*
uniform E_VS *and* valley-phase control are required — matching the consensus
that uniformity, not peak E_VS, is the bottleneck.

---

## 6. Core result: jointly robust CNOT design

Standard GRAPE maximises fidelity at one nominal operating point. We instead
maximise the **ensemble-averaged** fidelity over a joint distribution of both
dominant channels (`eo_pulse_ir/sim/robust.py`, `scripts/eo_robust_joint.py`):
per ensemble sample, a valley-phase mismatch Δφ ~ U[−δ, δ] scales the boundary
(inter-triple) exchange areas by cos²(Δφ/2), and an independent per-edge charge
noise multiplies every area by (1 + ε), ε ~ N(0, σ). The ensemble gradient
chains through the affine area transform a_eff = s·a, so grad_a =
mean_k(grad_eff(a_eff,k)·s_k); optimisation is Adam ascent on the ensemble mean,
seeded from the validated CNOT.

We set σ to the calibrated 0.91 % (Section 5) and δ = 0.3π, train on a 12-sample
joint ensemble, and evaluate baseline vs robust on fresh ensembles (200–400
samples) across the valley-phase spread.

**Result.** Under both error channels at calibrated strength:

| metric                          | standard CNOT | joint-robust CNOT |
|---------------------------------|--------------:|------------------:|
| mean F over the joint ensemble  | **0.826**     | **0.950**         |
| mean F, charge-noise only (Δφ=0)| 0.993         | 0.975             |

The standard gate is essentially optimal at the nominal point but falls off a
cliff as valley-phase mismatch grows; the robust gate gives up ≈ 1.8 percentage
points of peak fidelity at Δφ = 0 and buys back ≈ 12 points across the spread,
staying flat and charge-noise-tolerant (figure: `out/robust_joint/robust_joint.svg`).
Given that valley/E_VS uniformity — not the nominal exchange — is the recognised
scaling bottleneck, flat-across-spread is the correct objective.

For comparison, valley-robust design *alone* (charge noise off) lifts the mean
fidelity over ±0.3π from 0.783 to 0.943; adding the calibrated charge channel to
both training and evaluation gives the joint numbers above.

---

## 7. Scaling: optimal control on tensor networks

The same machinery runs on a matrix-product-state backend
(`eo_pulse_ir/sim/mps.py`). The MPS evolver reproduces dense CNOT/SWAP/CXSWAP to
F = 1.0 with small bond dimension (≤ 6) and applies EO schedules up to ≥ 12
logical qubits (36 dots, dense dimension 2^36) in tens of milliseconds with bond
dimension capped at 16. State-preparation **MPS-GRAPE**
(`eo_pulse_ir/sim/mps_grape.py`), with an O(N) adjoint gradient verified to
2×10^(−10) against finite differences, prepares an encoded GHZ on 2 logical
qubits to F = 1.0 and on 3 to F = 0.92, where dense GRAPE is already
expensive — uniting the optimal-control and tensor-network views for
low-entanglement targets at scale.

---

## 8. Discussion and limitations

The calibration is phenomenological (one knob to one number); valley splitting
itself is set by the heterostructure and temperature and is not raised by any
pulse — the robust design mitigates the *consequence* (valley-phase detuning of
exchange), not the cause. The valley model uses the cos²(Δφ/2) exchange
suppression and a 4-level-per-dot leakage space; it does not yet include thermal
excited-valley population (the ≳ 5 k_B T rule), the spin-valley hotspot at
E_VS = E_Zeeman, or correlated spatial E_VS disorder, all of which the research
synthesis (`docs/valley_splitting_research.md`) flags as natural next extensions.
Robustness is demonstrated for CNOT; extending the joint ensemble objective to
the full validated gate set and to multi-qubit blocks on the MPS backend is
straightforward with the existing engine.

---

## 9. Methods (reproducibility)

All numbers are reproducible from this repository. Key entry points:

- Native-gate validation / (J,τ) landscape: `scripts/eo_landscape.py`,
  `scripts/eo_optimize_2q.py`.
- Charge-noise Monte-Carlo and calibration: `scripts/eo_noise_mc.py`,
  `scripts/eo_calibrate.py`.
- Controllability and bifurcation: `scripts/eo_control_analysis.py`,
  `scripts/eo_bifurcation.py`.
- Valley model: `scripts/eo_valley.py`.
- Robust design (core result): `scripts/eo_robust.py`,
  `scripts/eo_robust_joint.py`.
- Tensor networks: `scripts/eo_mps.py`, `scripts/eo_mps_grape.py`.

The simulator depends only on numpy; the IR core uses the standard library only.
Continuous integration runs both test suites and smoke-tests every script on
Python 3.10–3.12.

---

## References

Primary sources (from the cited literature synthesis in
`docs/valley_splitting_research.md`; consult the arXiv PDFs for load-bearing
numbers):

1. Friesen et al., *Valley splitting theory of SiGe/Si/SiGe quantum wells*,
   PRB **75**, 115318 (2007), arXiv:cond-mat/0608229.
2. Losert et al., *Practical strategies for enhancing the valley splitting*,
   PRB **108**, 125405 (2023), arXiv:2303.02499.
3. Thayil, Ermoneit, Kantner, arXiv:2412.20618 (2024) — valley uniformity as a
   scaling bottleneck.
4. Wuetz et al., *Atomic fluctuations lifting the valley degeneracy*,
   Nat. Commun. **13**, 7730 (2022), arXiv:2112.09606.
5. McJunkin et al., *SiGe quantum wells with oscillating Ge (Wiggle Well)*,
   Nat. Commun. **13**, 7777 (2022), arXiv:2112.09765.
6. Wuetz & Friesen, PRB **104**, 085406 (2021), arXiv:2104.08232 — Ge spike.
7. Degli Esposti et al., *Low disorder and high valley splitting*,
   npj QI **10** (2024), arXiv:2309.02832.
8. Tariq & Hu, *Effects of valley-phase on exchange*, npj QI **8**, 53 (2022),
   arXiv:2107.00732. **(load-bearing: valley-phase exchange suppression)**
9. Buterakos & Das Sarma, *Spin-valley qubit leakage*, PRX Quantum **2**,
   040358 (2021), arXiv:2106.01391. **(E_VS ≫ J; hidden-at-2-qubit leakage)**
10. Mortemousque et al., PRX Quantum **2**, 020309 (2021), arXiv:2101.12594 —
    valley-dependent tunnel couplings in a triple dot.
11. Andrews et al., *Quantifying error and leakage in an EO qubit*,
    Nat. Nano. **14**, 747 (2019), arXiv:1812.02693.
12. Weinstein et al., *Universal logic with encoded spin qubits in silicon*,
    Nature **615**, 817 (2023), arXiv:2202.03605. **(HRL 6-dot EO)**
13. Penthorn et al., PR Applied **14**, 054015 (2020), arXiv:2007.08680 —
    intervalley relaxation.
14. Huang et al., *High-fidelity operation above 1 K*, Nature **627**, 772
    (2024), arXiv:2308.02111.
15. Borjans/Hollmann et al., PR Applied **13**, 034068 (2020), arXiv:1907.04146
    — electric-field valley tuning / hotspot avoidance.
16. Volmer et al., *Velocity-shaped spin shuttling*, npj QI **10**, 61 (2024),
    arXiv:2312.17694.
17. 300 mm Si-MOS foundry qubits, arXiv:2410.15590 (2024).

_Citation keys above are short forms; full author lists and DOIs to be completed
at submission._
