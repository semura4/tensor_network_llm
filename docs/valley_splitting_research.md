# Valley splitting in silicon exchange-only spin qubits — research synthesis

A cited literature synthesis (deep-research: 5 parallel search angles → cross-checked
claims → ranked report) to inform extending the `eo_pulse_ir` Heisenberg-exchange
toolkit with a **valley pseudo-spin** degree of freedom.

**Method / reliability caveat.** Five independent research passes were run (physics,
temperature, exchange effects, mitigation, requirements). Direct full-text fetch of
arXiv / Nature / APS returned HTTP 403 in this environment, so numeric values were
extracted from search-result summaries of the cited primary sources. Figures quoted
**consistently across multiple independent passes are flagged high-confidence**;
single-source or model-dependent numbers are flagged. Confirm load-bearing numbers
against the primary PDFs (arXiv IDs given).

Conversion anchors: **1 GHz ≈ 4.14 µeV**; **k_B T ≈ 86 µeV/K** (≈ 8.6 µeV at 100 mK,
86 µeV at 1 K, 340 µeV at 4 K).

---

## 1. Physics: 6 valleys → 2, and the interface-induced E_VS

- **Bulk Si has a 6-fold valley degeneracy** (Δ valleys along ⟨100⟩, at ~0.85 of the
  way to X); each is an ellipsoid with m_l = 0.98 m₀, m_t = 0.19 m₀. *[high confidence]*
- **Strain + (001) confinement lifts the 4 in-plane valleys by ~200 meV**, leaving the
  2 out-of-plane (z, "Δ2") valleys lowest. The user's framing — *"the 4-fold valleys are
  excluded because xy/z symmetry is broken"* — is **correct, stated precisely**: biaxial
  **strain (~200 meV) dominates**, vertical confinement is secondary (and selects z via
  the heavy m_l). Source: Friesen et al., PRB 75, 115318 (2007), arXiv:cond-mat/0608229.
  *[high confidence]*
- **The residual 2-fold z-valley degeneracy is split only by the sharp interface**
  (broken inversion symmetry coupling ±z). Typical **E_VS ≈ 10–300 µeV (~2–70 GHz)**,
  often 1–2 orders below ideal-well theory. *[high confidence — corroborated ×4]*
  Sources: Losert et al., PRB 108, 125405 (2023), arXiv:2303.02499; Thayil/Ermoneit/
  Kantner, arXiv:2412.20618 (2024).
- **Alloy disorder dominates and randomizes E_VS** → it is a random variable
  (Rayleigh in the pure-random limit, **Rice** with a deterministic component), so dots
  on one chip show a broad spread; low-E_VS "hotspots" are unavoidable. Sources:
  Wuetz et al., Nat. Commun. 13, 7730 (2022), arXiv:2112.09606; Losert 2023. *[high]*
- **Spatial variability is real and disorder-correlated**: in an Intel Si₀.₉₇₂Ge₀.₀₂₈
  well, E_VS correlations at <100 nm and >1 µm match alloy-disorder theory, near-Gaussian
  with correlation length ≈ dot size. Source: Nat. Commun. (2025), arXiv:2504.12455. *[high]*
- **Models**: sharp-interface effective-mass (valley coupling Δ as a δ-potential, E_VS
  ≈ linear in E-field) and atomistic tight-binding (sp³d⁵s\*, NEMO-3D) for random-alloy
  effects. Sources: Friesen 2007; arXiv:2412.20618. *[established]*

## 2. Temperature dependence

- **Design rule: energy scales (incl. E_VS) should be ≳ 5 k_B T** to avoid thermal
  population; "2|Δ| ≫ k_B T" for high-fidelity init/readout. *[high; the 5× is a rule of
  thumb]* Sources: Vahapoglu/Tosi review arXiv:1612.05936; arXiv:1001.5040.
- **E_VS < ~60 µeV → ~1% excited-valley population at 150 mK**, degrading Pauli-blockade
  readout. Source: arXiv:2504.12455 (2025). *[established]*
- **Spin-valley "hotspot": when E_VS = E_Zeeman, spin T1 collapses** (phonon-mediated
  spin-valley mixing; T1⁻¹ ∝ B⁵ donors / B⁷ dots). Hard constraint: keep E_VS away from
  the Zeeman energy. *[high confidence — corroborated ×3]* Sources: Nat. Commun. 4, 2069
  (2013), arXiv:1302.0983; arXiv:1907.04146.
- **Intervalley relaxation T1,valley ≈ 12 ms** (Si/SiGe, B-independent). Source: Penthorn
  et al., PR Applied 14, 054015 (2020), arXiv:2007.08680. *[established, direct measurement]*
- **Hot operation (>1 K) demonstrated**: single-qubit ≈ 99.85%, two-qubit ≈ 98.92%,
  readout/init ≈ 99.3% above 1 K (needs E_VS large enough to stay single-valley).
  Source: Huang et al., Nature 627, 772 (2024), arXiv:2308.02111. *[high — corroborated ×2]*
- Phonon-induced valley dephasing rates (~1.1 MHz valley-like vs ~140 kHz orbital-like)
  are **model-dependent — treat as contested**. Source: arXiv:1904.01852.

## 3. Effect on the exchange interaction (most relevant to this toolkit)

- **Valley-PHASE difference between dots suppresses exchange J — nulling at Δφ = π —
  even when E_VS is large in both dots.** Mechanism: valley-phase-dependent dressing by
  doubly-occupied states + modified Coulomb integrals. Source: Tariq & Hu, npj QI 8, 53
  (2022), arXiv:2107.00732. *[well-established theory — directly load-bearing for us]*
- **Inter- vs intra-valley tunnel couplings differ and vary across dots** (measured
  t′/t = 0.90 and 0.56 in adjacent pairs of a triple dot) → J is valley-dependent and
  non-uniform. Source: Mortemousque et al., PRX Quantum 2, 020309 (2021), arXiv:2101.12594.
  *[established, experiment]*
- **When E_VS is small the spin-only Heisenberg Hamiltonian is incomplete** — excited
  valley states enter the low-energy dynamics. Source: Tariq & Hu (2022). *[established]*
- **Spin-valley entanglement = leakage**; with improper valley init it grows, and for
  small E_VS it can be **hidden in 2-qubit benchmarks but appear at scale**. Source:
  Buterakos & Das Sarma, PRX Quantum 2, 040358 (2021), arXiv:2106.01391. *[established;
  the "hidden at 2-qubit level" point is a notable implication]*
- **Core requirement: E_VS ≫ J** (and ≫ Zeeman, ≫ k_B T) to avoid leakage — in tension
  with gate speed (∝ J). Source: Buterakos & Das Sarma (2021). *[high confidence]*
- **Exchange-only specifics**: HRL demonstrated universal encoded (3-spin) EO logic in a
  6-dot Si/SiGe SLEDGE device via nearest-neighbour partial-SWAP (Weinstein et al.,
  Nature 615, 817 (2023), arXiv:2202.03605). Measured EO leakage **0.17%/gate** — but
  attributed to **nuclear spins, not valley** (Andrews et al., Nat. Nano. 14, 747 (2019),
  arXiv:1812.02693). Valley leakage on SWAP is **sub-dominant to charge noise** in current
  devices (arXiv:2110.11329). *[established]*

## 4. Mitigation (device + control)

- **Wiggle Well** (oscillating Ge in the well): measured E_VS **54–239 µeV**, theory mean
  >200 µeV; enhancement is mostly **amplified random disorder**, not deterministic. Source:
  McJunkin et al., Nat. Commun. 13, 7777 (2022), arXiv:2112.09765. *[established, ×3]*
- **Ge spike in the well ≈ doubles E_VS**, ~independent of field/content/position. Source:
  Wuetz/Friesen, PRB 104, 085406 (2021), arXiv:2104.08232. *[established theory]*
- **Electric-field tuning**: E_VS gate-tunable (~15%; slope ~ −45 µeV/V corner dots); tune
  the spin-valley *hotspot* into a relaxation *cold spot*; T1 > 1 s. Source: PR Applied 13,
  034068 (2020), arXiv:1907.04146. *[established]*
- **Co-design SOTA**: mean E_VS **0.24(7) meV** with charge noise 0.9(3) µeV·Hz⁻¹ᐟ².
  Source: Degli Esposti et al., npj QI 10 (2024), arXiv:2309.02832. *[high — corroborated ×3]*
- **Control-level**: the dominant mitigations are (a) operate at large E_VS/k_B T (single-
  valley regime), (b) echo/CPMG for dephasing, (c) field operating points away from the
  hotspot, (d) velocity-shaped spin shuttling through E_VS minima (E_VS ≈ 20 µeV → error
  <10⁻³ over 10 µm at 8 m/s). Sources: arXiv:2411.11695 (2024); Volmer et al., npj QI 10,
  61 (2024), arXiv:2312.17694. There is **no established "valley-selective pulse gate"**
  beyond these. *[mixed: operate-large + echo established; shuttling schemes emerging]*

## 5. Requirements & consensus (2022–2026)

- **No single hard number; consensus is E_VS ≫ {k_B T, Zeeman, J}**, with a widely-used
  engineering target of **E_VS > 100 µeV at high yield (≳ 95% of dots)**. Achievable with
  ~5–15% Ge in/near the well (theory). Sources: Losert 2023 (arXiv:2303.02499); Thayil 2024
  (arXiv:2412.20618). *[high confidence]*
- **Valley splitting is a recognized scaling bottleneck — the problem is uniformity/yield
  across an array, not the peak value.** Sources: arXiv:2412.20618 (2024); arXiv:2512.18064
  (2025). *[consensus]*
- Foundry milestone: 300 mm Si-MOS unit cells >99% one- & two-qubit, SPAM >99.9%, valley
  variability flagged as a yield concern. Source: arXiv:2410.15590 / Nature (2025). *[established]*

---

## Implications for the `eo_pulse_ir` toolkit (how to add valley)

The current simulator is a **pure-spin Heisenberg model** (each dot = 2-level). The
literature points to a minimal, high-value valley extension:

1. **Per-dot valley state**: give each dot a valley pseudo-spin with a **splitting E_VS,i**
   and a **valley phase φ_i**. Draw E_VS,i from a **Rice/Rayleigh distribution** to model
   alloy disorder (matches §1) — this turns the toolkit into a tool for *valley-disorder*
   robustness studies.
2. **Valley-dependent exchange** (the load-bearing physics, §3): replace `J_ij` with an
   **effective coupling suppressed by valley-phase mismatch**, `J_ij^eff ≈ J_ij ·
   cos²((φ_i−φ_j)/2)`-type factor (nulls at Δφ = π). This is implementable directly in the
   existing scheduler/optimiser and immediately exposes a new failure mode (large E_VS but
   misaligned valley phase ⇒ dead exchange).
3. **Leakage channel** (§3): enlarge each dot's local space to 4 (spin⊗valley) and define
   leakage as population leaving the `spin ⊗ ground-valley` subspace; the validated CNOT/
   SWAP/CXSWAP fidelities can then be re-scored vs **E_VS/J** (expect collapse when
   E_VS ≲ J, recovering the E_VS ≫ J rule).
4. **Temperature** (§2): a thermal (density-matrix) initial state with excited-valley
   population `∝ exp(−E_VS/k_BT)`; sweep T to reproduce the ≳ 5 k_BT rule and the
   E_VS < 60 µeV ⇒ ~1% population result.
5. **Hotspot avoidance** (§2): flag operating points where **E_VS ≈ E_Zeeman**.

**Control priorities** (what matters most for pulse design): (i) **valley-phase robustness**
— optimise gate pulses to be insensitive to Δφ between dots (a new robustness axis for the
GRAPE/landscape layers, exactly the bifurcation/robustness analyses already in the repo);
(ii) **E_VS uniformity** as the dominant device constraint; (iii) keep **E_VS ≫ J** while
wanting fast gates (a genuine trade-off to map). The quaternion/SO(4) view is a clean
*description* of the combined spin⊗valley SU(2)×SU(2), but E_VS itself is set by the
heterostructure/temperature — quaternions organise the algebra, they do not raise E_VS.

### Primary sources
Friesen PRB 75 115318 (2007) arXiv:cond-mat/0608229 · Losert PRB 108 125405 (2023)
arXiv:2303.02499 · Thayil/Ermoneit/Kantner arXiv:2412.20618 (2024) · Wuetz Nat.Commun.
13 7730 (2022) arXiv:2112.09606 · McJunkin Nat.Commun. 13 7777 (2022) arXiv:2112.09765 ·
Wuetz/Friesen PRB 104 085406 (2021) arXiv:2104.08232 · Degli Esposti npj QI 10 (2024)
arXiv:2309.02832 · Tariq & Hu npj QI 8 53 (2022) arXiv:2107.00732 · Buterakos & Das Sarma
PRX Quantum 2 040358 (2021) arXiv:2106.01391 · Mortemousque PRX Quantum 2 020309 (2021)
arXiv:2101.12594 · Andrews Nat.Nano 14 747 (2019) arXiv:1812.02693 · Weinstein Nature 615
817 (2023) arXiv:2202.03605 · Penthorn PR Applied 14 054015 (2020) arXiv:2007.08680 ·
Huang Nature 627 772 (2024) arXiv:2308.02111 · Borjans/Hollmann PR Applied 13 034068 (2020)
arXiv:1907.04146 · Volmer npj QI 10 61 (2024) arXiv:2312.17694 · 300 mm foundry
arXiv:2410.15590 (2024).
