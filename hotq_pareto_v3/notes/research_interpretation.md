# Research interpretation

## The claim, stated narrowly

> Thermal noise may assist a **state-dependent threshold escape** in a
> **latching, metastable charge readout stage** located *after* spin-to-charge
> conversion.

The thermally assisted object is the **classical / semiclassical charge
amplifier**, not the qubit, not a quantum gate, and not quantum coherence.

## What this model can show

1. Whether an **optimal temperature / readout-time window** exists in which
   `Gamma_1 >> Gamma_0`, `Gamma_1 t ≳ 1`, `Gamma_0 t << 1` — and, crucially,
   the **conditions under which it does NOT exist** (Fig G, Fig H).
2. How that window depends on `DeltaU_0`, `DeltaU_1`, `Gamma_attempt`, and `A`.
3. The **bounded** conditions under which a latched nonlinear readout could
   beat a Johnson–Nyquist-limited linear RF channel (Fig I).
4. A first-order, honest sensitivity to **noise spectral shaping** via a
   simplified OU correction (Fig J).

## What this model CANNOT show

- It cannot show that heat improves qubit gates or coherence.
- It cannot show that a hot qubit is necessarily better than an mK qubit.
- It cannot establish a phonon-engineering advantage; Fig J is a surrogate.
- It cannot be used as a device prediction — every parameter is illustrative.

## Explicitly out of scope / disallowed framings

The following must NOT be stated anywhere in this repository:

- heat restores quantum coherence;
- heat improves quantum gates;
- heat is an unlimited energy source;
- phonons can be reused as a coherent wave;
- the RF resonator is itself a nonlinear bistable element from the start;
- classical/semiclassical post-latch charge amplification is the same thing as
  the quantum state itself;
- leakage error can be turned directly into a computational resource;
- a hot qubit is necessarily superior to an mK qubit;
- AIST or Mori-san's work has already demonstrated this hypothesis.

## Final-report answers (default illustrative parameters)

1. **Does an optimal temperature region exist?** Yes for the default
   parameters (interior optimum near `T* ≈ 7 K` at `t = 1 µs`), but it is
   **conditional**: it vanishes for near-equal barriers or very short `t`
   (Fig G left panel, Fig H regime (c) gives a boundary-only optimum).
   With smaller barriers (`DeltaU_0 = 1.5, DeltaU_1 = 0.8 meV`), the
   optimum shifts to `T* ≈ 0.8 K` inside the 1–4 K focus band (Fig G right
   panel, Fig H regime (d)).

2. **Dependence on `DeltaU_0, DeltaU_1, Gamma_attempt, A`** (see Fig K):
   - **Barrier gap `DeltaU_0 - DeltaU_1`:** larger gap → lower `T*`
     (gap ≈ 5 meV brings `T*` into the 1–4 K band at `DeltaU_0 = 6 meV`).
   - **Absolute barrier height:** `T*` rises approximately linearly with
     `DeltaU_0` (at fixed gap). Reaching the 1–4 K focus band requires
     barriers below ~4 meV (at gap = 2 meV).
   - **Attempt frequency `Gamma_attempt`:** higher → lower `T*` (100 GHz
     brings `T*` down to ~4 K with default barriers).
   - **Drive amplitude `A`:** larger → lower `T*` (A ≈ 3 meV brings `T*`
     to ~2.7 K); works because `eta_1 > eta_0` preferentially lowers the
     state-1 barrier.
   - Near-equal barriers destroy the window regardless of other parameters.

3. **When does nonlinear beat linear (JN)?** The 2-D advantage map (Fig I
   right panel) shows the nonlinear channel wins only in a **bounded island**
   in `(T, t)` space. For default parameters, the island spans roughly
   `T ∈ [5, 10] K` and `t ∈ [10^-7, 10^-5] s`. The boundary (black contour
   at equal error) sweeps from `T ≈ 5 K` at `t = 10^-5 s` to `T ≈ 10 K` at
   `t = 10^-7 s`. **In the 1–4 K focus band with default high barriers, the
   linear channel always wins** — the nonlinear channel's advantage requires
   either smaller barriers (to shift the island down in `T`) or different
   `V_sig`/`R` (to weaken the linear channel). This is an honest finding.

4. **When does heating break the scheme?** When `Gamma_0` grows enough that
   state 0 falsely escapes (`Gamma_1/Gamma_0 → 1`, `P_err → 0.5`). This is
   visible on Fig I as the advantage island closing at high `T`. In reality
   (not modelled here), higher `T` also shortens `T1` and the latch
   lifetime and raises JN noise — all of which narrow or eliminate the
   advantage window further.

5. **Central figure for AIST:** **Fig G** (two panels) — the `(t, T)` phase
   diagram, because it shows honestly that the useful window is conditional
   and its position depends on barrier size. Supplemented by Fig I (2-D
   advantage map) for the linear-vs-nonlinear comparison and Fig K for
   parameter design guidance.

6. **Parameters to replace with measured values:** `DeltaU_s`,
   `Gamma_attempt`, `T1`, latch hold time, RF readout SNR (`V_sig`, `R`), and
   the temperature dependence of the noise. Fig K provides design guidance:
   if the measured `DeltaU_s` are known, Fig K panel (b) tells where `T*`
   falls and whether the 1–4 K band is reachable.

7. **What the model shows / does not show:** it shows *whether and where* a
   thermally assisted latched-readout window can exist and beat a JN-limited
   linear channel, and *how to shift that window into the 1–4 K focus band*
   (smaller barriers, higher attempt frequency, or stronger drive). It does
   **not** show any thermal benefit to qubits, gates, or coherence, and it
   is not a device prediction.
