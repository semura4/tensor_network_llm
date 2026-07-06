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
   (Fig G, Fig H regime (c) gives a boundary-only optimum).
2. **Dependence on `DeltaU_0, DeltaU_1, Gamma_attempt, A`:** the window widens
   and moves to lower `T` when the barrier gap `DeltaU_0 - DeltaU_1` is larger,
   when `Gamma_attempt` is higher, and when `A` (weighted by `eta_s`,
   `eta_1 > eta_0`) lowers the state-1 barrier more than the state-0 barrier.
   Near-equal barriers destroy the window.
3. **When does nonlinear beat linear (JN)?** Only in a bounded temperature band
   (for defaults, roughly 6–8 K at `t = 1 µs`, `V_sig = 1 µV`, `R = 1 kΩ`),
   where JN noise has degraded the linear channel while `Gamma_1 t` has become
   ≳ 1 but `Gamma_0 t` is still small. Outside this band linear wins.
4. **When does heating break the scheme?** When `Gamma_0` grows enough that
   state 0 falsely escapes (`Gamma_1/Gamma_0 -> 1`, `P_err -> 0.5`), and — not
   modelled here but decisive in reality — when higher `T` shortens `T1` and
   the latch lifetime and raises JN noise.
5. **Central figure for AIST:** **Fig G** — the `(t, T)` phase diagram, because
   it shows honestly that the useful window is conditional.
6. **Parameters to replace with measured values:** `DeltaU_s`,
   `Gamma_attempt`, `T1`, latch hold time, RF readout SNR (`V_sig`, `R`), and
   the temperature dependence of the noise.
7. **What the model shows / does not show:** it shows *whether and where* a
   thermally assisted latched-readout window can exist and beat a JN-limited
   linear channel; it does **not** show any thermal benefit to qubits, gates,
   or coherence, and it is not a device prediction.
