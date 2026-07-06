# hotq_pareto_v3 — Thermally assisted nonlinear readout (minimal model)

## What this is (and what it is NOT)

**This model does NOT claim that heat improves qubit gates or restores quantum
coherence.** It is a **minimal model** that evaluates whether **thermal noise
can assist a state-dependent threshold escape in a latching, metastable charge
readout stage** that sits *after* spin-to-charge conversion (PSB / Elzerman
type). The object that escapes over a barrier is the **classical /
semiclassical latched charge state**, not the qubit.

RF reflectometry is treated as a **linear (or quasi-linear) observation port**
that reads out the *already-latched* charge state. **RF reflectometry is NOT
the nonlinear element**, and the RF resonator is *not* modelled as a bistable
nonlinear device.

> This model is not a model of thermally improved hot-qubit *gate* operation.
> It is a minimal model of a latching metastable charge readout stage, after
> PSB / spin-to-charge conversion, in which thermal noise may assist a
> state-dependent threshold escape.

### Physical hierarchy modelled

```
spin / PSB state
  -> spin-to-charge conversion
  -> latched metastable charge state          <-- Kramers barrier lives HERE
  -> thermally assisted nonlinear escape / threshold transition
  -> RF reflectometry / gate-based charge sensing   (linear OBSERVATION port)
  -> classifier
```

The Kramers escape is the escape of the **latched metastable charge state**,
not of a qubit coherence barrier.

## All parameters are ASSUMED / ILLUSTRATIVE

**Every numeric value in this repository is an assumed illustrative value, not
a measured value.** Before presenting this to AIST or any experimental group,
the following must be replaced with measured numbers:

- `DeltaU_s` — effective barriers of the latched charge readout stage
- `DeltaU_latch` — reverse barrier for de-latching
- `Gamma_attempt` — attempt frequency
- `T1(T)` — spin relaxation time vs temperature (bounds readout from above)
- `tau_latch(T)` — metastable charge hold time vs temperature (bounds readout from above)
- RF readout SNR
- temperature-dependent noise (including the resistance `R` in the JN term)

## Model

### Kramers escape rate

For state `s ∈ {0, 1}`, the latched-charge effective barrier is `DeltaU_s`, and
the drive/readout amplitude `A` lowers it with efficiency `eta_s`:

```
Gamma_s(T, A) = Gamma_attempt * exp[ -(DeltaU_s - eta_s * A) / (kB * T) ]
```

`DeltaU_s` is the effective barrier of the **latching charge readout stage**,
**not** a qubit-coherence barrier.

- `s = 1` should escape.
- `s = 0` should NOT escape.

### Readout error (each term pinned explicitly)

Over readout integration time `t`:

- miss of state 1 = `P(no escape by t | s=1) = exp(-Gamma_1 * t)`
- false escape of state 0 = `P(escape by t | s=0) = 1 - exp(-Gamma_0 * t)`

```
P_err(t, T, A) = 0.5 * [ exp(-Gamma_1 * t) + (1 - exp(-Gamma_0 * t)) ]
F_readout      = 1 - P_err
```

Too cold: `Gamma_1` is tiny, state 1 is missed. Too hot: `Gamma_0` grows too,
state 0 falsely escapes. An optimal window (`Gamma_1 >> Gamma_0`,
`Gamma_1 * t ≳ 1`, `Gamma_0 * t << 1`) **may or may not** exist depending on
parameters (see Fig G / Fig H).

### Linear (RF-reflectometry-only) comparison — Johnson–Nyquist noise made explicit

To avoid arbitrarily disadvantaging the linear channel, the Johnson–Nyquist
thermal noise is written out explicitly:

```
voltage noise PSD    S_v = 4 * kB * T * R            [V^2 / Hz]
noise bandwidth      B   = 1 / (2 * t)
rms noise voltage    V_n = sqrt(4 kB T R * B) = sqrt(2 kB T R / t)
SNR_linear(T, t)     = V_sig * sqrt(t) / sqrt(2 kB T R)
P_err_linear         = 0.5 * erfc( SNR_linear / (2 * sqrt(2)) )
```

This is the **primary** linear model used in Fig I. It reduces to the
**simplified form** `SNR_linear = S0 * sqrt(t) / sqrt(T)` with
`S0 = V_sig / sqrt(2 kB R)`; the simplified form is documented here only as the
scaling limit — the code uses the explicit JN expression. Because the linear
channel degrades with `T` through real JN noise, the comparison against the
nonlinear channel is fair rather than rigged.

`V_sig` is set to an illustrative value that places the comparison in a
**non-trivial regime** (both channels have visible, temperature-dependent
error). A larger `V_sig` would make the linear channel trivially perfect and
the comparison uninformative.

### Spin relaxation T1(T) — readout time ceiling

The spin can relax during readout. If it does, the charge state randomizes
and the measurement outcome is a coin flip. This bounds usable readout time
from above: `t` cannot exceed `T1(T)` without losing the signal.

Two-mechanism phenomenological model (all values **illustrative**):

```
1 / T1(T) = rate_phonon * T  +  rate_multi * T^5
```

- `rate_phonon = 906 Hz/K` — direct one-phonon process
- `rate_multi = 94.1 Hz/K^5` — Raman / multi-phonon process
- Illustrative values: `T1(1 K) ≈ 1 ms`, `T1(4 K) ≈ 10 µs`

### Latch lifetime τ_latch(T) — metastable charge hold time

The latched (escaped) charge state is metastable. It can de-latch (return to
the original well) via a reverse Kramers escape with barrier `DeltaU_latch`:

```
Gamma_delatch(T) = Gamma_attempt * exp(-DeltaU_latch / (kB * T))
tau_latch(T)     = 1 / Gamma_delatch(T)
```

- `DeltaU_latch = 8 meV` (illustrative) — the reverse barrier for de-latching
- Illustrative values: `tau_latch(4 K) ≈ 12 s` (no constraint),
  `tau_latch(10 K) ≈ 11 µs` (comparable to `t_readout`)

### Combined readout error (T1 + latch decay)

The readout signal is valid only if the spin survives **and** the latch
persists. If either fails, the outcome is a coin flip (`P_err = 0.5`):

```
p_signal = exp(-t / T1(T)) * exp(-t / tau_latch(T))
P_err    = p_signal * P_err_kramers  +  (1 - p_signal) * 0.5
```

The readout window is now bounded from **below** (Kramers rate too slow at
low `T`) and from **above** (T1 too short AND/OR latch decays at high `T`).
Fig G shows both ceilings: `T1(T)` (red dashed) and `τ_latch(T)` (orange
dotted); Fig H compares fidelity bare vs with both constraints.

### Colored (Ornstein–Uhlenbeck) noise — simplified

The OU process (with its stationary variance pinned explicitly):

```
d eta = -(eta / tau_c) dt + sqrt(2 D / tau_c) dW
=> stationary variance Var[eta] = D,   so sigma = sqrt(D)
```

`eta` is treated as a fluctuating **barrier lowering**. In the fast-noise
(motional-narrowing) limit the Kramers rate is boosted by the Gaussian average

```
<exp(eta / kB T)> = exp( sigma^2 / (2 (kB T)^2) )
```

`tau_c` enters **only** as a fast ⟷ quasi-static interpolation knob. This is a
deliberately minimal **noise-spectral-shaping surrogate**, not a solved
colored-noise Kramers problem, and it is used to evaluate *noise spectral
shaping* without exaggeration.

## Default illustrative parameters

| parameter | value | meaning |
|---|---|---|
| `DeltaU_0` | 6 meV | state-0 barrier (should not escape) |
| `DeltaU_1` | 4 meV | state-1 barrier (should escape) |
| `A` | 0.5 meV | drive as equivalent barrier lowering |
| `eta_0` | 0.5 | drive efficiency, state 0 |
| `eta_1` | 1.0 | drive efficiency, state 1 |
| `Gamma_attempt` | 1 GHz | attempt frequency |
| `t_readout` | 1 µs | integration time |
| `V_sig` | 1 µV | linear state-dependent RF signal (illustrative) |
| `R` | 1 kΩ | effective source resistance (illustrative) |
| `rate_phonon` | 906 Hz/K | one-phonon T1 relaxation rate (illustrative) |
| `rate_multi` | 94.1 Hz/K⁵ | multi-phonon T1 relaxation rate (illustrative) |
| `DeltaU_latch` | 8 meV | reverse barrier for de-latching (illustrative) |

Reference scales: `kB T ≈ 0.086 meV` at 1 K, `≈ 0.345 meV` at 4 K.
Sweep ranges: `T` 0.1–10 K (focus 1–4 K), `DeltaU_0` 1–20 meV,
`DeltaU_1` = `DeltaU_0 − (0.1…5) meV`, `Gamma_attempt` 1 MHz–100 GHz,
`t` 1 ns–10 ms, `A` 0–a few meV equivalent.

## Figures

| file | content |
|---|---|
| `figF_kramers_rates_vs_temperature.png` | Γ₀, Γ₁ and Γ₁/Γ₀ vs T |
| `figG_error_heatmap_time_temperature.png` | **CENTRAL**: P_err phase diagram over (t, T) with T1(T) + τ_latch(T) ceilings, two panels: high barriers (window at 5–9 K) and small barriers (window at 0.5–2 K in the focus band) |
| `figH_stochastic_resonance_optimum.png` | F_readout vs T — 2-panel: bare Kramers (left) vs with T1 + τ_latch (right); includes regimes with no interior optimum |
| `figI_compare_linear_vs_nonlinear_readout.png` | linear (JN) vs nonlinear latched readout (with T1 + τ_latch constraints); 2-D map includes T1 and τ_latch ceiling lines |
| `figJ_colored_noise_sensitivity.png` | OU colored-noise sensitivity (simplified) |
| `figK_parameter_dependence.png` | T* vs barrier gap, absolute barrier height, Γ_attempt, and drive A |

**Fig G is the central figure.** The 2-panel `(t, T)` phase diagram shows
honestly that an optimal window exists for *some* conditions and vanishes for
others. The left panel (default high barriers) has its window at 5–9 K; the
right panel (small barriers `DeltaU_0=1.5, DeltaU_1=0.8 meV`) demonstrates that
the window can fall **inside the 1–4 K focus band** when barriers are small
enough. Both panels now include two ceiling lines: `T1(T)` (red dashed) and
`τ_latch(T)` (orange dotted), and the error heatmap uses the fully constrained
`P_err` (T1 + latch decay). Fig H is a 2-panel comparison: left panel shows
bare Kramers fidelity, right panel shows fidelity with both T1 and latch
lifetime — the combined constraint degrades high-barrier regimes significantly
(F drops from ~0.94 to ~0.66) while small-barrier regimes are barely affected
(τ_latch is very long at 1–2 K). Fig H deliberately includes regimes where
**no interior optimum exists**.

**Fig K shows parameter dependence** (final report question 2). Key findings:
T* decreases with larger barrier gap, higher Γ_attempt, and larger drive A.
T* increases approximately linearly with the absolute barrier height. The
gray band marks the 1–4 K focus region — reaching it requires either small
barriers, large barrier gap, high attempt frequency, or strong drive.

## Run

```bash
pip install numpy scipy matplotlib
python nonlinear_readout_v3.py
```

Writes PNGs to `figures/` and CSVs to `data/`.

## See also

- `notes/assumptions.md`
- `notes/limitations.md`
- `notes/research_interpretation.md`
