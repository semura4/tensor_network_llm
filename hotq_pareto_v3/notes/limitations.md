# Limitations

This is a **minimal** model. It is intended to scope a hypothesis, not to
predict device performance.

## What is left out

- **T1 is modelled but simplified.** Spin relaxation `T1(T)` is now included
  as a two-mechanism power law (`1/T1 = rate_phonon*T + rate_multi*T^5`), but
  the rates are illustrative, and the model ignores valley splitting, magnetic
  field, spin-orbit coupling, and phonon spectral density.
- **Latch lifetime is modelled but simplified.** The reverse Kramers escape
  (`Gamma_delatch = Gamma_attempt * exp(-DeltaU_latch/(kBT))`) provides a
  first-order lifetime for the metastable latched charge state. The reverse
  barrier `DeltaU_latch` is illustrative. The model uses the same attempt
  frequency for forward and reverse escape, ignores charge noise, and treats
  latch decay as instantaneous loss of signal (coin flip) rather than
  modelling the actual charge relaxation dynamics.
- **Single attempt frequency, no prefactor physics.** `Gamma_attempt` is a
  constant. Real Kramers prefactors depend on curvature, friction/dissipation,
  and temperature.
- **No re-trapping or back-transitions.** Escape is one-way and irreversible in
  this model.
- **Colored noise is a surrogate.** The OU treatment is a fast-limit rate
  correction with a heuristic `tau_c` knob, not a solved colored-noise escape
  problem. It should not be read as a validated phonon-engineering result.
- **Linear model is a two-hypothesis Gaussian detector.** The JN model captures
  thermal voltage noise and integration time, but not amplifier back-action,
  1/f charge noise, resonator nonlinearity, or realistic matching.
- **Symmetric priors and a single decision event.** No classifier, no repeated
  sampling, no adaptive readout.
- **Parameters are illustrative.** Numbers were chosen to make the figures
  legible, not fitted to any device.

## Regimes where the model breaks down / self-defeats

- **Too hot:** `Gamma_0` grows until state 0 falsely escapes; discrimination
  `Gamma_1/Gamma_0` collapses and `P_err -> 0.5`. Higher device temperature
  also shortens `T1` (now modelled) and degrades the latch lifetime (now
  modelled). Both ceilings are visible in Fig G. JN noise in the linear
  channel also rises with `T`.
- **Too cold / too short `t`:** `Gamma_1 t << 1`, state 1 is missed and the
  optimum window in Fig H can disappear entirely (boundary optimum only).
- **Barriers too close (`DeltaU_0 - DeltaU_1` small):** discrimination is weak
  at all `T`; no useful window.

## Honest reading of the figures

- Fig G (central) shows the window is **conditional**: present for some
  `(t, T)` and absent for others. Two ceiling lines show where physical
  constraints cut off usable readout time: `T1(T)` (red dashed) and
  `τ_latch(T)` (orange dotted). The region above either ceiling is
  constraint-limited regardless of Kramers rates.
- Fig H (2-panel) shows fidelity bare (left) vs with T1 + τ_latch (right).
  High-barrier regimes degrade significantly (F ~ 0.94 → 0.66); small-barrier
  regimes are barely affected (τ_latch is very long at 1–2 K). Includes
  regimes with **no interior optimum**; do not read it as "there is always an
  optimal temperature."
- Fig I shows the nonlinear channel is advantageous only in a **bounded**
  temperature band, and only for the chosen illustrative `V_sig`, `R`, `t`.
