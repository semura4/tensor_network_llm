# Limitations

This is a **minimal** model. It is intended to scope a hypothesis, not to
predict device performance.

## What is left out

- **T1 is modelled but simplified.** Spin relaxation `T1(T)` is now included
  as a two-mechanism power law (`1/T1 = rate_phonon*T + rate_multi*T^5`), but
  the rates are illustrative, and the model ignores valley splitting, magnetic
  field, spin-orbit coupling, and phonon spectral density. **Latch lifetime**
  (metastable charge hold time) is still NOT modelled; it must be added and
  measured before any quantitative claim.
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
  also shortens `T1` (now modelled — the `T1(T)` ceiling is visible in Fig G)
  and raises JN noise in the linear channel. Latch lifetime degradation at
  high `T` is still not modelled — so the apparent high-`T` nonlinear
  advantage may still be somewhat optimistic.
- **Too cold / too short `t`:** `Gamma_1 t << 1`, state 1 is missed and the
  optimum window in Fig H can disappear entirely (boundary optimum only).
- **Barriers too close (`DeltaU_0 - DeltaU_1` small):** discrimination is weak
  at all `T`; no useful window.

## Honest reading of the figures

- Fig G (central) shows the window is **conditional**: present for some
  `(t, T)` and absent for others. The red dashed `T1(T)` ceiling shows where
  spin relaxation cuts off usable readout time — the region above the ceiling
  is T1-limited regardless of Kramers rates.
- Fig H (2-panel) shows fidelity *without* T1 (left) vs *with* T1 (right).
  High-barrier regimes degrade significantly (F ~ 0.94 → 0.66); small-barrier
  regimes are barely affected. Includes regimes with **no interior optimum**;
  do not read it as "there is always an optimal temperature."
- Fig I shows the nonlinear channel is advantageous only in a **bounded**
  temperature band, and only for the chosen illustrative `V_sig`, `R`, `t`.
