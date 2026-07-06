# Assumptions

All values are **illustrative**, not measured.

## Physical picture

- The nonlinear element is a **latching, metastable charge sensor** (a
  latching quantum dot / metastable charge state), placed **after**
  spin-to-charge conversion (PSB / Elzerman type).
- The barrier `DeltaU_s` belongs to this **classical / semiclassical charge
  readout stage**, not to qubit coherence and not to a quantum gate.
- RF reflectometry (or gate-based charge sensing) is a **linear observation
  port** that reads the already-latched charge state. It is not itself the
  nonlinear/bistable element.

## Rate model

- A single Kramers-type escape rate per state:
  `Gamma_s = Gamma_attempt * exp[-(DeltaU_s - eta_s A)/(kB T)]`.
- Single attempt frequency `Gamma_attempt`, temperature-independent.
- Drive `A` acts purely as a barrier lowering with efficiency `eta_s`.
- Escape is a Poisson process: `P(escape by t) = 1 - exp(-Gamma_s t)`.
- The barrier is clipped at 0 (a negative effective barrier escapes at the
  attempt rate); no re-trapping / back-transition is modelled.

## Error model

- Symmetric prior over the two states (factor 0.5 each).
- `P_err = 0.5 [exp(-Gamma_1 t) + (1 - exp(-Gamma_0 t))]`.
- A single threshold-escape event decides the outcome; no full trajectory or
  classifier training is modelled.

## Linear comparison (Fig I)

- Johnson–Nyquist voltage noise PSD `S_v = 4 kB T R` is used explicitly.
- Noise bandwidth `B = 1/(2t)` for integration time `t`.
- Signal is a fixed state-dependent voltage `V_sig`; Gaussian detection gives
  `P_err_linear = 0.5 erfc(SNR/(2 sqrt2))`.
- `V_sig`, `R` are illustrative and chosen to keep the comparison non-trivial.

## Spin relaxation T1(T) (Fig G, Fig H)

- Two-mechanism phenomenological model:
  `1/T1(T) = rate_phonon * T + rate_multi * T^5`.
- `rate_phonon = 906 Hz/K` (direct one-phonon), `rate_multi = 94.1 Hz/K^5`
  (Raman/multi-phonon). All values **illustrative**.
- Spin survives readout with probability `p_survive = exp(-t / T1(T))`.
  If it relaxes, charge state randomizes → coin flip (P_err = 0.5).
- The T1 model is a simple power-law fit. Real T1(T) depends on valley
  splitting, magnetic field, spin-orbit coupling, and phonon spectral density
  — none of which are modelled.

## Latch lifetime τ_latch(T) (Fig G, Fig H)

- Reverse Kramers escape from the latched charge well:
  `Gamma_delatch(T) = Gamma_attempt * exp(-DeltaU_latch / (kB T))`,
  `tau_latch(T) = 1 / Gamma_delatch(T)`.
- `DeltaU_latch = 8 meV` (**illustrative**). The reverse barrier is larger
  than the forward barriers, so the latch is metastable by design.
- If the latch decays before readout, the detector sees "no escape" regardless
  of the true state. In the model this is approximated as a coin flip.
- Same attempt frequency `Gamma_attempt` is used for both forward and reverse
  escape. Real devices may have different prefactors.

## Combined readout error

- `p_signal = exp(-t/T1) * exp(-t/tau_latch)`: signal is valid only if spin
  survives AND latch persists. If either fails, P_err = 0.5.
- `P_err = p_signal * P_err_kramers + (1 - p_signal) * 0.5`.

## Colored noise (Fig J)

- OU barrier fluctuation `eta` with stationary variance `Var[eta] = D`,
  `sigma = sqrt(D)`.
- Fast-noise (motional-narrowing) rate boost `exp(sigma^2/(2 (kB T)^2))`.
- `tau_c` folded in only as a fast ⟷ quasi-static interpolation knob; this is a
  surrogate, not a solved colored-noise Kramers calculation.

## Units

- Energies in meV, temperature in K, `kB = 0.0861733 meV/K`.
- The JN term uses SI `kB = 1.380649e-23 J/K` with `V_sig` in volts.
