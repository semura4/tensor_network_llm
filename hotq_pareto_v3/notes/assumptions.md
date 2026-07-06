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

## Colored noise (Fig J)

- OU barrier fluctuation `eta` with stationary variance `Var[eta] = D`,
  `sigma = sqrt(D)`.
- Fast-noise (motional-narrowing) rate boost `exp(sigma^2/(2 (kB T)^2))`.
- `tau_c` folded in only as a fast ⟷ quasi-static interpolation knob; this is a
  surrogate, not a solved colored-noise Kramers calculation.

## Units

- Energies in meV, temperature in K, `kB = 0.0861733 meV/K`.
- The JN term uses SI `kB = 1.380649e-23 J/K` with `V_sig` in volts.
