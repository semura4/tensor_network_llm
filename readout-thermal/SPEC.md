# SPEC — Self-consistent electro-thermal model of a quantum-dot charge sensor

Target experiment: Mills et al., PRApplied 18, 064028 (2022), Fig. 1.
Sensor dot "S1", AC excitation V_exc (peak-to-peak) across the sensor,
demodulated current measured; charge signal = Coulomb-peak shift when the
target dot occupation changes N2: 0 -> 1.

## §1. Notation and fixed inputs
- Te: electron temperature of the sensor [K]. Tph: phonon/bath temperature [K].
- eps: detuning of the sensor-dot level from the lead chemical potential [J].
- V(t) = (Vpp/2)·sin(2π f t): excitation, symmetric bias drop (η = 1/2).
- f(x) = 1/(1 + exp(x)): Fermi function (dimensionless argument).
- Fixed from the paper: Tph (fridge base temperature), demodulation frequency f,
  effective noise bandwidth B (default 500 kHz from 1 MS/s sampling; B only
  ever appears in the product S_I·B, see §4).

## §2. Sensor transport (single-level sequential tunneling, kB·Te >> hΓ)
### §2.1 Finite-bias current
I(eps, V; Te) = A · [ f((eps − eV/2)/(kB·Te)) − f((eps + eV/2)/(kB·Te)) ]
where A [amperes] = e·ΓL·ΓR/(ΓL+ΓR) is a single calibration constant.
### §2.2 Linear-response check (must hold numerically)
G(eps) ≡ dI/dV |_{V→0} = (A·e)/(4·kB·Te) · sech²( eps/(2·kB·Te) )
Peak FWHM in eps: 3.525·kB·Te.
### §2.3 Lock-in signal
X(eps; Te, Vpp) = (2/T)·∫₀ᵀ I(eps, V(t); Te)·sin(2πft) dt   (in-phase first
harmonic, numerical quadrature over one period; result in amperes).
### §2.4 Charge states and signal
State 0: eps = eps_b. State 1: eps = eps_b + Δeps (Coulomb-peak shift from the
mutual capacitance to the target dot; Δeps is a calibration constant taken from
the measured peak shift, Fig. 1(b)).
Signal: ΔX(Vpp) = | X(eps_b) − X(eps_b + Δeps) |.
Bias point eps_b: DEFAULT = fixed value chosen to maximize ΔX at the base
condition (Vpp small, Te = Te0), mirroring experimental practice. Provide a
flag `reoptimize_bias=True` that re-optimizes eps_b at every (Vpp, Te) point;
report both curves in M4.

## §3. Thermal model (self-heating)
### §3.1 Dissipated power
P_diss(Te, Vpp) = (1/T)·∫₀ᵀ I(eps, V(t); Te)·V(t) dt, evaluated for each charge
state; use the MEAN of the two states (documented approximation — occupancy
duty cycle unknown).
### §3.2 Heat balance
Electron-phonon cooling (3D deformation potential, non-polar Si):
P_diss + P_bg = Σ_eff · (Te^p − Tph^p),  p = 5 (default).
Sensitivity requirement: results must also be produced for p ∈ {4, 6}.
Optional Wiedemann-Franz lead term (flag `include_WF`, default False):
P_WF = κ_WF · (Te² − Tph²).
P_bg is fixed by requiring Te(Vpp → 0) = Te0 (base electron temperature,
calibration constant).
### §3.3 Self-consistency
For each Vpp: solve R(Te) = P_diss(Te, Vpp) + P_bg − Σ_eff·(Te^p − Tph^p) = 0
for Te ∈ [Tph, 10 K] with scipy.optimize.brentq. Assert a sign change exists in
the bracket and that the root is unique (R monotone decreasing in Te beyond the
root); raise on non-convergence, never silently clip.

## §4. Noise and SNR
σ_I = sqrt(S_I · B): white, Vpp-independent noise floor (amplifier-dominated,
consistent with the HEMT chain of the target experiment). S_I·B is ONE fitted
constant (σ_I directly). Optional shot-noise term (flag, default False):
σ_shot² = 2·e·Ī·B.
SNR(Vpp) = ΔX(Vpp) / σ_I.

## §5. Calibration and success criteria
Fitted constants (exactly these, nothing else): A, Δeps, Te0, Σ_eff, σ_I.
Fit protocol (M4): least squares on the digitized Fig. 1(d) data
(data/mills_fig1d_snr.csv, data/mills_fig1d_te.csv; columns vexc_uvpp, value).
Success criteria:
  (a) RMS relative error ≤ 20% on both curves over the measured Vpp range;
  (b) predicted argmax of SNR within ±25% of the experimental optimum
      (≈ 85 µVpp);
  (c) sensitivity table: criteria (a)-(b) evaluated for p ∈ {4,5,6} and
      include_WF ∈ {False, True}.
The SHAPE of both curves (SNR rollover, superlinear Te rise) must emerge from
the model, not from the fit — 5 scalars cannot encode two curve shapes.

## §6. Required analytic-limit tests
T1 Linear conductance: numerical dI/dV at V→0 matches §2.2 to rel. tol 1e-6.
T2 FWHM: conductance peak width = 3.525·kB·Te within 0.5%.
T3 Baseline: Te(Vpp→0) = Te0 within solver tolerance.
T4 Monotonicity: Te strictly increasing in Vpp.
T5 Fixed point: exactly one sign change of R(Te) in the bracket.
T6 Small-signal SNR: SNR ∝ Vpp within 2% for e·Vpp < 0.1·kB·Te0.
T7 Heating shifts the optimum: argmax_Vpp SNR with self-heating ON is strictly
   smaller than with heating OFF (Σ_eff → ∞ limit).
T8 P_diss ≥ 0 everywhere.
