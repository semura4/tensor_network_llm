"""
hotq_pareto_v3 : Thermally assisted nonlinear readout (minimal model)
=====================================================================

Scope (read this before touching the physics):

    This is NOT a model of thermally improved qubit gates or thermally
    recovered coherence. It is a minimal model of a LATCHING METASTABLE
    CHARGE readout stage that sits AFTER spin-to-charge conversion (PSB /
    Elzerman-style). The object that escapes over a barrier is the classical
    / semiclassical latched charge state, not the qubit itself.

    Physical hierarchy modelled here:

        spin / PSB state
          -> spin-to-charge conversion
          -> latched metastable charge state      <-- Kramers barrier lives here
          -> thermally assisted nonlinear escape / threshold transition
          -> RF reflectometry / gate-based charge sensing  (linear OBSERVATION port)
          -> classifier

    RF reflectometry is treated as a linear/quasi-linear observation port that
    reads out the ALREADY-LATCHED charge state. It is NOT the nonlinear element.

All parameters are ASSUMED / ILLUSTRATIVE, not measured. See README.md.

Math conventions are pinned explicitly in this file so they are not silently
"completed" differently elsewhere:

  Kramers escape rate (state s in {0,1}):
      Gamma_s(T, A) = Gamma_attempt * exp[ -(DeltaU_s - eta_s * A) / (kB * T) ]

  Readout error over integration time t.
    State 1 SHOULD escape; state 0 should NOT escape.
      - miss of state 1        = P(no escape by t | s=1) = exp(-Gamma_1 * t)
      - false escape of state 0 = P(escape by t   | s=0) = 1 - exp(-Gamma_0 * t)
      P_err(t,T,A) = 0.5 * [ exp(-Gamma_1 t) + (1 - exp(-Gamma_0 t)) ]
      F_readout    = 1 - P_err

  Linear (RF-reflectometry-only) comparison with Johnson-Nyquist thermal noise
  made EXPLICIT so the linear channel is not arbitrarily disadvantaged:
      voltage noise PSD    S_v = 4 kB T R                      [V^2 / Hz]
      noise bandwidth      B   = 1 / (2 t)                     (integration time t)
      rms noise voltage    V_n = sqrt(4 kB T R * B) = sqrt(2 kB T R / t)
      SNR_linear(T,t)      = V_sig * sqrt(t) / sqrt(2 kB T R)  (~ S0 * sqrt(t)/sqrt(T))
      P_err_linear         = 0.5 * erfc( SNR_linear / (2 * sqrt(2)) )

  Spin relaxation T1(T) -- bounds usable readout time from above:
      1/T1(T) = rate_phonon * T  +  rate_multi * T^5
      p_survive = exp(-t / T1(T))

  Latch lifetime tau_latch(T) -- metastable charge hold time:
      Gamma_delatch(T) = Gamma_attempt * exp(-DeltaU_latch / (kB T))
      tau_latch(T)     = 1 / Gamma_delatch(T)
      p_latch          = exp(-t / tau_latch(T))

  Combined readout error (spin relaxation + latch decay):
      p_signal = p_survive * p_latch = exp(-t/T1) * exp(-t/tau_latch)
      P_err    = p_signal * P_err_kramers + (1 - p_signal) * 0.5

  Colored (Ornstein-Uhlenbeck) barrier noise, simplified:
      d eta = -(eta / tau_c) dt + sqrt(2 D / tau_c) dW
      => stationary variance Var[eta] = D,  so sigma = sqrt(D)
      Fast-noise (motional-narrowing) limit rate correction:
          <Gamma> = Gamma_attempt * exp[-(DeltaU_s - eta_s A)/(kB T)]
                                  * exp[ sigma_eff^2 / (2 (kB T)^2) ]
      tau_c enters only as a fast<->quasi-static interpolation knob (see below).
"""

from __future__ import annotations

import csv
import os

import numpy as np
from scipy.special import erfc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Physical constants (in the meV / K unit system used throughout)
# --------------------------------------------------------------------------
KB_MEV_PER_K = 0.0861733  # Boltzmann constant in meV / K

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
DATA_DIR = os.path.join(HERE, "data")


# --------------------------------------------------------------------------
# Default ILLUSTRATIVE parameters (NOT measured values)
# --------------------------------------------------------------------------
class Params:
    # Latched-charge effective barriers [meV]
    DeltaU0 = 6.0   # state 0: should NOT escape (higher barrier)
    DeltaU1 = 4.0   # state 1: should escape     (lower barrier)
    # Drive / readout amplitude expressed as an equivalent barrier lowering [meV]
    A = 0.5
    # Efficiency with which drive lowers the barrier (dimensionless)
    eta0 = 0.5
    eta1 = 1.0
    # Attempt frequency [Hz]
    Gamma_attempt = 1.0e9
    # Default readout integration time [s]
    t_readout = 1.0e-6

    # --- Linear (Johnson-Nyquist) readout parameters (ILLUSTRATIVE) ---
    # Signal voltage difference between the two latched charge states [V].
    # Chosen so the linear-vs-nonlinear comparison sits in a NON-TRIVIAL regime
    # (both channels have visible, T-dependent error). A larger V_sig makes the
    # linear channel trivially perfect and the comparison uninformative.
    V_sig = 1.0e-6      # 1 uV state-dependent RF signal
    R = 1.0e3           # effective source resistance [Ohm]
    KB_SI = 1.380649e-23  # Boltzmann constant [J/K] for the JN noise term

    # --- Spin relaxation T1(T) parameters (ILLUSTRATIVE) ---
    # Two-mechanism phenomenological model:
    #   1/T1(T) = rate_phonon * T  +  rate_multi * T^5
    # (direct one-phonon process  +  Raman/multi-phonon process)
    # Tuned to give illustrative values roughly matching Si spin qubits:
    #   T1(0.1K) ~ 10 ms,  T1(1K) ~ 1 ms,  T1(4K) ~ 10 us
    rate_phonon = 906.0    # [Hz / K]   (one-phonon / direct process)
    rate_multi = 94.1      # [Hz / K^5] (Raman / multi-phonon process)

    # --- Latch lifetime parameters (ILLUSTRATIVE) ---
    # The latched (escaped) charge state is metastable. It can de-latch
    # (return to the original well) via a reverse Kramers escape with
    # barrier DeltaU_latch. If the latch decays before readout completes,
    # the detector sees "no escape" regardless of the true qubit state.
    #
    # Gamma_delatch(T) = Gamma_attempt * exp(-DeltaU_latch / (kB T))
    # tau_latch(T)     = 1 / Gamma_delatch(T)
    #
    # Illustrative: DeltaU_latch = 8 meV gives tau_latch >> t_readout at 1-4 K
    # but tau_latch ~ t_readout around 8-10 K, providing a visible constraint
    # in the high-barrier regime without affecting the small-barrier / low-T regime.
    DeltaU_latch = 8.0     # [meV]  reverse barrier for de-latching


# --------------------------------------------------------------------------
# Spin relaxation T1(T) — constrains the maximum usable readout time
# --------------------------------------------------------------------------
def spin_T1(T, p: Params):
    """T1(T) [s] = 1 / (rate_phonon * T + rate_multi * T^5).

    Two-mechanism phenomenological model; all values ILLUSTRATIVE.
    """
    T = np.asarray(T, dtype=float)
    rate = p.rate_phonon * T + p.rate_multi * T ** 5
    return 1.0 / rate


def readout_error_with_T1(Gamma0, Gamma1, t, T1):
    """Readout error including spin relaxation during readout.

    The spin survives with probability p_s = exp(-t / T1). If it relaxes,
    the charge state randomizes and the outcome is a coin flip (P_err = 0.5).

    P_err = p_s * P_err_kramers + (1 - p_s) * 0.5
    """
    p_survive = np.exp(-t / T1)
    perr_kramers = readout_error(Gamma0, Gamma1, t)
    return p_survive * perr_kramers + (1.0 - p_survive) * 0.5


# --------------------------------------------------------------------------
# Latch lifetime — constrains how long the latched charge state persists
# --------------------------------------------------------------------------
def latch_lifetime(T, p: Params):
    """tau_latch(T) [s] = 1 / Gamma_delatch(T).

    Gamma_delatch = Gamma_attempt * exp(-DeltaU_latch / (kB T)).
    Kramers de-latching rate over the reverse barrier.
    """
    T = np.asarray(T, dtype=float)
    exponent = -p.DeltaU_latch / (KB_MEV_PER_K * T)
    exponent = np.maximum(exponent, -700.0)
    return 1.0 / (p.Gamma_attempt * np.exp(exponent))


def readout_error_full(Gamma0, Gamma1, t, T1, tau_latch):
    """Readout error including BOTH spin relaxation AND latch decay.

    The readout signal is valid only if the spin survives (exp(-t/T1)) AND
    the latch persists (exp(-t/tau_latch)). If either fails, P_err = 0.5.

    p_signal = exp(-t/T1) * exp(-t/tau_latch)
    P_err    = p_signal * P_err_kramers + (1 - p_signal) * 0.5
    """
    p_signal = np.exp(-t / T1) * np.exp(-t / tau_latch)
    perr_kramers = readout_error(Gamma0, Gamma1, t)
    return p_signal * perr_kramers + (1.0 - p_signal) * 0.5


# --------------------------------------------------------------------------
# Core rate / error model
# --------------------------------------------------------------------------
def kramers_rate(DeltaU, eta, A, T, Gamma_attempt):
    """Kramers escape rate [Hz].

    Gamma = Gamma_attempt * exp[ -(DeltaU - eta*A) / (kB T) ].
    """
    T = np.asarray(T, dtype=float)
    barrier = DeltaU - eta * A            # effective barrier after drive [meV]
    barrier = np.maximum(barrier, 0.0)    # a negative barrier => escape at attempt rate
    return Gamma_attempt * np.exp(-barrier / (KB_MEV_PER_K * T))


def readout_error(Gamma0, Gamma1, t):
    """P_err = 0.5 * [ exp(-Gamma_1 t) + (1 - exp(-Gamma_0 t)) ].

    State 1 should escape (miss = exp(-Gamma_1 t));
    state 0 should not escape (false escape = 1 - exp(-Gamma_0 t)).
    """
    miss_1 = np.exp(-Gamma1 * t)
    false_escape_0 = 1.0 - np.exp(-Gamma0 * t)
    return 0.5 * (miss_1 + false_escape_0)


def readout_fidelity(Gamma0, Gamma1, t):
    return 1.0 - readout_error(Gamma0, Gamma1, t)


# --------------------------------------------------------------------------
# Linear (RF reflectometry only) readout with explicit Johnson-Nyquist noise
# --------------------------------------------------------------------------
def snr_linear_jn(T, t, p: Params):
    """SNR of a linear RF-reflectometry channel, Johnson-Nyquist limited.

    S_v = 4 kB T R  [V^2/Hz];  B = 1/(2t);  V_n = sqrt(2 kB T R / t).
    SNR = V_sig * sqrt(t) / sqrt(2 kB T R).
    """
    T = np.asarray(T, dtype=float)
    v_noise = np.sqrt(2.0 * p.KB_SI * T * p.R / t)  # rms noise voltage [V]
    return p.V_sig / v_noise


def perr_linear(T, t, p: Params):
    """P_err_linear = 0.5 * erfc( SNR / (2 sqrt(2)) )."""
    snr = snr_linear_jn(T, t, p)
    return 0.5 * erfc(snr / (2.0 * np.sqrt(2.0)))


# --------------------------------------------------------------------------
# Colored (Ornstein-Uhlenbeck) noise, simplified as an effective-rate boost
# --------------------------------------------------------------------------
def ou_rate_correction_factor(sigma, T, tau_c, gamma_char):
    """Simplified OU colored-noise correction to the Kramers rate.

    OU process d eta = -(eta/tau_c) dt + sqrt(2D/tau_c) dW has stationary
    variance D, i.e. sigma = sqrt(D). We add eta as a fluctuating barrier
    lowering. In the fast-noise (motional-narrowing) limit the rate is boosted
    by the Gaussian average

        <exp(eta / kB T)> = exp( sigma^2 / (2 (kB T)^2) ).

    tau_c is folded in ONLY as a fast<->quasi-static interpolation knob:
        adiab = gamma_char * tau_c / (1 + gamma_char * tau_c)  in [0,1)
    where adiab -> 0 is the fast limit (full Gaussian boost) and adiab -> 1 is
    the quasi-static limit (the boost is suppressed because the barrier is
    frozen during the escape attempt and merely broadens the rate distribution
    rather than lifting the mean in this minimal treatment).

    This is a deliberately minimal spectral-shaping surrogate, NOT a solved
    colored-noise Kramers problem.
    """
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    fast_boost_exponent = sigma ** 2 / (2.0 * (KB_MEV_PER_K * T) ** 2)
    adiab = (gamma_char * tau_c) / (1.0 + gamma_char * tau_c)
    effective_exponent = (1.0 - adiab) * fast_boost_exponent
    effective_exponent = np.minimum(effective_exponent, 500.0)
    return np.exp(effective_exponent)


# --------------------------------------------------------------------------
# CSV helper
# --------------------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------
# Fig F : Kramers rates vs temperature
# --------------------------------------------------------------------------
def fig_F(p: Params):
    T = np.linspace(0.1, 10.0, 400)
    G0 = kramers_rate(p.DeltaU0, p.eta0, p.A, T, p.Gamma_attempt)
    G1 = kramers_rate(p.DeltaU1, p.eta1, p.A, T, p.Gamma_attempt)
    ratio = G1 / np.maximum(G0, 1e-300)

    G0_clipped = np.clip(G0, 1e-15, None)
    G1_clipped = np.clip(G1, 1e-15, None)
    ratio_clipped = np.clip(ratio, 1e-1, None)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.semilogy(T, G0_clipped, color="tab:blue", label=r"$\Gamma_0$ (state 0, should NOT escape)")
    ax1.semilogy(T, G1_clipped, color="tab:red", label=r"$\Gamma_1$ (state 1, should escape)")
    ax1.set_xlabel("Temperature T [K]")
    ax1.set_ylabel("Kramers escape rate [Hz]")
    ax1.set_ylim(1e-15, 1e12)
    ax1.axvspan(1.0, 4.0, color="gray", alpha=0.12, label="focus band 1-4 K")
    ax1.axhline(1.0 / p.t_readout, color="black", ls=":", lw=0.8,
                label=f"1/t_readout = {1/p.t_readout:.0e} Hz")
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.semilogy(T, ratio_clipped, color="tab:green", ls="--", label=r"$\Gamma_1/\Gamma_0$")
    ax2.set_ylabel(r"discrimination ratio $\Gamma_1/\Gamma_0$", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=7)
    ax1.set_title("Fig F: Kramers rates vs temperature (illustrative parameters)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figF_kramers_rates_vs_temperature.png"), dpi=140)
    plt.close(fig)

    rows = [[f"{t:.5f}", f"{g0:.6e}", f"{g1:.6e}", f"{r:.6e}"]
            for t, g0, g1, r in zip(T, G0, G1, ratio)]
    write_csv(os.path.join(DATA_DIR, "kramers_rates.csv"),
              ["T_K", "Gamma0_Hz", "Gamma1_Hz", "ratio_G1_over_G0"], rows)
    return T, G0, G1, ratio


# --------------------------------------------------------------------------
# Fig G : Error heatmap over (t_readout, T)  --  the CENTRAL figure
#         Two panels: (left) default high barriers, (right) small barriers
#         that place the useful window inside the 1-4 K focus band.
# --------------------------------------------------------------------------
def _fig_G_panel(ax, T, t, DeltaU0, DeltaU1, eta0, eta1, A, Gamma_attempt, p, title):
    """Single panel of Fig G: P_err heatmap with T1 + latch lifetime."""
    TT, tt = np.meshgrid(T, t)
    G0 = kramers_rate(DeltaU0, eta0, A, TT, Gamma_attempt)
    G1 = kramers_rate(DeltaU1, eta1, A, TT, Gamma_attempt)

    T1_arr = spin_T1(TT, p)
    tau_latch_arr = latch_lifetime(TT, p)
    Perr = readout_error_full(G0, G1, tt, T1_arr, tau_latch_arr)

    pcm = ax.pcolormesh(T, t, Perr, shading="auto", cmap="viridis",
                        vmin=0.0, vmax=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T [K]")
    cs = ax.contour(T, t, Perr, levels=[0.01, 0.05, 0.1, 0.25], colors="white",
                    linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    ax.axvspan(1.0, 4.0, color="white", alpha=0.08)

    T1_line = spin_T1(T, p)
    T1_visible = np.clip(T1_line, t.min(), t.max())
    ax.plot(T, T1_visible, color="red", ls="--", lw=2.0,
            label=r"$T_1(T)$ ceiling")

    tau_latch_line = latch_lifetime(T, p)
    tau_latch_visible = np.clip(tau_latch_line, t.min(), t.max())
    ax.plot(T, tau_latch_visible, color="orange", ls=":", lw=2.0,
            label=r"$\tau_{\mathrm{latch}}(T)$ ceiling")

    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(title, fontsize=9)
    return pcm, Perr


def fig_G(p: Params):
    T = np.linspace(0.1, 10.0, 220)
    t = np.logspace(-9, -2, 220)  # 1 ns .. 10 ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    # left: default (high barriers, window at ~5-9 K)
    pcm1, Perr_default = _fig_G_panel(
        ax1, T, t, p.DeltaU0, p.DeltaU1, p.eta0, p.eta1, p.A, p.Gamma_attempt, p,
        r"$\Delta U_0$=6, $\Delta U_1$=4 meV (window at 5-9 K)")
    ax1.set_ylabel("Readout integration time t [s]")

    # right: small barriers, window in the 1-4 K focus band
    pcm2, Perr_small = _fig_G_panel(
        ax2, T, t, 1.5, 0.8, p.eta0, p.eta1, p.A, p.Gamma_attempt, p,
        r"$\Delta U_0$=1.5, $\Delta U_1$=0.8 meV (window at 0.5-2 K)")

    cbar = fig.colorbar(pcm2, ax=[ax1, ax2], shrink=0.85)
    cbar.set_label(r"$P_{\mathrm{err}}(t,T,A)$ [with $T_1$ + latch lifetime]")
    fig.suptitle("Fig G: readout-error phase diagram with $T_1(T)$ + "
                 r"$\tau_{\mathrm{latch}}(T)$ ceilings [CENTRAL figure]",
                 fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.12, wspace=0.08)
    fig.savefig(os.path.join(FIG_DIR, "figG_error_heatmap_time_temperature.png"), dpi=140)
    plt.close(fig)

    # CSV: down-sample to keep the file readable
    rows = []
    for it in range(0, len(t), 4):
        for iT in range(0, len(T), 4):
            rows.append([f"{t[it]:.6e}", f"{T[iT]:.5f}",
                         f"{Perr_default[it, iT]:.6e}",
                         f"{Perr_small[it, iT]:.6e}"])
    write_csv(os.path.join(DATA_DIR, "error_heatmap.csv"),
              ["t_readout_s", "T_K", "P_err_default_full",
               "P_err_small_barrier_full"], rows)
    return T, t, Perr_default


# --------------------------------------------------------------------------
# Fig H : Stochastic-resonance-like optimum -- shown WITH regimes where it
#         does NOT exist (do not assume an optimum always exists)
# --------------------------------------------------------------------------
def fig_H(p: Params):
    T = np.linspace(0.1, 10.0, 500)

    # Two key regimes, each shown WITHOUT and WITH physical constraints:
    #  (a) default high barriers
    #  (d) small barriers (1-4 K focus)
    regimes = [
        dict(label="(a) DeltaU0=6, DeltaU1=4",
             DeltaU0=6.0, DeltaU1=4.0, t=1e-6, color="tab:blue"),
        dict(label="(d) DeltaU0=1.5, DeltaU1=0.8",
             DeltaU0=1.5, DeltaU1=0.8, t=1e-6, color="tab:purple"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    opt_rows = []

    for rg in regimes:
        G0 = kramers_rate(rg["DeltaU0"], p.eta0, p.A, T, p.Gamma_attempt)
        G1 = kramers_rate(rg["DeltaU1"], p.eta1, p.A, T, p.Gamma_attempt)

        # without constraints
        F_bare = readout_fidelity(G0, G1, rg["t"])
        ax1.plot(T, F_bare, color=rg["color"], label=rg["label"])

        i = int(np.argmax(F_bare))
        interior = 0 < i < len(T) - 1
        ax1.plot(T[i], F_bare[i], "o", color=rg["color"], ms=6,
                 mfc="white" if not interior else rg["color"])
        opt_rows.append([rg["label"] + " (bare)", f"{rg['DeltaU0']:.3f}",
                         f"{rg['DeltaU1']:.3f}", f"{rg['t']:.3e}",
                         f"{T[i]:.4f}", f"{F_bare[i]:.6f}",
                         "interior" if interior else "boundary"])

        # with T1 + latch lifetime
        T1_arr = spin_T1(T, p)
        tau_latch_arr = latch_lifetime(T, p)
        F_full = 1.0 - readout_error_full(G0, G1, rg["t"], T1_arr, tau_latch_arr)
        ax2.plot(T, F_full, color=rg["color"],
                 label=rg["label"] + r" + $T_1$ + $\tau_{\mathrm{latch}}$")

        j = int(np.argmax(F_full))
        interior_full = 0 < j < len(T) - 1
        ax2.plot(T[j], F_full[j], "o", color=rg["color"], ms=6,
                 mfc="white" if not interior_full else rg["color"])
        opt_rows.append([rg["label"] + " (full)", f"{rg['DeltaU0']:.3f}",
                         f"{rg['DeltaU1']:.3f}", f"{rg['t']:.3e}",
                         f"{T[j]:.4f}", f"{F_full[j]:.6f}",
                         "interior" if interior_full else "boundary"])

    for ax, title in [(ax1, "bare Kramers (no constraints)"),
                      (ax2, r"with $T_1(T)$ + $\tau_{\mathrm{latch}}(T)$")]:
        ax.set_xlabel("Temperature T [K]")
        ax.axvspan(1.0, 4.0, color="gray", alpha=0.12)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower center")
        ax.set_title(title, fontsize=10)
    ax1.set_ylabel(r"$F_{\mathrm{readout}} = 1 - P_{\mathrm{err}}$")
    fig.suptitle(r"Fig H: effect of $T_1(T)$ + $\tau_{\mathrm{latch}}(T)$ "
                 "on the readout fidelity window", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, "figH_stochastic_resonance_optimum.png"), dpi=140)
    plt.close(fig)

    write_csv(os.path.join(DATA_DIR, "optimum_points.csv"),
              ["regime", "DeltaU0_meV", "DeltaU1_meV", "t_readout_s",
               "T_opt_K", "F_readout_at_opt", "optimum_type"], opt_rows)
    return opt_rows


# --------------------------------------------------------------------------
# Fig I : Linear (Johnson-Nyquist) vs nonlinear (latched) readout
#         Two panels: (left) 1-D comparison at fixed t, (right) 2-D advantage
#         map over (T, t) showing exactly where nonlinear beats linear.
# --------------------------------------------------------------------------
def fig_I(p: Params):
    T_1d = np.linspace(0.1, 10.0, 400)
    t_fixed = p.t_readout

    # --- 1-D slice (left panel) ---
    G0_1d = kramers_rate(p.DeltaU0, p.eta0, p.A, T_1d, p.Gamma_attempt)
    G1_1d = kramers_rate(p.DeltaU1, p.eta1, p.A, T_1d, p.Gamma_attempt)
    Perr_nl_1d = readout_error(G0_1d, G1_1d, t_fixed)
    Perr_lin_1d = perr_linear(T_1d, t_fixed, p)

    # --- 2-D advantage map (right panel) ---
    T_2d = np.linspace(0.1, 10.0, 200)
    t_2d = np.logspace(-9, -2, 200)
    TT, tt = np.meshgrid(T_2d, t_2d)
    G0_2d = kramers_rate(p.DeltaU0, p.eta0, p.A, TT, p.Gamma_attempt)
    G1_2d = kramers_rate(p.DeltaU1, p.eta1, p.A, TT, p.Gamma_attempt)
    Perr_nl_2d = readout_error(G0_2d, G1_2d, tt)
    Perr_lin_2d = perr_linear(TT, tt, p)
    # advantage ratio: < 1 means nonlinear wins, > 1 means linear wins
    # use log10 for symmetric color scale
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log10(np.clip(Perr_nl_2d, 1e-30, 1) /
                         np.clip(Perr_lin_2d, 1e-30, 1))
    ratio = np.clip(ratio, -5, 5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # left: 1-D comparison
    ax1.semilogy(T_1d, np.clip(Perr_lin_1d, 1e-16, 1),
                 color="tab:purple", label="linear RF (JN limited)")
    ax1.semilogy(T_1d, np.clip(Perr_nl_1d, 1e-16, 1),
                 color="tab:red", label="nonlinear latched")
    ax1.set_xlabel("Temperature T [K]")
    ax1.set_ylabel(r"$P_{\mathrm{err}}$")
    ax1.axvspan(1.0, 4.0, color="gray", alpha=0.12, label="focus 1-4 K")
    ax1.grid(True, which="both", alpha=0.25)
    better = Perr_nl_1d < Perr_lin_1d
    if better.any():
        ax1.fill_between(T_1d, 1e-16, 1, where=better, color="tab:red", alpha=0.06,
                         label="nonlinear wins")
    ax1.legend(fontsize=7, loc="upper center")
    ax1.set_title(f"1-D comparison, t = {t_fixed:.0e} s", fontsize=9)

    # right: 2-D advantage map
    cmap = plt.cm.RdBu_r
    pcm = ax2.pcolormesh(T_2d, t_2d, ratio, shading="auto", cmap=cmap,
                         vmin=-5, vmax=5)
    ax2.set_yscale("log")
    ax2.set_xlabel("Temperature T [K]")
    ax2.set_ylabel("Readout integration time t [s]")
    ax2.contour(T_2d, t_2d, ratio, levels=[0], colors="black", linewidths=1.5)
    ax2.axvspan(1.0, 4.0, color="white", alpha=0.08)
    cbar = fig.colorbar(pcm, ax=ax2)
    cbar.set_label(r"$\log_{10}(P_{\mathrm{err,NL}} / P_{\mathrm{err,lin}})$"
                   "\n(blue = nonlinear wins, red = linear wins)")
    ax2.set_title("2-D advantage map (NL vs linear JN)", fontsize=9)

    fig.suptitle("Fig I: linear (Johnson-Nyquist) vs nonlinear latched readout", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, "figI_compare_linear_vs_nonlinear_readout.png"), dpi=140)
    plt.close(fig)
    return T_1d, Perr_lin_1d, Perr_nl_1d


# --------------------------------------------------------------------------
# Fig J : Colored (OU) noise sensitivity
# --------------------------------------------------------------------------
def fig_J(p: Params):
    # start from 1 K: the fast-noise OU approximation is invalid when sigma >> kBT,
    # which happens at very low T and produces unphysical spikes
    T = np.linspace(1.0, 10.0, 300)
    t = p.t_readout
    gamma_char = p.Gamma_attempt  # characteristic escape-attempt scale for the knob

    # sweep OU noise strength sigma [meV] (barrier fluctuation std) and tau_c
    # sigma must be comparable to kBT (~0.1-0.9 meV in the active range) to have
    # a visible effect on the rate via exp(sigma^2/(2 (kBT)^2))
    sigmas = [0.0, 0.2, 0.5]            # meV  (0.5 meV ~ kBT at 6 K)
    tau_cs = [1e-12, 1e-9]              # s  (fast vs slower correlation)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rows = []
    for sigma in sigmas:
        for tau_c in tau_cs:
            corr = ou_rate_correction_factor(sigma, T, tau_c, gamma_char)
            G0 = np.minimum(kramers_rate(p.DeltaU0, p.eta0, p.A, T, p.Gamma_attempt) * corr,
                            p.Gamma_attempt)
            G1 = np.minimum(kramers_rate(p.DeltaU1, p.eta1, p.A, T, p.Gamma_attempt) * corr,
                            p.Gamma_attempt)
            F = readout_fidelity(G0, G1, t)
            lbl = f"sigma={sigma:.2f} meV, tau_c={tau_c:.0e}s"
            ls = "-" if tau_c == tau_cs[0] else "--"
            ax.plot(T, F, ls=ls, label=lbl)
            for iT in range(0, len(T), 10):
                rows.append([f"{sigma:.3f}", f"{tau_c:.3e}", f"{T[iT]:.4f}",
                             f"{F[iT]:.6f}"])

    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel(r"$F_{\mathrm{readout}} = 1 - P_{\mathrm{err}}$")
    ax.axvspan(1.0, 4.0, color="gray", alpha=0.12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="lower center", ncol=2)
    ax.set_title("Fig J: colored (OU) noise sensitivity -- simplified rate correction")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figJ_colored_noise_sensitivity.png"), dpi=140)
    plt.close(fig)

    write_csv(os.path.join(DATA_DIR, "colored_noise_sweep.csv"),
              ["sigma_meV", "tau_c_s", "T_K", "F_readout"], rows)
    return rows


# --------------------------------------------------------------------------
# Parameter dependence: how T* and F* depend on barrier gap, Gamma_attempt, A
# --------------------------------------------------------------------------
def fig_param_dependence(p: Params):
    """Systematic sweep of T* vs key parameters (final report question 2)."""
    T = np.linspace(0.1, 10.0, 2000)
    t = p.t_readout

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) T* vs barrier gap  (DeltaU0 fixed, vary DeltaU1)
    ax = axes[0, 0]
    gaps = np.linspace(0.3, 5.0, 50)
    DeltaU0_fixed = 6.0
    Topt_gap, Fopt_gap = [], []
    for gap in gaps:
        DU1 = DeltaU0_fixed - gap
        if DU1 < 0:
            Topt_gap.append(np.nan)
            Fopt_gap.append(np.nan)
            continue
        G0 = kramers_rate(DeltaU0_fixed, p.eta0, p.A, T, p.Gamma_attempt)
        G1 = kramers_rate(DU1, p.eta1, p.A, T, p.Gamma_attempt)
        F = readout_fidelity(G0, G1, t)
        i = int(np.argmax(F))
        Topt_gap.append(T[i] if 0 < i < len(T) - 1 else np.nan)
        Fopt_gap.append(F[i] if 0 < i < len(T) - 1 else np.nan)
    ax.plot(gaps, Topt_gap, "o-", ms=3, color="tab:blue")
    ax.set_xlabel(r"barrier gap $\Delta U_0 - \Delta U_1$ [meV]")
    ax.set_ylabel(r"$T^*$ [K]")
    ax.set_title(r"(a) $T^*$ vs barrier gap ($\Delta U_0$=6 meV)")
    ax.axhspan(1, 4, color="gray", alpha=0.1)
    ax.grid(True, alpha=0.25)

    # (b) T* vs DeltaU0 (keeping gap = 2 meV fixed)
    ax = axes[0, 1]
    DU0s = np.linspace(1.0, 15.0, 50)
    gap_fixed = 2.0
    Topt_u0, Fopt_u0 = [], []
    for du0 in DU0s:
        du1 = du0 - gap_fixed
        G0 = kramers_rate(du0, p.eta0, p.A, T, p.Gamma_attempt)
        G1 = kramers_rate(du1, p.eta1, p.A, T, p.Gamma_attempt)
        F = readout_fidelity(G0, G1, t)
        i = int(np.argmax(F))
        Topt_u0.append(T[i] if 0 < i < len(T) - 1 else np.nan)
        Fopt_u0.append(F[i] if 0 < i < len(T) - 1 else np.nan)
    ax.plot(DU0s, Topt_u0, "s-", ms=3, color="tab:red")
    ax.set_xlabel(r"$\Delta U_0$ [meV]  (gap fixed at 2 meV)")
    ax.set_ylabel(r"$T^*$ [K]")
    ax.set_title(r"(b) $T^*$ vs absolute barrier height")
    ax.axhspan(1, 4, color="gray", alpha=0.1)
    ax.grid(True, alpha=0.25)

    # (c) T* vs Gamma_attempt
    ax = axes[1, 0]
    Gatts = np.logspace(6, 11, 40)  # 1 MHz .. 100 GHz
    Topt_ga, Fopt_ga = [], []
    for ga in Gatts:
        G0 = kramers_rate(p.DeltaU0, p.eta0, p.A, T, ga)
        G1 = kramers_rate(p.DeltaU1, p.eta1, p.A, T, ga)
        F = readout_fidelity(G0, G1, t)
        i = int(np.argmax(F))
        Topt_ga.append(T[i] if 0 < i < len(T) - 1 else np.nan)
        Fopt_ga.append(F[i] if 0 < i < len(T) - 1 else np.nan)
    ax.semilogx(Gatts, Topt_ga, "^-", ms=3, color="tab:green")
    ax.set_xlabel(r"$\Gamma_{\mathrm{attempt}}$ [Hz]")
    ax.set_ylabel(r"$T^*$ [K]")
    ax.set_title(r"(c) $T^*$ vs attempt frequency")
    ax.axhspan(1, 4, color="gray", alpha=0.1)
    ax.grid(True, alpha=0.25)

    # (d) T* vs drive amplitude A
    ax = axes[1, 1]
    As = np.linspace(0.0, 3.0, 40)
    Topt_a, Fopt_a = [], []
    for Aval in As:
        G0 = kramers_rate(p.DeltaU0, p.eta0, Aval, T, p.Gamma_attempt)
        G1 = kramers_rate(p.DeltaU1, p.eta1, Aval, T, p.Gamma_attempt)
        F = readout_fidelity(G0, G1, t)
        i = int(np.argmax(F))
        Topt_a.append(T[i] if 0 < i < len(T) - 1 else np.nan)
        Fopt_a.append(F[i] if 0 < i < len(T) - 1 else np.nan)
    ax.plot(As, Topt_a, "D-", ms=3, color="tab:purple")
    ax.set_xlabel("drive A [meV equivalent barrier lowering]")
    ax.set_ylabel(r"$T^*$ [K]")
    ax.set_title(r"(d) $T^*$ vs drive amplitude")
    ax.axhspan(1, 4, color="gray", alpha=0.1)
    ax.grid(True, alpha=0.25)

    fig.suptitle("Parameter dependence of optimal temperature T*", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "figK_parameter_dependence.png"), dpi=140)
    plt.close(fig)

    # CSV
    rows = []
    for g, to, fo in zip(gaps, Topt_gap, Fopt_gap):
        rows.append(["barrier_gap", f"{g:.3f}", f"{to}" if not np.isnan(to) else "nan",
                      f"{fo}" if not np.isnan(fo) else "nan"])
    for du0, to, fo in zip(DU0s, Topt_u0, Fopt_u0):
        rows.append(["DeltaU0_sweep", f"{du0:.3f}", f"{to}" if not np.isnan(to) else "nan",
                      f"{fo}" if not np.isnan(fo) else "nan"])
    for ga, to, fo in zip(Gatts, Topt_ga, Fopt_ga):
        rows.append(["Gamma_attempt", f"{ga:.3e}", f"{to}" if not np.isnan(to) else "nan",
                      f"{fo}" if not np.isnan(fo) else "nan"])
    for a, to, fo in zip(As, Topt_a, Fopt_a):
        rows.append(["drive_A", f"{a:.3f}", f"{to}" if not np.isnan(to) else "nan",
                      f"{fo}" if not np.isnan(fo) else "nan"])
    write_csv(os.path.join(DATA_DIR, "parameter_dependence.csv"),
              ["sweep_type", "param_value", "T_opt_K", "F_opt"], rows)


# --------------------------------------------------------------------------
# Final-report style summary printed to stdout
# --------------------------------------------------------------------------
def summarize(p: Params):
    T = np.linspace(0.1, 10.0, 2000)
    G0 = kramers_rate(p.DeltaU0, p.eta0, p.A, T, p.Gamma_attempt)
    G1 = kramers_rate(p.DeltaU1, p.eta1, p.A, T, p.Gamma_attempt)

    F_no_T1 = readout_fidelity(G0, G1, p.t_readout)
    i_no = int(np.argmax(F_no_T1))
    interior_no = 0 < i_no < len(T) - 1

    T1_vals = spin_T1(T, p)
    tau_latch_vals = latch_lifetime(T, p)
    Perr_full = readout_error_full(G0, G1, p.t_readout, T1_vals, tau_latch_vals)
    F_full = 1.0 - Perr_full
    i_full = int(np.argmax(F_full))
    interior_full = 0 < i_full < len(T) - 1

    Perr_nl = Perr_full
    Perr_lin = perr_linear(T, p.t_readout, p)
    adv = Perr_nl < Perr_lin

    print("=" * 70)
    print("hotq_pareto_v3 summary (ILLUSTRATIVE parameters, not measured)")
    print("=" * 70)
    print(f"default: DeltaU0={p.DeltaU0} meV, DeltaU1={p.DeltaU1} meV, "
          f"A={p.A} meV, eta0={p.eta0}, eta1={p.eta1}, "
          f"Gamma_attempt={p.Gamma_attempt:.1e} Hz, t={p.t_readout:.1e} s")
    print(f"T1 model: 1/T1 = {p.rate_phonon:.1f}*T + {p.rate_multi:.1f}*T^5  "
          f"[T1(1K)={spin_T1(1.0, p):.1e} s, T1(4K)={spin_T1(4.0, p):.1e} s]")
    print(f"Latch:   DeltaU_latch={p.DeltaU_latch} meV  "
          f"[tau_latch(4K)={latch_lifetime(4.0, p):.1e} s, "
          f"tau_latch(10K)={latch_lifetime(10.0, p):.1e} s]")
    print()
    if interior_no:
        print(f"[1a] BARE Kramers:   interior optimum at T*={T[i_no]:.2f} K, "
              f"F_readout={F_no_T1[i_no]:.4f}")
    else:
        print("[1a] BARE Kramers:   no interior temperature optimum "
              "(optimum sits at a T boundary).")
    if interior_full:
        print(f"[1b] WITH T1+latch:  interior optimum at T*={T[i_full]:.2f} K, "
              f"F_readout={F_full[i_full]:.4f}")
    else:
        print("[1b] WITH T1+latch:  no interior temperature optimum "
              "(optimum sits at a T boundary).")
    if interior_no and interior_full:
        dF = F_no_T1[i_no] - F_full[i_full]
        print(f"     penalty: F drops by {dF:.4f} "
              f"(T* shifts {T[i_no]:.2f} -> {T[i_full]:.2f} K)")
    print()
    if adv.any():
        print(f"[3] nonlinear (with T1+latch) beats linear (JN) over "
              f"T in [{T[adv].min():.2f}, {T[adv].max():.2f}] K "
              f"at t={p.t_readout:.0e} s")
    else:
        print("[3] nonlinear (with T1+latch) never beats linear (JN) "
              "at these defaults.")
    print()
    print("Figures written to figures/, data written to data/.")
    print("=" * 70)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    p = Params()
    fig_F(p)
    fig_G(p)
    fig_H(p)
    fig_I(p)
    fig_J(p)
    fig_param_dependence(p)
    summarize(p)


if __name__ == "__main__":
    main()
