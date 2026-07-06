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

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.semilogy(T, G0, color="tab:blue", label=r"$\Gamma_0$ (state 0, should NOT escape)")
    ax1.semilogy(T, G1, color="tab:red", label=r"$\Gamma_1$ (state 1, should escape)")
    ax1.set_xlabel("Temperature T [K]")
    ax1.set_ylabel("Kramers escape rate [Hz]")
    ax1.axvspan(1.0, 4.0, color="gray", alpha=0.12, label="focus band 1-4 K")
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.semilogy(T, ratio, color="tab:green", ls="--", label=r"$\Gamma_1/\Gamma_0$")
    ax2.set_ylabel(r"discrimination ratio $\Gamma_1/\Gamma_0$", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
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
# --------------------------------------------------------------------------
def fig_G(p: Params):
    T = np.linspace(0.1, 10.0, 220)
    t = np.logspace(-9, -2, 220)  # 1 ns .. 10 ms
    TT, tt = np.meshgrid(T, t)

    G0 = kramers_rate(p.DeltaU0, p.eta0, p.A, TT, p.Gamma_attempt)
    G1 = kramers_rate(p.DeltaU1, p.eta1, p.A, TT, p.Gamma_attempt)
    Perr = readout_error(G0, G1, tt)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pcm = ax.pcolormesh(T, t, Perr, shading="auto", cmap="viridis",
                        vmin=0.0, vmax=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("Readout integration time t [s]")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$P_{\mathrm{err}}(t,T,A)$")

    # contour at a "useful" error level
    cs = ax.contour(T, t, Perr, levels=[0.05, 0.1, 0.25], colors="white",
                    linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    ax.axvspan(1.0, 4.0, color="white", alpha=0.08)
    ax.set_title("Fig G: readout-error phase diagram (t vs T) [CENTRAL figure]")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figG_error_heatmap_time_temperature.png"), dpi=140)
    plt.close(fig)

    # CSV: down-sample to keep the file readable
    rows = []
    for it in range(0, len(t), 4):
        for iT in range(0, len(T), 4):
            rows.append([f"{t[it]:.6e}", f"{T[iT]:.5f}", f"{Perr[it, iT]:.6e}"])
    write_csv(os.path.join(DATA_DIR, "error_heatmap.csv"),
              ["t_readout_s", "T_K", "P_err"], rows)
    return T, t, Perr


# --------------------------------------------------------------------------
# Fig H : Stochastic-resonance-like optimum -- shown WITH regimes where it
#         does NOT exist (do not assume an optimum always exists)
# --------------------------------------------------------------------------
def fig_H(p: Params):
    T = np.linspace(0.1, 10.0, 500)

    # Three illustrative regimes:
    #  (a) default -> an interior optimum exists
    #  (b) barriers nearly equal -> weak / no useful optimum
    #  (c) very close barriers + short time -> optimum washed out
    regimes = [
        dict(label="(a) DeltaU0=6, DeltaU1=4 (clear window)",
             DeltaU0=6.0, DeltaU1=4.0, t=1e-6, color="tab:blue"),
        dict(label="(b) DeltaU0=6, DeltaU1=5.7 (weak/absent window)",
             DeltaU0=6.0, DeltaU1=5.7, t=1e-6, color="tab:orange"),
        dict(label="(c) DeltaU0=6, DeltaU1=4, t=1 ns (window washed out)",
             DeltaU0=6.0, DeltaU1=4.0, t=1e-9, color="tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    opt_rows = []
    for rg in regimes:
        G0 = kramers_rate(rg["DeltaU0"], p.eta0, p.A, T, p.Gamma_attempt)
        G1 = kramers_rate(rg["DeltaU1"], p.eta1, p.A, T, p.Gamma_attempt)
        F = readout_fidelity(G0, G1, rg["t"])
        ax.plot(T, F, color=rg["color"], label=rg["label"])

        # locate interior optimum, but only report it if it is genuinely interior
        i = int(np.argmax(F))
        interior = 0 < i < len(T) - 1
        ax.plot(T[i], F[i], "o", color=rg["color"], ms=6,
                mfc="white" if not interior else rg["color"])
        opt_rows.append([rg["label"], f"{rg['DeltaU0']:.3f}", f"{rg['DeltaU1']:.3f}",
                         f"{rg['t']:.3e}", f"{T[i]:.4f}", f"{F[i]:.6f}",
                         "interior" if interior else "boundary(no interior optimum)"])

    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel(r"$F_{\mathrm{readout}} = 1 - P_{\mathrm{err}}$")
    ax.axvspan(1.0, 4.0, color="gray", alpha=0.12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower center")
    ax.set_title("Fig H: temperature optimum EXISTS only in some regimes")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figH_stochastic_resonance_optimum.png"), dpi=140)
    plt.close(fig)

    write_csv(os.path.join(DATA_DIR, "optimum_points.csv"),
              ["regime", "DeltaU0_meV", "DeltaU1_meV", "t_readout_s",
               "T_opt_K", "F_readout_at_opt", "optimum_type"], opt_rows)
    return opt_rows


# --------------------------------------------------------------------------
# Fig I : Linear (Johnson-Nyquist) vs nonlinear (latched) readout
# --------------------------------------------------------------------------
def fig_I(p: Params):
    T = np.linspace(0.1, 10.0, 400)
    t = p.t_readout

    # nonlinear latched readout error
    G0 = kramers_rate(p.DeltaU0, p.eta0, p.A, T, p.Gamma_attempt)
    G1 = kramers_rate(p.DeltaU1, p.eta1, p.A, T, p.Gamma_attempt)
    Perr_nl = readout_error(G0, G1, t)

    # linear RF-reflectometry readout error with EXPLICIT Johnson-Nyquist noise
    Perr_lin = perr_linear(T, t, p)
    snr_lin = snr_linear_jn(T, t, p)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.semilogy(T, np.clip(Perr_lin, 1e-16, 1),
                color="tab:purple", label="linear RF (Johnson-Nyquist limited)")
    ax.semilogy(T, np.clip(Perr_nl, 1e-16, 1),
                color="tab:red", label="nonlinear latched readout")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel(r"$P_{\mathrm{err}}$")
    ax.axvspan(1.0, 4.0, color="gray", alpha=0.12, label="focus band 1-4 K")
    ax.grid(True, which="both", alpha=0.25)

    # mark crossover region where nonlinear beats linear
    better = Perr_nl < Perr_lin
    if better.any():
        ax.fill_between(T, 1e-16, 1, where=better, color="tab:red", alpha=0.06,
                        label="nonlinear advantageous")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_title(f"Fig I: linear (JN) vs nonlinear readout, t={t:.0e} s")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figI_compare_linear_vs_nonlinear_readout.png"), dpi=140)
    plt.close(fig)
    return T, Perr_lin, Perr_nl, snr_lin


# --------------------------------------------------------------------------
# Fig J : Colored (OU) noise sensitivity
# --------------------------------------------------------------------------
def fig_J(p: Params):
    T = np.linspace(0.1, 10.0, 300)
    t = p.t_readout
    gamma_char = p.Gamma_attempt  # characteristic escape-attempt scale for the knob

    # sweep OU noise strength sigma [meV] (barrier fluctuation std) and tau_c
    sigmas = [0.0, 0.05, 0.15]          # meV
    tau_cs = [1e-11, 1e-9]              # s  (fast vs slower correlation)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rows = []
    for sigma in sigmas:
        for tau_c in tau_cs:
            corr = ou_rate_correction_factor(sigma, T, tau_c, gamma_char)
            G0 = kramers_rate(p.DeltaU0, p.eta0, p.A, T, p.Gamma_attempt) * corr
            G1 = kramers_rate(p.DeltaU1, p.eta1, p.A, T, p.Gamma_attempt) * corr
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
# Final-report style summary printed to stdout
# --------------------------------------------------------------------------
def summarize(p: Params):
    T = np.linspace(0.1, 10.0, 2000)
    G0 = kramers_rate(p.DeltaU0, p.eta0, p.A, T, p.Gamma_attempt)
    G1 = kramers_rate(p.DeltaU1, p.eta1, p.A, T, p.Gamma_attempt)
    F = readout_fidelity(G0, G1, p.t_readout)
    i = int(np.argmax(F))
    interior = 0 < i < len(T) - 1

    Perr_nl = readout_error(G0, G1, p.t_readout)
    Perr_lin = perr_linear(T, p.t_readout, p)
    adv = Perr_nl < Perr_lin

    print("=" * 70)
    print("hotq_pareto_v3 summary (ILLUSTRATIVE parameters, not measured)")
    print("=" * 70)
    print(f"default: DeltaU0={p.DeltaU0} meV, DeltaU1={p.DeltaU1} meV, "
          f"A={p.A} meV, eta0={p.eta0}, eta1={p.eta1}, "
          f"Gamma_attempt={p.Gamma_attempt:.1e} Hz, t={p.t_readout:.1e} s")
    if interior:
        print(f"[1] interior temperature optimum FOUND at T*={T[i]:.2f} K, "
              f"F_readout={F[i]:.4f}")
    else:
        print("[1] NO interior temperature optimum for the default parameters "
              "(optimum sits at a T boundary).")
    if adv.any():
        print(f"[3] nonlinear beats linear (JN) over "
              f"T in [{T[adv].min():.2f}, {T[adv].max():.2f}] K at t={p.t_readout:.0e}s")
    else:
        print("[3] nonlinear never beats linear (JN) at these defaults.")
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
    summarize(p)


if __name__ == "__main__":
    main()
