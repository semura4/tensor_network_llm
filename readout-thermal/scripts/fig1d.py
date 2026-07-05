"""Generate a Mills Fig. 1(d)-style dual-axis figure (SNR left, Te right,
x = V_exc in µVpp) from the self-consistent model with PLACEHOLDER
calibration constants (M3). The calibrated overlay is produced in M4.

Run: python scripts/fig1d.py  (writes figures/fig1d_placeholder.png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from readout_thermal.constants import KB
from readout_thermal.sensor import default_bias
from readout_thermal.sweep import k_to_mk, sweep, uvpp_to_v, v_to_uvpp
from readout_thermal.thermal import ModelParams

# --- PLACEHOLDER calibration constants (the five of SPEC §5) --------------
# These are order-of-magnitude stand-ins for direct visual comparison only;
# M4 replaces them by the least-squares fit to the digitized data.
A_CONST = 1e-9  # A  (§2.1)
TE0 = 0.100  # K  (§3.2)
DELTA_EPS = 2.0 * KB * TE0  # J  (§2.4)
SIGMA_EFF = 5e-11  # W/K^5 (§3.2)
SIGMA_I = 2e-13  # A  (§4)
TPH = 0.050  # K (§1 fixed input Tph is ALSO a placeholder here: the paper's
# fridge base temperature must be filled in for M4; not cited yet)


def main() -> None:
    params = ModelParams(
        a=A_CONST,
        delta_eps=DELTA_EPS,
        eps_b=default_bias(TE0, A_CONST, DELTA_EPS),
        te0=TE0,
        tph=TPH,
        sigma_eff=SIGMA_EFF,
    )
    vexc_uvpp = np.logspace(np.log10(2.0), np.log10(1000.0), 60)
    result = sweep(uvpp_to_v(vexc_uvpp), params, SIGMA_I)

    fig, ax_snr = plt.subplots(figsize=(5.0, 3.6))
    ax_te = ax_snr.twinx()

    ax_snr.plot(v_to_uvpp(result.vpp), result.snr, "o-", color="tab:blue",
                ms=3, lw=1.2, label="SNR (model)")
    ax_te.plot(v_to_uvpp(result.vpp), k_to_mk(result.te), "s-",
               color="tab:red", ms=3, lw=1.2, label=r"$T_e$ (model)")

    ax_snr.set_xscale("log")
    ax_snr.set_xlabel(r"$V_\mathrm{exc}$ ($\mu$V$_\mathrm{pp}$)")
    ax_snr.set_ylabel("SNR", color="tab:blue")
    ax_te.set_ylabel(r"$T_e$ (mK)", color="tab:red")
    ax_snr.tick_params(axis="y", labelcolor="tab:blue")
    ax_te.tick_params(axis="y", labelcolor="tab:red")

    i_max = int(np.argmax(result.snr))
    ax_snr.axvline(float(v_to_uvpp(result.vpp[i_max])), color="gray",
                   ls=":", lw=1)
    ax_snr.set_title(
        "Mills Fig. 1(d) style — placeholder constants (M3)\n"
        f"model SNR optimum at {v_to_uvpp(result.vpp[i_max]):.0f} "
        r"$\mu$V$_\mathrm{pp}$",
        fontsize=9,
    )
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    path = out / "fig1d_placeholder.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
