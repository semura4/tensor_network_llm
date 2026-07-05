# QUESTIONS — for the human

## Q1 (M2, non-blocking — default include_WF=False is unaffected)
SPEC §3.2 gives the optional Wiedemann-Franz lead term as
P_WF = kappa_WF * (Te^2 - Tph^2) but does not state on which side of the heat
balance it enters. Implemented reading (thermal.py cooling_power): it is an
additional COOLING channel, i.e. P_diss + P_bg = Sigma_eff*(Te^p - Tph^p)
+ kappa_WF*(Te^2 - Tph^2) when include_WF=True, since a lead out-flow term
vanishes at Te = Tph and removes heat for Te > Tph. Note the induced changes:
with include_WF=True the §3.3 residual gains the same -kappa_WF*(Te^2 - Tph^2)
cooling term and the §3.2 P_bg condition becomes
P_bg = Sigma_eff*(Te0^p - Tph^p) + kappa_WF*(Te0^2 - Tph^2).
Please confirm before the M4 sensitivity table (criterion (c)) is interpreted.

## Q2 (M3, non-blocking — shot-noise flag defaults to False)
SPEC §4 defines the optional shot-noise term sigma_shot^2 = 2*e*Ibar*B but
does not define (i) how the mean current Ibar is averaged (over the drive
period? over the two charge states? absolute value?), (ii) how sigma_shot
combines with sigma_I (quadrature sum?), or (iii) whether SNR — literally
defined in §4 as DeltaX/sigma_I — uses the combined noise when the flag is
on. noise.shot_sigma therefore only evaluates the literal §4 expression,
takes Ibar as an explicit caller-supplied input, raises for Ibar < 0, and NO
combination rule or SNR redefinition is implemented. Please specify all three
before the shot-noise flag is used in M4+.

## Q4 (M3, non-blocking — affects only the reoptimize_bias=True curve)
With reoptimize_bias=True (SPEC §2.4), which eps_b enters the §3.1/§3.3 heat
balance? Implemented reading (sweep.sweep): Te is solved at the FIXED
params.eps_b, then eps_b is re-optimized for the signal at that Te — i.e. no
joint (Te, eps_b) self-consistent fixed point. A device truly operated at the
re-optimized bias would dissipate at that bias, which would couple the two.
Please confirm which reading the M4 reoptimize_bias curve should use.

## Q3 (M2, documented constraint)
thermal.background_power raises for Te0 < Tph: the §3.2 condition
Te(Vpp -> 0) = Te0 then requires P_bg < 0, i.e. a background heat SINK, which
we take to be unphysical. SPEC §3.2 does not state Te0 >= Tph explicitly —
flagging the added restriction per golden rule 2. Calibration (M4) should
constrain Te0 >= Tph accordingly.
