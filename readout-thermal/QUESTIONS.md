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
does not define how the mean current Ibar is averaged (over the drive period?
over the two charge states? absolute value?). noise.shot_sigma therefore takes
Ibar as an explicit caller-supplied input and no averaging convention is baked
in. Please specify Ibar before the shot-noise flag is used in M4+.

## Q3 (M2, documented constraint)
thermal.background_power raises for Te0 < Tph: the §3.2 condition
Te(Vpp -> 0) = Te0 then requires P_bg < 0, i.e. a background heat SINK, which
we take to be unphysical. SPEC §3.2 does not state Te0 >= Tph explicitly —
flagging the added restriction per golden rule 2. Calibration (M4) should
constrain Te0 >= Tph accordingly.
