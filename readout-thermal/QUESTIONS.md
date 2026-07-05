# QUESTIONS — for the human

## Q1 (M2, non-blocking — default include_WF=False is unaffected)
SPEC §3.2 gives the optional Wiedemann-Franz lead term as
P_WF = kappa_WF * (Te^2 - Tph^2) but does not state on which side of the heat
balance it enters. Implemented reading (thermal.py cooling_power): it is an
additional COOLING channel, i.e. P_diss + P_bg = Sigma_eff*(Te^p - Tph^p)
+ kappa_WF*(Te^2 - Tph^2) when include_WF=True, since a lead out-flow term
vanishes at Te = Tph and removes heat for Te > Tph. Please confirm before the
M4 sensitivity table (criterion (c)) is interpreted.
