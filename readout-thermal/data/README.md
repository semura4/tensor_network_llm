# data/ — digitized Mills Fig. 1(d) (human-provided, required for M4)

M4 calibration (SPEC §5) needs two CSVs digitized from Mills et al.,
PRApplied 18, 064028 (2022) / arXiv:2204.09551 (v2), Fig. 1(d):

- `mills_fig1d_snr.csv` — columns `vexc_uvpp,value` (SNR, dimensionless)
- `mills_fig1d_te.csv`  — columns `vexc_uvpp,value` (Te in mK)

Per the project plan this digitization is a HUMAN task (check the APS
supplemental material for machine-readable data first; otherwise
WebPlotDigitizer on the arXiv PDF). Keep the raw digitizer project file in
`data/raw/`. M4 does not start until these files exist.
