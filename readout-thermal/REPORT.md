# REPORT — readout-thermal Phase 1

## M0 — scaffold (2026-07-05)
Built: src layout (constants, sensor, thermal, selfconsistent, noise, sweep,
calibrate — physics modules are placeholders), tests/, scripts/, data/raw/,
pyproject.toml with pytest config, REPORT.md, QUESTIONS.md. constants.py holds
kB and e (CODATA 2018 exact SI values). Smoke test imports every module and
pins the constant values.
Pytest: 9 passed.
Open issues: none. No physics implemented, so no physics-auditor run for M0.
Note: environment provides Python 3.12.3 via a project-local venv (.venv/,
gitignored).
