# Project: readout-thermal (Phase 1)

Goal: quantitatively reproduce Fig. 1(d) of Mills et al., Phys. Rev. Applied 18,
064028 (2022) / arXiv:2204.09551 — charge-readout SNR and electron temperature
Te as functions of sensor excitation amplitude V_exc — from a self-consistent
electro-thermal model.

## Golden rules (highest priority, override everything else)
1. NEVER invent physics. Every equation you implement must exist in SPEC.md
   with a section number. Every physics function's docstring must cite that
   section (e.g. "Implements SPEC §2.1").
2. If SPEC.md is ambiguous or silent on something you need, STOP. Write the
   question to QUESTIONS.md and end the session. Do not guess.
3. NEVER modify SPEC.md. (A PreToolUse hook enforces this; do not try to
   bypass it via bash.)
4. Tests first. Each milestone lists required tests; write them before or with
   the implementation. A PostToolUse hook runs pytest after every edit — treat
   any failure output it shows you as a blocker. A milestone is DONE only when
   `pytest -q` is fully green.
5. After each milestone, append to REPORT.md: what was built, pytest summary,
   open issues. Then invoke the physics-auditor subagent on the milestone's
   changes and address blockers before committing.

## Stack & conventions
- Python 3.12. Dependencies: numpy, scipy, matplotlib, pytest ONLY.
- Layout: src/readout_thermal/{constants,sensor,thermal,selfconsistent,noise,
  sweep,calibrate}.py ; tests/ ; scripts/ (figure generation) ; data/ .
- SI units internally everywhere. Convert to/from µVpp, mK only at I/O
  boundaries. kB and e live in constants.py and nowhere else.
- Pure functions, full type hints, no global state, vectorize over V_exc grids
  with numpy where natural.
- No notebooks in core; scripts/ writes PNGs.
- One commit per milestone, message "M<n>: <summary>".
