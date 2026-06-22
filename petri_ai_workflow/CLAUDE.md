# CLAUDE.md — petri_ai_workflow

Guidance for AI agents (Claude Code / Codex / LLM workers) operating in this
subproject.

## What this is

A **minimal** Petri-net engine for managing AI workflow state. It is an
experiment, not a production agent platform. Keep it small.

## Hard constraints — do NOT add (yet)

- ❌ GUI / web app
- ❌ Mathematica integration
- ❌ LLM API connections
- ❌ async / parallel execution
- ❌ external dependencies beyond `pyyaml` and `pytest`

## Conventions

- Python 3.11+; **type hints on every function**.
- Keep modules small and readable; prefer dataclasses.
- Tests come first: `python -m pytest tests/` must stay green.
- All illegal Petri-net operations raise `PetriNetError`.

## Key files

- `petri_engine.py` — `Token`, `Place`, `Transition`, `PetriNet`. The marking
  is `dict[str, list[Token]]`. Firing consumes one token per input place and
  produces one token per output place.
- `workflow.yaml` — declarative workflow: `places`, `initial_marking`,
  `transitions` (each with `input`, `output`, `action`).
- `orchestrator.py` — loads YAML, builds the net, runs the demo, writes JSON
  firing logs to `artifacts/`.

## Safety invariant

`committed` must only be reachable through a transition that consumes a
`human_approved` token. Preserve this when editing `workflow.yaml`.

## Before committing

1. `python -m pytest tests/ -v` — all tests pass.
2. `python orchestrator.py` — demo runs end-to-end to `committed`.
