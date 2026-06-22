# Petri-net AI Workflow Orchestrator (minimal)

A tiny, dependency-light **Petri net** engine for safely managing the state of
an AI worker (Claude Code / Codex / LLM) workflow. This is a *minimal
experiment* — no GUI, no web app, no LLM API calls, no async/parallel execution.

The modelled pipeline:

```
paper_loaded → hypothesis_extracted → code_generated → test_passed
             → human_approved → committed
```

The key safety property: **`committed` can only be reached after a
`human_approved` token exists.** The Petri-net structure enforces this — there
is no edge into `committed` that does not consume from `human_approved`.

## Layout

```
petri_ai_workflow/
├─ CLAUDE.md                  # guidance for AI agents working in this folder
├─ README.md                  # this file
├─ workflow.yaml              # workflow definition (places + transitions)
├─ petri_engine.py            # Place / Transition / Token / PetriNet
├─ orchestrator.py            # load YAML, build net, run demo, log to artifacts/
├─ artifacts/                 # JSON firing logs (generated)
├─ notebooks/
│  └─ 01_demo.ipynb
└─ tests/
   └─ test_petri_engine.py
```

## Requirements

- Python 3.11+
- External deps: `pyyaml` and (for tests) `pytest` only.

```bash
pip install pyyaml pytest
```

## Run the demo

From inside `petri_ai_workflow/`:

```bash
python orchestrator.py
```

This builds the net, fires each enabled transition in order, prints the marking
at every step, and writes one JSON log per firing into `artifacts/`.

## Run the tests

From inside `petri_ai_workflow/`:

```bash
python -m pytest tests/ -v
```

## Run the notebook

```bash
jupyter notebook notebooks/01_demo.ipynb
```

The notebook loads `workflow.yaml`, shows the current marking, fires the
transitions in order, and prints the resulting `artifacts/` logs.

## Concepts

- **Token** — a movable marker carrying optional `data` metadata.
- **Place** — a named state location holding zero or more tokens.
- **Transition** — consumes one token from each `input_places` and produces one
  token in each `output_places`; may be gated by an optional `guard` predicate.
- **PetriNet** — owns places, transitions, and the `marking`
  (`dict[str, list[Token]]`). Use `is_enabled`, `enabled_transitions`, and
  `fire_transition`. Illegal firings raise `PetriNetError`.
