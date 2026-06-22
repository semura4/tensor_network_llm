"""Orchestrator: build a PetriNet from YAML and run a demo firing sequence.

The orchestrator is deliberately synchronous and single-threaded. It loads a
workflow definition, constructs a :class:`PetriNet`, then fires the enabled
transitions in order, writing a JSON log to ``artifacts/`` for every firing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from petri_engine import PetriNet, Token, Transition

HERE = Path(__file__).resolve().parent
DEFAULT_WORKFLOW = HERE / "workflow.yaml"
ARTIFACTS_DIR = HERE / "artifacts"


def load_workflow(path: Path = DEFAULT_WORKFLOW) -> dict[str, Any]:
    """Load and return the raw workflow definition from a YAML file."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_net(spec: dict[str, Any]) -> PetriNet:
    """Build a :class:`PetriNet` from a parsed workflow specification."""
    net = PetriNet()

    for place_name in spec.get("places", []):
        net.add_place(place_name)

    for transition in spec.get("transitions", []):
        net.add_transition(
            Transition(
                name=transition["name"],
                input_places=list(transition.get("input", [])),
                output_places=list(transition.get("output", [])),
                action_name=transition.get("action"),
            )
        )

    for place_name, count in (spec.get("initial_marking") or {}).items():
        for _ in range(int(count)):
            net.add_token(place_name, Token(name=place_name))

    return net


def marking_summary(net: PetriNet) -> dict[str, int]:
    """Return a ``{place: token_count}`` snapshot of the current marking."""
    return {name: len(tokens) for name, tokens in net.marking.items()}


def log_firing(
    transition: Transition,
    net: PetriNet,
    step: int,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Path:
    """Write a JSON log entry for a firing and return the file path."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transition": transition.name,
        "action": transition.action_name,
        "input_places": transition.input_places,
        "output_places": transition.output_places,
        "marking_after": marking_summary(net),
    }
    path = artifacts_dir / f"step_{step:02d}_{transition.name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2)
    return path


def run_demo(
    path: Path = DEFAULT_WORKFLOW,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> PetriNet:
    """Run the full demo: fire enabled transitions until the net is quiescent."""
    spec = load_workflow(path)
    net = build_net(spec)

    print(f"workflow: {spec.get('name', '<unnamed>')}")
    print(f"initial marking: {marking_summary(net)}")

    step = 0
    while True:
        enabled = net.enabled_transitions()
        if not enabled:
            break
        name = enabled[0]
        transition = net.fire_transition(name)
        step += 1
        log_path = log_firing(transition, net, step, artifacts_dir)
        print(
            f"[step {step}] fired {name!r} "
            f"-> marking {marking_summary(net)} "
            f"(log: {log_path.name})"
        )

    print(f"final marking: {marking_summary(net)}")
    return net


if __name__ == "__main__":
    run_demo()
