"""Tests for the minimal Petri-net engine and orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``petri_engine`` / ``orchestrator`` importable when running pytest
# from anywhere (the package is a flat directory, not installed).
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from orchestrator import build_net, load_workflow  # noqa: E402
from petri_engine import PetriNet, PetriNetError, Token, Transition  # noqa: E402


@pytest.fixture
def net() -> PetriNet:
    """A PetriNet built from the shipped workflow.yaml."""
    spec = load_workflow(PKG_ROOT / "workflow.yaml")
    return build_net(spec)


def test_initial_token_in_source_place(net: PetriNet) -> None:
    """The single initial token starts in ``paper_loaded`` and nowhere else."""
    assert len(net.marking["paper_loaded"]) == 1
    other_counts = {
        name: len(tokens)
        for name, tokens in net.marking.items()
        if name != "paper_loaded"
    }
    assert all(count == 0 for count in other_counts.values()), other_counts


def test_enabled_transitions_are_correct(net: PetriNet) -> None:
    """Only ``extract_hypothesis`` is enabled from the initial marking."""
    assert net.enabled_transitions() == ["extract_hypothesis"]


def test_token_moves_on_fire(net: PetriNet) -> None:
    """Firing moves the token from the input place to the output place."""
    net.fire_transition("extract_hypothesis")
    assert len(net.marking["paper_loaded"]) == 0
    assert len(net.marking["hypothesis_extracted"]) == 1
    assert net.enabled_transitions() == ["generate_code"]


def test_cannot_commit_without_human_approval(net: PetriNet) -> None:
    """``commit`` stays disabled until a ``human_approved`` token exists."""
    # Advance up to test_passed but NOT through request_approval.
    net.fire_transition("extract_hypothesis")
    net.fire_transition("generate_code")
    net.fire_transition("run_tests")

    assert len(net.marking["human_approved"]) == 0
    assert "commit" not in net.enabled_transitions()
    with pytest.raises(PetriNetError):
        net.fire_transition("commit")

    # After approval, commit becomes possible and reaches ``committed``.
    net.fire_transition("request_approval")
    assert "commit" in net.enabled_transitions()
    net.fire_transition("commit")
    assert len(net.marking["committed"]) == 1


def test_invalid_transition_raises(net: PetriNet) -> None:
    """Unknown transition names and disabled firings raise PetriNetError."""
    with pytest.raises(PetriNetError):
        net.fire_transition("does_not_exist")

    # ``commit`` is not enabled from the initial marking.
    with pytest.raises(PetriNetError):
        net.fire_transition("commit")


def test_guard_blocks_firing() -> None:
    """A guard returning False keeps an otherwise-ready transition disabled."""
    net = PetriNet()
    net.add_place("a")
    net.add_place("b")
    net.add_token("a", Token(name="a"))
    net.add_transition(
        Transition(
            name="t",
            input_places=["a"],
            output_places=["b"],
            guard=lambda marking: False,
        )
    )
    assert net.enabled_transitions() == []
    with pytest.raises(PetriNetError):
        net.fire_transition("t")


def test_add_transition_with_unknown_place_raises() -> None:
    """Referencing a non-existent place when wiring a transition fails fast."""
    net = PetriNet()
    net.add_place("a")
    with pytest.raises(PetriNetError):
        net.add_transition(
            Transition(name="t", input_places=["a"], output_places=["ghost"])
        )
