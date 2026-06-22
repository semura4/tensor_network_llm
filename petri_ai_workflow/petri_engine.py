"""Minimal Petri-net engine for AI workflow orchestration.

This module provides a tiny, dependency-free Petri-net implementation used to
model and safely manage the state of an AI worker (e.g. Claude Code / Codex)
workflow. It is intentionally small: no async, no parallelism, no GUI.

Core concepts:
    * ``Token``    - a movable marker carrying optional metadata.
    * ``Place``    - a named location that holds tokens.
    * ``Transition`` - a rule that consumes tokens from input places and
      produces tokens in output places, optionally gated by a ``guard``.
    * ``PetriNet`` - the container that owns places, transitions and the
      current ``marking`` (which tokens sit in which place).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# A guard receives the current marking and returns whether firing is allowed.
Marking = dict[str, list["Token"]]
Guard = Callable[[Marking], bool]


class PetriNetError(Exception):
    """Raised for any invalid Petri-net operation (e.g. illegal firing)."""


@dataclass
class Token:
    """A single token that flows through the net.

    Attributes:
        name: A short identifier for the token (not required to be unique).
        data: Arbitrary metadata attached to the token.
    """

    name: str = "token"
    data: dict[str, object] = field(default_factory=dict)


@dataclass
class Place:
    """A named place that can hold tokens.

    Attributes:
        name: Unique identifier of the place.
    """

    name: str


@dataclass
class Transition:
    """A transition that moves tokens between places.

    Attributes:
        name: Unique identifier of the transition.
        input_places: Places from which one token each is consumed.
        output_places: Places into which one token each is produced.
        guard: Optional predicate over the marking; firing is only allowed
            when it returns ``True``.
        action_name: Optional label naming the side effect / action that the
            orchestrator should associate with this transition.
    """

    name: str
    input_places: list[str]
    output_places: list[str]
    guard: Optional[Guard] = None
    action_name: Optional[str] = None


@dataclass
class PetriNet:
    """A Petri net: places, transitions and the current marking."""

    places: dict[str, Place] = field(default_factory=dict)
    transitions: dict[str, Transition] = field(default_factory=dict)
    marking: Marking = field(default_factory=dict)

    # -- construction helpers ------------------------------------------------

    def add_place(self, name: str) -> Place:
        """Register a place and initialise its (empty) marking."""
        if name in self.places:
            raise PetriNetError(f"place already exists: {name}")
        place = Place(name=name)
        self.places[name] = place
        self.marking.setdefault(name, [])
        return place

    def add_transition(self, transition: Transition) -> Transition:
        """Register a transition, validating that referenced places exist."""
        if transition.name in self.transitions:
            raise PetriNetError(f"transition already exists: {transition.name}")
        for place_name in (*transition.input_places, *transition.output_places):
            if place_name not in self.places:
                raise PetriNetError(
                    f"transition {transition.name!r} references unknown "
                    f"place {place_name!r}"
                )
        self.transitions[transition.name] = transition
        return transition

    def add_token(self, place_name: str, token: Optional[Token] = None) -> Token:
        """Place a token into ``place_name`` (creating a default token if needed)."""
        if place_name not in self.places:
            raise PetriNetError(f"unknown place: {place_name}")
        token = token if token is not None else Token()
        self.marking[place_name].append(token)
        return token

    # -- queries -------------------------------------------------------------

    def is_enabled(self, name: str) -> bool:
        """Return whether transition ``name`` can currently fire.

        A transition is enabled when every input place holds at least one
        token and its guard (if any) accepts the current marking.
        """
        transition = self._get_transition(name)
        for place_name in transition.input_places:
            if not self.marking.get(place_name):
                return False
        if transition.guard is not None and not transition.guard(self.marking):
            return False
        return True

    def enabled_transitions(self) -> list[str]:
        """Return the names of all currently enabled transitions."""
        return [name for name in self.transitions if self.is_enabled(name)]

    # -- firing --------------------------------------------------------------

    def fire_transition(self, name: str) -> Transition:
        """Fire transition ``name``, moving tokens from inputs to outputs.

        One token is consumed from each input place and one token is produced
        in each output place. Consumed-token metadata is forwarded to the
        produced tokens so that workflow data can flow downstream.

        Raises:
            PetriNetError: If the transition is unknown or not enabled.
        """
        transition = self._get_transition(name)
        if not self.is_enabled(name):
            raise PetriNetError(f"transition not enabled: {name}")

        consumed: list[Token] = []
        for place_name in transition.input_places:
            consumed.append(self.marking[place_name].pop(0))

        # Merge metadata from consumed tokens to carry context forward.
        carried: dict[str, object] = {}
        for token in consumed:
            carried.update(token.data)

        for place_name in transition.output_places:
            self.marking[place_name].append(
                Token(name=place_name, data=dict(carried))
            )
        return transition

    # -- internals -----------------------------------------------------------

    def _get_transition(self, name: str) -> Transition:
        if name not in self.transitions:
            raise PetriNetError(f"unknown transition: {name}")
        return self.transitions[name]
