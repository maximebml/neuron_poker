"""Common interface implemented by every agent (random, rule-based, RL)."""
from poker.actions import Action
from poker.engine import PokerEngine


class Agent:
    """A policy that can act for a seat in a `PokerEngine` hand."""

    name = "agent"

    def act(self, engine: PokerEngine, seat: int) -> Action:
        """Return one of the actions in `engine.legal_actions(seat)`."""
        raise NotImplementedError
