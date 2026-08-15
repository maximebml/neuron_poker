"""Discrete action space used by the environment and agents."""
from enum import IntEnum


class Action(IntEnum):
    """A discrete decision a player can make on their turn.

    Raises are expressed as a fraction of the pot rather than a fixed chip
    amount, so the same action id is meaningful at any stack depth or pot
    size. ``engine.legal_actions`` translates each into a concrete chip
    contribution for the current state.
    """

    FOLD = 0
    CHECK_CALL = 1
    RAISE_33 = 2
    RAISE_75 = 3
    RAISE_150 = 4
    ALL_IN = 5


ALL_ACTIONS = list(Action)
N_ACTIONS = len(ALL_ACTIONS)

# Fraction of the pot (after calling) added on top of the call amount.
RAISE_POT_FRACTIONS = {
    Action.RAISE_33: 0.33,
    Action.RAISE_75: 0.75,
    Action.RAISE_150: 1.5,
}
