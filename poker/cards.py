"""Card representation and deck handling.

Cards are plain two-character strings such as ``"Ah"`` or ``"Td"``
(rank + lowercase suit), which is exactly the format the `treys` hand
evaluator expects. No custom Card class is needed.
"""
import random

RANKS = "23456789TJQKA"
SUITS = "shdc"

FULL_DECK = [r + s for r in RANKS for s in SUITS]

RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}
SUIT_TO_INT = {s: i for i, s in enumerate(SUITS)}


class Deck:
    """A shuffled deck that cards can be dealt from."""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._cards = list(FULL_DECK)
        self._rng.shuffle(self._cards)

    def deal(self, n: int) -> list:
        """Deal (remove and return) ``n`` cards from the top of the deck."""
        dealt, self._cards = self._cards[:n], self._cards[n:]
        return dealt

    def __len__(self) -> int:
        return len(self._cards)


def rank_of(card: str) -> int:
    """0-12 index of a card's rank (2 lowest, ace highest)."""
    return RANK_TO_INT[card[0]]


def suit_of(card: str) -> int:
    """0-3 index of a card's suit."""
    return SUIT_TO_INT[card[1]]
