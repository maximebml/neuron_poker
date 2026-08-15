"""Hand strength evaluation, built on top of `treys`.

`treys.Evaluator.evaluate` returns an integer score where **lower is
better** (1 = royal flush, 7462 = worst high card). This module keeps
that convention internally but also exposes `estimate_equity`, a Monte
Carlo win-probability estimator used by the rule-based agent and the
situation-input decision tool to explain *why* a decision was made.
"""
import random
from functools import lru_cache

from treys import Card, Evaluator

from poker.cards import FULL_DECK

_evaluator = Evaluator()


@lru_cache(maxsize=None)
def _to_int(card: str) -> int:
    return Card.new(card)


def score_hand(hole_cards: list, board: list) -> int:
    """Score a 2-card hand + board (3-5 cards). Lower is better."""
    board_ints = [_to_int(c) for c in board]
    hole_ints = [_to_int(c) for c in hole_cards]
    return _evaluator.evaluate(board_ints, hole_ints)


def rank_players(hole_cards_by_player: list, board: list) -> list:
    """Rank players by hand strength at showdown.

    Args:
        hole_cards_by_player: list of (player_index, [card, card]) for
            players still eligible to win (not folded).
        board: 5 community cards.

    Returns:
        List of (player_index, score) sorted best hand first. Ties share
        the same score.
    """
    scored = [(idx, score_hand(cards, board)) for idx, cards in hole_cards_by_player]
    scored.sort(key=lambda t: t[1])
    return scored


def hand_class_name(hole_cards: list, board: list) -> str:
    """Human-readable hand class, e.g. 'Two Pair'. Requires >=3 board cards."""
    score = score_hand(hole_cards, board)
    return _evaluator.class_to_string(_evaluator.get_rank_class(score))


def estimate_equity(hole_cards: list, board: list, num_opponents: int,
                     n_sims: int = 300, rng: random.Random = None) -> float:
    """Monte Carlo estimate of win probability (ties count as a split).

    Deals random opponent hole cards and remaining community cards from
    the cards not already known (hero's hole cards + current board), and
    averages hero's showdown share across simulations.
    """
    rng = rng or random.Random()
    known = set(hole_cards) | set(board)
    remaining_deck = [c for c in FULL_DECK if c not in known]
    cards_needed_on_board = 5 - len(board)

    total_share = 0.0
    for _ in range(n_sims):
        pool = list(remaining_deck)
        rng.shuffle(pool)
        draw_idx = 0

        opp_hands = []
        for _ in range(num_opponents):
            opp_hands.append(pool[draw_idx:draw_idx + 2])
            draw_idx += 2

        sim_board = board + pool[draw_idx:draw_idx + cards_needed_on_board]

        hero_score = score_hand(hole_cards, sim_board)
        opp_scores = [score_hand(h, sim_board) for h in opp_hands]
        best_opp = min(opp_scores) if opp_scores else float("inf")

        if hero_score < best_opp:
            total_share += 1.0
        elif hero_score == best_opp:
            winners = 1 + sum(1 for s in opp_scores if s == hero_score)
            total_share += 1.0 / winners
        # else hero loses, contributes 0

    return total_share / n_sims
