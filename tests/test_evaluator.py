import random

from poker.evaluator import estimate_equity, rank_players, score_hand


def test_higher_hand_class_scores_better():
    board = ["2h", "5d", "9c", "Kh", "3s"]
    two_pair = score_hand(["2c", "5h"], board)       # two pair 5s and 2s
    trips = score_hand(["9h", "9d"], board)           # trip 9s
    assert trips < two_pair  # lower treys score = stronger hand


def test_rank_players_orders_best_first():
    board = ["Ah", "Kh", "Qh", "2c", "3d"]
    hands = [(0, ["Jh", "Th"]), (1, ["2h", "3h"]), (2, ["4c", "4d"])]
    ranking = rank_players(hands, board)
    # player 0 has royal flush on a heart-heavy board, should win
    assert ranking[0][0] == 0


def test_pocket_aces_heads_up_equity_is_favoured():
    rng = random.Random(0)
    equity = estimate_equity(["Ah", "Ac"], [], num_opponents=1, n_sims=800, rng=rng)
    assert 0.75 < equity < 0.95  # AA vs random hand heads-up is ~85%


def test_equity_drops_with_more_opponents():
    rng = random.Random(0)
    heads_up = estimate_equity(["Kh", "Kc"], [], num_opponents=1, n_sims=600, rng=rng)
    five_way = estimate_equity(["Kh", "Kc"], [], num_opponents=5, n_sims=600, rng=rng)
    assert five_way < heads_up


def test_equity_is_one_with_no_opponents():
    rng = random.Random(0)
    equity = estimate_equity(["Ah", "Ac"], ["2h", "3h", "4h", "5h", "9c"], num_opponents=0, n_sims=10, rng=rng)
    assert equity == 1.0
