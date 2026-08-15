import random

import pytest

from poker.actions import Action
from poker.engine import PokerEngine, Street


def play_randomly(engine, rng):
    steps = 0
    while not engine.done:
        seat = engine.current_player()
        legal = engine.legal_actions(seat)
        assert legal, f"seat {seat} has no legal actions"
        engine.step(rng.choice(list(legal.keys())))
        steps += 1
        assert steps < 1000, "runaway hand"


@pytest.mark.parametrize("trial", range(300))
def test_chip_conservation_random_play(trial):
    rng = random.Random(trial)
    n = rng.randint(2, 6)
    stacks = [rng.choice([0, rng.randint(1, 10), rng.randint(1, 500)]) for _ in range(n)]
    if sum(s > 0 for s in stacks) < 2:
        for i in rng.sample(range(n), 2):
            stacks[i] = rng.randint(1, 200)

    engine = PokerEngine(num_players=n, small_blind=1.0, big_blind=2.0, rng=rng)
    start_total = sum(stacks)
    engine.start_hand(stacks, button=rng.randrange(n))
    play_randomly(engine, rng)

    assert abs(sum(p.stack for p in engine.players) - start_total) < 1e-6
    assert abs(sum(engine.payouts.values())) < 1e-6
    assert all(p.stack >= -1e-6 for p in engine.players)


def test_heads_up_button_is_small_blind():
    engine = PokerEngine(num_players=2, small_blind=1.0, big_blind=2.0, rng=random.Random(0))
    engine.start_hand([100, 100], button=0)
    assert engine.sb_seat == 0
    assert engine.bb_seat == 1
    # heads-up preflop: SB/button acts first
    assert engine.current_player() == 0


def test_six_max_preflop_action_starts_utg():
    engine = PokerEngine(num_players=6, small_blind=1.0, big_blind=2.0, rng=random.Random(0))
    engine.start_hand([100] * 6, button=0)
    assert engine.sb_seat == 1
    assert engine.bb_seat == 2
    assert engine.current_player() == 3  # UTG, left of BB


def test_everyone_folds_awards_uncontested_pot():
    engine = PokerEngine(num_players=3, small_blind=1.0, big_blind=2.0, rng=random.Random(0))
    engine.start_hand([100, 100, 100], button=0)
    seat = engine.current_player()
    while not engine.done:
        legal = engine.legal_actions(seat)
        if Action.FOLD in legal:
            engine.step(Action.FOLD)
        else:
            engine.step(Action.CHECK_CALL)
        seat = engine.current_player()
    assert engine.done
    winner = max(engine.payouts, key=engine.payouts.get)
    assert engine.payouts[winner] > 0
    assert sum(engine.payouts.values()) == pytest.approx(0.0, abs=1e-6)


def test_check_is_only_option_when_no_bet_faced():
    engine = PokerEngine(num_players=2, small_blind=1.0, big_blind=2.0, rng=random.Random(0))
    engine.start_hand([100, 100], button=0)
    # play preflop out to the flop with calls/checks only
    while engine.street == Street.PREFLOP:
        seat = engine.current_player()
        legal = engine.legal_actions(seat)
        engine.step(Action.CHECK_CALL if Action.CHECK_CALL in legal else Action.FOLD)
    seat = engine.current_player()
    legal = engine.legal_actions(seat)
    assert Action.FOLD not in legal
    assert legal[Action.CHECK_CALL] == 0.0


def test_short_all_in_raise_does_not_reopen_action():
    # Three players. P0 raises big, P1 calls, P2 goes all-in for less than
    # a full raise over P0's bet -> P0 (who already acted/called-in-effect
    # as raiser) should only be allowed fold/call, not another raise, when
    # action returns to them for the extra chips.
    engine = PokerEngine(num_players=3, small_blind=1.0, big_blind=2.0, rng=random.Random(1))
    engine.start_hand([1000, 1000, 12], button=0)
    # seats: 0=button/SB, 1=BB, 2=UTG (since n=3, sb=next after button=1? let's just drive via actual seats)
    seat = engine.current_player()
    # UTG opens to 20 (a real raise)
    legal = engine.legal_actions(seat)
    raise_action = next(a for a in (Action.RAISE_150, Action.RAISE_75, Action.RAISE_33) if a in legal)
    raiser_seat = seat
    engine.step(raise_action)

    # next player calls
    seat = engine.current_player()
    engine.step(Action.CHECK_CALL)

    # last player (short stack) goes all-in for less than a full raise
    seat = engine.current_player()
    legal = engine.legal_actions(seat)
    engine.step(Action.ALL_IN)

    # action should return only to players who owe chips and haven't
    # folded/all-in; if it's the original raiser again, they must be capped
    if not engine.done and engine.current_player() == raiser_seat:
        legal = engine.legal_actions(raiser_seat)
        assert Action.RAISE_33 not in legal
        assert Action.RAISE_75 not in legal
        assert Action.RAISE_150 not in legal
        assert Action.ALL_IN not in legal
        assert Action.CHECK_CALL in legal
        assert Action.FOLD in legal


def test_side_pot_distribution_three_way_all_in():
    # P0 short-stacked all-in for 10, P1 and P2 both put in 100.
    # P0 can only win a pot capped at 10*3=30 (main pot); the remaining
    # 90*2=180 side pot is contested only between P1 and P2.
    engine = PokerEngine(num_players=3, small_blind=1.0, big_blind=2.0, rng=random.Random(0))
    engine.players = []
    from poker.engine import PlayerState
    engine.players = [
        PlayerState(seat=0, stack=0.0, hole_cards=["2c", "2d"], total_committed=10.0, bet_street=0.0, all_in=True),
        PlayerState(seat=1, stack=0.0, hole_cards=["Ah", "Ad"], total_committed=100.0, bet_street=0.0, all_in=True),
        PlayerState(seat=2, stack=0.0, hole_cards=["3h", "3d"], total_committed=100.0, bet_street=0.0, all_in=True),
    ]
    engine.board = ["4h", "9d", "Kc", "2s", "7h"]  # no straight/flush possible for any hand below
    engine.street = Street.RIVER
    engine.button = 0
    engine._resolve_showdown()

    assert engine.done
    # P1 (aces) wins both main pot and side pot outright: highest pair, and
    # the board can't complete a straight or flush for anyone here.
    assert engine.payouts[1] > 0
    assert engine.payouts[2] < 0
    assert sum(engine.payouts.values()) == pytest.approx(0.0, abs=1e-6)


def test_legal_actions_amounts_never_exceed_stack():
    rng = random.Random(5)
    for _ in range(200):
        n = rng.randint(2, 6)
        stacks = [rng.randint(1, 300) for _ in range(n)]
        engine = PokerEngine(num_players=n, small_blind=1.0, big_blind=2.0, rng=rng)
        engine.start_hand(stacks, button=rng.randrange(n))
        while not engine.done:
            seat = engine.current_player()
            p = engine.players[seat]
            legal = engine.legal_actions(seat)
            for amount in legal.values():
                assert amount <= p.stack + 1e-6
            engine.step(rng.choice(list(legal.keys())))
