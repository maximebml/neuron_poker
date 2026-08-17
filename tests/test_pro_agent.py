import random

import pytest

from agents.pro_agent import PARAM_BOUNDS, PARAM_DEFAULTS, ProAgent, Tier, board_texture, chen_score, \
    position_quality, postflop_tier
from agents.random_agent import RandomAgent
from poker.engine import PokerEngine


# --------------------------------------------------------------- parameters

def test_default_agent_uses_param_defaults():
    agent = ProAgent()
    assert agent.params == PARAM_DEFAULTS


def test_overriding_a_param_leaves_the_rest_at_defaults():
    agent = ProAgent(cbet_freq=0.9)
    assert agent.params["cbet_freq"] == 0.9
    for name, value in PARAM_DEFAULTS.items():
        if name != "cbet_freq":
            assert agent.params[name] == value


def test_unknown_param_raises():
    with pytest.raises(TypeError):
        ProAgent(not_a_real_param=1.0)


def test_param_bounds_cover_exactly_the_same_names_as_defaults():
    assert set(PARAM_BOUNDS) == set(PARAM_DEFAULTS)


def test_param_defaults_fall_within_their_own_bounds():
    for name, default in PARAM_DEFAULTS.items():
        low, high, _sigma = PARAM_BOUNDS[name]
        assert low <= default <= high, f"{name} default {default} outside bounds [{low}, {high}]"


# ------------------------------------------------------------- Chen formula

@pytest.mark.parametrize("hole,expected", [
    (["Ah", "Ac"], 20.0),
    (["Kh", "Kc"], 16.0),
    (["2h", "2c"], 5.0),
    (["Ah", "Kh"], 12.0),   # AKs
    (["Ah", "Kc"], 10.0),   # AKo
    (["Jh", "Th"], 9.0),    # JTs
    (["7h", "6h"], 6.5),    # 76s
    (["7h", "2c"], 0.0),    # worst hand, floored at 0
])
def test_chen_score_matches_known_values(hole, expected):
    assert chen_score(hole) == expected


def test_chen_score_suited_beats_offsuit():
    assert chen_score(["Ah", "Kh"]) > chen_score(["Ah", "Kc"])


def test_chen_score_pair_beats_non_pair_of_same_high_card():
    assert chen_score(["9h", "9c"]) > chen_score(["9h", "2c"])


# ---------------------------------------------------------- position_quality

def test_button_is_best_position():
    assert position_quality(seat=3, button=3, num_players=6) == 1.0


def test_small_blind_is_worst_position():
    # seat 4 is next after button 3 in a 6-max table -> small blind
    assert position_quality(seat=4, button=3, num_players=6) == 0.0


def test_position_quality_monotonic_around_the_table():
    button = 0
    n = 6
    qualities = [position_quality(seat, button, n) for seat in range(n)]
    # seat 0 (button) should be strictly the best
    assert qualities[0] == max(qualities)


def test_position_quality_heads_up():
    # heads-up: button acts last postflop too (special rule the engine enforces)
    assert position_quality(seat=0, button=0, num_players=2) == 1.0
    assert position_quality(seat=1, button=0, num_players=2) == 0.0


# ------------------------------------------------------------- draw detection

def test_open_ended_straight_draw_detected():
    from agents.pro_agent import _draw_features
    flush, oesd, gutshot = _draw_features(["Th", "Jd"], ["9c", "8s", "2h"])
    assert (flush, oesd, gutshot) == (False, True, False)


def test_gutshot_detected():
    from agents.pro_agent import _draw_features
    flush, oesd, gutshot = _draw_features(["Kh", "Qd"], ["Tc", "9s", "2h"])
    assert (flush, oesd, gutshot) == (False, False, True)


def test_flush_draw_detected():
    from agents.pro_agent import _draw_features
    flush, oesd, gutshot = _draw_features(["Ah", "Kh"], ["2h", "7h", "9c"])
    assert (flush, oesd, gutshot) == (True, False, False)


def test_wheel_straight_draw_ace_plays_low():
    from agents.pro_agent import _draw_features
    flush, oesd, gutshot = _draw_features(["Ah", "2d"], ["3c", "4s", "9h"])
    assert oesd is True


def test_no_draw_on_disconnected_board():
    from agents.pro_agent import _draw_features
    assert _draw_features(["Ah", "2d"], ["7c", "Ks", "9h"]) == (False, False, False)


def test_no_draw_flagged_on_the_river():
    from agents.pro_agent import _draw_features
    # same cards that make a flush draw on the flop -- but river means no more cards coming
    assert _draw_features(["Ah", "Kh"], ["2h", "7h", "9c", "3d", "5s"]) == (False, False, False)


# ------------------------------------------------------------- postflop tier

def test_trips_is_strong_made():
    assert postflop_tier(["7h", "7d"], ["7c", "2s", "9h"]) == Tier.STRONG_MADE


def test_top_pair_is_medium_made():
    assert postflop_tier(["Ah", "9d"], ["Ac", "2s", "5h"]) == Tier.MEDIUM_MADE


def test_weak_kicker_pair_is_weak_made():
    assert postflop_tier(["2h", "9d"], ["Ac", "9s", "5h"]) == Tier.WEAK_MADE


def test_two_pair_is_medium_made():
    assert postflop_tier(["Ah", "5d"], ["Ac", "5s", "9h"]) == Tier.MEDIUM_MADE


def test_nothing_is_air():
    assert postflop_tier(["2h", "3d"], ["Ac", "Ks", "9h"]) == Tier.AIR


def test_flush_draw_with_no_made_hand_is_strong_draw():
    assert postflop_tier(["2h", "3h"], ["9h", "Kh", "5c"]) == Tier.STRONG_DRAW


def test_pair_using_only_the_boards_own_pair_is_weak():
    # board itself is paired (77) -- hero's KQ barely plays, not a real hand
    assert postflop_tier(["Kh", "Qd"], ["7c", "7s", "2h"]) == Tier.WEAK_MADE


def test_combo_pair_plus_draw_takes_the_better_tier():
    # bottom pair (weak) + flush draw (strong draw) -> strong draw wins out
    tier = postflop_tier(["2h", "9h"], ["9c", "Kh", "5h"])
    assert tier == Tier.STRONG_DRAW


# --------------------------------------------------------------- board texture

def test_monotone_connected_board_is_wet():
    wetness, paired = board_texture(["9h", "Th", "Jh"])
    assert wetness == 1.0
    assert paired is False


def test_rainbow_disconnected_board_is_dry():
    wetness, _ = board_texture(["2h", "9c", "Ks"])
    assert wetness == 0.0


def test_paired_board_flagged():
    _, paired = board_texture(["7h", "7c", "2s"])
    assert paired is True


# ---------------------------------------------------------------- end to end

def play_randomly_except_hero(engine, hero_seat, hero_agent, rng):
    steps = 0
    while not engine.done:
        seat = engine.current_player()
        agent = hero_agent if seat == hero_seat else None
        if agent is not None:
            action = agent.act(engine, seat)
        else:
            action = rng.choice(list(engine.legal_actions(seat).keys()))
        engine.step(action)
        steps += 1
        assert steps < 1000, "runaway hand"


@pytest.mark.parametrize("trial", range(60))
def test_pro_agent_always_returns_legal_actions(trial):
    rng = random.Random(trial)
    n = rng.randint(2, 6)
    stacks = [rng.randint(20, 400) for _ in range(n)]
    hero_seat = rng.randrange(n)
    agent = ProAgent(rng=rng, equity_sims=30)  # small n_sims: speed, not precision, is what's tested here

    engine = PokerEngine(num_players=n, small_blind=1.0, big_blind=2.0, rng=rng)
    engine.start_hand(stacks, button=rng.randrange(n))
    play_randomly_except_hero(engine, hero_seat, agent, rng)

    assert engine.done
    assert abs(sum(p.stack for p in engine.players) - sum(stacks)) < 1e-6


def test_pro_agent_beats_random_agent_over_many_hands():
    """A sanity/regression check, not a strength claim: if the heuristics are
    badly broken (e.g. folding premiums, raising air constantly), this is
    the kind of test that would catch it -- beating pure random play is a
    very low bar for a structured, position- and hand-strength-aware agent."""
    rng = random.Random(7)
    pro = ProAgent(rng=rng, equity_sims=40)
    villain = RandomAgent(rng=rng)

    n_hands = 400
    total_bb = 0.0
    big_blind = 2.0
    for i in range(n_hands):
        n = rng.choice([2, 3, 6])
        stacks = [100.0 * big_blind] * n
        hero_seat = rng.randrange(n)
        agents = {hero_seat: pro}
        engine = PokerEngine(num_players=n, small_blind=1.0, big_blind=big_blind, rng=rng)
        engine.start_hand(stacks, button=rng.randrange(n))
        while not engine.done:
            seat = engine.current_player()
            action = agents.get(seat, villain).act(engine, seat)
            engine.step(action)
        total_bb += engine.payouts[hero_seat] / big_blind

    assert total_bb / n_hands > 0.0  # positive bb/hand vs. pure random opponents
