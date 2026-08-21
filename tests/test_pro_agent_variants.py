import random

import pytest

from agents.pro_agent import PARAM_DEFAULTS
from agents.pro_agent_v2 import TUNED_PARAMS as V2_PARAMS, ProAgentV2
from agents.pro_agent_v3 import TUNED_PARAMS as V3_PARAMS, ProAgentV3
from agents.random_agent import RandomAgent
from poker.engine import PokerEngine
from tests.test_pro_agent import play_randomly_except_hero


@pytest.mark.parametrize("agent_cls,tuned_params,expected_name", [
    (ProAgentV2, V2_PARAMS, "pro_v2"),
    (ProAgentV3, V3_PARAMS, "pro_v3"),
])
def test_variant_applies_its_tuned_params_over_defaults(agent_cls, tuned_params, expected_name):
    agent = agent_cls()
    assert agent.name == expected_name
    for name, value in tuned_params.items():
        assert agent.params[name] == value
    for name, default in PARAM_DEFAULTS.items():
        if name not in tuned_params:
            assert agent.params[name] == default


@pytest.mark.parametrize("agent_cls", [ProAgentV2, ProAgentV3])
@pytest.mark.parametrize("trial", range(20))
def test_variant_always_returns_legal_actions(agent_cls, trial):
    rng = random.Random(trial)
    n = rng.randint(2, 6)
    stacks = [rng.randint(20, 400) for _ in range(n)]
    hero_seat = rng.randrange(n)
    agent = agent_cls(rng=rng, equity_sims=30)  # small n_sims: speed, not precision, is what's tested here

    engine = PokerEngine(num_players=n, small_blind=1.0, big_blind=2.0, rng=rng)
    engine.start_hand(stacks, button=rng.randrange(n))
    play_randomly_except_hero(engine, hero_seat, agent, rng)

    assert engine.done
    assert abs(sum(p.stack for p in engine.players) - sum(stacks)) < 1e-6


@pytest.mark.parametrize("agent_cls", [ProAgentV2, ProAgentV3])
def test_variant_beats_random_agent_over_many_hands(agent_cls):
    """Same low-bar sanity check as test_pro_agent.py's equivalent: a tuned
    ProAgent variant should still comfortably beat pure random play."""
    rng = random.Random(7)
    hero = agent_cls(rng=rng, equity_sims=40)
    villain = RandomAgent(rng=rng)

    n_hands = 400
    total_bb = 0.0
    big_blind = 2.0
    for i in range(n_hands):
        n = rng.choice([2, 3, 6])
        stacks = [100.0 * big_blind] * n
        hero_seat = rng.randrange(n)
        agents = {hero_seat: hero}
        engine = PokerEngine(num_players=n, small_blind=1.0, big_blind=big_blind, rng=rng)
        engine.start_hand(stacks, button=rng.randrange(n))
        while not engine.done:
            seat = engine.current_player()
            action = agents.get(seat, villain).act(engine, seat)
            engine.step(action)
        total_bb += engine.payouts[hero_seat] / big_blind

    assert total_bb / n_hands > 0.0
