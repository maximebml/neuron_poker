import pytest

from inference.decide import build_engine_from_situation, decide
from poker.actions import Action


def make_situation(**overrides):
    situation = {
        "num_players": 3,
        "big_blind": 2,
        "small_blind": 1,
        "button_seat": 0,
        "street": "flop",
        "board": ["Qs", "Jd", "2c"],
        "pot_before_street": 20,
        "hero_seat": 0,
        "hero_cards": ["Ah", "Kd"],
        "players": [
            {"seat": 0, "stack": 200, "bet": 0},
            {"seat": 1, "stack": 200, "bet": 10},
            {"seat": 2, "stack": 0, "folded": True},
        ],
    }
    situation.update(overrides)
    return situation


def test_build_engine_computes_correct_pot_and_to_call():
    engine, hero_seat = build_engine_from_situation(make_situation())
    assert hero_seat == 0
    assert engine.pot_total() == pytest.approx(30.0)  # 20 carried over + 10 bet
    legal = engine.legal_actions(hero_seat)
    assert legal[Action.CHECK_CALL] == pytest.approx(10.0)
    assert Action.FOLD in legal


def test_decide_without_model_uses_rule_based_fallback():
    result = decide(make_situation(), model_path=None)
    assert result["action"] in {a.name for a in Action}
    assert result["action"] in result["legal_actions"]
    assert result["action_probabilities"] is None
    assert "rule_based" in result["used_model"]


def test_decide_raises_when_hero_has_no_legal_actions():
    situation = make_situation()
    situation["players"][0]["folded"] = True
    with pytest.raises(ValueError):
        decide(situation, model_path=None)


def test_decide_preflop_situation():
    situation = make_situation(street="preflop", board=[], pot_before_street=3,
                                players=[
                                    {"seat": 0, "stack": 200, "bet": 0},
                                    {"seat": 1, "stack": 199, "bet": 2},
                                    {"seat": 2, "stack": 0, "folded": True},
                                ])
    result = decide(situation, model_path=None, n_equity_sims=50)
    assert result["pot_before_action"] == pytest.approx(5.0)
    assert 0.0 <= result["hero_equity_vs_live_opponents"] <= 1.0
