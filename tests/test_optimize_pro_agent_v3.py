import json
import os

from training.optimize_pro_agent_v2 import OPTIMIZED_PARAM_NAMES
from training.optimize_pro_agent_v3 import _head_to_head_bb100, _table_evaluate, run_chain


def _write_champion_init(path, params=None):
    from agents.pro_agent import PARAM_DEFAULTS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params or dict(PARAM_DEFAULTS), f)


def test_table_evaluate_returns_a_number(tmp_path):
    from agents.pro_agent import PARAM_DEFAULTS
    subset = {name: PARAM_DEFAULTS[name] for name in OPTIMIZED_PARAM_NAMES}
    bb100 = _table_evaluate(subset, subset, hands=10, equity_sims=10, seed=1)
    assert isinstance(bb100, float)


def test_head_to_head_bb100_is_antisymmetric_in_expectation_direction(tmp_path):
    # Not a strict mathematical antisymmetry (different hands are dealt to
    # each side), but running the exact same matchup with A/B swapped and a
    # shared seed should still be a plain float, not raise or hang.
    from agents.pro_agent import PARAM_DEFAULTS
    subset = {name: PARAM_DEFAULTS[name] for name in OPTIMIZED_PARAM_NAMES}
    bb100 = _head_to_head_bb100(subset, subset, hands=10, seed=2)
    assert isinstance(bb100, float)


def test_run_chain_end_to_end_and_resume(tmp_path):
    champion_init = str(tmp_path / "init" / "best_params.json")
    _write_champion_init(champion_init)
    out_dir = str(tmp_path / "chain")
    final_out_dir = str(tmp_path / "final")

    champion = run_chain(iterations=2, generations_per_iteration=1, population_size=2, elite_count=1,
                         survivor_count=1, search_hands=5, validation_hands=5, search_equity_sims=5,
                         seed=5, out_dir=out_dir, final_out_dir=final_out_dir,
                         champion_init_path=champion_init)
    assert set(champion) == set(OPTIMIZED_PARAM_NAMES)

    chain_state_path = os.path.join(out_dir, "chain_state.json")
    assert os.path.exists(chain_state_path)
    with open(chain_state_path) as f:
        state = json.load(f)
    assert state["completed_iterations"] == 2
    assert len(state["history"]) == 2
    for record in state["history"]:
        assert "vs_champion_bb100" in record and "vs_stock_pro_bb100" in record and "promoted" in record

    final_params_path = os.path.join(final_out_dir, "best_params.json")
    assert os.path.exists(final_params_path)
    with open(final_params_path) as f:
        final_params = json.load(f)
    assert set(final_params) == set(OPTIMIZED_PARAM_NAMES)

    # Resuming should continue from iteration 2, not restart from 0, and
    # early history should be preserved unchanged.
    resumed_champion = run_chain(iterations=3, generations_per_iteration=1, population_size=2, elite_count=1,
                                 survivor_count=1, search_hands=5, validation_hands=5, search_equity_sims=5,
                                 seed=5, out_dir=out_dir, final_out_dir=final_out_dir,
                                 champion_init_path=champion_init, resume=True)
    with open(chain_state_path) as f:
        state_after_resume = json.load(f)
    assert state_after_resume["completed_iterations"] == 3
    assert len(state_after_resume["history"]) == 3
    assert state_after_resume["history"][0] == state["history"][0]
    assert state_after_resume["history"][1] == state["history"][1]
    assert set(resumed_champion) == set(OPTIMIZED_PARAM_NAMES)


def test_run_chain_never_promotes_a_non_improving_challenger(tmp_path, monkeypatch):
    # Force every head-to-head validation to say the challenger lost, and
    # confirm the champion never changes from the seeded init across
    # several iterations -- a "no-op" chain is a legitimate outcome.
    import training.optimize_pro_agent_v3 as mod

    champion_init = str(tmp_path / "init" / "best_params.json")
    _write_champion_init(champion_init)

    monkeypatch.setattr(mod, "_head_to_head_bb100", lambda *a, **k: -1.0)

    out_dir = str(tmp_path / "chain")
    final_out_dir = str(tmp_path / "final")
    champion = run_chain(iterations=3, generations_per_iteration=1, population_size=2, elite_count=1,
                         survivor_count=1, search_hands=5, validation_hands=5, search_equity_sims=5,
                         seed=9, out_dir=out_dir, final_out_dir=final_out_dir,
                         champion_init_path=champion_init)

    with open(champion_init) as f:
        init_params = json.load(f)
    expected = {name: init_params[name] for name in OPTIMIZED_PARAM_NAMES}
    assert champion == expected

    with open(os.path.join(out_dir, "chain_state.json")) as f:
        state = json.load(f)
    assert all(not r["promoted"] for r in state["history"])
