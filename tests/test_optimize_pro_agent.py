import json
import os
import random

from agents.pro_agent import PARAM_BOUNDS, PARAM_DEFAULTS
from training.optimize_pro_agent import crossover_params, mutate_params, run_evolution, sample_random_params


def test_sample_random_params_respects_bounds():
    rng = random.Random(0)
    for _ in range(20):
        params = sample_random_params(rng)
        assert set(params) == set(PARAM_DEFAULTS)
        for name, value in params.items():
            low, high, _ = PARAM_BOUNDS[name]
            assert low <= value <= high


def test_mutate_params_stays_within_bounds_even_with_large_sigma_scale():
    rng = random.Random(1)
    for _ in range(20):
        mutated = mutate_params(PARAM_DEFAULTS, rng, sigma_scale=50.0)  # deliberately huge, to hit the clamps
        for name, value in mutated.items():
            low, high, _ = PARAM_BOUNDS[name]
            assert low <= value <= high


def test_crossover_params_picks_each_value_from_one_parent_or_the_other():
    rng = random.Random(2)
    a = {name: 0.0 for name in PARAM_DEFAULTS}
    b = {name: 1.0 for name in PARAM_DEFAULTS}
    child = crossover_params(a, b, rng)
    assert set(child) == set(PARAM_DEFAULTS)
    assert all(v in (0.0, 1.0) for v in child.values())
    # with enough parameters, a 50/50 coin flip per gene should draw from both parents at least once
    assert 0.0 in child.values()
    assert 1.0 in child.values()


def test_run_evolution_end_to_end_and_resume(tmp_path):
    out_dir = str(tmp_path / "opt")

    best = run_evolution(generations=2, population_size=3, elite_count=1, survivor_count=2,
                         baseline_hands=20, sparring_hands=20, sparring_opponents=1,
                         search_equity_sims=10, seed=5, out_dir=out_dir)
    assert "params" in best and "fitness" in best
    assert set(best["params"]) == set(PARAM_DEFAULTS)

    state_path = os.path.join(out_dir, "state.json")
    assert os.path.exists(state_path)
    with open(state_path) as f:
        state = json.load(f)
    assert state["generation"] == 1  # 0-indexed, ran generations 0 and 1
    assert len(state["history"]) == 2

    best_params_path = os.path.join(out_dir, "best_params.json")
    assert os.path.exists(best_params_path)

    # Resuming should continue from generation 2, not restart from 0.
    resumed_best = run_evolution(generations=4, population_size=3, elite_count=1, survivor_count=2,
                                 baseline_hands=20, sparring_hands=20, sparring_opponents=1,
                                 search_equity_sims=10, seed=5, out_dir=out_dir, resume=True)
    with open(state_path) as f:
        state_after_resume = json.load(f)
    assert state_after_resume["generation"] == 3
    assert len(state_after_resume["history"]) == 4
    assert state_after_resume["history"][0] == state["history"][0]  # early history preserved, not redone
    assert resumed_best["fitness"] >= best["fitness"] - 1e-9  # best-ever never regresses
