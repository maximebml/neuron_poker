import json
import os
import random
import statistics

from agents.pro_agent import PARAM_BOUNDS, PARAM_DEFAULTS
from training.optimize_pro_agent_v2 import (OPTIMIZED_PARAM_NAMES, _combine_scores, _default_params, _zscores,
                                            crossover_params, mutate_params, run_evolution, sample_random_params)


def test_optimized_param_names_are_a_real_subset_of_param_defaults():
    assert 0 < len(OPTIMIZED_PARAM_NAMES) < len(PARAM_DEFAULTS)
    assert set(OPTIMIZED_PARAM_NAMES).issubset(set(PARAM_DEFAULTS))


def test_default_params_matches_param_defaults_for_the_optimized_subset():
    defaults = _default_params()
    assert set(defaults) == set(OPTIMIZED_PARAM_NAMES)
    for name in OPTIMIZED_PARAM_NAMES:
        assert defaults[name] == PARAM_DEFAULTS[name]


def test_sample_random_params_respects_bounds_and_only_touches_optimized_names():
    rng = random.Random(0)
    for _ in range(20):
        params = sample_random_params(rng)
        assert set(params) == set(OPTIMIZED_PARAM_NAMES)
        for name, value in params.items():
            low, high, _ = PARAM_BOUNDS[name]
            assert low <= value <= high


def test_mutate_params_stays_within_bounds_even_with_large_sigma_scale():
    rng = random.Random(1)
    base = _default_params()
    for _ in range(20):
        mutated = mutate_params(base, rng, sigma_scale=50.0)  # deliberately huge, to hit the clamps
        assert set(mutated) == set(OPTIMIZED_PARAM_NAMES)
        for name, value in mutated.items():
            low, high, _ = PARAM_BOUNDS[name]
            assert low <= value <= high


def test_crossover_params_picks_each_value_from_one_parent_or_the_other():
    rng = random.Random(2)
    a = {name: 0.0 for name in OPTIMIZED_PARAM_NAMES}
    b = {name: 1.0 for name in OPTIMIZED_PARAM_NAMES}
    child = crossover_params(a, b, rng)
    assert set(child) == set(OPTIMIZED_PARAM_NAMES)
    assert all(v in (0.0, 1.0) for v in child.values())
    assert 0.0 in child.values()
    assert 1.0 in child.values()


def test_zscores_degenerate_population_returns_zeros():
    assert _zscores([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_zscores_mean_zero_stdev_one():
    values = [10.0, 20.0, 30.0, 40.0]
    z = _zscores(values)
    assert abs(statistics.mean(z)) < 1e-9
    assert abs(statistics.pstdev(z) - 1.0) < 1e-9


def test_combine_scores_not_dominated_by_larger_scale_others_signal():
    # vs_others_bb100 runs ~10x the scale of vs_pro_bb100 here, as seen in
    # real runs (weak opponents let a candidate run to huge bb/100). A
    # correctly-normalized blend should not let that raw scale silently
    # dominate: candidates that trade off oppositely on the two signals
    # should land close together, not with the big-vs_others one running away.
    raw_results = [
        {"vs_pro_bb100": -40.0, "vs_others_bb100": 900.0},
        {"vs_pro_bb100": 10.0, "vs_others_bb100": 500.0},
        {"vs_pro_bb100": 60.0, "vs_others_bb100": 100.0},
    ]
    combined = _combine_scores(raw_results, pro_weight=0.7)
    assert len(combined) == 3
    assert max(combined) - min(combined) < 3.0


def test_combine_scores_respects_weight_direction():
    raw_results = [
        {"vs_pro_bb100": -40.0, "vs_others_bb100": 900.0},
        {"vs_pro_bb100": 60.0, "vs_others_bb100": 100.0},
    ]
    pro_leaning = _combine_scores(raw_results, pro_weight=0.9)
    others_leaning = _combine_scores(raw_results, pro_weight=0.1)
    # candidate 1 has the better vs-pro result but worse vs-others: weighting
    # toward pro should favor it more than weighting toward others does.
    assert (pro_leaning[1] - pro_leaning[0]) > (others_leaning[1] - others_leaning[0])


def test_run_evolution_end_to_end_and_resume(tmp_path):
    out_dir = str(tmp_path / "opt_v2")

    best = run_evolution(generations=2, population_size=3, elite_count=1, survivor_count=2,
                         pro_hands=20, other_hands=10, pro_weight=0.7,
                         search_equity_sims=10, seed=5, out_dir=out_dir, model_path="no/such/model.zip")
    assert "params" in best and "fitness" in best
    assert set(best["params"]) == set(OPTIMIZED_PARAM_NAMES)

    state_path = os.path.join(out_dir, "state.json")
    assert os.path.exists(state_path)
    with open(state_path) as f:
        state = json.load(f)
    assert state["generation"] == 1  # 0-indexed, ran generations 0 and 1
    assert len(state["history"]) == 2
    assert "best_other_scores" in state["history"][0]

    best_params_path = os.path.join(out_dir, "best_params.json")
    assert os.path.exists(best_params_path)
    with open(best_params_path) as f:
        saved_best_params = json.load(f)
    assert set(saved_best_params) == set(OPTIMIZED_PARAM_NAMES)

    # Resuming should continue from generation 2, not restart from 0.
    resumed_best = run_evolution(generations=4, population_size=3, elite_count=1, survivor_count=2,
                                 pro_hands=20, other_hands=10, pro_weight=0.7,
                                 search_equity_sims=10, seed=5, out_dir=out_dir,
                                 model_path="no/such/model.zip", resume=True)
    with open(state_path) as f:
        state_after_resume = json.load(f)
    assert state_after_resume["generation"] == 3
    assert len(state_after_resume["history"]) == 4
    assert state_after_resume["history"][0] == state["history"][0]  # early history preserved, not redone
    assert resumed_best["fitness"] >= best["fitness"] - 1e-9  # best-ever never regresses


def test_run_evolution_gracefully_skips_missing_rl_checkpoint(tmp_path):
    # model_path pointing at a nonexistent file should just omit the RL
    # opponent, not raise -- most sessions won't have a checkpoint on disk.
    out_dir = str(tmp_path / "opt_v2_norl")
    best = run_evolution(generations=1, population_size=2, elite_count=1, survivor_count=1,
                         pro_hands=10, other_hands=10, pro_weight=0.7,
                         search_equity_sims=10, seed=1, out_dir=out_dir, model_path="definitely/missing.zip")
    assert "params" in best
