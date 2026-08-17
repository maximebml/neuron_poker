import json
import os
import random
import statistics

from agents.pro_agent import PARAM_BOUNDS, PARAM_DEFAULTS
from training.optimize_pro_agent import (_combine_scores, _zscores, crossover_params, mutate_params, run_evolution,
                                         sample_random_params)


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


def test_zscores_degenerate_population_returns_zeros():
    assert _zscores([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_zscores_mean_zero_stdev_one():
    values = [10.0, 20.0, 30.0, 40.0]
    z = _zscores(values)
    assert abs(statistics.mean(z)) < 1e-9
    assert abs(statistics.pstdev(z) - 1.0) < 1e-9


def test_combine_scores_not_dominated_by_larger_scale_signal():
    # baseline_bb100 is ~10x the scale of sparring_bb100, as seen in real runs.
    # Candidate 0 "wins" on baseline but is worst on sparring; candidate 2 is the
    # reverse. A correctly-normalized 50/50 blend should not let the larger raw
    # scale of baseline silently dominate: the two should land close together,
    # not with candidate 0 far ahead purely because its raw numbers are bigger.
    raw_results = [
        {"baseline_bb100": 800.0, "sparring_bb100": -40.0},
        {"baseline_bb100": 500.0, "sparring_bb100": 10.0},
        {"baseline_bb100": 200.0, "sparring_bb100": 60.0},
    ]
    combined = _combine_scores(raw_results, baseline_weight=0.5)
    assert len(combined) == 3
    # Both signals contribute roughly equally once standardized: since baseline
    # and sparring rank in exactly opposite order across these three candidates,
    # a genuinely-balanced 50/50 blend should score them all close to zero/tied,
    # not have candidate 0 running away with it as a raw average would.
    assert max(combined) - min(combined) < 1.0


def test_combine_scores_respects_weight_direction():
    raw_results = [
        {"baseline_bb100": 800.0, "sparring_bb100": -40.0},
        {"baseline_bb100": 200.0, "sparring_bb100": 60.0},
    ]
    baseline_leaning = _combine_scores(raw_results, baseline_weight=0.9)
    sparring_leaning = _combine_scores(raw_results, baseline_weight=0.1)
    # candidate 0 has the better baseline but worse sparring: weighting toward
    # baseline should favor it more than weighting toward sparring does.
    assert (baseline_leaning[0] - baseline_leaning[1]) > (sparring_leaning[0] - sparring_leaning[1])


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
