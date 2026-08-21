"""Evolve a curated subset of ProAgent's highest-leverage parameters against
fixed, known-strength opponents -- mostly the current stock ProAgent (the
strongest agent evaluated so far in this project), with a smaller amount of
weight spread across a handful of weaker fixed agents so the result doesn't
overfit to beating one specific opponent.

Unlike training/optimize_pro_agent.py (which searches all 20 strategy
parameters against a hybrid of a weak fixed pool + evolving population
sparring), this is deliberately narrower on two axes:

- Parameters: only OPTIMIZED_PARAM_NAMES below are searched -- six of the
  twenty, picked for how often they actually fire. Preflop opening
  (open_intercept/open_position_coef) and facing-a-single-raise
  (facing_raise_call_intercept/facing_raise_raise_intercept) decisions
  happen on nearly every hand; continuation betting (cbet_freq) and
  bluff-catching (bluffcatch_extra_margin) are the two highest-frequency
  postflop levers. Every other parameter stays locked at PARAM_DEFAULTS,
  which keeps the search space small enough to converge reliably and
  makes any improvement attributable to a specific, named strategic knob
  rather than 20 simultaneously-drifting numbers.
- Opponents: all fixed and already-known-strength (no self-play, no
  population sparring), so unlike the original optimizer's fixed pool none
  of these are being gamed by an evolving population.

That said, vs_pro_bb100 and vs_others_bb100 still live on very different
raw scales -- vs_pro is measured against an opponent of comparable skill
and stays in a tight range, while vs_others averages in weak, highly
exploitable opponents (random, the RL model) where a good candidate can
run to 500-1500+ bb/100. An early version of this script blended the raw
numbers with pro_weight directly, which quietly let vs_others dominate
regardless of the requested weight and produced parameters that hit their
search bounds by over-exploiting the weak opponents (verified: a large
head-to-head against stock ProAgent came back statistically even, despite
a large claimed "improvement"). Both signals are now z-score standardized
within each generation's population before blending -- the same fix
optimize_pro_agent.py needed for the same underlying reason.

Usage:
    python -m training.optimize_pro_agent_v2 --generations 8
"""
import argparse
import json
import os
import random
import statistics

from agents.pro_agent import PARAM_BOUNDS, PARAM_DEFAULTS, ProAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from training.evaluate import evaluate_agent

OPTIMIZED_PARAM_NAMES = [
    "open_intercept", "open_position_coef",
    "facing_raise_call_intercept", "facing_raise_raise_intercept",
    "cbet_freq", "bluffcatch_extra_margin",
]


def _default_params() -> dict:
    return {name: PARAM_DEFAULTS[name] for name in OPTIMIZED_PARAM_NAMES}


def _other_opponents(model_path: str = None) -> list:
    """Fixed, weaker benchmark opponents -- distinct from the stock ProAgent
    the search mostly plays against, so a good result has to generalize a
    little, not just exploit one specific reference strategy."""
    opponents = [
        ("random", RandomAgent()),
        ("rule_0.2", RuleBasedAgent(aggression=0.2)),
        ("rule_0.5", RuleBasedAgent(aggression=0.5)),
        ("rule_0.8", RuleBasedAgent(aggression=0.8)),
    ]
    if model_path and os.path.exists(model_path):
        from agents.rl_agent import RLAgent
        opponents.append(("rl", RLAgent(model_path, deterministic=True)))
    return opponents


def sample_random_params(rng: random.Random) -> dict:
    return {name: rng.uniform(*PARAM_BOUNDS[name][:2]) for name in OPTIMIZED_PARAM_NAMES}


def mutate_params(params: dict, rng: random.Random, sigma_scale: float = 1.0) -> dict:
    mutated = dict(params)
    for name in OPTIMIZED_PARAM_NAMES:
        low, high, sigma = PARAM_BOUNDS[name]
        mutated[name] = min(high, max(low, params[name] + rng.gauss(0, sigma * sigma_scale)))
    return mutated


def crossover_params(a: dict, b: dict, rng: random.Random) -> dict:
    return {name: (a[name] if rng.random() < 0.5 else b[name]) for name in OPTIMIZED_PARAM_NAMES}


def _evaluate_candidate(candidate_params: dict, pro_hands: int, other_hands: int, other_opponents: list,
                        search_equity_sims: int, seed: int) -> dict:
    candidate = ProAgent(rng=random.Random(seed), equity_sims=search_equity_sims, **candidate_params)

    stock_pro = ProAgent(rng=random.Random(seed + 1), equity_sims=search_equity_sims)
    vs_pro = evaluate_agent(candidate, n_hands=pro_hands, num_players_range=(2, 2),
                            seed=seed, opponents=[stock_pro])["bb_per_100"]

    other_scores = {}
    for i, (name, opponent) in enumerate(other_opponents):
        result = evaluate_agent(candidate, n_hands=other_hands, num_players_range=(2, 2),
                                seed=seed + 10 + i, opponents=[opponent])
        other_scores[name] = result["bb_per_100"]
    vs_others_mean = statistics.mean(other_scores.values()) if other_scores else 0.0

    return {"vs_pro_bb100": vs_pro, "vs_others_bb100": vs_others_mean, "other_scores": other_scores}


def _zscores(values: list) -> list:
    """Standardize to mean 0, stdev 1 (0.0 for everyone if the population is degenerate)."""
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev < 1e-9:
        return [0.0 for _ in values]
    return [(v - mean) / stdev for v in values]


def _combine_scores(raw_results: list, pro_weight: float) -> list:
    """Blend vs-pro and vs-others bb/100 into one selection score.

    vs_pro_bb100 is measured against an opponent of comparable skill (the
    stock ProAgent), so it naturally stays in a tight range. vs_others_bb100
    is a mean that includes weak, highly exploitable opponents (random, the
    RL model) where a good candidate can run to 500-1500+ bb/100 -- a much
    larger and noisier scale. Blending the raw numbers with pro_weight would
    make that weight a lie: whichever signal has the bigger swing that
    generation dominates regardless of the requested split. Standardizing
    both within the current population before blending is the same fix
    optimize_pro_agent.py needed for the same reason.
    """
    pro_z = _zscores([r["vs_pro_bb100"] for r in raw_results])
    others_z = _zscores([r["vs_others_bb100"] for r in raw_results])
    return [pro_weight * pz + (1 - pro_weight) * oz for pz, oz in zip(pro_z, others_z)]


def _save_state(out_dir, generation, population, history, best_ever):
    os.makedirs(out_dir, exist_ok=True)
    state = {"generation": generation, "population": population, "history": history, "best_ever": best_ever}
    tmp_path = os.path.join(out_dir, "state.json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, os.path.join(out_dir, "state.json"))
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_ever["params"], f, indent=2)


def run_evolution(generations: int, population_size: int = 8, elite_count: int = 2, survivor_count: int = 4,
                  pro_hands: int = 400, other_hands: int = 150, pro_weight: float = 0.7,
                  search_equity_sims: int = 60, seed: int = 0, out_dir: str = "models/pro_agent_v2",
                  model_path: str = "models/final_model.zip", resume: bool = False) -> dict:
    rng = random.Random(seed)
    other_opponents = _other_opponents(model_path)
    state_path = os.path.join(out_dir, "state.json")

    if resume and os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        population = state["population"]
        history = state["history"]
        best_ever = state["best_ever"]
        start_gen = state["generation"] + 1
        print(f"Resumed from {state_path} at generation {start_gen}")
    else:
        defaults = _default_params()
        population = [dict(defaults)] + [
            mutate_params(defaults, rng, sigma_scale=rng.uniform(1.0, 3.0))
            for _ in range(population_size - 1)]
        history = []
        best_ever = {"params": dict(defaults), "fitness": float("-inf")}
        start_gen = 0

    for gen in range(start_gen, generations):
        raw_results = [
            _evaluate_candidate(candidate_params, pro_hands, other_hands, other_opponents,
                                search_equity_sims, seed=seed + gen * 1000 + i)
            for i, candidate_params in enumerate(population)]
        combined_scores = _combine_scores(raw_results, pro_weight)
        scored = [(population[i], {**raw_results[i], "combined": combined_scores[i]})
                 for i in range(len(population))]

        scored.sort(key=lambda t: -t[1]["combined"])
        best_params, best_result = scored[0]
        fitnesses = [r["combined"] for _, r in scored]
        gen_summary = {
            "generation": gen,
            "best_combined": best_result["combined"],
            "best_vs_pro_bb100": best_result["vs_pro_bb100"],
            "best_vs_others_bb100": best_result["vs_others_bb100"],
            "best_other_scores": best_result["other_scores"],
            "mean_combined": statistics.mean(fitnesses),
        }
        history.append(gen_summary)
        print(f"[gen {gen}] {gen_summary}")

        if best_result["combined"] > best_ever["fitness"]:
            best_ever = {"params": best_params, "fitness": best_result["combined"]}
            print(f"[gen {gen}] new best ever: combined={best_ever['fitness']:.2f} (z-score blend, "
                 f"vs pro={best_result['vs_pro_bb100']:.1f} bb/100, vs others={best_result['vs_others_bb100']:.1f} bb/100)")

        survivors = [p for p, _ in scored[:survivor_count]]
        next_population = [p for p, _ in scored[:elite_count]]  # elitism
        while len(next_population) < population_size:
            parent_a, parent_b = rng.sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0])
            child = mutate_params(crossover_params(parent_a, parent_b, rng), rng)
            next_population.append(child)
        population = next_population

        _save_state(out_dir, gen, population, history, best_ever)

    print(f"Optimization complete. Best ever combined score: {best_ever['fitness']:.2f} "
         f"(z-score blend, {pro_weight:.0%} vs stock ProAgent / {1 - pro_weight:.0%} vs others, "
         f"not itself a bb/100 figure)")
    print(f"Best params saved -> {os.path.join(out_dir, 'best_params.json')}")
    return best_ever


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--survivor-count", type=int, default=4)
    parser.add_argument("--pro-hands", type=int, default=400, help="Hands per candidate vs. the stock ProAgent")
    parser.add_argument("--other-hands", type=int, default=150, help="Hands per candidate vs. each other opponent")
    parser.add_argument("--pro-weight", type=float, default=0.7,
                        help="Fitness weight on beating the stock ProAgent (rest split across the others)")
    parser.add_argument("--search-equity-sims", type=int, default=60,
                        help="Lower than ProAgent's default 200: speed during search, not final precision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="models/pro_agent_v2")
    parser.add_argument("--model-path", type=str, default="models/final_model.zip",
                        help="Optional trained RL checkpoint to include among the 'other' opponents, if present")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evolution(generations=args.generations, population_size=args.population_size,
                 elite_count=args.elite_count, survivor_count=args.survivor_count,
                 pro_hands=args.pro_hands, other_hands=args.other_hands, pro_weight=args.pro_weight,
                 search_equity_sims=args.search_equity_sims, seed=args.seed, out_dir=args.out_dir,
                 model_path=args.model_path, resume=args.resume)
