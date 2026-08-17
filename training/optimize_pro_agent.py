"""Evolve ProAgent's strategy parameters (see agents/pro_agent.py PARAM_*).

Fitness for each candidate is a hybrid of two signals, matching how the
RL training pool mixes fixed opponents with self-play rather than
relying on either alone:

- performance against PokerEnv's fixed baseline pool (random + 3
  RuleBasedAgents) -- an anchor to sane, generalizable play, so the
  search can't drift into something that only beats its own population.
- "sparring": performance against a few other members of the current
  population, sampled fresh each generation -- rewards being robust
  against similarly-skilled, evolving opponents, not just weak scripted
  ones. This is a Monte Carlo approximation of a full round-robin
  (which would be O(population^2) matches per generation): each
  candidate only plays a handful of sampled peers, not everyone.

The search itself is a simple (mu, lambda)-style evolutionary loop:
elitism keeps the top performers unchanged, the rest of each new
generation is bred by crossover + Gaussian mutation of the survivors.
No gradients, no neural net -- each "fitness evaluation" is just fast
heuristic-vs-heuristic play, so this is far cheaper than RL training,
but fitness is still noisy (poker's per-hand variance is high), so
don't expect precise convergence over many parameters from a short run
-- treat the result as a nudge toward a locally better setting, not a
global optimum.

Usage:
    python -m training.optimize_pro_agent --generations 6 --population-size 8
"""
import argparse
import json
import os
import random
import statistics

from agents.pro_agent import PARAM_BOUNDS, PARAM_DEFAULTS, ProAgent
from training.evaluate import evaluate_agent

PARAM_NAMES = sorted(PARAM_DEFAULTS)


def sample_random_params(rng: random.Random) -> dict:
    return {name: rng.uniform(*PARAM_BOUNDS[name][:2]) for name in PARAM_NAMES}


def mutate_params(params: dict, rng: random.Random, sigma_scale: float = 1.0) -> dict:
    mutated = {}
    for name, value in params.items():
        low, high, sigma = PARAM_BOUNDS[name]
        mutated[name] = min(high, max(low, value + rng.gauss(0, sigma * sigma_scale)))
    return mutated


def crossover_params(a: dict, b: dict, rng: random.Random) -> dict:
    return {name: (a[name] if rng.random() < 0.5 else b[name]) for name in PARAM_NAMES}


def _fitness(candidate_params: dict, population: list, exclude_idx: int, rng: random.Random,
            baseline_hands: int, sparring_hands: int, sparring_opponents: int, baseline_weight: float,
            search_equity_sims: int, seed: int) -> dict:
    candidate = ProAgent(rng=random.Random(seed), equity_sims=search_equity_sims, **candidate_params)
    baseline = evaluate_agent(candidate, n_hands=baseline_hands, seed=seed)["bb_per_100"]

    peer_indices = [i for i in range(len(population)) if i != exclude_idx]
    sampled = rng.sample(peer_indices, k=min(sparring_opponents, len(peer_indices))) if peer_indices else []
    sparring_scores = []
    for peer_idx in sampled:
        opponent = ProAgent(rng=random.Random(seed + peer_idx + 1), equity_sims=search_equity_sims,
                            **population[peer_idx])
        sparring_scores.append(evaluate_agent(candidate, n_hands=sparring_hands, num_players_range=(2, 2),
                                              seed=seed + peer_idx, opponents=[opponent])["bb_per_100"])
    sparring = statistics.mean(sparring_scores) if sparring_scores else 0.0

    combined = baseline_weight * baseline + (1 - baseline_weight) * sparring
    return {"combined": combined, "baseline_bb100": baseline, "sparring_bb100": sparring}


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
                  baseline_hands: int = 500, sparring_hands: int = 250, sparring_opponents: int = 2,
                  baseline_weight: float = 0.5, search_equity_sims: int = 60, seed: int = 0,
                  out_dir: str = "models/pro_agent_optimization", resume: bool = False) -> dict:
    rng = random.Random(seed)
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
        population = [dict(PARAM_DEFAULTS)] + [
            mutate_params(PARAM_DEFAULTS, rng, sigma_scale=rng.uniform(1.0, 3.0))
            for _ in range(population_size - 1)]
        history = []
        best_ever = {"params": dict(PARAM_DEFAULTS), "fitness": float("-inf")}
        start_gen = 0

    for gen in range(start_gen, generations):
        scored = []
        for i, candidate_params in enumerate(population):
            result = _fitness(candidate_params, population, i, rng, baseline_hands, sparring_hands,
                              sparring_opponents, baseline_weight, search_equity_sims, seed=seed + gen * 1000 + i)
            scored.append((candidate_params, result))

        scored.sort(key=lambda t: -t[1]["combined"])
        best_params, best_result = scored[0]
        fitnesses = [r["combined"] for _, r in scored]
        gen_summary = {
            "generation": gen,
            "best_combined": best_result["combined"],
            "best_baseline_bb100": best_result["baseline_bb100"],
            "best_sparring_bb100": best_result["sparring_bb100"],
            "mean_combined": statistics.mean(fitnesses),
        }
        history.append(gen_summary)
        print(f"[gen {gen}] {gen_summary}")

        if best_result["combined"] > best_ever["fitness"]:
            best_ever = {"params": best_params, "fitness": best_result["combined"]}
            print(f"[gen {gen}] new best ever: combined={best_ever['fitness']:.1f} bb/100")

        survivors = [p for p, _ in scored[:survivor_count]]
        next_population = [p for p, _ in scored[:elite_count]]  # elitism
        while len(next_population) < population_size:
            parent_a, parent_b = rng.sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0])
            child = mutate_params(crossover_params(parent_a, parent_b, rng), rng)
            next_population.append(child)
        population = next_population

        _save_state(out_dir, gen, population, history, best_ever)

    print(f"Optimization complete. Best ever: {best_ever['fitness']:.1f} bb/100 (combined baseline+sparring)")
    print(f"Best params saved -> {os.path.join(out_dir, 'best_params.json')}")
    return best_ever


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--survivor-count", type=int, default=4)
    parser.add_argument("--baseline-hands", type=int, default=500)
    parser.add_argument("--sparring-hands", type=int, default=250)
    parser.add_argument("--sparring-opponents", type=int, default=2)
    parser.add_argument("--baseline-weight", type=float, default=0.5)
    parser.add_argument("--search-equity-sims", type=int, default=60,
                        help="Lower than ProAgent's default 200: speed during search, not final precision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="models/pro_agent_optimization")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evolution(generations=args.generations, population_size=args.population_size,
                 elite_count=args.elite_count, survivor_count=args.survivor_count,
                 baseline_hands=args.baseline_hands, sparring_hands=args.sparring_hands,
                 sparring_opponents=args.sparring_opponents, baseline_weight=args.baseline_weight,
                 search_equity_sims=args.search_equity_sims, seed=args.seed, out_dir=args.out_dir,
                 resume=args.resume)
