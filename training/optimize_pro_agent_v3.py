"""Iteratively refine ProAgent's six highest-leverage parameters (the same
subset optimize_pro_agent_v2.py searches -- see that module's docstring for
why these six) through a chain of evolutionary search rounds, each played
out at one full 6-max table rather than several separate matches: hero
(the candidate) + the stock ProAgent (a fixed anchor, every round) + the
current "champion" (a ProAgent tuned with the best parameters the chain has
found so far, starting from ProAgent v2's tuned parameters) + three more
fixed agents (random, two rule-based aggressions) filling the rest of the
table. That's "and others if the game is large enough" taken literally: a
6-max table is the largest this project supports, so it's the one table
size where the stock agent, the champion, and a spread of weaker agents
can all sit down together. It also sidesteps the raw-scale-mismatch bug
both the v1 and v2 optimizers had, since fitness here is a single number
(bb/100 at that one table) rather than a blend of differently-scaled
components.

Each of ITERATIONS rounds runs its own small evolutionary search
(generations_per_iteration generations, seeded near the current champion
rather than the global defaults), then the round's winner is validated
head-to-head against both the current champion and the stock ProAgent on a
held-out sample. The champion is only replaced if the challenger actually
won that validation match -- "the search's own fitness said it's better"
was already shown not to be good enough on its own, twice, while building
v1 and v2. A round that doesn't produce a genuine improvement is a no-op
(the champion carries over unchanged), so the chain can only ever get
better or stay flat, never silently regress, and a chain that stalls out
early on plateauing improvements is a legitimate, honestly-reported result,
not a failure to hide.

Deliberately NOT pure sequential self-play (where each round only ever
faces its immediate predecessor and the fixed reference points are dropped
entirely): that's exactly the self-play drift failure mode already flagged
for the main RL training run and both prior ProAgent optimizers in this
project. Keeping the stock ProAgent as a constant anchor across all ten
rounds, alongside the evolving champion, is a deliberate choice to avoid
repeating that mistake a third time -- not an oversight of "replace
everything with the latest winner."

The whole chain is resumable at iteration granularity, and each iteration's
own evolutionary search is separately resumable/checkpointed (same atomic
temp-file-plus-rename pattern as v1/v2), since this is meant to run for
hours unattended in an environment with a history of unannounced restarts.

Usage:
    python -m training.optimize_pro_agent_v3 --iterations 10
"""
import argparse
import json
import os
import random
import statistics

from agents.pro_agent import ProAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from training.evaluate import evaluate_agent
from training.optimize_pro_agent_v2 import OPTIMIZED_PARAM_NAMES, crossover_params, mutate_params

TABLE_SIZE = 6


def _make_agent(params: dict, rng: random.Random, equity_sims: int = 200) -> ProAgent:
    return ProAgent(rng=rng, equity_sims=equity_sims, **params)


def _table_evaluate(candidate_params: dict, champion_params: dict, hands: int, equity_sims: int,
                    seed: int) -> float:
    """bb/100 for a candidate at a full 6-max table: stock ProAgent + the
    current champion + random + two rule-based aggressions."""
    candidate = _make_agent(candidate_params, random.Random(seed), equity_sims)
    opponents = [
        _make_agent({}, random.Random(seed + 1), equity_sims),
        _make_agent(champion_params, random.Random(seed + 2), equity_sims),
        RandomAgent(rng=random.Random(seed + 3)),
        RuleBasedAgent(aggression=0.3, rng=random.Random(seed + 4)),
        RuleBasedAgent(aggression=0.7, rng=random.Random(seed + 5)),
    ]
    result = evaluate_agent(candidate, n_hands=hands, num_players_range=(TABLE_SIZE, TABLE_SIZE),
                            seed=seed, opponents=opponents)
    return result["bb_per_100"]


def _head_to_head_bb100(a_params: dict, b_params: dict, hands: int, seed: int) -> float:
    """Heads-up bb/100 for A against B (positive means A is ahead)."""
    a = _make_agent(a_params, random.Random(seed))
    b = _make_agent(b_params, random.Random(seed + 1))
    result = evaluate_agent(a, n_hands=hands, num_players_range=(2, 2), seed=seed, opponents=[b])
    return result["bb_per_100"]


def _save_iter_state(out_dir, generation, population, history, best_ever):
    os.makedirs(out_dir, exist_ok=True)
    state = {"generation": generation, "population": population, "history": history, "best_ever": best_ever}
    tmp_path = os.path.join(out_dir, "state.json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, os.path.join(out_dir, "state.json"))


def _save_chain_state(out_dir, completed_iterations, champion_params, history):
    os.makedirs(out_dir, exist_ok=True)
    state = {"completed_iterations": completed_iterations, "champion_params": champion_params,
             "history": history}
    tmp_path = os.path.join(out_dir, "chain_state.json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, os.path.join(out_dir, "chain_state.json"))


def run_iteration_search(champion_params: dict, generations: int, population_size: int, elite_count: int,
                         survivor_count: int, hands: int, search_equity_sims: int, seed: int, out_dir: str,
                         resume: bool = False) -> dict:
    """One round's evolutionary search: find the best challenger to the
    current champion. Seeds the initial population near the champion
    (rather than the global PARAM_DEFAULTS) since each round should refine
    the chain's current best, not restart the search from scratch."""
    rng = random.Random(seed)
    state_path = os.path.join(out_dir, "state.json")

    if resume and os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        population = state["population"]
        history = state["history"]
        best_ever = state["best_ever"]
        start_gen = state["generation"] + 1
        print(f"    resumed inner search at generation {start_gen}")
    else:
        population = [dict(champion_params)] + [
            mutate_params(champion_params, rng, sigma_scale=rng.uniform(0.5, 2.0))
            for _ in range(population_size - 1)]
        history = []
        best_ever = {"params": dict(champion_params), "fitness": float("-inf")}
        start_gen = 0

    for gen in range(start_gen, generations):
        scored = [(params, _table_evaluate(params, champion_params, hands, search_equity_sims,
                                           seed=seed + gen * 1000 + i))
                 for i, params in enumerate(population)]
        scored.sort(key=lambda t: -t[1])
        best_params, best_bb100 = scored[0]
        gen_summary = {"generation": gen, "best_bb100": best_bb100,
                       "mean_bb100": statistics.mean(b for _, b in scored)}
        history.append(gen_summary)
        print(f"    [gen {gen}] best={best_bb100:.1f} bb/100  mean={gen_summary['mean_bb100']:.1f} bb/100")

        if best_bb100 > best_ever["fitness"]:
            best_ever = {"params": best_params, "fitness": best_bb100}

        survivors = [p for p, _ in scored[:survivor_count]]
        next_population = [p for p, _ in scored[:elite_count]]  # elitism
        while len(next_population) < population_size:
            parent_a, parent_b = rng.sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0])
            next_population.append(mutate_params(crossover_params(parent_a, parent_b, rng), rng))
        population = next_population
        _save_iter_state(out_dir, gen, population, history, best_ever)

    return best_ever["params"]


def run_chain(iterations: int, generations_per_iteration: int = 5, population_size: int = 6,
             elite_count: int = 2, survivor_count: int = 3, search_hands: int = 200,
             validation_hands: int = 1200, search_equity_sims: int = 50, seed: int = 0,
             out_dir: str = "models/pro_agent_v3_chain", final_out_dir: str = "models/pro_agent_v3",
             champion_init_path: str = "models/pro_agent_v2_fixed/best_params.json",
             resume: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    chain_state_path = os.path.join(out_dir, "chain_state.json")

    if resume and os.path.exists(chain_state_path):
        with open(chain_state_path) as f:
            chain_state = json.load(f)
        champion_params = chain_state["champion_params"]
        history = chain_state["history"]
        start_iter = chain_state["completed_iterations"]
        print(f"Resumed chain at iteration {start_iter + 1}/{iterations}")
    else:
        with open(champion_init_path) as f:
            init_params = json.load(f)
        champion_params = {name: init_params[name] for name in OPTIMIZED_PARAM_NAMES}
        history = []
        start_iter = 0
        print(f"Starting chain from {champion_init_path}")

    for it in range(start_iter, iterations):
        print(f"=== iteration {it + 1}/{iterations} ===")
        iter_dir = os.path.join(out_dir, f"iter_{it}")
        challenger_params = run_iteration_search(
            champion_params, generations=generations_per_iteration, population_size=population_size,
            elite_count=elite_count, survivor_count=survivor_count, hands=search_hands,
            search_equity_sims=search_equity_sims, seed=seed + it * 100000, out_dir=iter_dir,
            resume=resume and it == start_iter)

        vs_champion = _head_to_head_bb100(challenger_params, champion_params, validation_hands,
                                          seed=seed + it * 100000 + 9001)
        vs_stock_pro = _head_to_head_bb100(challenger_params, {}, validation_hands,
                                           seed=seed + it * 100000 + 9002)
        promoted = vs_champion > 0
        record = {"iteration": it, "vs_champion_bb100": vs_champion, "vs_stock_pro_bb100": vs_stock_pro,
                 "promoted": promoted, "params": challenger_params}
        history.append(record)
        verdict = "PROMOTED to champion" if promoted else "kept previous champion (no improvement)"
        print(f"iteration {it + 1}: challenger vs champion={vs_champion:.1f} bb/100, "
             f"vs stock ProAgent={vs_stock_pro:.1f} bb/100 -> {verdict}")

        if promoted:
            champion_params = challenger_params

        _save_chain_state(out_dir, it + 1, champion_params, history)

    os.makedirs(final_out_dir, exist_ok=True)
    with open(os.path.join(final_out_dir, "best_params.json"), "w") as f:
        json.dump(champion_params, f, indent=2)
    promotions = sum(1 for r in history if r["promoted"])
    print(f"Chain complete: {iterations} iterations, {promotions} promoted. "
         f"ProAgent_v3 params saved -> {os.path.join(final_out_dir, 'best_params.json')}")
    return champion_params


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--generations-per-iteration", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--survivor-count", type=int, default=3)
    parser.add_argument("--search-hands", type=int, default=200, help="Hands per candidate per generation")
    parser.add_argument("--validation-hands", type=int, default=1200,
                        help="Hands per validation head-to-head (vs champion and vs stock ProAgent)")
    parser.add_argument("--search-equity-sims", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="models/pro_agent_v3_chain")
    parser.add_argument("--final-out-dir", type=str, default="models/pro_agent_v3")
    parser.add_argument("--champion-init-path", type=str, default="models/pro_agent_v2_fixed/best_params.json")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_chain(iterations=args.iterations, generations_per_iteration=args.generations_per_iteration,
             population_size=args.population_size, elite_count=args.elite_count,
             survivor_count=args.survivor_count, search_hands=args.search_hands,
             validation_hands=args.validation_hands, search_equity_sims=args.search_equity_sims,
             seed=args.seed, out_dir=args.out_dir, final_out_dir=args.final_out_dir,
             champion_init_path=args.champion_init_path, resume=args.resume)
