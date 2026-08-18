"""neuron_poker: a no-limit hold'em bot (2-6 players) trained with RL.

Usage:
    python main.py train        [--timesteps N] [--n-envs N] [--min-players N] [--max-players N] ...
    python main.py evaluate     --agent {rl,pro,rule_based,random} [--model ...] [--hands N]
    python main.py decide       --model models/final_model.zip --situation situation.json
    python main.py play         [--agent {rl,pro,rule_based,random}] [--model ...] [--hands N]
    python main.py optimize-pro    [--generations N] [--population-size N] ...
    python main.py optimize-pro-v2 [--generations N] [--pro-weight 0.7] ...

`train`, `decide`, `optimize-pro`, and `optimize-pro-v2` accept the same
flags as their `python -m training.train` / `python -m inference.decide` /
`python -m training.optimize_pro_agent` / `python -m
training.optimize_pro_agent_v2` equivalents (run with --help for the full
list); `main.py` is just a single convenient entry point.
"""
import argparse
import json
import random
import sys


def cmd_train(argv):
    # Call training.train.train() directly rather than routing through
    # runpy: with --vec-env subproc, SubprocVecEnv's forkserver/spawn
    # workers need to re-import "the main module" to start up, and doing
    # that through runpy's simulated __main__ (rather than this file
    # being the real one) measured ~9x slower end to end -- each worker
    # re-executing main.py's own dispatch logic on top of its real work.
    from training.train import _parse_args, train
    sys.argv = ["training.train"] + argv
    args = _parse_args()
    train(total_timesteps=args.timesteps, n_envs=args.n_envs,
         num_players_range=(args.min_players, args.max_players), out_dir=args.out_dir,
         checkpoint_every=args.checkpoint_every, self_play_every=args.self_play_every,
         self_play_pool_cap=args.self_play_pool_cap, eval_every=args.eval_every,
         eval_hands=args.eval_hands, seed=args.seed, tensorboard_log=args.tensorboard_log,
         resume_from=args.resume_from, vec_env_type=args.vec_env, rule_based_n_sims=args.opponent_n_sims)


def cmd_decide(argv):
    from inference.decide import _parse_args, decide
    sys.argv = ["inference.decide"] + argv
    args = _parse_args()
    if args.interactive:
        from inference.decide import _prompt_situation
        situation = _prompt_situation()
    elif args.situation:
        with open(args.situation) as f:
            situation = json.load(f)
    else:
        situation = json.loads(args.json)
    result = decide(situation, model_path=args.model, deterministic=not args.stochastic)
    print(json.dumps(result, indent=2))


def cmd_evaluate(argv):
    parser = argparse.ArgumentParser(prog="main.py evaluate",
                                      description="Evaluate an agent's bb/100 and win rate against a fixed "
                                                  "baseline opponent pool (random + 3 rule-based agents).")
    parser.add_argument("--model", type=str, default=None, help="Path to a trained MaskablePPO .zip checkpoint")
    parser.add_argument("--agent", type=str, default=None, choices=["rl", "pro", "rule_based", "random"],
                        help="Which agent to evaluate, overriding the --model-presence default (implied 'rl')")
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from training.evaluate import evaluate_agent

    agent_kind = args.agent or ("rl" if args.model else "rule_based")
    if agent_kind == "rl":
        from agents.rl_agent import RLAgent
        agent = RLAgent(args.model, deterministic=True)
    elif agent_kind == "pro":
        from agents.pro_agent import ProAgent
        agent = ProAgent()
    elif agent_kind == "random":
        from agents.random_agent import RandomAgent
        agent = RandomAgent()
    else:
        from agents.rule_based_agent import RuleBasedAgent
        agent = RuleBasedAgent(aggression=0.5)

    stats = evaluate_agent(agent, n_hands=args.hands, num_players_range=(args.min_players, args.max_players),
                            seed=args.seed)
    print(json.dumps(stats, indent=2))


def cmd_play(argv):
    parser = argparse.ArgumentParser(prog="main.py play",
                                      description="Watch the bot play a few hands against baseline opponents.")
    parser.add_argument("--model", type=str, default=None,
                        help="RL model controlling seat 0 (equivalent to --agent rl); "
                             "without it, seat 0 uses the rule-based baseline")
    parser.add_argument("--agent", type=str, default=None, choices=["rl", "pro", "rule_based", "random"],
                        help="Explicitly pick seat 0's agent, overriding the --model-presence default")
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--hands", type=int, default=5)
    parser.add_argument("--stack-bb", type=float, default=200.0, help="Starting stack, in big blinds")
    parser.add_argument("--big-blind", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from agents.pro_agent import ProAgent
    from agents.random_agent import RandomAgent
    from agents.rule_based_agent import RuleBasedAgent
    from poker.engine import PokerEngine

    rng = random.Random(args.seed)
    agent_kind = args.agent or ("rl" if args.model else "rule_based")
    if agent_kind == "rl":
        from agents.rl_agent import RLAgent
        seat0_agent = RLAgent(args.model, deterministic=True)
    elif agent_kind == "pro":
        seat0_agent = ProAgent(rng=rng)
    elif agent_kind == "random":
        seat0_agent = RandomAgent(rng=rng)
    else:
        seat0_agent = RuleBasedAgent(aggression=0.5, rng=rng)
    opponent_choices = [RandomAgent(rng=rng), RuleBasedAgent(aggression=0.3, rng=rng),
                        RuleBasedAgent(aggression=0.7, rng=rng), ProAgent(rng=rng)]
    agents_by_seat = [seat0_agent] + [rng.choice(opponent_choices) for _ in range(args.num_players - 1)]

    stacks = [args.stack_bb * args.big_blind] * args.num_players
    button = 0
    for hand_no in range(1, args.hands + 1):
        if sum(s > 0 for s in stacks) < 2:
            print("A player is out of chips -- stopping.")
            break
        engine = PokerEngine(num_players=args.num_players, small_blind=args.big_blind / 2,
                             big_blind=args.big_blind, rng=rng)
        engine.start_hand(stacks, button=button)
        while not engine.done:
            seat = engine.current_player()
            engine.step(agents_by_seat[seat].act(engine, seat))
        payouts = {s: round(v, 1) for s, v in engine.payouts.items()}
        print(f"Hand {hand_no}: payouts={payouts}")
        stacks = [p.stack for p in engine.players]
        button = (button + 1) % args.num_players

    print("Final stacks:", {i: round(s, 1) for i, s in enumerate(stacks)})


def cmd_optimize_pro(argv):
    from training.optimize_pro_agent import _parse_args, run_evolution
    sys.argv = ["training.optimize_pro_agent"] + argv
    args = _parse_args()
    run_evolution(generations=args.generations, population_size=args.population_size,
                  elite_count=args.elite_count, survivor_count=args.survivor_count,
                  baseline_hands=args.baseline_hands, sparring_hands=args.sparring_hands,
                  sparring_opponents=args.sparring_opponents, baseline_weight=args.baseline_weight,
                  search_equity_sims=args.search_equity_sims, seed=args.seed, out_dir=args.out_dir,
                  resume=args.resume)


def cmd_optimize_pro_v2(argv):
    from training.optimize_pro_agent_v2 import _parse_args, run_evolution
    sys.argv = ["training.optimize_pro_agent_v2"] + argv
    args = _parse_args()
    run_evolution(generations=args.generations, population_size=args.population_size,
                  elite_count=args.elite_count, survivor_count=args.survivor_count,
                  pro_hands=args.pro_hands, other_hands=args.other_hands, pro_weight=args.pro_weight,
                  search_equity_sims=args.search_equity_sims, seed=args.seed, out_dir=args.out_dir,
                  model_path=args.model_path, resume=args.resume)


COMMANDS = {"train": cmd_train, "evaluate": cmd_evaluate, "decide": cmd_decide, "play": cmd_play,
           "optimize-pro": cmd_optimize_pro, "optimize-pro-v2": cmd_optimize_pro_v2}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
