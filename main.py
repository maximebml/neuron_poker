"""neuron_poker: a no-limit hold'em bot (2-6 players) trained with RL.

Usage:
    python main.py train    [--timesteps N] [--n-envs N] [--min-players N] [--max-players N] ...
    python main.py evaluate --model models/final_model.zip [--hands N]
    python main.py decide   --model models/final_model.zip --situation situation.json
    python main.py play     [--model models/final_model.zip] [--num-players N] [--hands N]

`train` and `decide` accept the same flags as `python -m training.train` /
`python -m inference.decide` (run with --help for the full list); `main.py`
is just a single convenient entry point for all four workflows.
"""
import argparse
import json
import random
import sys


def cmd_train(argv):
    import runpy
    sys.argv = ["training.train"] + argv
    runpy.run_module("training.train", run_name="__main__")


def cmd_decide(argv):
    import runpy
    sys.argv = ["inference.decide"] + argv
    runpy.run_module("inference.decide", run_name="__main__")


def cmd_evaluate(argv):
    parser = argparse.ArgumentParser(prog="main.py evaluate")
    parser.add_argument("--model", required=True, help="Path to a trained MaskablePPO .zip checkpoint")
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from agents.rl_agent import RLAgent
    from training.evaluate import evaluate_agent

    agent = RLAgent(args.model, deterministic=True)
    stats = evaluate_agent(agent, n_hands=args.hands, num_players_range=(args.min_players, args.max_players),
                            seed=args.seed)
    print(json.dumps(stats, indent=2))


def cmd_play(argv):
    parser = argparse.ArgumentParser(prog="main.py play",
                                      description="Watch the bot play a few hands against baseline opponents.")
    parser.add_argument("--model", type=str, default=None,
                        help="RL model controlling seat 0; without it, seat 0 uses the rule-based baseline")
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--hands", type=int, default=5)
    parser.add_argument("--stack-bb", type=float, default=200.0, help="Starting stack, in big blinds")
    parser.add_argument("--big-blind", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from agents.random_agent import RandomAgent
    from agents.rule_based_agent import RuleBasedAgent
    from poker.engine import PokerEngine

    rng = random.Random(args.seed)
    if args.model:
        from agents.rl_agent import RLAgent
        seat0_agent = RLAgent(args.model, deterministic=True)
    else:
        seat0_agent = RuleBasedAgent(aggression=0.5, rng=rng)
    opponent_choices = [RandomAgent(rng=rng), RuleBasedAgent(aggression=0.3, rng=rng),
                        RuleBasedAgent(aggression=0.7, rng=rng)]
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


COMMANDS = {"train": cmd_train, "evaluate": cmd_evaluate, "decide": cmd_decide, "play": cmd_play}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
