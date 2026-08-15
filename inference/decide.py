"""Feed the bot an arbitrary poker situation and get back a decision.

A "situation" is a plain dict/JSON object describing one decision point:
who's at the table, their stacks and current-street bets, the board, and
whose turn it is. It does not need to come from a hand actually played
by this engine -- any legal-looking table state works, which is what
lets you ask "what would the bot do here?" for a hand you saw elsewhere.

Example situation JSON::

    {
      "num_players": 6,
      "big_blind": 2,
      "small_blind": 1,
      "button_seat": 3,
      "street": "flop",
      "board": ["Qs", "Jd", "2c"],
      "pot_before_street": 60,
      "hero_seat": 0,
      "hero_cards": ["Ah", "Kd"],
      "players": [
        {"seat": 0, "stack": 860, "bet": 0},
        {"seat": 1, "stack": 500, "bet": 40},
        {"seat": 2, "stack": 0, "folded": true},
        {"seat": 3, "stack": 640, "bet": 0},
        {"seat": 4, "stack": 300, "bet": 0, "folded": true},
        {"seat": 5, "stack": 900, "bet": 40}
      ]
    }

Usage:
    python -m inference.decide --situation situation.json --model models/final_model.zip
    python -m inference.decide --interactive --model models/final_model.zip
"""
import argparse
import json

from poker.engine import PlayerState, PokerEngine, Street
from poker.evaluator import estimate_equity

STREET_NAME_TO_ENUM = {s.name.lower(): s for s in Street if s != Street.SHOWDOWN}


def build_engine_from_situation(situation: dict) -> tuple:
    """Construct a `PokerEngine` whose current state matches `situation`.

    Only what's needed to compute legal actions, pot odds and the
    observation is reconstructed -- this engine is for a single decision,
    not for continuing play (no deck, no rest of the hand).
    """
    n = situation["num_players"]
    big_blind = float(situation.get("big_blind", 2.0))
    small_blind = float(situation.get("small_blind", big_blind / 2))
    hero_seat = situation["hero_seat"]

    players_in = {p["seat"]: p for p in situation["players"]}
    if set(players_in) != set(range(n)):
        raise ValueError(f"'players' must have exactly one entry per seat 0..{n - 1}")

    engine = PokerEngine(num_players=n, small_blind=small_blind, big_blind=big_blind)
    engine.board = list(situation.get("board", []))
    engine.street = STREET_NAME_TO_ENUM[situation.get("street", "preflop").lower()]
    engine.button = situation.get("button_seat", 0)
    engine.min_raise = float(situation.get("min_raise", big_blind))

    players = []
    for seat in range(n):
        data = players_in[seat]
        stack = float(data.get("stack", 0.0))
        folded = bool(data.get("folded", False))
        in_hand = bool(data.get("in_hand", stack > 0 or folded))
        bet = float(data.get("bet", 0.0))
        all_in = bool(data.get("all_in", in_hand and not folded and stack <= 0))
        players.append(PlayerState(
            seat=seat,
            stack=stack,
            hole_cards=list(situation["hero_cards"]) if seat == hero_seat else [],
            bet_street=bet,
            total_committed=bet,
            folded=folded,
            all_in=all_in,
            in_hand=in_hand,
        ))
    # The pot carried over from previous streets isn't attributable to any
    # one player from this input format; dumping it onto seat 0's
    # total_committed keeps pot_total() correct for sizing/odds without
    # affecting anything (this engine is never used to resolve a showdown).
    players[0].total_committed += float(situation.get("pot_before_street", 0.0))
    engine.players = players
    engine.max_bet = max((p.bet_street for p in players), default=0.0)
    engine.current_seat = hero_seat

    return engine, hero_seat


def decide(situation: dict, agent=None, model_path: str = None, deterministic: bool = True,
          n_equity_sims: int = 500) -> dict:
    """Return the bot's decision for `situation` (see module docstring).

    Args:
        agent: a ready-to-use `Agent` instance (e.g. `RandomAgent()`,
            `RuleBasedAgent(aggression=0.7)`, `RLAgent(path)`) to decide
            with. Takes priority over `model_path` -- lets callers (e.g.
            the dashboard) pick any agent, not just "RL model or the
            default rule-based fallback".
        model_path: path to a trained MaskablePPO checkpoint; ignored if
            `agent` is given. If neither is given, falls back to a
            default rule-based heuristic agent.
    """
    engine, hero_seat = build_engine_from_situation(situation)
    legal = engine.legal_actions(hero_seat)
    if not legal:
        raise ValueError("Hero has no legal actions in this situation (already folded or all-in)")

    hero = engine.players[hero_seat]
    num_opponents = sum(1 for p in engine.players if p.active and p.seat != hero_seat)
    equity = estimate_equity(hero.hole_cards, engine.board, num_opponents, n_sims=n_equity_sims)

    if agent is None:
        if model_path:
            from agents.rl_agent import RLAgent
            agent = RLAgent(model_path, deterministic=deterministic)
        else:
            from agents.rule_based_agent import RuleBasedAgent
            agent = RuleBasedAgent(aggression=0.5, n_sims=n_equity_sims)

    if hasattr(agent, "act_with_probabilities"):
        action, probs = agent.act_with_probabilities(engine, hero_seat)
        action_probabilities = {a.name: round(p, 4) for a, p in probs.items()}
    else:
        action = agent.act(engine, hero_seat)
        action_probabilities = None

    return {
        "action": action.name,
        "amount": round(legal[action], 2),
        "action_probabilities": action_probabilities,
        "legal_actions": {a.name: round(amt, 2) for a, amt in legal.items()},
        "pot_before_action": round(engine.pot_total(), 2),
        "hero_equity_vs_live_opponents": round(equity, 4),
        "used_model": model_path or getattr(agent, "name", agent.__class__.__name__),
    }


def _prompt_situation() -> dict:
    def ask(prompt, cast=str, default=None):
        raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
        if not raw and default is not None:
            return default
        return cast(raw)

    n = ask("Number of players at the table (2-6)", int)
    big_blind = ask("Big blind size", float, 2.0)
    small_blind = ask("Small blind size", float, big_blind / 2)
    button_seat = ask("Button seat (0-indexed)", int, 0)
    street = ask("Street (preflop/flop/turn/river)", str, "preflop").lower()
    board = []
    if street != "preflop":
        board = ask("Board cards, e.g. 'Ah Kd 2c'", str).split()
    pot_before_street = ask("Pot size carried over from previous streets", float, 0.0)
    hero_seat = ask("Hero's seat", int)
    hero_cards = ask("Hero's hole cards, e.g. 'Ah Kd'", str).split()

    players = []
    for seat in range(n):
        print(f"-- Seat {seat}{' (hero)' if seat == hero_seat else ''} --")
        stack = ask("  Stack", float)
        bet = ask("  Current bet this street", float, 0.0)
        folded = ask("  Folded? (y/n)", str, "n").lower().startswith("y")
        players.append({"seat": seat, "stack": stack, "bet": bet, "folded": folded})

    return {
        "num_players": n, "big_blind": big_blind, "small_blind": small_blind,
        "button_seat": button_seat, "street": street, "board": board,
        "pot_before_street": pot_before_street, "hero_seat": hero_seat,
        "hero_cards": hero_cards, "players": players,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None, help="Path to a trained MaskablePPO .zip checkpoint")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--situation", type=str, help="Path to a situation JSON file")
    src.add_argument("--json", type=str, help="Situation JSON given inline")
    src.add_argument("--interactive", action="store_true", help="Build the situation via interactive prompts")
    parser.add_argument("--stochastic", action="store_true", help="Sample from the policy instead of taking the argmax")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.interactive:
        situation = _prompt_situation()
    elif args.situation:
        with open(args.situation) as f:
            situation = json.load(f)
    else:
        situation = json.loads(args.json)

    result = decide(situation, model_path=args.model, deterministic=not args.stochastic)
    print(json.dumps(result, indent=2))
