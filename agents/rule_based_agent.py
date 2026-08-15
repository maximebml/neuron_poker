"""Monte-Carlo-equity heuristic agent.

Not meant to be strong poker strategy -- it exists so the RL agent has a
pool of reasonably sensible, varied opponents to train and be evaluated
against (pure random play is too weak/degenerate to train a good policy
against on its own), and to give a non-RL baseline to compare against.
"""
import random

from agents.base_agent import Agent
from poker.actions import Action
from poker.evaluator import estimate_equity


class RuleBasedAgent(Agent):
    """Bets/calls/folds based on Monte Carlo hand equity vs. live opponents.

    `aggression` in [0, 1] shifts the agent from a tight calling style
    (0.0) to a loose, high-frequency-raising style (1.0), which gives the
    training opponent pool useful diversity.
    """

    def __init__(self, aggression: float = 0.5, n_sims: int = 150, rng: random.Random = None):
        self.aggression = aggression
        self.n_sims = n_sims
        self.rng = rng or random.Random()
        self.name = f"rule_based(agg={aggression:.1f})"

    def act(self, engine, seat):
        legal = engine.legal_actions(seat)
        player = engine.players[seat]
        num_opponents = sum(1 for q in engine.players if q.active and q.seat != seat)
        equity = estimate_equity(player.hole_cards, engine.board, num_opponents,
                                  n_sims=self.n_sims, rng=self.rng)

        facing_bet = Action.FOLD in legal
        if facing_bet:
            call_amount = legal[Action.CHECK_CALL]
            pot_odds = call_amount / (engine.pot_total() + call_amount)
            if equity < pot_odds * (1.0 - 0.15 * self.aggression):
                return Action.FOLD
            raise_threshold = 0.65 - 0.25 * self.aggression
            raise_freq = 0.2 + 0.5 * self.aggression
        else:
            raise_threshold = 0.55 - 0.2 * self.aggression
            raise_freq = 0.3 + 0.5 * self.aggression

        if equity > raise_threshold and self.rng.random() < raise_freq:
            for act in (Action.RAISE_33, Action.RAISE_75, Action.RAISE_150, Action.ALL_IN):
                if act in legal:
                    return act
        return Action.CHECK_CALL
