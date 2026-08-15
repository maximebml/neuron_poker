"""Uniformly-random legal action. Useful as a training opponent and a floor baseline."""
import random

from agents.base_agent import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()

    def act(self, engine, seat):
        legal = engine.legal_actions(seat)
        return self.rng.choice(list(legal.keys()))
