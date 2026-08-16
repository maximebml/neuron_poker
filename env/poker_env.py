"""Single-agent gymnasium environment: hero plays one hand vs. 1-5 opponents.

Each episode is one hand of no-limit hold'em with a randomly sampled
table size (2-6 players), stack depths and button position, so a policy
trained here has to cope with arbitrary short-handed/full-ring and
shallow/deep-stack situations rather than a single fixed configuration.
Seats other than the hero are auto-played synchronously inside `step`
by pluggable opponent agents (see `agents/`).
"""
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.observation import OBSERVATION_SIZE, build_observation, legal_action_mask
from poker.actions import N_ACTIONS, Action
from poker.engine import PokerEngine


class PokerEnv(gym.Env):
    """Gymnasium environment for training a poker-playing policy."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players_range=(2, 6), small_blind: float = 1.0, big_blind: float = 2.0,
                 min_stack_bb: float = 20.0, max_stack_bb: float = 200.0, opponents: list = None,
                 seed: int = None):
        super().__init__()
        self.num_players_range = num_players_range
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.min_stack_bb = min_stack_bb
        self.max_stack_bb = max_stack_bb
        self._rng = random.Random(seed)

        if opponents is None:
            from agents.random_agent import RandomAgent
            from agents.rule_based_agent import RuleBasedAgent
            opponents = [
                RandomAgent(rng=self._rng),
                RuleBasedAgent(aggression=0.2, rng=self._rng),
                RuleBasedAgent(aggression=0.5, rng=self._rng),
                RuleBasedAgent(aggression=0.8, rng=self._rng),
            ]
        self.opponent_pool = opponents

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-1e4, high=1e4, shape=(OBSERVATION_SIZE,), dtype=np.float32)

        self.engine: PokerEngine = None
        self.hero_seat = 0
        self.num_players = None
        self._seat_agents = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        options = options or {}

        # Opponent pools that support it (DiskBackedOpponentPool, used with
        # SubprocVecEnv self-play -- see training/opponent_pool.py) get a
        # chance to pick up newly published self-play snapshots here, once
        # per hand rather than once per action.
        if hasattr(self.opponent_pool, "refresh"):
            self.opponent_pool.refresh()

        # A hand can end with everyone folding to the hero's blind before
        # the hero ever gets a turn (a "walk"). Gymnasium requires reset()
        # to hand back a live (non-terminated) episode, and a hand with
        # zero hero decisions has nothing to train on anyway, so redeal
        # until the hero actually faces a decision.
        for _ in range(50):
            n = options.get("num_players") or self._rng.randint(*self.num_players_range)
            self.num_players = n
            self.hero_seat = options.get("hero_seat", self._rng.randrange(n))
            stacks = [round(self._rng.uniform(self.min_stack_bb, self.max_stack_bb) * self.big_blind, 2)
                      for _ in range(n)]
            button = self._rng.randrange(n)

            self.engine = PokerEngine(num_players=n, small_blind=self.small_blind, big_blind=self.big_blind,
                                       rng=self._rng)
            self._seat_agents = [self._rng.choice(self.opponent_pool) for _ in range(n)]
            self.engine.start_hand(stacks, button)
            self._play_opponents_until_hero_or_done()
            if not self.engine.done:
                break
        else:
            raise RuntimeError("Could not deal a hand where the hero gets a decision after 50 tries")

        return self._build_obs(), self._info()

    def step(self, action):
        action = Action(int(action))
        legal = self.engine.legal_actions(self.hero_seat)
        if action not in legal:
            action = Action.CHECK_CALL if Action.CHECK_CALL in legal else next(iter(legal))

        self.engine.step(action)
        self._play_opponents_until_hero_or_done()

        terminated = self.engine.done
        reward = self.engine.payouts[self.hero_seat] / self.big_blind if terminated else 0.0
        return self._build_obs(), reward, terminated, False, self._info()

    def action_masks(self) -> np.ndarray:
        """Legal-action mask for the hero's current turn (sb3-contrib MaskablePPO)."""
        if self.engine.done:
            return np.ones(N_ACTIONS, dtype=bool)
        return legal_action_mask(self.engine, self.hero_seat)

    def render(self):
        e = self.engine
        lines = [f"Street: {e.street.name}  Board: {e.board}  Pot: {e.pot_total():.1f}"]
        for p in e.players:
            marker = " <- hero" if p.seat == self.hero_seat else ""
            marker += " (button)" if p.seat == e.button else ""
            status = "folded" if p.folded else ("all-in" if p.all_in else "in")
            lines.append(f"  seat {p.seat}: stack={p.stack:.1f} bet={p.bet_street:.1f} "
                         f"[{status}] cards={p.hole_cards if p.seat == self.hero_seat else '??'}{marker}")
        print("\n".join(lines))

    def _play_opponents_until_hero_or_done(self):
        while not self.engine.done and self.engine.current_player() != self.hero_seat:
            seat = self.engine.current_player()
            action = self._seat_agents[seat].act(self.engine, seat)
            self.engine.step(action)

    def _build_obs(self):
        return build_observation(self.engine, self.hero_seat)

    def _info(self):
        return {
            "hero_seat": self.hero_seat,
            "num_players": self.num_players,
            "street": self.engine.street.name,
            "pot": self.engine.pot_total(),
            "done": self.engine.done,
        }


gym.register(id="neuron_poker-v0", entry_point="env.poker_env:PokerEnv")
