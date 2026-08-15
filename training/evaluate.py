"""Evaluate an agent's win rate (bb/100) against the baseline opponent pool."""
import random

from env.poker_env import PokerEnv


def evaluate_agent(agent, n_hands: int = 2000, num_players_range=(2, 6), seed: int = 0) -> dict:
    """Play `agent` as hero for `n_hands` hands against the default opponent pool.

    Returns win rate in big blinds per hand, and per hand won/lost stats.
    """
    env = PokerEnv(num_players_range=num_players_range, seed=seed)
    rng = random.Random(seed)

    total_bb = 0.0
    wins = 0
    for i in range(n_hands):
        obs, info = env.reset(seed=rng.randint(0, 2**31 - 1))
        done = False
        while not done:
            action = agent.act(env.engine, env.hero_seat)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        total_bb += reward
        if reward > 0:
            wins += 1

    return {
        "hands": n_hands,
        "bb_per_hand": total_bb / n_hands,
        "bb_per_100": 100 * total_bb / n_hands,
        "win_rate": wins / n_hands,
    }
