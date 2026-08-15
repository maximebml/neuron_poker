import numpy as np

from env.observation import OBSERVATION_SIZE
from env.poker_env import PokerEnv
from poker.actions import N_ACTIONS


def test_reset_returns_valid_observation():
    env = PokerEnv(num_players_range=(2, 6), seed=1)
    obs, info = env.reset()
    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32
    assert 2 <= info["num_players"] <= 6
    assert not info["done"]


def test_action_mask_matches_legal_actions():
    env = PokerEnv(num_players_range=(2, 6), seed=2)
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (N_ACTIONS,)
    assert mask.dtype == bool
    assert mask.any()
    legal = env.engine.legal_actions(env.hero_seat)
    for i in range(N_ACTIONS):
        assert mask[i] == (i in {int(a) for a in legal})


def test_full_random_episodes_across_table_sizes():
    rng = np.random.default_rng(3)
    for num_players in range(2, 7):
        env = PokerEnv(num_players_range=(num_players, num_players), seed=int(rng.integers(1e6)))
        for _ in range(10):
            obs, info = env.reset(options={"num_players": num_players})
            assert info["num_players"] == num_players
            done = False
            steps = 0
            while not done:
                mask = env.action_masks()
                action = rng.choice(np.flatnonzero(mask))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                assert steps < 200
            assert isinstance(reward, float)


def test_reset_is_seed_reproducible():
    env_a = PokerEnv(num_players_range=(6, 6), seed=42)
    env_b = PokerEnv(num_players_range=(6, 6), seed=42)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_illegal_action_falls_back_gracefully():
    env = PokerEnv(num_players_range=(2, 2), seed=4)
    env.reset()
    mask = env.action_masks()
    illegal_ids = np.flatnonzero(~mask)
    if len(illegal_ids) > 0:
        obs, reward, terminated, truncated, info = env.step(illegal_ids[0])
        assert obs.shape == (OBSERVATION_SIZE,)
