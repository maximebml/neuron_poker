import json
import os
import time

import pytest

from agents.random_agent import RandomAgent
from training.opponent_pool import DiskBackedOpponentPool, write_manifest


def test_pool_behaves_like_a_list_with_only_static_opponents(tmp_path):
    static = [RandomAgent(), RandomAgent()]
    pool = DiskBackedOpponentPool(static, str(tmp_path / "manifest.json"))
    assert len(pool) == 2
    assert pool[0] is static[0]
    assert pool[1] is static[1]


def test_refresh_is_a_noop_without_a_published_manifest(tmp_path):
    pool = DiskBackedOpponentPool([RandomAgent()], str(tmp_path / "missing_manifest.json"))
    pool.refresh()  # must not raise
    assert len(pool) == 1


def test_refresh_loads_new_snapshots_from_manifest(tmp_path):
    # Build a tiny real checkpoint to load as a "self-play snapshot".
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from env.poker_env import PokerEnv

    env = ActionMasker(PokerEnv(num_players_range=(2, 2), seed=0), lambda e: e.action_masks())
    model = MaskablePPO("MlpPolicy", env, n_steps=32, batch_size=16, verbose=0)
    snapshot_path = str(tmp_path / "snapshot.zip")
    model.save(snapshot_path)

    manifest_path = str(tmp_path / "manifest.json")
    pool = DiskBackedOpponentPool([RandomAgent()], manifest_path)

    write_manifest([snapshot_path], manifest_path)
    pool.refresh()

    assert len(pool) == 2  # 1 static + 1 loaded snapshot
    from agents.rl_agent import RLAgent
    assert isinstance(pool[1], RLAgent)


def test_refresh_evicts_snapshots_no_longer_in_manifest(tmp_path):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from env.poker_env import PokerEnv

    env = ActionMasker(PokerEnv(num_players_range=(2, 2), seed=0), lambda e: e.action_masks())
    model = MaskablePPO("MlpPolicy", env, n_steps=32, batch_size=16, verbose=0)
    path_a = str(tmp_path / "a.zip")
    path_b = str(tmp_path / "b.zip")
    model.save(path_a)
    model.save(path_b)

    manifest_path = str(tmp_path / "manifest.json")
    pool = DiskBackedOpponentPool([], manifest_path)

    write_manifest([path_a, path_b], manifest_path)
    pool.refresh()
    assert len(pool) == 2

    write_manifest([path_b], manifest_path)  # 'a' evicted
    pool.refresh()
    assert len(pool) == 1


def test_refresh_skips_reload_when_manifest_unchanged(tmp_path, monkeypatch):
    manifest_path = str(tmp_path / "manifest.json")
    write_manifest([], manifest_path)
    pool = DiskBackedOpponentPool([], manifest_path)
    pool.refresh()

    calls = []
    real_getmtime = os.path.getmtime

    def tracking_getmtime(path):
        calls.append(path)
        return real_getmtime(path)

    monkeypatch.setattr(os.path, "getmtime", tracking_getmtime)
    pool.refresh()
    pool.refresh()
    assert len(calls) == 2  # mtime checked each time, but no re-read/re-parse of an unchanged manifest


def test_write_manifest_is_atomic_no_partial_file_left_behind(tmp_path):
    manifest_path = str(tmp_path / "manifest.json")
    write_manifest(["a.zip", "b.zip"], manifest_path)
    assert not os.path.exists(manifest_path + ".tmp")
    with open(manifest_path) as f:
        assert json.load(f) == ["a.zip", "b.zip"]
