"""Self-play opponent pool that works across a process boundary.

`DummyVecEnv` runs every sub-environment in the same process, so a plain
shared Python list works for self-play: appending a new snapshot to it
is immediately visible everywhere (see `train.py`). `SubprocVecEnv` runs
each sub-environment in its own OS process, so there's no shared memory
-- a Python object mutated in the main process is invisible to workers
that are already running.

`DiskBackedOpponentPool` sidesteps that: the trainer saves new self-play
snapshots to disk and atomically publishes a manifest file listing the
currently active ones; each worker's pool independently polls that
manifest (cheap -- just an mtime check, most of the time) and reloads
only when it has actually changed.
"""
import json
import os


def write_manifest(snapshot_paths: list, manifest_path: str):
    """Atomically publish the current set of active self-play snapshot paths."""
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(snapshot_paths, f)
    os.replace(tmp_path, manifest_path)  # atomic on POSIX, avoids a torn read


class DiskBackedOpponentPool:
    """List-like: fixed opponents + self-play snapshots refreshed from disk.

    Implements __len__/__getitem__ so it's a drop-in replacement anywhere
    a plain opponents list was used (e.g. `random.choice(pool)`, which
    only needs those two).
    """

    def __init__(self, static_opponents: list, manifest_path: str):
        self.static_opponents = static_opponents
        self.manifest_path = manifest_path
        self._snapshot_agents = {}  # path -> RLAgent
        self._manifest_mtime = None

    def refresh(self):
        """Reload snapshot agents if the manifest has changed since last checked."""
        try:
            mtime = os.path.getmtime(self.manifest_path)
        except OSError:
            return  # no manifest published yet
        if mtime == self._manifest_mtime:
            return

        try:
            with open(self.manifest_path) as f:
                active_paths = json.load(f)
        except (OSError, ValueError):
            return  # transient: shouldn't happen given the atomic replace, but don't crash a worker over it
        self._manifest_mtime = mtime

        from agents.rl_agent import RLAgent  # local: avoid importing sb3 in workers that never need it

        for path in active_paths:
            if path not in self._snapshot_agents:
                try:
                    self._snapshot_agents[path] = RLAgent(path, deterministic=False)
                except (OSError, RuntimeError):
                    pass  # e.g. read mid-write despite the atomic rename; picked up next refresh
        for path in list(self._snapshot_agents):
            if path not in active_paths:
                del self._snapshot_agents[path]

    def __len__(self):
        return len(self.static_opponents) + len(self._snapshot_agents)

    def __getitem__(self, idx):
        return (self.static_opponents + list(self._snapshot_agents.values()))[idx]
