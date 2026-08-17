"""Train a poker-playing policy with MaskablePPO (sb3-contrib).

Usage:
    python -m training.train --timesteps 2000000 --n-envs 8 --out-dir models

The hero trains against a pool of opponents (random + equity-heuristic
agents at varying aggression). With --self-play-every > 0, frozen
snapshots of the training policy itself are periodically added to that
pool, so later training also has to beat earlier versions of itself.

--vec-env dummy (default) runs every sub-environment in this one process
(simple, well-tested, but only ever uses one CPU core's worth of Python
execution for env stepping). --vec-env subproc runs each sub-environment
in its own OS process for real multi-core parallelism -- worthwhile since
env stepping is pure Python/CPU-bound -- at the cost of self-play needing
to go through disk (see training/opponent_pool.py) instead of a shared
in-memory list, since subprocesses don't share memory with the trainer.
"""
import argparse
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agents.base_agent import Agent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from env.observation import build_observation, legal_action_mask
from env.poker_env import PokerEnv
from poker.actions import Action
from training.evaluate import evaluate_agent
from training.opponent_pool import DiskBackedOpponentPool, write_manifest


def default_opponent_pool(rng=None, rule_based_n_sims: int = 150):
    """A small but diverse pool of non-RL opponents.

    `rule_based_n_sims` controls the Monte Carlo equity estimate each
    RuleBasedAgent runs on every decision -- and dominates training
    throughput far more than anything else in the pipeline (measured:
    ~6300 env steps/sec against random opponents vs. ~50/sec at the
    default n_sims=150, a ~127x difference). Lower is faster but noisier;
    150 is a reasonable default for a strong-ish baseline pool, but long
    runs may prefer trading some of that precision for speed.
    """
    return [
        RandomAgent(rng=rng),
        RuleBasedAgent(aggression=0.1, n_sims=rule_based_n_sims, rng=rng),
        RuleBasedAgent(aggression=0.4, n_sims=rule_based_n_sims, rng=rng),
        RuleBasedAgent(aggression=0.7, n_sims=rule_based_n_sims, rng=rng),
        RuleBasedAgent(aggression=1.0, n_sims=rule_based_n_sims, rng=rng),
    ]


class _LiveModelAgent(Agent):
    """Wraps an in-memory (not-yet-saved) MaskablePPO model as an Agent."""

    def __init__(self, model, deterministic=False):
        self.model = model
        self.deterministic = deterministic
        self.name = "self-play-snapshot"

    def act(self, engine, seat):
        obs = build_observation(engine, seat)
        mask = legal_action_mask(engine, seat)
        raw_action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return Action(int(raw_action))


def _mask_fn(env):
    return env.action_masks()


def make_vec_env(n_envs: int, num_players_range, base_seed: int, vec_env_type: str, rule_based_n_sims: int,
                 self_play_manifest_path: str = None):
    """Build the training vec env.

    Returns (vec_env, shared_opponent_pool). For "dummy", shared_opponent_pool
    is the actual live list every sub-env holds a reference to -- append to
    it and every env sees the change, which is how self-play snapshots get
    added. For "subproc" it's None: each subprocess builds its own
    DiskBackedOpponentPool pointed at self_play_manifest_path instead, since
    there's no memory to share across processes -- see training/opponent_pool.py.
    """
    if vec_env_type == "dummy":
        shared_pool = default_opponent_pool(rule_based_n_sims=rule_based_n_sims)

        def make_one(i):
            def _init():
                e = PokerEnv(num_players_range=num_players_range, opponents=shared_pool, seed=base_seed + i)
                return ActionMasker(e, _mask_fn)
            return _init

        return DummyVecEnv([make_one(i) for i in range(n_envs)]), shared_pool

    elif vec_env_type == "subproc":
        def make_one(i):
            def _init():
                pool = DiskBackedOpponentPool(default_opponent_pool(rule_based_n_sims=rule_based_n_sims),
                                              self_play_manifest_path)
                e = PokerEnv(num_players_range=num_players_range, opponents=pool, seed=base_seed + i)
                return ActionMasker(e, _mask_fn)
            return _init

        # No start_method override: SB3 defaults to forkserver/spawn, not
        # fork -- forking a process that already has PyTorch's internal
        # threads running is a known deadlock/crash hazard.
        return SubprocVecEnv([make_one(i) for i in range(n_envs)]), None

    raise ValueError(f"vec_env_type must be 'dummy' or 'subproc', got {vec_env_type!r}")


def train(total_timesteps: int, n_envs: int = 8, num_players_range=(2, 6), out_dir: str = "models",
          checkpoint_every: int = 50_000, self_play_every: int = 0, self_play_pool_cap: int = 6,
          eval_every: int = 100_000, eval_hands: int = 500, seed: int = 0, tensorboard_log: str = None,
          resume_from: str = None, vec_env_type: str = "dummy", rule_based_n_sims: int = 150):
    """Train (or resume training) a MaskablePPO policy.

    `total_timesteps` is always the absolute target for `model.num_timesteps`
    -- when resuming, training runs until the real step count (not the
    number of *additional* steps) reaches it, so re-running with the same
    `--timesteps` after an interruption just picks up where it left off.

    `self_play_pool_cap` bounds how many frozen self-play snapshots can
    accumulate in the opponent pool (oldest evicted first): with no cap, a
    long run keeps adding snapshots forever, and since every snapshot opponent
    runs full NN inference (unlike the cheap random/rule-based agents), an
    ever-growing pool would steadily and unboundedly slow down training.
    """
    os.makedirs(out_dir, exist_ok=True)
    self_play_manifest_path = os.path.join(out_dir, "self_play_manifest.json")
    self_play_dir = os.path.join(out_dir, "self_play")
    active_snapshot_paths = []  # subproc mode only: main-process bookkeeping for the manifest/cap
    if vec_env_type == "subproc" and self_play_every:
        os.makedirs(self_play_dir, exist_ok=True)
        write_manifest([], self_play_manifest_path)  # self-play doesn't persist across restarts; start empty

    vec_env, opponent_pool = make_vec_env(n_envs, num_players_range, base_seed=seed, vec_env_type=vec_env_type,
                                          rule_based_n_sims=rule_based_n_sims,
                                          self_play_manifest_path=self_play_manifest_path)
    if resume_from:
        model = MaskablePPO.load(resume_from, env=vec_env)
        print(f"Resumed from {resume_from} at step {model.num_timesteps}")
    else:
        model = MaskablePPO("MlpPolicy", vec_env, verbose=1, seed=seed, tensorboard_log=tensorboard_log)

    # Resuming partway through: the next checkpoint/self-play/eval point is
    # the next multiple of its interval *after* the current step, not 0.
    def _next_multiple_after(step, interval):
        return (step // interval + 1) * interval if interval else None

    next_checkpoint = _next_multiple_after(model.num_timesteps, checkpoint_every)
    next_self_play = _next_multiple_after(model.num_timesteps, self_play_every)
    next_eval = _next_multiple_after(model.num_timesteps, eval_every)

    # Cap each learn() call at the *smallest* active interval so checkpoints/
    # self-play snapshots/evals normally land at meaningfully different
    # points in training, not all backdated onto the same end-of-chunk model
    # state. This is a best effort, not a guarantee: PPO only checks progress
    # between whole rollouts (n_steps * n_envs), so a single call can still
    # overshoot past more than one interval if an interval is set smaller
    # than one rollout -- see the while-loops below for why that stays correct.
    active_intervals = [x for x in (checkpoint_every, self_play_every, eval_every) if x]
    step_interval = min(active_intervals) if active_intervals else total_timesteps

    while model.num_timesteps < total_timesteps:
        # MaskablePPO.learn(total_timesteps=N, reset_num_timesteps=False)
        # runs N *more* steps from model.num_timesteps -- but only in whole
        # rollouts of n_steps * n_envs, overshooting whatever N asks for.
        # Track progress off model.num_timesteps (the real counter), not a
        # separately incremented one, or this loop keeps requesting chunks
        # long after the real step count has already passed total_timesteps.
        chunk = min(step_interval, total_timesteps - model.num_timesteps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done_steps = model.num_timesteps

        # Each action below fires at most once per learn() call (there's
        # only one current model state to checkpoint/snapshot/evaluate,
        # however many interval boundaries got crossed) but the `while`
        # advances the pointer past *every* crossed boundary -- otherwise,
        # if a single call jumps past more than one interval, the pointer
        # would only move past one and immediately re-fire next iteration.
        if checkpoint_every and done_steps >= next_checkpoint:
            path = os.path.join(out_dir, f"checkpoint_{done_steps}")
            model.save(path)
            print(f"[{done_steps}] checkpoint saved -> {path}.zip")
            while done_steps >= next_checkpoint:
                next_checkpoint += checkpoint_every

        if self_play_every and done_steps >= next_self_play:
            if vec_env_type == "dummy":
                # Mutate opponent_pool in place (pop/append), never rebind
                # it: every sub-environment holds a reference to this exact
                # list object, and reassigning it here wouldn't reach them.
                if self_play_pool_cap:
                    snapshot_indices = [i for i, a in enumerate(opponent_pool) if isinstance(a, _LiveModelAgent)]
                    while len(snapshot_indices) >= self_play_pool_cap:
                        opponent_pool.pop(snapshot_indices.pop(0))
                opponent_pool.append(_LiveModelAgent(model))
                pool_size = len(opponent_pool)
            else:
                snapshot_path = os.path.join(self_play_dir, f"step_{done_steps}.zip")
                model.save(snapshot_path)  # completes fully before the manifest can reference it
                active_snapshot_paths.append(snapshot_path + ".zip")
                if self_play_pool_cap:
                    active_snapshot_paths = active_snapshot_paths[-self_play_pool_cap:]
                write_manifest(active_snapshot_paths, self_play_manifest_path)
                pool_size = len(active_snapshot_paths)
            print(f"[{done_steps}] added self-play snapshot to opponent pool (pool size={pool_size})")
            while done_steps >= next_self_play:
                next_self_play += self_play_every

        if eval_every and done_steps >= next_eval:
            stats = evaluate_agent(_LiveModelAgent(model, deterministic=True), n_hands=eval_hands,
                                    num_players_range=num_players_range, seed=seed)
            print(f"[{done_steps}] eval vs baseline pool: {stats}")
            while done_steps >= next_eval:
                next_eval += eval_every

    final_path = os.path.join(out_dir, "final_model")
    model.save(final_path)
    print(f"Training complete. Final model saved -> {final_path}.zip")
    return model


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--self-play-every", type=int, default=0,
                        help="Add a frozen self-play snapshot to the opponent pool every N steps (0=off)")
    parser.add_argument("--self-play-pool-cap", type=int, default=6,
                        help="Max self-play snapshots kept in the pool at once (oldest evicted first, 0=unbounded)")
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-hands", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from a saved checkpoint .zip instead of training from scratch")
    parser.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"],
                        help="'subproc' uses real multi-core parallelism but self-play goes through disk")
    parser.add_argument("--opponent-n-sims", type=int, default=150,
                        help="Monte Carlo rollouts per decision for rule-based opponents (lower=faster/noisier)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(total_timesteps=args.timesteps, n_envs=args.n_envs,
          num_players_range=(args.min_players, args.max_players), out_dir=args.out_dir,
          checkpoint_every=args.checkpoint_every, self_play_every=args.self_play_every,
          self_play_pool_cap=args.self_play_pool_cap, eval_every=args.eval_every,
          eval_hands=args.eval_hands, seed=args.seed, tensorboard_log=args.tensorboard_log,
          resume_from=args.resume_from, vec_env_type=args.vec_env, rule_based_n_sims=args.opponent_n_sims)
