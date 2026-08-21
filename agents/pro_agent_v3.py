"""ProAgent tuned by `training/optimize_pro_agent_v3.py` -- see that module's
docstring for how the chain works (a sequence of evolutionary search rounds
seeded near the current champion, each round's winner promoted only if it
beats the champion head-to-head on a held-out validation sample, so the
chain can only improve or stay flat, never silently regress).

TUNED_PARAMS below is the `best_params.json` produced by running that chain
with its default settings (10 iterations, 5 generations/iteration, seeded
2026-08-21, starting from ProAgentV2's TUNED_PARAMS as the initial
champion). Only 2 of the 10 rounds actually promoted a challenger
(iterations 7 and 8); the rest were no-ops that correctly kept the previous
champion. Baked in here as a plain dict rather than loaded from `models/`
at runtime, since `models/` is gitignored and this agent should be usable
from a fresh checkout without re-running the chain. Every ProAgent
parameter not in TUNED_PARAMS stays at PARAM_DEFAULTS.
"""
import random

from agents.pro_agent import ProAgent

TUNED_PARAMS = {
    "open_intercept": 11.71163463777752,
    "open_position_coef": 2.6470746508842007,
    "facing_raise_call_intercept": 4.659268509591221,
    "facing_raise_raise_intercept": 10.757821680091732,
    "cbet_freq": 0.7228920307098777,
    "bluffcatch_extra_margin": 0.1411922737399857,
}


class ProAgentV3(ProAgent):
    """ProAgent with TUNED_PARAMS instead of PARAM_DEFAULTS. See module docstring."""

    def __init__(self, rng: random.Random = None, equity_sims: int = 200):
        super().__init__(rng=rng, equity_sims=equity_sims, name="pro_v3", **TUNED_PARAMS)
