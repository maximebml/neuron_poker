"""ProAgent tuned by `training/optimize_pro_agent_v2.py` -- see that module's
docstring for how the search works (evolves the six highest-leverage
strategy parameters mostly against stock ProAgent, with a smaller weight on
a few weaker fixed opponents, both signals z-score standardized before
blending).

TUNED_PARAMS below is the `best_params.json` produced by running that
search with its default settings (8 generations, population 8, seeded
2026-08-21). Baked in here as a plain dict rather than loaded from
`models/` at runtime, since `models/` is gitignored and this agent should
be usable from a fresh checkout without re-running the search. Every
ProAgent parameter not in TUNED_PARAMS stays at PARAM_DEFAULTS.
"""
import random

from agents.pro_agent import ProAgent

TUNED_PARAMS = {
    "open_intercept": 11.713112623743285,
    "open_position_coef": 4.509808143561855,
    "facing_raise_call_intercept": 6.053156465619619,
    "facing_raise_raise_intercept": 14.0,
    "cbet_freq": 0.30602503369791745,
    "bluffcatch_extra_margin": 0.009662970632652888,
}


class ProAgentV2(ProAgent):
    """ProAgent with TUNED_PARAMS instead of PARAM_DEFAULTS. See module docstring."""

    def __init__(self, rng: random.Random = None, equity_sims: int = 200):
        super().__init__(rng=rng, equity_sims=equity_sims, name="pro_v2", **TUNED_PARAMS)
