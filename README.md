# neuron_poker

A from-scratch, no-limit Texas Hold'em bot for 2-6 players, trained with
reinforcement learning. Give it any table situation and it returns a
decision.

This is a clean rebuild of the original `neuron_poker` project on a
modern stack: `gymnasium` (not the abandoned `gym`), `stable-baselines3`
+ `sb3-contrib` `MaskablePPO` (not `keras-rl2`/TF1-era Keras), and
`treys` for hand evaluation (no C++/Cython build step required).

## Layout

```
poker/        Core no-limit hold'em engine (2-6 players): betting rounds,
              blinds, min-raise rules, side pots, showdown evaluation.
env/          gymnasium.Env wrapping the engine for RL training, plus the
              fixed-size observation encoder and action-legality masking.
agents/       Pluggable policies sharing one `act(engine, seat)` interface:
              RandomAgent, RuleBasedAgent (single Monte-Carlo-equity
              heuristic), ProAgent (structured pro-style rule system --
              see below) plus its evolved variants ProAgentV2/ProAgentV3,
              RLAgent (loads a trained MaskablePPO checkpoint).
training/     Self-play training loop and a baseline-pool evaluator.
inference/    `decide.py` -- feed the bot an arbitrary situation, get a
              decision back.
tests/        pytest suite: engine correctness (side pots, all-ins, min-raise
              rules, chip conservation), evaluator, env, inference.
```

## Setup

```
pip install -r requirements.txt
pytest       # ~320 tests, a few seconds
```

## How a hand is modelled

`PokerEngine` runs exactly one hand at a time (dealt fresh with given
stacks/button), which keeps it simple while still supporting anything
that can happen *within* a hand: 2-6 players, arbitrary stack depths,
multi-way all-ins and side pots, and the official rule that an all-in
raise for less than a full raise does not reopen betting for players who
already acted. Multi-hand sessions (bankroll tracked hand to hand,
players busting out) are the caller's job -- the RL environment resets
to a fresh hand every episode, which is what lets it randomize table
size, stack depth and position on every episode instead of tracking a
whole session.

Betting is discretized into 6 actions so the network doesn't have to
learn continuous bet sizing from scratch: `FOLD`, `CHECK_CALL`, and
raises sized at 33%/75%/150% of the pot, plus `ALL_IN`. Raise sizes that
would leave the player fewer chips than a valid raise collapse into
`ALL_IN` automatically; illegal actions for the current player are
excluded via an action mask (`env.action_masks()`), which
`MaskablePPO` uses so the policy only ever samples legal moves.

## Training

```
python main.py train --timesteps 2000000 --n-envs 8 --out-dir models
```

The hero trains against a pool of `RandomAgent` and `RuleBasedAgent`
(Monte-Carlo-equity heuristic, several aggression levels) opponents,
with table size, stack depths, seat and button position all randomized
every hand. Pass `--self-play-every 200000` to also periodically freeze
the current policy and add it to the opponent pool, so training also has
to beat earlier versions of itself. Useful flags:

```
--min-players / --max-players   restrict the table sizes trained on (default 2-6)
--checkpoint-every N            save a checkpoint every N steps
--eval-every N / --eval-hands N  periodically report bb/100 vs. the baseline pool
```

2,000,000 timesteps on CPU is enough to move well past random play but
not to reach a strong bot -- this is a base you can scale up (more
timesteps, more parallel envs, a bigger network, self-play) rather than
a finished superhuman agent; poker-solving at that level takes orders of
magnitude more compute than fits in one training run here.

Evaluate a checkpoint's win rate against the baseline pool:

```
python main.py evaluate --model models/final_model.zip --hands 2000
```

Watch it (or the rule-based baseline, if you omit `--model`) play a few
hands with commentary:

```
python main.py play --model models/final_model.zip --num-players 6 --hands 10
```

Pass `--agent {rl,pro,rule_based,random}` to explicitly pick seat 0's
policy (`--model` alone implies `rl`); every hand also seats a mix of
`ProAgent`, `RuleBasedAgent`, and `RandomAgent` as opponents.

## The rule-based pro agent

`agents/pro_agent.py` (`ProAgent`) is a second, more structured
non-RL baseline, built to play closer to how a strategy-literate human
would rather than off a single equity number:

- **Preflop**: the Chen formula (a fast, standard hand-strength score --
  no Monte Carlo needed) drives position-scaled opening, calling, and
  3-betting ranges, plus a short-stack push/fold mode below 15bb.
- **Postflop**: classifies the hand into a strength tier from made-hand
  category (trips+, two pair, top pair vs. a weak kicker pair, etc.) and
  detected draws (flush draws, open-ended straights, gutshots), then
  acts on that tier -- continuation betting as the preflop raiser,
  board-texture-scaled bet sizing (bigger on wet/coordinated boards),
  pot-odds-driven continuing decisions that tighten as more opponents
  are in the pot, and getting stacks in rather than min-raising when
  short relative to the pot (low stack-to-pot ratio).

It is **not a solver** -- no search, no real opponent modeling beyond
what a single hand's own action history shows, and no exact
game-theoretic betting frequencies, just explicit, explainable rules.
Use it via `main.py play --agent pro`, the dashboard's "Pro (sophisticated
rules)" option, or directly as `ProAgent()` anywhere an `Agent` is
expected (e.g. `inference.decide.decide(situation, agent=ProAgent())`).

### Evolved variants: ProAgentV2 / ProAgentV3

`training/optimize_pro_agent_v2.py` and `training/optimize_pro_agent_v3.py`
evolve ProAgent's six highest-leverage strategy parameters (preflop
opening, facing-a-raise, c-bet frequency, bluff-catching margin) against
stock ProAgent and a mix of weaker fixed opponents -- v3 chains several
rounds of that search, validating each round's winner head-to-head before
promoting it, so the result can only improve or stay flat.

`agents/pro_agent_v2.py` (`ProAgentV2`) and `agents/pro_agent_v3.py`
(`ProAgentV3`) are ProAgent subclasses with one such search's output baked
in as a plain params dict (not loaded from `models/` at runtime, since that
directory is gitignored) -- so they're ready to use immediately, no need to
re-run the optimizer. Use them exactly like `ProAgent`: `main.py evaluate
--agent pro_v2`, `main.py play --agent pro_v3`, the dashboard's "Pro v2
(evolved)" / "Pro v3 (evolved, chained)" options, or directly as
`ProAgentV2()` / `ProAgentV3()`. Re-running the optimizer scripts and
copying their `best_params.json` output into these files' `TUNED_PARAMS`
is how you'd refresh them with a new search.

## Asking it for a decision

`inference/decide.py` takes a plain description of a table state -- it
doesn't have to be a hand the engine actually played, any legal-looking
situation works:

```
python main.py decide --model models/final_model.zip --situation situation.json
python main.py decide --model models/final_model.zip --interactive
```

`situation.json`:

```json
{
  "num_players": 6,
  "big_blind": 2,
  "small_blind": 1,
  "button_seat": 3,
  "street": "flop",
  "board": ["Qs", "Jd", "2c"],
  "pot_before_street": 60,
  "hero_seat": 0,
  "hero_cards": ["Ah", "Kd"],
  "players": [
    {"seat": 0, "stack": 860, "bet": 0},
    {"seat": 1, "stack": 500, "bet": 40},
    {"seat": 2, "stack": 0, "folded": true},
    {"seat": 3, "stack": 640, "bet": 0},
    {"seat": 4, "stack": 300, "bet": 0, "folded": true},
    {"seat": 5, "stack": 900, "bet": 40}
  ]
}
```

`pot_before_street` is whatever was already in the pot before this
street's betting started (0 preflop); each player's `bet` is what
they've put in *this street* only. Output:

```json
{
  "action": "RAISE_75",
  "amount": 175.0,
  "action_probabilities": {"FOLD": 0.02, "CHECK_CALL": 0.10, "...": "..."},
  "legal_actions": {"FOLD": 0.0, "CHECK_CALL": 40.0, "...": "..."},
  "pot_before_action": 140.0,
  "hero_equity_vs_live_opponents": 0.34,
  "used_model": "models/final_model.zip"
}
```

`--model` is optional -- without it, `decide` falls back to the
rule-based equity heuristic, so the tool is still useful before you've
trained anything.

## Local dashboard

A local Streamlit dashboard lets you pick an agent and build a table
situation with widgets instead of hand-writing JSON:

```
streamlit run dashboard/app.py
```

In the sidebar, choose the agent (a trained RL checkpoint -- auto-discovered
from `models/`, the sophisticated pro-style rule agent or its evolved
ProAgentV2/ProAgentV3 variants, the simpler rule-based heuristic at any
aggression, or random), then in the main panel set the
number of players, blinds, button, street and
board, each seat's stack/current bet/folded/all-in state, and the hero's
seat and hole cards. "Get decision" runs the same `inference.decide.decide`
pipeline the CLI uses and shows the action, sizing, legal actions, hand
equity, and (for the RL agent) the full action-probability breakdown. An
expander at the bottom shows the equivalent situation JSON, so anything you
build in the dashboard can be replayed with `main.py decide --situation`.

A second page, "Play vs ProAgent" (in the sidebar page picker), lets you
actually play live hands instead of querying single decisions: you act as
the hero via buttons, opponent seats (ProAgent by default, or any mix of
rule-based/random/RL agents) act automatically between your turns, and
hands run end-to-end through the real engine -- multi-street betting,
showdown reveals, and a running session bb chart. Any seat that busts is
reloaded to its starting stack before the next hand, so it's a practice
table, not a tournament.

## Design notes / known simplifications

- Bet sizes are continuous chip amounts (not integer chips); this is
  fine for training and for the numbers `decide.py` reports, but real
  chip stacks would need rounding.
- The observation is table-size-invariant: seats are encoded relative
  to the hero (padding empty seat slots for tables under 6 players), so
  the same network handles 2-6 players.
- Side-pot resolution follows the standard "contribution level" algorithm
  and is covered by dedicated multi-way-all-in tests.
