"""A hand-crafted, position- and texture-aware rule-based agent.

Plays a solid, tight-aggressive style closer to how a strategy-literate
human professional approaches no-limit hold'em -- in contrast to
`RuleBasedAgent`'s single "Monte Carlo equity vs. an aggression
threshold" heuristic. It combines several well-known, explicit rules
rather than one number:

- Preflop: the Chen formula (a fast, standard hand-strength estimate --
  no Monte Carlo needed) for position-scaled opening/calling/3-betting
  ranges, plus a short-stack push/fold mode.
- Postflop: made-hand strength (top pair vs. weak pair vs. two pair+ vs.
  sets/better) and draw detection (flush draws, open-ended straight
  draws, gutshots) from the actual hole cards and board, combined into
  one strength tier.
- Continuation betting as the preflop raiser, board-texture-scaled bet
  sizing, pot-odds-driven continuing decisions that tighten as more
  opponents are in the pot, and stack-to-pot-ratio commitment (getting
  it in rather than min-raising when short relative to the pot).

This is *not* a solver: there's no search, no real opponent modeling
beyond what a single hand's own action history shows, and no exact
game-theoretic betting frequencies -- just structured heuristics, with
a documented, deliberate scope (e.g. kicker strength beyond "paired the
top board card or not" isn't tracked; board run-outs where the board
itself is already paired/trips get coarser treatment since attributing
hand strength correctly there requires real card-by-card hand reading).
"""
import random
from collections import Counter
from enum import IntEnum

from agents.base_agent import Agent
from poker.actions import Action
from poker.cards import RANK_TO_INT, RANKS, rank_of, suit_of
from poker.engine import Street
from poker.evaluator import estimate_equity, hand_class_name

_RAISE_PREFERENCE = (Action.RAISE_75, Action.RAISE_33, Action.RAISE_150, Action.ALL_IN)
_STRONG_HAND_CLASSES = {"Straight Flush", "Four of a Kind", "Full House", "Flush", "Straight",
                        "Three of a Kind"}
_CHEN_HIGH_CARD = {"A": 10, "K": 8, "Q": 7, "J": 6, "T": 5}
_CHEN_GAP_PENALTY = {0: 0, 1: 1, 2: 2, 3: 4}


class Tier(IntEnum):
    """Coarse postflop hand-strength buckets, weakest to strongest."""

    AIR = 0
    WEAK_DRAW = 1       # gutshot only
    WEAK_MADE = 2        # bottom/middle pair, or a pair that's mostly "the board's"
    STRONG_DRAW = 3      # flush draw or open-ended straight draw
    MEDIUM_MADE = 4      # top pair, overpair, or two pair
    STRONG_MADE = 5      # trips or better


def chen_score(hole_cards: list) -> float:
    """The Chen formula: a fast, standard preflop hand-strength estimate.

    Deterministic and requires no Monte Carlo rollout, so every preflop
    decision stays cheap no matter how many hands are played.
    """
    r1, r2 = hole_cards[0][0], hole_cards[1][0]
    suited = hole_cards[0][1] == hole_cards[1][1]
    high, low = sorted((RANK_TO_INT[r1], RANK_TO_INT[r2]), reverse=True)

    def high_card_value(rank_idx):
        rank_char = RANKS[rank_idx]
        return _CHEN_HIGH_CARD.get(rank_char, (rank_idx + 2) / 2)

    if r1 == r2:
        score = max(high_card_value(high) * 2, 5.0)
    else:
        score = high_card_value(high)
        if suited:
            score += 2
        gap = high - low - 1
        score -= _CHEN_GAP_PENALTY.get(gap, 5)
        if gap <= 1 and high <= RANK_TO_INT["Q"]:
            score += 1  # extra straight potential with both cards Q or lower

    return max(0.0, round(score * 2) / 2)  # round to the nearest 0.5


def position_quality(seat: int, button: int, num_players: int) -> float:
    """1.0 = button (best postflop position), 0.0 = small blind (worst).

    Postflop action always starts just after the button and wraps around
    to the button last (true heads-up too -- the engine enforces this),
    so "how many players act after me" is a position-agnostic way to
    rank seats that works the same for 2 to 6 players.
    """
    if num_players <= 1:
        return 1.0
    offset = (seat - button) % num_players  # 0 = button
    order_position = offset if offset != 0 else num_players
    players_after_me = num_players - order_position
    return 1.0 - players_after_me / (num_players - 1)


def _draw_features(hole: list, board: list):
    """(has_flush_draw, has_oesd, has_gutshot). Only meaningful on flop/turn --
    there's nothing left to draw to once the river card is out."""
    if not 3 <= len(board) <= 4:
        return False, False, False

    all_cards = hole + board
    suits = [suit_of(c) for c in all_cards]
    hole_suits = {suit_of(c) for c in hole}
    suit_counts = Counter(suits)
    flush_draw = any(count == 4 and s in hole_suits for s, count in suit_counts.items())

    ranks = {rank_of(c) for c in all_cards}
    hole_ranks = {rank_of(c) for c in hole}
    ace_idx = RANK_TO_INT["A"]
    if ace_idx in ranks:
        ranks = ranks | {-1}  # ace also plays low, for the wheel (A-2-3-4-5)
    if ace_idx in hole_ranks:
        hole_ranks = hole_ranks | {-1}

    oesd = gutshot = False
    for start in range(-1, RANK_TO_INT["A"]):  # every 5-wide window from A-5 up to T-A
        window = set(range(start, start + 5))
        present = window & ranks
        missing = window - present
        if len(present) == 4 and len(missing) == 1 and (hole_ranks & present):
            missing_rank = next(iter(missing))
            if missing_rank in (start, start + 4):
                oesd = True
            else:
                gutshot = True

    return flush_draw, oesd, gutshot


def _pair_tier(hole: list, board: list) -> Tier:
    board_ranks = [rank_of(c) for c in board]
    top_board_rank = max(board_ranks)
    board_has_pair = any(count >= 2 for count in Counter(board_ranks).values())
    hole_ranks = [rank_of(c) for c in hole]

    if board_has_pair:
        return Tier.WEAK_MADE  # the pair is mostly (or entirely) the board's, not hero's
    if hole_ranks[0] == hole_ranks[1]:
        return Tier.MEDIUM_MADE if hole_ranks[0] >= top_board_rank else Tier.WEAK_MADE
    matched = [r for r in hole_ranks if r in board_ranks]
    if not matched:
        return Tier.WEAK_MADE  # defensive: shouldn't happen if the class really is "Pair"
    return Tier.MEDIUM_MADE if matched[0] >= top_board_rank else Tier.WEAK_MADE


def postflop_tier(hole: list, board: list) -> Tier:
    """Combine made-hand strength and draw potential into one tier."""
    class_name = hand_class_name(hole, board)
    if class_name in _STRONG_HAND_CLASSES:
        made_tier = Tier.STRONG_MADE
    elif class_name == "Two Pair":
        made_tier = Tier.MEDIUM_MADE
    elif class_name == "Pair":
        made_tier = _pair_tier(hole, board)
    else:
        made_tier = Tier.AIR

    flush_draw, oesd, gutshot = _draw_features(hole, board)
    draw_tier = Tier.STRONG_DRAW if (flush_draw or oesd) else (Tier.WEAK_DRAW if gutshot else Tier.AIR)

    return max(made_tier, draw_tier)


def board_texture(board: list):
    """(wetness in [0, 1], is_paired). Higher wetness = more flush/straight potential."""
    if len(board) < 3:
        return 0.0, False
    ranks = [rank_of(c) for c in board]
    suits = [suit_of(c) for c in board]
    paired = len(set(ranks)) < len(ranks)
    wetness = 0.0
    if max(Counter(suits).values()) >= 3:
        wetness += 0.5
    if max(ranks) - min(ranks) <= 4:
        wetness += 0.5
    return wetness, paired


class ProAgent(Agent):
    """Rule-based agent playing a solid, pro-influenced TAG style. See module docstring."""

    def __init__(self, rng: random.Random = None, equity_sims: int = 200, name: str = "pro"):
        self.rng = rng or random.Random()
        self.equity_sims = equity_sims
        self.name = name

    def act(self, engine, seat: int) -> Action:
        legal = engine.legal_actions(seat)
        action = (self._preflop_action(engine, seat, legal) if engine.street == Street.PREFLOP
                  else self._postflop_action(engine, seat, legal))
        return action if action in legal else self._fallback(legal)

    @staticmethod
    def _fallback(legal: dict) -> Action:
        return Action.CHECK_CALL if Action.CHECK_CALL in legal else next(iter(legal))

    @staticmethod
    def _best_raise(legal: dict):
        return next((act for act in _RAISE_PREFERENCE if act in legal), None)

    # ------------------------------------------------------------- preflop

    def _preflop_action(self, engine, seat, legal):
        player = engine.players[seat]
        chen = chen_score(player.hole_cards)
        pos = position_quality(seat, engine.button, engine.num_players)
        effective_bb = player.stack / engine.big_blind
        facing_bet = Action.FOLD in legal
        raises_this_street = sum(
            1 for h in engine.history
            if h.street == Street.PREFLOP and h.action not in (Action.FOLD, Action.CHECK_CALL))

        if effective_bb <= 15:
            # Short-stacked: get it in or fold rather than min-raise into an
            # unworkable stack-to-pot ratio.
            push_threshold = 10.0 - pos * 3.5
            if chen >= push_threshold:
                return Action.ALL_IN if Action.ALL_IN in legal else (self._best_raise(legal) or Action.CHECK_CALL)
            return Action.FOLD if facing_bet else Action.CHECK_CALL

        if not facing_bet:
            open_threshold = 9.0 - pos * 3.0  # ~9 from early position down to ~6 on the button
            if chen >= open_threshold:
                return Action.RAISE_75 if Action.RAISE_75 in legal else (self._best_raise(legal) or Action.CHECK_CALL)
            return Action.CHECK_CALL

        if raises_this_street >= 2:
            if chen >= 10.5:
                return self._best_raise(legal) or Action.CHECK_CALL
            return Action.CHECK_CALL if chen >= 9.0 else Action.FOLD

        # Facing a single open.
        if chen >= 10.0 - pos:
            raise_action = Action.RAISE_150 if Action.RAISE_150 in legal else self._best_raise(legal)
            if raise_action:
                return raise_action
        if chen >= 8.0 - pos * 1.5:
            return Action.CHECK_CALL
        return Action.FOLD

    # ------------------------------------------------------------ postflop

    def _postflop_action(self, engine, seat, legal):
        player = engine.players[seat]
        tier = postflop_tier(player.hole_cards, engine.board)
        num_opponents = sum(1 for p in engine.players if p.active and p.seat != seat)

        if Action.FOLD in legal:
            return self._facing_bet(engine, seat, legal, tier, num_opponents)
        wetness, _ = board_texture(engine.board)
        was_aggressor = self._was_preflop_aggressor(engine, seat)
        return self._can_check(engine, legal, tier, wetness, num_opponents, was_aggressor)

    @staticmethod
    def _was_preflop_aggressor(engine, seat) -> bool:
        preflop_raises = [h for h in engine.history if h.street == Street.PREFLOP
                          and h.action not in (Action.FOLD, Action.CHECK_CALL)]
        return bool(preflop_raises) and preflop_raises[-1].seat == seat

    def _can_check(self, engine, legal, tier, wetness, num_opponents, was_aggressor):
        big_size = Action.RAISE_150 if wetness > 0.5 else Action.RAISE_75

        if tier == Tier.STRONG_MADE:
            return (big_size if big_size in legal else self._best_raise(legal)) or Action.CHECK_CALL
        if tier == Tier.MEDIUM_MADE and num_opponents <= 2 and Action.RAISE_33 in legal:
            return Action.RAISE_33
        if tier == Tier.STRONG_DRAW and Action.RAISE_33 in legal and self.rng.random() < 0.6:
            return Action.RAISE_33  # semi-bluff
        if (was_aggressor and engine.street == Street.FLOP and num_opponents <= 2
                and Action.RAISE_33 in legal and self.rng.random() < 0.65):
            return Action.RAISE_33  # standard continuation bet
        if tier == Tier.AIR and num_opponents <= 1 and Action.RAISE_33 in legal and self.rng.random() < 0.15:
            return Action.RAISE_33  # rare, heads-up-only bluff
        return Action.CHECK_CALL

    def _facing_bet(self, engine, seat, legal, tier, num_opponents):
        player = engine.players[seat]
        call_amount = legal[Action.CHECK_CALL]
        pot_after_call = engine.pot_total() + call_amount
        pot_odds = call_amount / pot_after_call if pot_after_call > 0 else 0.0
        spr = player.stack / max(engine.pot_total(), 1e-6)
        multiway_margin = 0.03 * num_opponents

        if tier == Tier.STRONG_MADE or (tier == Tier.MEDIUM_MADE and spr <= 1.5):
            if Action.ALL_IN in legal and spr <= 2.5:
                return Action.ALL_IN
            return self._best_raise(legal) or Action.CHECK_CALL

        equity = estimate_equity(player.hole_cards, engine.board, num_opponents,
                                 n_sims=self.equity_sims, rng=self.rng)

        if tier == Tier.STRONG_DRAW:
            if equity > pot_odds + multiway_margin and self.rng.random() < 0.35:
                raise_action = self._best_raise(legal)
                if raise_action:
                    return raise_action
            return Action.CHECK_CALL if equity > pot_odds else Action.FOLD

        if tier in (Tier.MEDIUM_MADE, Tier.WEAK_DRAW):
            return Action.CHECK_CALL if equity > pot_odds + multiway_margin else Action.FOLD

        # Weak made hand or air: only a clean-odds bluff-catch.
        return Action.CHECK_CALL if equity > pot_odds + 0.10 + multiway_margin else Action.FOLD
