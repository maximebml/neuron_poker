"""No-limit Texas Hold'em engine for a single hand, 2-6 players.

`PokerEngine` is deliberately scoped to *one hand*: it is created (or
reset) with starting stacks and a button position, and runs until the
hand is over (someone wins the pot). Multi-hand sessions, bankroll
tracking across hands, and player elimination are the caller's
responsibility (see `env/poker_env.py`, which starts a fresh hand each
episode) -- keeping that out of the engine avoids a large amount of
incidental complexity while still supporting arbitrary stack depths,
button positions and player counts within a hand.

Betting is modelled with a per-seat `acted_this_round` / `capped` flag
pair so that the official no-limit rule "a raise that is all-in for
less than a full raise does not reopen the betting for players who
already acted" is implemented correctly, including side pots for
multi-way all-ins.
"""
from dataclasses import dataclass, field
from enum import IntEnum

from poker.actions import Action, RAISE_POT_FRACTIONS
from poker.cards import Deck
from poker.evaluator import rank_players


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


_NEW_CARDS_PER_STREET = {Street.FLOP: 3, Street.TURN: 1, Street.RIVER: 1}


@dataclass
class PlayerState:
    seat: int
    stack: float
    hole_cards: list = field(default_factory=list)
    bet_street: float = 0.0
    total_committed: float = 0.0
    folded: bool = False
    all_in: bool = False
    in_hand: bool = True
    acted_this_round: bool = False
    capped: bool = False  # facing a short all-in raise: may only fold/call

    @property
    def active(self) -> bool:
        """Dealt into this hand and hasn't folded (still eligible to win)."""
        return self.in_hand and not self.folded

    @property
    def can_act(self) -> bool:
        """Can still make a betting decision this hand."""
        return self.active and not self.all_in and self.stack > 0


@dataclass
class ActionRecord:
    seat: int
    street: Street
    action: Action
    amount: float


class PokerEngine:
    """Runs a single hand of no-limit hold'em."""

    def __init__(self, num_players: int, small_blind: float = 1.0, big_blind: float = 2.0, rng=None):
        if not 2 <= num_players <= 6:
            raise ValueError("num_players must be between 2 and 6")
        import random
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rng = rng or random.Random()

        self.players: list = []
        self.button = 0
        self.sb_seat = None
        self.bb_seat = None
        self.board: list = []
        self.deck = None
        self.street = Street.PREFLOP
        self.max_bet = 0.0
        self.min_raise = big_blind
        self.current_seat = None
        self.history: list = []
        self.done = False
        self.payouts = None
        self.showdown_ranking = None

    # ------------------------------------------------------------------ setup

    def start_hand(self, stacks: list, button: int):
        """Deal a brand-new hand.

        Args:
            stacks: starting stack for every seat 0..num_players-1. A seat
                with stack <= 0 sits out of this hand entirely.
            button: seat index the button should be on (rounded forward to
                the next seat with chips if that seat is sitting out).
        """
        if len(stacks) != self.num_players:
            raise ValueError("stacks must have length num_players")
        self.players = [PlayerState(seat=i, stack=float(s), in_hand=s > 0) for i, s in enumerate(stacks)]
        if sum(p.in_hand for p in self.players) < 2:
            raise ValueError("Need at least 2 players with chips to start a hand")

        self.button = self._adjust_to_dealt_in(button)
        self.board = []
        self.deck = Deck(self.rng)
        self.street = Street.PREFLOP
        self.history = []
        self.done = False
        self.payouts = None
        self.showdown_ranking = None

        for p in self.players:
            if p.in_hand:
                p.hole_cards = self.deck.deal(2)

        self._post_blinds()
        first_seat = self._next_dealt_in_seat(self.bb_seat)
        self.current_seat = self._find_next_actable(first_seat)
        if self.current_seat is None:
            self._advance()

    def _post_blinds(self):
        dealt_in = [p.seat for p in self.players if p.in_hand]
        if len(dealt_in) == 2:
            self.sb_seat = self.button
        else:
            self.sb_seat = self._next_dealt_in_seat(self.button)
        self.bb_seat = self._next_dealt_in_seat(self.sb_seat)

        self._contribute(self.sb_seat, min(self.small_blind, self.players[self.sb_seat].stack))
        self._contribute(self.bb_seat, min(self.big_blind, self.players[self.bb_seat].stack))
        self.max_bet = max(p.bet_street for p in self.players)
        self.min_raise = self.big_blind

    # -------------------------------------------------------------- seat math

    def _adjust_to_dealt_in(self, seat: int) -> int:
        seat %= self.num_players
        for _ in range(self.num_players):
            if self.players[seat].in_hand:
                return seat
            seat = (seat + 1) % self.num_players
        raise RuntimeError("No players with chips")

    def _next_dealt_in_seat(self, from_seat: int) -> int:
        seat = (from_seat + 1) % self.num_players
        for _ in range(self.num_players):
            if self.players[seat].in_hand:
                return seat
            seat = (seat + 1) % self.num_players
        raise RuntimeError("No dealt-in players")

    def _find_next_actable(self, start_seat: int):
        for i in range(self.num_players):
            s = (start_seat + i) % self.num_players
            p = self.players[s]
            if p.can_act and not p.acted_this_round:
                return s
        return None

    # ----------------------------------------------------------------- query

    def current_player(self):
        """Seat to act now, or None if the hand is over."""
        return None if self.done else self.current_seat

    def pot_total(self) -> float:
        """All chips committed to the pot so far this hand (any street)."""
        return sum(p.total_committed for p in self.players)

    def legal_actions(self, seat=None) -> dict:
        """Legal actions -> chip contribution for the given seat (default: current)."""
        seat = self.current_seat if seat is None else seat
        p = self.players[seat]
        if not p.can_act:
            return {}
        to_call = max(0.0, self.max_bet - p.bet_street)
        call_amount = min(to_call, p.stack)
        legal = {}

        if to_call > 1e-9:
            legal[Action.FOLD] = 0.0
        legal[Action.CHECK_CALL] = call_amount

        remaining_after_call = p.stack - call_amount
        if remaining_after_call > 1e-9 and not p.capped:
            pot_after_call = self.pot_total() + call_amount
            for act, frac in RAISE_POT_FRACTIONS.items():
                increment = max(frac * pot_after_call, self.min_raise)
                contribution = (self.max_bet + increment) - p.bet_street
                if contribution < p.stack - 1e-9:
                    legal[act] = contribution
            legal[Action.ALL_IN] = p.stack

        return legal

    # ------------------------------------------------------------------ step

    def step(self, action: Action):
        """Apply `action` for the current player and advance the hand."""
        if self.done:
            raise RuntimeError("Hand is already over")
        seat = self.current_seat
        p = self.players[seat]
        legal = self.legal_actions(seat)
        if action not in legal:
            raise ValueError(f"Illegal action {action!r} for seat {seat}; legal={list(legal)}")
        contribution = legal[action]

        if action == Action.FOLD:
            p.folded = True
        else:
            old_max_bet = self.max_bet
            self._contribute(seat, contribution)
            if p.bet_street > old_max_bet + 1e-9:
                increment = p.bet_street - old_max_bet
                self.max_bet = p.bet_street
                full_raise = increment >= self.min_raise - 1e-9
                if full_raise:
                    self.min_raise = increment
                self._reopen_action(raiser_seat=seat, full_raise=full_raise)

        p.acted_this_round = True
        p.capped = False
        self.history.append(ActionRecord(seat=seat, street=self.street, action=action, amount=contribution))
        self._advance()

    def _contribute(self, seat: int, amount: float):
        p = self.players[seat]
        amount = min(amount, p.stack)
        p.stack -= amount
        p.bet_street += amount
        p.total_committed += amount
        if p.stack <= 1e-9:
            p.stack = 0.0
            p.all_in = True

    def _reopen_action(self, raiser_seat: int, full_raise: bool):
        for p in self.players:
            if p.seat == raiser_seat or not p.can_act:
                continue
            if p.bet_street < self.max_bet - 1e-9:
                if full_raise:
                    p.acted_this_round = False
                    p.capped = False
                elif p.acted_this_round:
                    # short all-in: only players who already acted must
                    # respond again, and only with fold/call (no re-raise)
                    p.acted_this_round = False
                    p.capped = True

    # --------------------------------------------------------------- advance

    def _advance(self):
        active_players = [p for p in self.players if p.active]
        if len(active_players) == 1:
            self._finish_uncontested(active_players[0].seat)
            return

        can_act_players = [p for p in active_players if not p.all_in]
        round_complete = all(p.acted_this_round for p in can_act_players)

        if round_complete:
            if len(can_act_players) <= 1:
                self._run_out_and_showdown()
            elif self.street == Street.RIVER:
                self._go_to_showdown()
            else:
                self._advance_street()
        else:
            self.current_seat = self._find_next_actable((self.current_seat + 1) % self.num_players)

    def _advance_street(self):
        self.street = Street(self.street + 1)
        self.board += self.deck.deal(_NEW_CARDS_PER_STREET[self.street])
        for p in self.players:
            p.bet_street = 0.0
            p.acted_this_round = False
            p.capped = False
        self.max_bet = 0.0
        self.min_raise = self.big_blind

        first_seat = self._next_dealt_in_seat(self.button)
        self.current_seat = self._find_next_actable(first_seat)
        if self.current_seat is None:
            self._advance()

    def _run_out_and_showdown(self):
        while self.street < Street.RIVER:
            street = Street(self.street + 1)
            self.board += self.deck.deal(_NEW_CARDS_PER_STREET[street])
            self.street = street
        self._go_to_showdown()

    def _go_to_showdown(self):
        self.street = Street.SHOWDOWN
        self._resolve_showdown()

    # --------------------------------------------------------------- payouts

    def _finish_uncontested(self, winner_seat: int):
        net = {p.seat: -p.total_committed for p in self.players}
        net[winner_seat] += self.pot_total()
        self._finalize(net, showdown_ranking=None)

    def _resolve_showdown(self):
        active = [p for p in self.players if p.active]
        ranking = rank_players([(p.seat, p.hole_cards) for p in active], self.board)
        best_score_by_seat = dict(ranking)

        net = {p.seat: -p.total_committed for p in self.players}
        levels = sorted({p.total_committed for p in self.players if p.total_committed > 1e-9})

        prev_level = 0.0
        for level in levels:
            layer = level - prev_level
            contributors = [p for p in self.players if p.total_committed >= level - 1e-9]
            pot_layer = layer * len(contributors)
            prev_level = level
            if pot_layer <= 1e-9:
                continue

            eligible = [p.seat for p in contributors if p.active]
            if not eligible:
                # Every contributor at this level folded (only possible if
                # nobody left in the hand ever matched it) - refund pro-rata.
                for p in contributors:
                    net[p.seat] += layer
                continue

            best_score = min(best_score_by_seat[s] for s in eligible)
            winners = [s for s in eligible if best_score_by_seat[s] == best_score]
            share = pot_layer / len(winners)
            for w in winners:
                net[w] += share

        self.showdown_ranking = ranking
        self._finalize(net, showdown_ranking=ranking)

    def _finalize(self, net_payouts: dict, showdown_ranking):
        for p in self.players:
            winnings = net_payouts[p.seat] + p.total_committed
            p.stack += winnings
        self.payouts = net_payouts
        self.showdown_ranking = showdown_ranking
        self.done = True
        self.current_seat = None
