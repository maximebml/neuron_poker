"""Fixed-size observation encoding, independent of table size (2-6 players).

Every hand is encoded from the acting player's ("hero") point of view:
seat slots are ordered starting at hero and walking clockwise, so the
network sees "me, then the next player to act after me, etc." rather
than raw absolute seat numbers -- the policy should generalise across
seat counts and button positions instead of memorising them.
"""
import numpy as np

from poker.actions import N_ACTIONS
from poker.cards import RANK_TO_INT, SUIT_TO_INT

MAX_PLAYERS = 6
CARD_DIM = len(RANK_TO_INT) + len(SUIT_TO_INT)  # 13 + 4 = 17
HERO_CARD_SLOTS = 2
BOARD_CARD_SLOTS = 5
SEAT_FEATURES = 7   # occupied, in_hand, folded, all_in, stack_bb, bet_bb, is_button
SCALAR_FEATURES = 7  # pot, to_call, hero_stack, hero_bet, min_raise, num_active, num_can_act
STREET_FEATURES = 4  # preflop/flop/turn/river one-hot

OBSERVATION_SIZE = (
    HERO_CARD_SLOTS * CARD_DIM
    + BOARD_CARD_SLOTS * CARD_DIM
    + SCALAR_FEATURES
    + STREET_FEATURES
    + MAX_PLAYERS * SEAT_FEATURES
)


def _encode_card(card: str) -> list:
    vec = [0.0] * CARD_DIM
    vec[RANK_TO_INT[card[0]]] = 1.0
    vec[len(RANK_TO_INT) + SUIT_TO_INT[card[1]]] = 1.0
    return vec


def _encode_cards(cards: list, n_slots: int) -> list:
    out = []
    for i in range(n_slots):
        out += _encode_card(cards[i]) if i < len(cards) else [0.0] * CARD_DIM
    return out


def build_observation(engine, hero_seat: int) -> np.ndarray:
    """Encode `engine`'s current state from `hero_seat`'s perspective."""
    bb = engine.big_blind
    hero = engine.players[hero_seat]

    features = []
    features += _encode_cards(hero.hole_cards, HERO_CARD_SLOTS)
    features += _encode_cards(engine.board, BOARD_CARD_SLOTS)

    to_call = max(0.0, engine.max_bet - hero.bet_street)
    num_active = sum(1 for p in engine.players if p.active)
    num_can_act = sum(1 for p in engine.players if p.can_act)
    features += [
        engine.pot_total() / bb,
        to_call / bb,
        hero.stack / bb,
        hero.bet_street / bb,
        engine.min_raise / bb,
        num_active / MAX_PLAYERS,
        num_can_act / MAX_PLAYERS,
    ]

    street_onehot = [0.0] * STREET_FEATURES
    street_onehot[min(int(engine.street), STREET_FEATURES - 1)] = 1.0
    features += street_onehot

    n = engine.num_players
    seat_order = [(hero_seat + i) % n for i in range(n)]
    for slot in range(MAX_PLAYERS):
        if slot < n:
            p = engine.players[seat_order[slot]]
            features += [
                1.0,
                float(p.in_hand),
                float(p.folded),
                float(p.all_in),
                p.stack / bb,
                p.bet_street / bb,
                float(p.seat == engine.button),
            ]
        else:
            features += [0.0] * SEAT_FEATURES

    obs = np.array(features, dtype=np.float32)
    assert obs.shape == (OBSERVATION_SIZE,), f"expected {OBSERVATION_SIZE}, got {obs.shape}"
    return obs


def legal_action_mask(engine, seat: int) -> np.ndarray:
    """Boolean mask over `Action` for `seat`'s current legal moves."""
    legal = engine.legal_actions(seat)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for act in legal:
        mask[int(act)] = True
    return mask
