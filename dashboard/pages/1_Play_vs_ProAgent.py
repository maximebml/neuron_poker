"""Play live hands against ProAgent (or a mixed table of bots).

Run with:
    streamlit run dashboard/app.py
then open the "Play vs ProAgent" page from the sidebar.

Unlike the main decision-tool page (which answers "what would the bot do
in this one spot?"), this page runs real hands end-to-end through
`PokerEngine`: you act as the hero via buttons, bot seats act
automatically between your turns, and the hand plays out to showdown or
an early fold -- same engine, same discrete pot-fraction action space
the bots and the RL training environment use.
"""
import glob
import os
import random
import sys

# Streamlit's multipage mechanism doesn't add the repo root to sys.path for
# scripts under pages/ the way it does for the entry script (dashboard/app.py),
# so top-level imports like "agents.pro_agent" fail without this.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from agents.pro_agent import ProAgent
from agents.pro_agent_v2 import ProAgentV2
from agents.pro_agent_v3 import ProAgentV3
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from poker.actions import Action
from poker.engine import PokerEngine, Street
from poker.evaluator import estimate_equity

st.set_page_config(page_title="neuron_poker -- play", layout="wide")
st.title("neuron_poker -- play a hand")
st.caption("You're the hero (seat 0). Opponent seats act automatically between your turns. "
          "Any seat that busts is reloaded to the starting stack before the next hand -- "
          "this is a practice table, not a tournament.")

AGENT_CHOICES = ["Pro (sophisticated rules)", "Pro v2 (evolved)", "Pro v3 (evolved, chained)",
                "Rule-based (equity heuristic)", "Random", "Trained RL model"]
ACTION_LABELS = {
    Action.FOLD: "Fold", Action.CHECK_CALL: "Check/Call", Action.RAISE_33: "Raise 33% pot",
    Action.RAISE_75: "Raise 75% pot", Action.RAISE_150: "Raise 150% pot", Action.ALL_IN: "All-in",
}
SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


def format_card(card: str) -> str:
    return card[0] + SUIT_SYMBOLS[card[1]]


def format_cards(cards: list) -> str:
    return " ".join(format_card(c) for c in cards) if cards else "--"


def describe_action(action: Action, amount: float) -> str:
    if action == Action.FOLD:
        return "folds"
    if action == Action.CHECK_CALL:
        return "checks" if amount <= 1e-9 else f"calls ({amount:.1f})"
    if action == Action.ALL_IN:
        return f"goes all-in ({amount:.1f})"
    return f"{ACTION_LABELS[action].lower()} (+{amount:.1f})"


# ----------------------------------------------------------------- sidebar

st.sidebar.header("Table setup")
num_players = st.sidebar.slider("Players (you + opponents)", 2, 6, 2)
starting_stack_bb = st.sidebar.number_input("Starting stack (big blinds)", min_value=10.0, value=100.0, step=10.0)
big_blind = st.sidebar.number_input("Big blind", min_value=0.1, value=2.0)
small_blind = st.sidebar.number_input("Small blind", min_value=0.0, value=1.0, max_value=big_blind)
show_equity = st.sidebar.checkbox("Show my equity vs live opponents", value=True)

st.sidebar.subheader("Opponents")
opponent_cfgs = []
for i in range(num_players - 1):
    st.sidebar.markdown(f"**Seat {i + 1}**")
    kind = st.sidebar.selectbox("Agent", AGENT_CHOICES, index=0, key=f"opp_kind_{i}", label_visibility="collapsed")
    cfg = {"kind": kind}
    if kind == "Rule-based (equity heuristic)":
        cfg["aggression"] = st.sidebar.slider("Aggression", 0.0, 1.0, 0.5, 0.05, key=f"opp_agg_{i}")
    elif kind == "Trained RL model":
        checkpoints = sorted(glob.glob(os.path.join("models", "*.zip")), key=os.path.getmtime)
        if checkpoints:
            cfg["checkpoint"] = st.sidebar.selectbox("Checkpoint", checkpoints[::-1], key=f"opp_ckpt_{i}")
        else:
            st.sidebar.warning("No .zip checkpoints found in models/.")
            cfg["checkpoint"] = None
    opponent_cfgs.append(cfg)

st.sidebar.caption("Opponent/table changes apply next time you click Start/restart below.")
restart_clicked = st.sidebar.button("Start / restart session", type="primary", use_container_width=True)


def build_agent(cfg):
    kind = cfg["kind"]
    if kind == "Random":
        return RandomAgent()
    if kind == "Rule-based (equity heuristic)":
        return RuleBasedAgent(aggression=cfg["aggression"])
    if kind == "Trained RL model":
        if not cfg.get("checkpoint"):
            return None
        from agents.rl_agent import RLAgent
        return RLAgent(cfg["checkpoint"], deterministic=True)
    if kind == "Pro v2 (evolved)":
        return ProAgentV2()
    if kind == "Pro v3 (evolved, chained)":
        return ProAgentV3()
    return ProAgent()


# ------------------------------------------------------------------- state

def deal_next_hand():
    ss = st.session_state
    reloaded = [i for i, s in enumerate(ss.play_stacks) if s <= 1e-9]
    for i in reloaded:
        ss.play_stacks[i] = ss.play_stack_target
    ss.play_reload_notes = reloaded

    engine = PokerEngine(num_players=ss.play_num_players, small_blind=ss.play_small_blind,
                         big_blind=ss.play_big_blind, rng=random.Random())
    engine.start_hand(list(ss.play_stacks), button=ss.play_button)
    ss.play_engine = engine
    ss.play_hand_no += 1
    ss.play_hand_over = False
    ss.play_log = []
    _run_bot_turns()


def _run_bot_turns():
    ss = st.session_state
    engine = ss.play_engine
    while not engine.done and engine.current_player() != 0:
        seat = engine.current_player()
        agent = ss.play_opponents[seat - 1]
        action = agent.act(engine, seat)
        amount = engine.legal_actions(seat)[action]
        engine.step(action)
        ss.play_log.append(f"Seat {seat} ({agent.__class__.__name__}) {describe_action(action, amount)}")
    if engine.done:
        _finish_hand()


def apply_hero_action(action):
    ss = st.session_state
    engine = ss.play_engine
    amount = engine.legal_actions(0)[action]
    engine.step(action)
    ss.play_log.append(f"You {describe_action(action, amount)}")
    if engine.done:
        _finish_hand()
    else:
        _run_bot_turns()


def _finish_hand():
    ss = st.session_state
    engine = ss.play_engine
    ss.play_stacks = [p.stack for p in engine.players]
    ss.play_session_log.append({"hand": ss.play_hand_no, "hero_net_bb": engine.payouts[0] / ss.play_big_blind,
                                "hero_stack": ss.play_stacks[0]})
    ss.play_hand_over = True
    ss.play_button = (ss.play_button + 1) % ss.play_num_players


if restart_clicked or "play_engine" not in st.session_state:
    opponents = [build_agent(cfg) for cfg in opponent_cfgs]
    if any(o is None for o in opponents):
        st.warning("Pick a valid checkpoint for every 'Trained RL model' opponent seat, then restart.")
        st.stop()
    st.session_state.play_num_players = num_players
    st.session_state.play_opponents = opponents
    st.session_state.play_stack_target = starting_stack_bb * big_blind
    st.session_state.play_big_blind = big_blind
    st.session_state.play_small_blind = small_blind
    st.session_state.play_stacks = [starting_stack_bb * big_blind] * num_players
    st.session_state.play_button = 0
    st.session_state.play_hand_no = 0
    st.session_state.play_session_log = []
    st.session_state.play_engine = None
    st.session_state.play_hand_over = True
    st.session_state.play_reload_notes = []

ss = st.session_state

# ------------------------------------------------------------------ render

if ss.play_hand_over:
    if ss.play_session_log:
        last = ss.play_session_log[-1]
        st.subheader(f"Hand {last['hand']} result")
        if last["hero_net_bb"] > 1e-9:
            st.success(f"You won {last['hero_net_bb']:.1f} bb this hand.")
        elif last["hero_net_bb"] < -1e-9:
            st.error(f"You lost {-last['hero_net_bb']:.1f} bb this hand.")
        else:
            st.info("Chopped -- no net change.")
        engine = ss.play_engine
        if engine.showdown_ranking:
            st.write("Showdown:")
            for seat, _score in sorted(engine.showdown_ranking, key=lambda t: t[1]):
                who = "You" if seat == 0 else f"Seat {seat}"
                st.write(f"- {who}: {format_cards(engine.players[seat].hole_cards)}")
        for line in ss.play_log:
            st.caption(line)
    if ss.play_reload_notes:
        st.info("Reloaded to starting stack: " + ", ".join(
            "You" if i == 0 else f"Seat {i}" for i in ss.play_reload_notes))
    if st.button("Deal next hand", type="primary", use_container_width=True):
        deal_next_hand()
        st.rerun()

else:
    engine = ss.play_engine
    st.subheader(f"Hand {ss.play_hand_no} -- {engine.street.name.title()}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Pot", f"{engine.pot_total():.1f}")
    m2.metric("Board", format_cards(engine.board))
    m3.metric("Your stack", f"{engine.players[0].stack:.1f}")

    if show_equity:
        hero = engine.players[0]
        live_opponents = sum(1 for p in engine.players if p.active and p.seat != 0)
        if hero.active and live_opponents:
            equity = estimate_equity(hero.hole_cards, engine.board, live_opponents, n_sims=400)
            st.caption(f"Your equity vs {live_opponents} live opponent(s): {equity:.1%}")

    st.write("Seats")
    cols = st.columns(engine.num_players)
    for seat in range(engine.num_players):
        p = engine.players[seat]
        with cols[seat]:
            tags = " ".join(t for t in [
                "you" if seat == 0 else "",
                "button" if seat == engine.button else "",
                "SB" if seat == engine.sb_seat else "",
                "BB" if seat == engine.bb_seat else "",
                "to act" if seat == engine.current_seat else "",
                "folded" if p.folded else "",
                "all-in" if p.all_in else "",
            ] if t)
            st.markdown(f"**Seat {seat}**" + (f" _{tags}_" if tags else ""))
            st.write(format_cards(p.hole_cards) if seat == 0 else "?? ??")
            st.write(f"stack {p.stack:.1f}")
            st.write(f"bet {p.bet_street:.1f}")

    if ss.play_log:
        with st.expander("Action log (this hand)", expanded=True):
            for line in ss.play_log:
                st.caption(line)

    st.divider()
    if engine.current_player() == 0:
        st.subheader("Your turn")
        legal = engine.legal_actions(0)
        hero_bet = engine.players[0].bet_street
        action_cols = st.columns(len(legal))
        for col, (action, amount) in zip(action_cols, legal.items()):
            if action == Action.CHECK_CALL:
                label = "Check" if amount <= 1e-9 else f"Call {amount:.1f}"
            elif action == Action.ALL_IN:
                label = f"All-in ({amount:.1f})"
            elif action == Action.FOLD:
                label = "Fold"
            else:
                label = f"{ACTION_LABELS[action]} (to {hero_bet + amount:.1f})"
            if col.button(label, key=f"act_{action.name}", use_container_width=True):
                apply_hero_action(action)
                st.rerun()
    else:
        st.info("Waiting on opponents...")

if ss.play_session_log:
    st.divider()
    st.subheader("Session")
    cumulative = []
    total = 0.0
    for row in ss.play_session_log:
        total += row["hero_net_bb"]
        cumulative.append(total)
    st.line_chart({"cumulative bb won": cumulative})
    st.caption(f"{len(ss.play_session_log)} hands played this session, {total:+.1f} bb net.")
