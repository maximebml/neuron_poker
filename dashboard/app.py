"""Local dashboard: pick an agent and a table situation, ask it to decide.

Run with:
    streamlit run dashboard/app.py

Lets you choose which agent decides (random, the rule-based equity
heuristic at any aggression, or a trained RL checkpoint), configure a
table (2-6 players, blinds, per-seat stacks/bets/folded/all-in, street,
board), and set the hero's hole cards -- then shows the agent's action,
sizing, legal-action list, hand equity, and (for the RL agent) the full
action-probability distribution.

This is a thin UI over `inference.decide.decide`, the same function the
`main.py decide` CLI uses, so a situation built here matches the CLI's
`--situation` JSON exactly (shown at the bottom of the page).
"""
import glob
import os

import streamlit as st

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from inference.decide import decide
from poker.cards import FULL_DECK

st.set_page_config(page_title="neuron_poker decision dashboard", layout="wide")
st.title("neuron_poker -- decision dashboard")

CARD_OPTIONS = [""] + FULL_DECK
STREET_BOARD_SIZE = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

# --------------------------------------------------------------------- agent

st.sidebar.header("Agent")
agent_kind = st.sidebar.radio("Agent type", ["Trained RL model", "Rule-based (equity heuristic)", "Random"])

agent = None
if agent_kind == "Random":
    agent = RandomAgent()

elif agent_kind == "Rule-based (equity heuristic)":
    aggression = st.sidebar.slider("Aggression", 0.0, 1.0, 0.5, 0.05,
                                   help="0 = tight/calling-station, 1 = loose/aggressive")
    agent = RuleBasedAgent(aggression=aggression)

else:
    models_dir = st.sidebar.text_input("Checkpoints directory", "models")
    # Sort by modification time, not filename: "checkpoint_50000.zip" sorts
    # *after* "checkpoint_150000.zip" lexicographically, which would silently
    # pick the wrong file as "latest".
    checkpoints = sorted(glob.glob(os.path.join(models_dir, "*.zip")), key=os.path.getmtime)
    if checkpoints:
        model_path = st.sidebar.selectbox("Checkpoint (newest first)", checkpoints[::-1])
    else:
        st.sidebar.warning(f"No .zip checkpoints found in '{models_dir}/'.")
        model_path = st.sidebar.text_input("...or enter a checkpoint path directly", "")
    deterministic = st.sidebar.checkbox("Deterministic (always take the top action)", value=True)
    if model_path:
        from agents.rl_agent import RLAgent
        try:
            agent = RLAgent(model_path, deterministic=deterministic)
        except Exception as e:  # noqa: BLE001 - surface any load error in the UI
            st.sidebar.error(f"Couldn't load checkpoint: {e}")

st.sidebar.divider()
n_equity_sims = st.sidebar.slider("Equity simulation rollouts", 100, 2000, 500, 100,
                                  help="More rollouts = more accurate equity estimate, slower")

# --------------------------------------------------------------------- table

st.header("Table")
c1, c2, c3, c4 = st.columns(4)
num_players = c1.number_input("Players", min_value=2, max_value=6, value=6, step=1)
big_blind = c2.number_input("Big blind", min_value=0.1, value=2.0)
small_blind = c3.number_input("Small blind", min_value=0.0, value=1.0)
button_seat = c4.number_input("Button seat", min_value=0, max_value=int(num_players) - 1, value=0, step=1)

street = st.selectbox("Street", list(STREET_BOARD_SIZE), index=1)
n_board_cards = STREET_BOARD_SIZE[street]
board = []
if n_board_cards:
    st.write("Board")
    board_cols = st.columns(n_board_cards)
    for i in range(n_board_cards):
        board.append(board_cols[i].selectbox(f"Board {i + 1}", CARD_OPTIONS, key=f"board_{i}"))

pot_before_street = st.number_input("Pot carried over from previous streets", min_value=0.0, value=0.0,
                                    help="0 preflop; the pot size before this street's betting began otherwise")

st.header("Hero")
h1, h2, h3 = st.columns(3)
hero_seat = h1.number_input("Hero seat", min_value=0, max_value=int(num_players) - 1, value=0, step=1)
hero_card_1 = h2.selectbox("Hero card 1", CARD_OPTIONS, key="hero_card_1")
hero_card_2 = h3.selectbox("Hero card 2", CARD_OPTIONS, key="hero_card_2")

st.header("Players")
st.caption("Bet = what that seat has put in during the current street only.")
header_cols = st.columns([1.4, 1, 1, 1, 1])
for col, label in zip(header_cols, ["Seat", "Stack", "Bet", "Folded", "All-in"]):
    col.markdown(f"**{label}**")

players = []
for seat in range(int(num_players)):
    cols = st.columns([1.4, 1, 1, 1, 1])
    tags = " ".join(t for t in [
        "hero" if seat == hero_seat else "",
        "button" if seat == button_seat else "",
    ] if t)
    cols[0].markdown(f"Seat {seat}" + (f" _{tags}_" if tags else ""))
    stack = cols[1].number_input("stack", min_value=0.0, value=200.0, key=f"stack_{seat}",
                                 label_visibility="collapsed")
    bet = cols[2].number_input("bet", min_value=0.0, value=0.0, key=f"bet_{seat}", label_visibility="collapsed")
    folded = cols[3].checkbox("folded", value=False, key=f"folded_{seat}", label_visibility="collapsed")
    all_in = cols[4].checkbox("all-in", value=False, key=f"allin_{seat}", label_visibility="collapsed")
    players.append({"seat": seat, "stack": stack, "bet": bet, "folded": folded, "all_in": all_in})

st.divider()

if st.button("Get decision", type="primary", use_container_width=True):
    hero_cards = [c for c in (hero_card_1, hero_card_2) if c]
    board_cards = [c for c in board if c]
    all_picked_cards = hero_cards + board_cards

    if len(hero_cards) != 2:
        st.error("Pick both of the hero's hole cards.")
    elif len(board_cards) != n_board_cards:
        st.error(f"'{street}' needs exactly {n_board_cards} board card(s); {len(board_cards)} picked.")
    elif len(set(all_picked_cards)) != len(all_picked_cards):
        st.error("The same card was picked twice -- each card can only appear once.")
    elif agent is None:
        st.error("Pick a valid agent (or checkpoint) in the sidebar first.")
    else:
        situation = {
            "num_players": int(num_players), "big_blind": big_blind, "small_blind": small_blind,
            "button_seat": int(button_seat), "street": street, "board": board_cards,
            "pot_before_street": pot_before_street, "hero_seat": int(hero_seat),
            "hero_cards": hero_cards, "players": players,
        }
        try:
            result = decide(situation, agent=agent, n_equity_sims=n_equity_sims)
        except ValueError as e:
            st.error(str(e))
        else:
            st.subheader("Decision")
            amount_note = f" -- {result['amount']:.1f} chips" if result["amount"] > 0 else ""
            st.success(f"### {result['action']}{amount_note}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Pot before action", f"{result['pot_before_action']:.1f}")
            m2.metric("Hero equity vs. live opponents", f"{result['hero_equity_vs_live_opponents']:.1%}")
            m3.metric("Agent", result["used_model"])

            st.write("Legal actions and their chip cost:")
            st.table(result["legal_actions"])

            if result["action_probabilities"]:
                st.write("Policy probabilities:")
                st.bar_chart(result["action_probabilities"])

            with st.expander("Situation JSON (matches `main.py decide --situation`)"):
                st.json(situation)
