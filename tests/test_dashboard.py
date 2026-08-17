"""Headless regression tests for the Streamlit dashboard (dashboard/app.py)."""
import os
import time

from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")


def _fill_flop_hand(at):
    at.selectbox(key="hero_card_1").set_value("Ah").run(timeout=30)
    at.selectbox(key="hero_card_2").set_value("Kd").run(timeout=30)
    at.selectbox(key="board_0").set_value("Qs").run(timeout=30)
    at.selectbox(key="board_1").set_value("Jd").run(timeout=30)
    at.selectbox(key="board_2").set_value("2c").run(timeout=30)
    return at


def test_dashboard_renders_without_exceptions():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_rule_based_flow_produces_a_decision():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("Rule-based (equity heuristic)").run(timeout=30)
    _fill_flop_hand(at)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert len(at.success) == 1
    assert any(m.label == "Agent" for m in at.metric)


def test_pro_agent_flow_produces_a_decision():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("Pro (sophisticated rules)").run(timeout=30)
    _fill_flop_hand(at)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert len(at.success) == 1
    assert any(m.label == "Agent" and m.value == "pro" for m in at.metric)


def test_random_agent_flow_produces_a_decision():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("Random").run(timeout=30)
    _fill_flop_hand(at)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert len(at.success) == 1


def test_duplicate_card_selection_is_rejected():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("Random").run(timeout=30)
    at.selectbox(key="hero_card_1").set_value("Ah").run(timeout=30)
    at.selectbox(key="hero_card_2").set_value("Kd").run(timeout=30)
    at.selectbox(key="board_0").set_value("Ah").run(timeout=30)  # duplicate of hero_card_1
    at.selectbox(key="board_1").set_value("Jd").run(timeout=30)
    at.selectbox(key="board_2").set_value("2c").run(timeout=30)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert len(at.error) == 1
    assert "picked twice" in at.error[0].value.lower()


def test_checkpoint_dropdown_sorts_by_recency_not_filename(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # Deliberately write the "50000" checkpoint *after* the "150000" one so a
    # plain filename sort ("checkpoint_150000" < "checkpoint_50000") would
    # misorder them; only an mtime-based sort gets this right.
    (models_dir / "checkpoint_150000.zip").write_bytes(b"x")
    time.sleep(0.01)
    (models_dir / "checkpoint_50000.zip").write_bytes(b"x")

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    dir_input = next(t for t in at.sidebar.text_input if "directory" in (t.label or "").lower())
    dir_input.set_value(str(models_dir)).run(timeout=30)

    checkpoint_box = next(s for s in at.sidebar.selectbox if "checkpoint" in (s.label or "").lower())
    assert checkpoint_box.options[0] == str(models_dir / "checkpoint_50000.zip")
