import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import kart_rl  # noqa: F401
from kart_rl.envs.strategy_env import KartStrategyEnv, StrategyConfig
from scripts.evaluate_strategy import run_strategy, write_report


def test_registered_strategy_env_and_seeded_reset():
    env = gym.make("KartStrategy-v0", config={"laps": 2})
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_pit_cost_and_tyres_reset_without_free_start_stop():
    env = KartStrategyEnv({"laps": 3, "wear_per_lap": 0.2})
    try:
        env.reset(seed=0)
        _, _, term, trunc, info = env.step(1)
        assert not term and not trunc
        assert info["pit_stops"] == 0
        assert info["wear"] == pytest.approx(0.2, abs=0.002)
        _, reward, term, trunc, info = env.step(1)
        lap = info["history"][-1]
        assert info["pit_stops"] == 1 and lap["wear_start"] == 0
        assert lap["duration_s"] >= env.config.pit_loss_s
        assert reward == pytest.approx(-lap["duration_s"])
        assert not term and not trunc
        obs, info = env.reset(seed=0)
        assert info["pit_stops"] == 0 and info["wear"] == 0 and info["grip"] == 1
        assert env.observation_space.contains(obs)
        assert env.driving.kart.grip_scale == 1
    finally:
        env.close()


def test_failure_cannot_win_by_ending_early():
    env = KartStrategyEnv({"max_time_s": 1.0})
    try:
        env.reset()
        _, reward, term, trunc, info = env.step(0)
        assert trunc and not term and not info["finished"]
        assert reward <= -1
        with pytest.raises(RuntimeError):
            env.step(0)
    finally:
        env.close()


def test_completed_race_return_equals_negative_time_and_report(tmp_path):
    config = StrategyConfig(laps=2, wear_per_lap=0)
    result = run_strategy(config, lambda obs, info: 0)
    assert result["finished"] and result["laps_done"] == 2
    assert result["return"] == pytest.approx(-result["elapsed_s"])
    with_stop = run_strategy(config, lambda obs, info: info["laps_done"] == 1)
    assert with_stop["finished"] and with_stop["pit_stops"] == 1
    assert with_stop["elapsed_s"] - result["elapsed_s"] == pytest.approx(config.pit_loss_s)
    write_report(tmp_path, config, {"<test>": result})
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "&lt;test&gt;" in page and "<test>" not in page
    assert (tmp_path / "laps.csv").exists()


@pytest.mark.parametrize(
    "config", [{"laps": 0}, {"laps": 1.5}, {"pit_loss_s": -1}, {"min_grip": 0}, {"wear_per_lap": np.nan}]
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        KartStrategyEnv(config)
