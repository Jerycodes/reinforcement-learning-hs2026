import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import kart_rl  # noqa: F401
from kart_rl.envs.strategy_env import KartStrategyEnv, StrategyConfig, reward_components
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
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        assert info["reward_components"]["time"] == pytest.approx(
            -lap["duration_s"] / env.config.reward_time_scale_s
        )
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
        assert term and not trunc and not info["finished"]
        assert reward <= -env.config.max_time_s / env.config.reward_time_scale_s - env.config.reward_failure
        with pytest.raises(RuntimeError):
            env.step(0)
    finally:
        env.close()


def test_completed_race_return_equals_negative_time_and_report(tmp_path):
    config = StrategyConfig(laps=2, wear_per_lap=0)
    result = run_strategy(config, lambda obs, info: 0)
    assert result["finished"] and result["laps_done"] == 2
    assert result["return"] == pytest.approx(
        config.reward_finish - result["elapsed_s"] / config.reward_time_scale_s
    )
    assert result["reward_totals"]["progress"] == pytest.approx(0)
    with_stop = run_strategy(config, lambda obs, info: info["laps_done"] == 1)
    assert with_stop["finished"] and with_stop["pit_stops"] == 1
    assert with_stop["elapsed_s"] - result["elapsed_s"] == pytest.approx(config.pit_loss_s)
    assert result["return"] - with_stop["return"] == pytest.approx(
        config.pit_loss_s / config.reward_time_scale_s
    )
    write_report(tmp_path, config, {"<test>": result})
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "&lt;test&gt;" in page and "<test>" not in page
    assert (tmp_path / "laps.csv").exists()


@pytest.mark.parametrize(
    "config",
    [
        {"laps": 0},
        {"laps": 1.5},
        {"pit_loss_s": -1},
        {"min_grip": 0},
        {"wear_per_lap": np.nan},
        {"reward_time_scale_s": 0},
        {"reward_finish": -1},
        {"reward_progress": np.inf},
        {"reward_failure": -1},
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        KartStrategyEnv(config)


@pytest.mark.parametrize("failure_lap", [1, 4, 8])
def test_progress_shaping_cancels_on_success_and_failure(failure_lap):
    config = StrategyConfig()
    terms = [
        reward_components(config, 100, lap * 100, lap - 1, lap, lap == failure_lap, False)
        for lap in range(1, failure_lap + 1)
    ]
    assert sum(t["progress"] for t in terms) == pytest.approx(0)
    # Same return for early or late failure, no reward for surviving longer.
    assert sum(sum(t.values()) for t in terms) == pytest.approx(
        -config.max_time_s / config.reward_time_scale_s - config.reward_failure
    )


def test_shaping_does_not_change_time_ranking():
    for progress_weight in (0, 2, 100):
        config = StrategyConfig(laps=2, reward_progress=progress_weight)

        def race_return(lap_time, config=config):
            return sum(
                sum(
                    reward_components(
                        config, lap_time, lap * lap_time, lap - 1, lap, lap == 2, lap == 2
                    ).values()
                )
                for lap in (1, 2)
            )

        assert race_return(100) - race_return(106) == pytest.approx(0.12)
        assert race_return(100) == pytest.approx(config.reward_finish - 2)


def test_failure_is_worse_than_finishing_at_deadline():
    config = StrategyConfig()
    success = sum(reward_components(config, config.max_time_s, config.max_time_s, 0, 8, True, True).values())
    failure = sum(reward_components(config, 0, 0, 0, 0, True, False).values())
    assert success > failure
