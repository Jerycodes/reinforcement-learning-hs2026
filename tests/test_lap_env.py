import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import kart_rl  # noqa: F401  (registers the envs)
from kart_rl.baselines import PurePursuitDriver
from kart_rl.envs import KartLapEnv, LapEnvConfig


def test_registered_env_passes_gymnasium_checker():
    env = gym.make("KartLap-v0")
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_spaces_and_reset():
    env = KartLapEnv()
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert len(env.obs_names) == obs.shape[0]
    assert info["laps_done"] == 0 and not info["off_track"]


def test_reset_is_reproducible():
    env = KartLapEnv(LapEnvConfig(random_start=True))
    o1, _ = env.reset(seed=123)
    o2, _ = env.reset(seed=123)
    assert np.allclose(o1, o2)


def test_standing_start_is_at_the_line():
    env = KartLapEnv(LapEnvConfig(random_start=False))
    _, info = env.reset()
    assert info["s"] < 1.0 and info["speed"] == 0.0


def test_full_throttle_straight_leaves_track_eventually():
    env = KartLapEnv(LapEnvConfig(random_start=False, max_time=60))
    env.reset()
    done, reason = False, None
    while not done:
        _, _, term, trunc, info = env.step(np.array([0.0, 1.0]))
        done = term or trunc
        reason = info.get("episode_done")
    assert reason == "off_track"


def test_standing_still_terminates_as_stuck():
    env = KartLapEnv(LapEnvConfig(random_start=False, stuck_time=2.0))
    env.reset()
    total = 0.0
    for _ in range(100):
        _, r, term, trunc, info = env.step(np.array([0.0, 0.0]))
        total += r
        if term:
            break
    assert term and info["episode_done"] == "stuck"
    assert total < 0  # penalised


def test_pure_pursuit_completes_laps_and_progress_reward_matches_track_length():
    env = KartLapEnv(LapEnvConfig(laps=2, random_start=False, max_time=300.0, reward_steer_change=0.0))
    drv = PurePursuitDriver(env.track)
    env.reset(seed=0)
    done, total = False, 0.0
    while not done:
        _, r, term, trunc, info = env.step(drv.act(env))
        total += r
        done = term or trunc
    assert info["episode_done"] == "laps"
    assert len(info["lap_times"]) == 2
    assert 40.0 < info["lap_times"][1] < 120.0
    # progress reward over two laps = two track lengths
    assert total == pytest.approx(2 * env.track.length, abs=5.0)


def test_rgb_render_returns_frame():
    env = KartLapEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.step(np.array([0.0, 0.5]))
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    env.close()


def test_trajectory_recording():
    env = KartLapEnv(record=True)
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.array([0.1, 0.5]))
    traj = env.get_trajectory()
    assert len(traj["t"]) == 10 and "x" in traj and "steer" in traj
