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


# ---------------------------------------------------------------- track limits / barrier contact
def _place(env, s0, lat, speed=5.0):
    """Put the kart at arc length s0 with lateral offset lat (+ = left), heading along the track."""
    x, y, yaw = env.track.start_pose(s0, lat)
    env.kart.reset(x, y, yaw, speed)
    env.s, env.lateral, env.idx = env.track.localize(x, y)


def test_body_contact_uses_the_whole_kart():
    env = KartLapEnv(LapEnvConfig(random_start=False))
    env.reset(seed=0)
    s0 = 120.0
    half_left = float(env.track.half_width_left[env.track.nearest_index(*env.track.point_at(s0))])
    # centre 0.5 m from the left barrier: the body (0.7 m half width) sticks out -> contact
    _place(env, s0, half_left - 0.5)
    _, r, terminated, _, info = env.step([0.0, 0.3])
    assert info["wall_contact"] and terminated and info["episode_done"] == "off_track"
    assert r < -10
    # after the collision handling the kart is drawn at the barrier, not inside it
    assert info["clearance"] >= 0.0
    # centre 1.2 m from the barrier: whole body clear -> no contact
    env.reset(seed=0)
    _place(env, s0, half_left - 1.2)
    _, _, terminated, _, info = env.step([0.0, 0.3])
    assert not info["off_track"] and not terminated and info["clearance"] > 0.3


def test_centre_mode_is_the_old_behaviour():
    env = KartLapEnv(LapEnvConfig(random_start=False, wall_check="centre", off_track_margin=0.4))
    env.reset(seed=0)
    s0 = 120.0
    half_left = float(env.track.half_width_left[env.track.nearest_index(*env.track.point_at(s0))])
    _place(env, s0, half_left - 0.6)  # centre 0.6 m from the edge: body overlaps, centre mode does not care
    _, _, terminated, _, info = env.step([0.0, 0.3])
    assert not info["off_track"] and not terminated


def test_penalty_mode_barrier_is_an_obstacle():
    env = KartLapEnv(LapEnvConfig(random_start=False, off_track_mode="penalty", reward_off_track=-5.0))
    env.reset(seed=0)
    s0 = 120.0
    half_left = float(env.track.half_width_left[env.track.nearest_index(*env.track.point_at(s0))])
    _place(env, s0, half_left - 0.3, speed=8.0)
    _, r, terminated, truncated, info = env.step([0.0, 0.5])
    assert info["wall_contact"] and not terminated and not truncated
    assert r < -4                                   # penalty applied this step
    assert info["clearance"] >= 0.05                # pushed back onto the track
    assert info["speed"] < 8.0                      # lost speed on the barrier
    assert env.track.on_track(env.kart.x, env.kart.y)
    for _ in range(40):                             # keep driving straight: never ends up inside the barrier
        _, _, terminated, truncated, info = env.step([0.0, 1.0])
        assert env.track.on_track(env.kart.x, env.kart.y)
        if terminated or truncated:
            break


def test_random_starts_are_clear_of_the_barrier():
    env = KartLapEnv(LapEnvConfig(random_start=True))
    for seed in range(30):
        _, info = env.reset(seed=seed)
        assert not info["off_track"] and info["clearance"] >= env.cfg.off_track_margin


def test_optional_drift_penalty():
    def reward_while_sliding(reward_drift):
        env = KartLapEnv(LapEnvConfig(random_start=False, reward_drift=reward_drift, drift_threshold_deg=6.0))
        env.reset(seed=0)
        env.kart.state[3], env.kart.state[4] = 10.0, 3.0      # 10 m/s forward, 3 m/s sideways = 16.7° slip
        _, r, _, _, info = env.step([0.0, 0.0])
        return r, info["slip_deg"]

    r_off, slip = reward_while_sliding(0.0)
    r_on, _ = reward_while_sliding(-0.02)
    assert 10 < slip < 20
    assert r_on < r_off - 0.1                                # penalty applied only when switched on
    env = KartLapEnv(LapEnvConfig(random_start=False, reward_drift=-0.02))
    env.reset(seed=0)
    _, _, _, _, info = env.step([0.0, 0.5])                  # driving straight: no slip, no penalty
    assert info["slip_deg"] < 1.0
