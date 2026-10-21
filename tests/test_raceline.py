import numpy as np
import pytest

from kart_rl.baselines import PurePursuitDriver
from kart_rl.envs import KartLapEnv, LapEnvConfig
from kart_rl.raceline import RaceLine, minimum_curvature_line, speed_profile
from kart_rl.track import Track


@pytest.fixture(scope="module")
def track():
    return Track("race_inn_roggwil")


def test_stored_raceline_is_inside_the_track(track):
    line = RaceLine.load()
    assert track.on_track(line.points[:, 0], line.points[:, 1]).all()
    d = track.distance_to_edge(line.points[:, 0], line.points[:, 1])
    assert d.min() > 0.6  # margin 1 m minus raster resolution
    assert line.length < track.length  # cuts the corners
    assert np.abs(line.curvature).max() < np.abs(track.curvature).max()


def test_speed_profile_is_slower_in_corners(track):
    line = RaceLine.load()
    v = speed_profile(line)
    assert v.min() > 3.0 and v.max() <= 18.0 + 1e-6
    tight = np.abs(line.curvature) > 0.15
    assert v[tight].mean() < v[~tight].mean()
    centre = RaceLine.from_track_centerline(track)
    assert line.lap_time_estimate < centre.lap_time_estimate


def test_one_optimisation_iteration_reduces_curvature(track):
    pts = minimum_curvature_line(track, margin=1.0, iterations=1)
    line = RaceLine(pts)
    assert np.sum(line.curvature ** 2) < np.sum(track.curvature ** 2)
    assert track.on_track(pts[:, 0], pts[:, 1]).all()


def test_driver_on_raceline_beats_centreline_driver(track):
    env = KartLapEnv(LapEnvConfig(laps=2, random_start=False, max_time=300.0))
    line = RaceLine.load()
    times = {}
    for name, drv in (("centre", PurePursuitDriver(track)),
                      ("race", PurePursuitDriver(line, speed=line.speed, a_brake=4.0))):
        env.reset()
        drv.reset()
        done = False
        while not done:
            _, _, term, trunc, info = env.step(drv.act(env))
            done = term or trunc
        assert info["episode_done"] == "laps", name
        times[name] = info["lap_times"][-1]
    assert times["race"] < times["centre"] - 5.0
