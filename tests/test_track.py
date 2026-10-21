import numpy as np
import pytest

from kart_rl.track import Track


@pytest.fixture(scope="module")
def track():
    return Track("race_inn_roggwil")


def test_geometry_basics(track):
    assert track.n == 640
    assert abs(track.length - 640.0) < 1.0
    seg = np.linalg.norm(np.diff(np.vstack([track.centerline, track.centerline[:1]]), axis=0), axis=1)
    assert seg.min() > 0.9 and seg.max() < 1.1
    assert 5.0 < track.width.mean() < 9.0
    assert np.abs(track.curvature).max() < 0.5  # tightest corner radius > 2 m


def test_centreline_is_on_track(track):
    on = track.on_track(track.centerline[:, 0], track.centerline[:, 1])
    assert on.all()
    d = track.distance_to_edge(track.centerline[:, 0], track.centerline[:, 1])
    assert d.min() > 1.0


def test_far_away_is_off_track(track):
    assert not track.on_track(np.array([-50.0]), np.array([-50.0]))[0]
    assert not track.on_track(np.array([1e4]), np.array([1e4]))[0]
    # the paddock (inside the pit-lane corner) is not drivable
    assert not track.on_track(np.array([105.0]), np.array([35.0]))[0]


def test_localize_round_trip(track):
    for s in np.linspace(0, track.length - 1, 50):
        p = track.point_at(s)
        s2, lat, _ = track.localize(p[0], p[1])
        err = abs((s2 - s + track.length / 2) % track.length - track.length / 2)
        assert err < 0.05
        assert abs(lat) < 0.1


def test_lateral_sign(track):
    s = 100.0
    h = track.heading_at(s)
    left = track.point_at(s) + 1.5 * np.array([-np.sin(h), np.cos(h)])
    assert track.localize(left[0], left[1])[1] == pytest.approx(1.5, abs=0.1)


def test_lidar_ranges(track):
    x, y, yaw = track.start_pose(0.0)
    angles = np.linspace(-np.pi / 2, np.pi / 2, 9)
    d = track.lidar(x, y, yaw, angles, max_range=30.0)
    assert d.shape == (9,)
    assert (d > 0).all() and (d <= 30.0).all()
    # sideways rays hit the walls within half a width + a bit
    assert d[0] < 8.0 and d[-1] < 8.0
    # straight ahead on the start straight is long
    assert d[4] > 15.0


def test_pit_lane_connects(track):
    p = track.pit
    assert np.linalg.norm(p.centerline[0] - track.centerline[p.entry_index]) < 0.5
    assert np.linalg.norm(p.centerline[-1] - track.centerline[p.exit_index]) < 0.5
    assert p.entry_s < p.exit_s
    assert 50 < p.length < 120
