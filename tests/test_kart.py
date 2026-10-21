import numpy as np

from kart_rl.kart import Kart, KartParams


def drive(kart, steer, throttle, seconds, dt=0.05):
    for _ in range(int(seconds / dt)):
        kart.step(steer, throttle, dt)


def test_straight_line_acceleration_and_top_speed():
    k = Kart()
    k.reset(0, 0, 0, 0)
    drive(k, 0.0, 1.0, 5.0)
    assert 10.0 < k.vx < 16.0  # ~40-55 km/h after 5 s
    drive(k, 0.0, 1.0, 40.0)
    assert 16.0 < k.vx < 20.0  # top speed 58-72 km/h
    assert abs(k.y) < 1e-6 and abs(k.yaw) < 1e-6  # drives straight


def test_braking_stops_the_kart():
    k = Kart()
    k.reset(0, 0, 0, 15.0)
    drive(k, 0.0, -1.0, 6.0)
    assert k.vx == 0.0
    assert k.x > 10.0  # it needed some distance


def test_no_reverse_driving():
    k = Kart()
    k.reset(0, 0, 0, 0.0)
    drive(k, 0.0, -1.0, 2.0)
    assert k.vx == 0.0 and k.x == 0.0


def test_steady_state_cornering_is_grip_limited():
    k = Kart(KartParams(mu=1.0))
    k.reset(0, 0, 0, 8.0)
    drive(k, 0.6, 0.4, 30.0)
    a_lat = k.speed * abs(k.yaw_rate)
    assert 5.0 < a_lat < 10.5  # 0.5 .. 1.07 g with mu = 1
    assert k.yaw_rate > 0  # positive steer = left turn = positive yaw rate


def test_less_grip_means_slower_corner():
    results = {}
    for mu in (0.8, 1.1):
        k = Kart(KartParams(mu=mu))
        k.reset(0, 0, 0, 8.0)
        drive(k, 0.6, 0.4, 30.0)
        results[mu] = k.speed * abs(k.yaw_rate)
    assert results[0.8] < results[1.1]


def test_grip_and_power_scale_hooks():
    k = Kart()
    k.reset(0, 0, 0, 0.0)
    k.power_scale = 1.4
    drive(k, 0.0, 1.0, 3.0)
    v_boost = k.vx
    k2 = Kart()
    k2.reset(0, 0, 0, 0.0)
    drive(k2, 0.0, 1.0, 3.0)
    assert v_boost > k2.vx


def test_state_stays_finite_under_abuse():
    k = Kart()
    k.reset(0, 0, 0, 18.0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        k.step(rng.uniform(-1, 1), rng.uniform(-1, 1), 0.05)
        assert np.isfinite(k.state).all()


def test_steering_rate_limit():
    k = Kart()
    k.reset(0, 0, 0, 5.0)
    k.step(1.0, 0.0, 0.05)
    assert k.steer < k.p.max_steer  # cannot reach full lock in one step
    drive(k, 1.0, 0.0, 1.0)
    assert k.steer == pytest_approx(k.p.max_steer)


def pytest_approx(v):
    import pytest

    return pytest.approx(v, abs=1e-6)
