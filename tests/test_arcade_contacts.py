"""Independent geometry checks on published physics ticks."""

import math
import random
from itertools import combinations

import pytest

from kart_rl.game.race import RaceRoom, advance_field

GAS = [0, 1, -1, -1]
BOOST = [0, 1, -1, 1]


def place(player, s, lateral=0, speed=0, angle=0):
    env = player.racer.env
    x, y, heading = env.track.start_pose(s, lateral)
    env.kart.reset(x, y, heading + angle, speed)
    env.s, env.lateral, env.idx = env.track.localize(x, y)
    env.total_progress = env.max_progress = env.s
    player.racer.obs = env._observe()


def check_field(players):
    for p in players:
        env = p.racer.env
        assert env._clearance() >= env.cfg.off_track_margin
    for a, b in combinations(players, 2):
        pa, pb = a.racer.env.kart.footprint(), b.racer.env.kart.footprint()
        for direction in (pa[0] - pa[1], pa[1] - pa[2], pb[0] - pb[1], pb[1] - pb[2]):
            ua, ub = pa @ direction, pb @ direction
            if ua.max() <= ub.min() + 1e-8 or ub.max() <= ua.min() + 1e-8:
                break
        else:
            pytest.fail("Published overlapping rectangles")


def test_boost_pass_in_two_lanes():
    room = RaceRoom()
    quick, slow = room.join(), room.join()
    place(quick, 3, -0.9, 24)
    place(slow, 7, 0.9, 7)
    for i in range(12):
        advance_field([quick, slow], [BOOST, GAS], room.dt, 3, (i + 1) * room.dt)
        check_field([quick, slow])
    assert quick.racer.env.s > slow.racer.env.s + 1.55


@pytest.mark.parametrize("angle,lead_speed", [(0, 0), (math.pi, 26)])
def test_high_speed_contacts_keep_order(angle, lead_speed):
    room = RaceRoom()
    a, b = room.join(), room.join()
    place(a, 5, 0, 28)
    place(b, 7, 0, lead_speed, angle)
    for i in range(8):
        advance_field([a, b], [BOOST, GAS], room.dt, 3, (i + 1) * room.dt)
        check_field([a, b])
        assert a.racer.env.s < b.racer.env.s
    assert a.racer.env.kart.vx < 27


@pytest.mark.parametrize("s,lat", [(25, 1.6), (10, 2.86)])
def test_oblique_contact_and_wall_squeeze(s, lat):
    room = RaceRoom()
    inside, outside = room.join(), room.join()
    place(outside, s, lat, 3)
    place(inside, s - 1.7, lat - 1.35, 18, 0.65)
    check_field([inside, outside])
    sideways = []
    for i in range(15):
        advance_field([inside, outside], [GAS, GAS], room.dt, 3, (i + 1) * room.dt)
        check_field([inside, outside])
        sideways.append(abs(outside.racer.env.kart.vy))
    assert max(sideways) > 1  # the straight-running kart received a lateral impulse
    assert outside.racer.env.kart.speed > 3


def test_contact_near_line_scored_only_after_resolution():
    room = RaceRoom()
    a, b = room.join(), room.join()
    length = a.racer.env.track.length
    place(a, length - 1.8, 0, 28)
    place(b, length - 0.18, 0, 0)
    a.next_gate = b.next_gate = 4  # both have passed the three sector gates
    for i in range(4):
        advance_field([a, b], [BOOST, GAS], room.dt, 1, (i + 1) * room.dt)
        check_field([a, b])
        if a.finish_time is not None:
            assert b.finish_time is not None
        if b.finish_time is not None:
            break
    assert b.finish_time is not None and a.finish_time is None


def test_five_kart_start_and_first_corners(tmp_path, monkeypatch):
    room = RaceRoom(results_path=tmp_path / "race.csv")
    host = room.join()
    host.bot = host.ready = True
    for _ in range(4):
        room.join(bot=True)
    monkeypatch.setattr(random, "shuffle", random.Random(7).shuffle)
    room.start(host.id)
    room.countdown = 0
    room.tick()
    for _ in range(330):
        room.tick()
        check_field(list(room.players.values()))
    assert min(p.racer.env.total_progress for p in room.players.values()) > 65


def test_legacy_footprint_and_new_env():
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    old = gym.make("KartArcade-v0").unwrapped
    new = gym.make("KartArcade-v1").unwrapped
    assert (old.player.racer.env.kart.p.length, old.player.racer.env.kart.p.width) == (1.9, 1.4)
    assert (new.player.racer.env.kart.p.length, new.player.racer.env.kart.p.width) == (1.55, 1.02)
    check_env(new, skip_render_check=True)
    old.close()
    new.close()


def test_bot_uses_normal_boost_budget_and_responds_to_traffic():
    from kart_rl.game.race import rule_action

    room = RaceRoom()
    bot, leader = room.join(bot=True), room.join()
    place(bot, 2, 0, 14)
    place(leader, 7, 0, 9)
    assert bot.racer.env.kart.p == leader.racer.env.kart.p
    bot.level = "Club"
    assert rule_action(bot)[3] < 0
    bot.level = "Pro"
    action = rule_action(bot)
    assert action[3] > 0
    advance_field([bot], [action], room.dt, 3, room.dt)
    assert bot.energy < 100
    passing = rule_action(bot, [bot, leader])
    assert bot.passing_lane and abs(passing[0] - action[0]) > 0.1


def test_side_by_side_at_narrowest_cross_section():
    room = RaceRoom()
    players = [room.join(), room.join()]
    track = players[0].racer.env.track
    section = track.s[int(track.width.argmin())]
    for player, lateral in zip(players, [-1.05, 1.05]):
        place(player, section - 2, lateral, 12)
    for tick in range(11):
        advance_field(players, [GAS, GAS], room.dt, 3, (tick + 1) * room.dt)
        check_field(players)
    assert all(p.racer.env.s > section for p in players)
