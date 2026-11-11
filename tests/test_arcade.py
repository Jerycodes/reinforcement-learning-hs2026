import asyncio
import json
import time

import numpy as np
import pytest

from kart_rl.game.arcade import ArcadeKart, ArcadeServer
from kart_rl.kart import Kart


def test_arcade_accelerates_and_turns_more_quickly_without_changing_training_kart():
    arcade, training = ArcadeKart(), Kart()
    for kart in (arcade, training):
        kart.reset(0, 0, 0, 0)
        for _ in range(30):
            kart.step(0, 1, 1 / 30)
    assert arcade.vx > training.vx + 3
    assert arcade.vx == pytest.approx(12)
    arcade.step(1, 1, 1 / 30)
    assert arcade.yaw_rate > 0.8
    assert training.p.max_steer == 0.4
    for _ in range(100):
        arcade.step(0, -1, 1 / 30)
    assert arcade.vx == 0 and np.isfinite(arcade.state).all()


def test_each_player_has_own_controls_and_host_only_start(tmp_path):
    server = ArcadeServer(results_path=tmp_path / "races.csv")
    one, two = server.join(), server.join()
    assert not server.start(two.id)
    assert server.start(one.id)
    server.countdown = 0
    server.tick()
    server.handle(one.id, {"type": "keys", "up": True})
    for _ in range(30):
        one.last_input = time.perf_counter()
        server.tick()
    assert one.racer.env.total_progress > 3
    assert two.racer.env.kart.speed == 0
    assert one.racer.env.kart is not two.racer.env.kart
    server.leave(one.id)
    assert server.host_id == two.id and one.id not in server.players


def test_boost_drift_and_stale_input_safety(tmp_path, monkeypatch):
    server = ArcadeServer(results_path=tmp_path / "races.csv")
    player = server.join()
    player.racer.env.kart.reset(*player.racer.env.track.start_pose(0), speed=8)
    # Keep this mechanics test away from track barriers; physical movement and
    # player isolation are exercised separately above.
    monkeypatch.setattr(player.racer, "step", lambda action: None)
    server.handle(player.id, {"type": "keys", "up": True, "left": True, "drift": True})
    for _ in range(30):
        player.last_input = time.perf_counter()
        player.step(server.dt, 3, 1)
    assert player.drift_charge >= 0.7
    server.handle(player.id, {"type": "keys", "up": True})
    player.step(server.dt, 3, 1)
    assert player.mini_boost > 0 and player.racer.env.kart.boosting
    server.handle(player.id, {"type": "keys", "up": True, "boost": True})
    before = player.energy
    player.step(server.dt, 3, 1)
    assert player.energy < before
    # NumPy comparisons must not leak numpy.bool into the WebSocket payload.
    assert json.loads(json.dumps(server.message(), allow_nan=False))["states"][0]["boost"] is True
    player.last_input = time.perf_counter() - 2
    player.step(server.dt, 3, 1)
    assert not any(player.keyboard.keys.values()) and not player.boost


def test_race_distance_and_respawn_penalty_cannot_be_spoofed(tmp_path):
    server = ArcadeServer(laps=1, results_path=tmp_path / "races.csv")
    player = server.join()
    server.start(player.id)
    server.phase = "racing"
    # Spoofing a lap counter or position over the wire has no effect.
    server.handle(player.id, {"type": "state", "lap": 99, "x": 999})
    player.racer.env.laps_done = 4  # line crossing alone does not finish an arcade lap
    server.tick()
    assert player.finish_time is None and not player.laps
    progress = player.racer.env.total_progress
    server.handle(player.id, {"type": "control", "action": "respawn"})
    assert player.penalty == 3 and player.racer.env.total_progress == progress
    player.racer.env.total_progress = player.racer.env.track.length + 1
    server.tick()
    assert player.finish_time is not None and server.phase == "finished"
    assert player.finish_time == pytest.approx(server.elapsed + 3)
    assert (tmp_path / "races.csv").read_text().count("arcade") == 1
    server.tick()
    assert (tmp_path / "races.csv").read_text().count("arcade") == 1


def test_late_join_capacity_and_name_sanitization(tmp_path):
    server = ArcadeServer(max_players=2, results_path=tmp_path / "races.csv")
    one = server.join()
    server.start(one.id)
    two = server.join()
    assert two.waiting
    with pytest.raises(ValueError):
        server.join()
    server.handle(two.id, {"type": "set_name", "name": "<img src=x>Guest"})
    assert "<" not in two.name and ">" not in two.name
    server.handle(one.id, [])
    server.handle(one.id, {"type": "keys", "up": "false"})
    assert not one.keyboard.keys["up"]


def test_two_websocket_players_are_isolated_and_disconnect_cleans_up(tmp_path):
    import websockets

    async def scenario():
        server = ArcadeServer(results_path=tmp_path / "races.csv")
        async with websockets.serve(server.handler, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as one:
                first = json.loads(await one.recv())
                async with websockets.connect(f"ws://127.0.0.1:{port}") as two:
                    second = json.loads(await two.recv())
                    assert first["player_id"] != second["player_id"]
                    await one.send(json.dumps({"type": "keys", "up": True}))
                    await two.send(json.dumps({"type": "keys", "left": True}))
                    await asyncio.sleep(0.03)
                    assert server.players[first["player_id"]].keyboard.keys["up"]
                    assert not server.players[second["player_id"]].keyboard.keys["up"]
                await asyncio.sleep(0.03)
                assert second["player_id"] not in server.players

    asyncio.run(scenario())
