"""Responsive arcade kart racing, with one authoritative kart per browser."""

import asyncio
import csv
import functools
import http.server
import json
import math
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from ..kart import Kart, KartParams
from .server import GAME_ENV, ROOT, KeyboardDriver, Racer

COLORS = [0xFF5A36, 0x3FD0FF, 0xFFB000, 0x7CFF6B, 0xD36BFF, 0xFF7FBB, 0xFFFFFF, 0x41E4CB]


class ArcadeKart(Kart):
    """Kinematic handling for the game only; training physics stays unchanged."""

    def __init__(self):
        super().__init__(KartParams(max_steer=0.6, steer_rate=10))
        self.drifting = False
        self.boosting = False

    def step(self, steer_cmd, throttle_cmd, dt):
        steer = float(np.clip(steer_cmd, -1, 1))
        throttle = float(np.clip(throttle_cmd, -1, 1))
        x, y, yaw, vx, vy, _, _ = self.state
        acceleration = (20 if self.boosting else 12) * throttle if throttle > 0 else 22 * throttle
        if abs(throttle) < 0.05:
            acceleration -= 3.5
        limit = 28.0 if self.boosting else 22.0
        vx = float(np.clip(vx + acceleration * dt, 0, max(limit, vx - 8 * dt)))
        yaw_rate = steer * 3.5 / (1 + vx / 18) * min(vx / 2.5, 1)
        if self.drifting:
            yaw_rate *= 1.2
        lateral_target = -steer * vx * 0.2 if self.drifting else 0.0
        vy += (lateral_target - vy) * min(dt * (7 if self.drifting else 20), 1)
        yaw = (yaw + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        self.state[:] = (x, y, yaw, vx, vy, yaw_rate, steer * self.p.max_steer)
        self.last = {"ax": acceleration, "ay": vx * yaw_rate}


@dataclass
class Player:
    id: str
    name: str
    color: int
    racer: Racer
    keyboard: KeyboardDriver = field(default_factory=lambda: KeyboardDriver(10, 15, 18))
    drift: bool = False
    boost: bool = False
    energy: float = 100.0
    drift_charge: float = 0.0
    mini_boost: float = 0.0
    last_input: float = field(default_factory=time.perf_counter)
    laps: list = field(default_factory=list)
    last_lap_at: float = 0.0
    finish_time: float | None = None
    penalty: float = 0.0
    waiting: bool = False

    def release(self):
        self.keyboard.release()
        self.drift = self.boost = False

    def reset(self):
        self.racer.reset()
        self.release()
        self.keyboard.steer = self.keyboard.throttle = 0.0
        self.energy, self.drift_charge, self.mini_boost = 100.0, 0.0, 0.0
        self.laps, self.last_lap_at, self.finish_time, self.penalty = [], 0.0, None, 0.0
        self.waiting = False
        self.racer.env.kart.drifting = self.racer.env.kart.boosting = False

    def step(self, dt, race_laps, elapsed):
        if self.waiting or self.finish_time is not None:
            return
        if time.perf_counter() - self.last_input > 1.0:
            self.release()
        action = self.keyboard.action(dt)
        kart = self.racer.env.kart
        drifting = bool(self.drift and kart.speed > 4 and abs(action[0]) > 0.2)
        if drifting:
            self.drift_charge = min(1.6, self.drift_charge + dt)
        elif self.drift_charge:
            if self.drift_charge >= 0.7:
                self.mini_boost = 1.2
            self.drift_charge = 0.0
        self.mini_boost = max(0.0, self.mini_boost - dt)
        manual_boost = bool(self.boost and self.energy > 1 and action[1] > 0)
        self.energy = float(np.clip(self.energy + (-38 if manual_boost else 12) * dt, 0, 100))
        kart.drifting = drifting
        kart.boosting = bool((manual_boost or self.mini_boost > 0) and action[1] > 0)
        self.racer.step(action)
        # Count net travelled distance, not repeated forward/backward crossings
        # of the start line. Respawn never gives progress.
        progress = self.racer.env.total_progress
        length = self.racer.env.track.length
        if progress >= (len(self.laps) + 1) * length:
            self.laps.append(self.racer.env.t - self.last_lap_at)
            self.last_lap_at = self.racer.env.t
        if len(self.laps) >= race_laps:
            self.finish_time = elapsed + self.penalty
            self.release()
            kart.boosting = kart.drifting = False

    def state(self):
        return {
            **self.racer.state(self.id),
            "name": self.name,
            "color": self.color,
            "lap": len(self.laps),
            "boost": self.racer.env.kart.boosting,
            "drifting": self.racer.env.kart.drifting,
            "energy": round(self.energy, 1),
        }

    def hud(self):
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "lap": len(self.laps),
            "progress": max(0, self.racer.env.total_progress),
            "energy": round(self.energy, 1),
            "drift_charge": round(self.drift_charge / 1.6, 2),
            "finish_time": self.finish_time,
            "best": min(self.laps) if self.laps else None,
            "penalty": self.penalty,
            "waiting": self.waiting,
        }


class ArcadeServer:
    def __init__(self, laps=3, max_players=8, results_path=None):
        self.laps, self.max_players = laps, max_players
        self.players, self.clients = {}, {}
        self.counter = 0
        self.host_id = None
        self.phase, self.elapsed, self.countdown = "lobby", 0.0, 0.0
        self.dt = 1 / 30
        self.race_id, self.saved = "", False
        self.invite_url = None
        self.results_path = Path(results_path) if results_path else ROOT / "runs/arcade/races.csv"

    def join(self):
        if len(self.players) >= self.max_players:
            raise ValueError("Die Lobby ist voll (maximal acht Spieler).")
        self.counter += 1
        pid = f"p{self.counter}"
        racer = Racer(pid, {**GAME_ENV, "dt": self.dt, "wall_friction": 0.14})
        racer.env.record = False
        racer.env.kart = ArcadeKart()
        player = Player(pid, f"Spieler {self.counter}", COLORS[(self.counter - 1) % len(COLORS)], racer)
        player.reset()
        player.waiting = self.phase in ("countdown", "racing")
        self.players[pid] = player
        if self.host_id is None:
            self.host_id = pid
        return player

    def leave(self, pid):
        player = self.players.pop(pid, None)
        if player:
            player.release()
            player.racer.env.close()
        if self.host_id == pid:
            self.host_id = next(iter(self.players), None)
        if not self.players:
            self.phase, self.elapsed, self.countdown = "lobby", 0.0, 0.0
            self.counter = 0

    def start(self, pid):
        if pid != self.host_id:
            return False
        for player in self.players.values():
            player.reset()
        self.phase, self.elapsed, self.countdown = "countdown", 0.0, 3.0
        self.race_id, self.saved = uuid.uuid4().hex[:12], False
        return True

    def handle(self, pid, message):
        if not isinstance(message, dict) or pid not in self.players:
            return
        player = self.players[pid]
        kind = message.get("type")
        if kind == "keys":
            player.keyboard.set_keys({key: message.get(key) is True for key in player.keyboard.keys})
            player.drift, player.boost = message.get("drift") is True, message.get("boost") is True
            player.last_input = time.perf_counter()
        elif kind == "set_name":
            player.name = (
                "".join(c for c in str(message.get("name", "")) if c.isalnum() or c in " _-")[:24].strip()
                or player.name
            )
        elif kind == "control":
            action = message.get("action")
            if action == "start_race":
                self.start(pid)
            elif (
                action == "respawn"
                and self.phase == "racing"
                and not player.waiting
                and player.finish_time is None
            ):
                player.racer.respawn()
                player.release()
                player.keyboard.steer = player.keyboard.throttle = 0.0
                player.penalty += 3.0
                player.drift_charge = player.mini_boost = 0.0

    def tick(self):
        if self.phase == "countdown":
            self.countdown = max(0.0, self.countdown - self.dt)
            if self.countdown <= 1e-9:
                self.phase, self.countdown = "racing", 0.0
        elif self.phase == "racing":
            self.elapsed += self.dt
            active = [p for p in self.players.values() if not p.waiting]
            for player in active:
                player.step(self.dt, self.laps, self.elapsed)
            if not active or all(p.finish_time is not None for p in active) or self.elapsed >= 600:
                self.phase = "finished"
                self.save_results()
        return self.message()

    def message(self):
        board = [p.hud() for p in self.players.values()]
        board.sort(
            key=lambda p: (
                p["waiting"],
                p["finish_time"] is None,
                p["finish_time"] if p["finish_time"] is not None else -p["progress"],
            )
        )
        for i, row in enumerate(board):
            row["rank"] = i + 1
        return {
            "type": "states",
            "states": [p.state() for p in self.players.values()],
            "game": {
                "arcade": True,
                "phase": self.phase,
                "host_id": self.host_id,
                "countdown": round(self.countdown, 2),
                "elapsed": round(self.elapsed, 2),
                "race_laps": self.laps,
                "players": board,
                "policy": "Arcade Multiplayer",
            },
        }

    def save_results(self):
        if self.saved:
            return
        self.saved = True
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.results_path.exists()
        with self.results_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if new:
                writer.writerow(["race", "profile", "driver", "laps", "finish_s", "penalty_s"])
            for p in self.players.values():
                if not p.waiting:
                    writer.writerow([self.race_id, "arcade", p.name, len(p.laps), p.finish_time, p.penalty])

    async def handler(self, ws):
        try:
            player = self.join()
        except ValueError as exc:
            await ws.send(json.dumps({"type": "error", "message": str(exc)}))
            await ws.close()
            return
        self.clients[ws] = player.id
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "hello",
                        "mode": "game",
                        "arcade": True,
                        "player_id": player.id,
                        "host_id": self.host_id,
                        "track": "race_inn_roggwil",
                        "driver": player.name,
                        "policy": "Arcade Multiplayer",
                        "policies": [],
                        "schema": [],
                        "invite_url": self.invite_url,
                    }
                )
            )
            async for raw in ws:
                try:
                    self.handle(player.id, json.loads(raw))
                except (json.JSONDecodeError, TypeError):
                    continue
        finally:
            self.clients.pop(ws, None)
            self.leave(player.id)

    async def run(self):
        while True:
            start = time.perf_counter()
            message = self.tick()
            if self.clients:
                data = json.dumps(message, allow_nan=False)
                await asyncio.gather(*(ws.send(data) for ws in list(self.clients)), return_exceptions=True)
            await asyncio.sleep(max(0.001, self.dt - (time.perf_counter() - start)))


class ArcadeAssets(http.server.SimpleHTTPRequestHandler):
    """Serve only the viewer and track assets when the game is shared over LAN."""

    def send_head(self):
        path = Path(self.translate_path(urlsplit(self.path).path)).resolve()
        if not any(path.is_relative_to((ROOT / prefix).resolve()) for prefix in ("viewer", "kart_rl/data")):
            self.send_error(404)
            return None
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404)


async def serve(args):
    import websockets

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server = ArcadeServer(laps=args.race_laps)
    handler = functools.partial(ArcadeAssets, directory=str(ROOT))
    httpd = http.server.ThreadingHTTPServer((host, args.http), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://localhost:{args.http}/viewer/?game&arcade&ws={args.ws}"
    server.invite_url = url
    if args.lan:
        try:
            # UDP route selection only: no packet is sent to this documentation address.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                route.connect(("192.0.2.1", 9))
                address = route.getsockname()[0]
            server.invite_url = f"http://{address}:{args.http}/viewer/?game&arcade&ws={args.ws}"
        except OSError:
            pass
    try:
        async with websockets.serve(server.handler, host, args.ws, max_size=2048):
            print(f"Arcade ready: {url}", flush=True)
            print(f"Invite: {server.invite_url}", flush=True)
            if args.open:
                import webbrowser

                webbrowser.open(url)
            await server.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
        for player in server.players.values():
            player.racer.env.close()
