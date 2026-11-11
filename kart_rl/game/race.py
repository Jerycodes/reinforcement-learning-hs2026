"""Headless arcade race rules shared by network sessions and Gymnasium."""

import csv
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..kart import Kart, KartParams
from .collisions import resolve, swept_contact
from .driver import GAME_ENV, KeyboardDriver, Racer

ROOT = Path(__file__).resolve().parents[2]

RULES = "KartArcade-v1"
BOT_LEVELS = {"Club": 0.76, "Sport": 0.90, "Pro": 1.0}

AVATARS = ("Fuchs", "Panda", "Luchs", "Drache")

COLORS = [0xFF5A36, 0x3FD0FF, 0xFFB000, 0x7CFF6B, 0xD36BFF, 0xFF7FBB, 0xFFFFFF, 0x41E4CB]


# acceleration (m/s²), top speed (m/s), steering multiplier
CARS = {"Club": (12, 22, 1), "Sprint": (14, 20, 1.08), "GT": (10, 24, 0.92)}
# corner grip, acceleration / rolling performance
TYRES = {"Standard": (1, 1), "Grip": (1.16, 0.92), "Speed": (0.89, 1.08)}


class ArcadeKart(Kart):
    """Kinematic handling for the game only; training physics stays unchanged."""

    def __init__(self, legacy=False):
        super().__init__(
            KartParams(
                max_steer=0.6, steer_rate=10, length=1.9 if legacy else 1.55, width=1.4 if legacy else 1.02
            )
        )
        self.configure("Club", "Standard")
        self.drifting = False
        self.boosting = False

    def configure(self, car, tyre):
        self.car, self.tyre = car, tyre
        self.accel, self.limit, self.handling = CARS[car]
        self.grip, self.drive = TYRES[tyre]

    def step(self, steer_cmd, throttle_cmd, dt):
        steer = float(np.clip(steer_cmd, -1, 1))
        throttle = float(np.clip(throttle_cmd, -1, 1))
        x, y, yaw, vx, vy, _, _ = self.state
        acceleration = (
            ((self.accel + (8 if self.boosting else 0)) * self.drive) * throttle
            if throttle > 0
            else 22 * throttle
        )
        if abs(throttle) < 0.05:
            acceleration -= 3.5
        limit = self.limit + (6 if self.boosting else 0)
        vx = float(np.clip(vx + acceleration * dt, 0, max(limit, vx - 8 * dt)))
        yaw_rate = steer * 3.5 * self.handling * self.grip / (1 + vx / 18) * min(vx / 2.5, 1)
        if self.drifting:
            yaw_rate *= 1.2
        lateral_target = -steer * vx * 0.2 / self.grip if self.drifting else 0.0
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
    ready: bool = False
    bot: bool = False
    avatar: str = "Fuchs"
    level: str = "Sport"
    passing_lane: float = 0.0
    best_flying: float | None = None
    lap_penalty: float = 0.0

    def release(self):
        self.keyboard.release()
        self.drift = self.boost = False

    def reset(self):
        self.racer.reset()
        self.release()
        self.keyboard.steer = self.keyboard.throttle = 0.0
        self.energy, self.drift_charge, self.mini_boost = 100.0, 0.0, 0.0
        self.laps, self.last_lap_at, self.finish_time, self.penalty = [], 0.0, None, 0.0
        self.best_flying = None
        self.passing_lane = self.lap_penalty = 0.0
        self.waiting = False
        self.racer.env.kart.drifting = self.racer.env.kart.boosting = False

    def step(self, dt, race_laps, elapsed, controls=None, *, defer_laps=False):
        if self.waiting or self.finish_time is not None:
            return
        if time.perf_counter() - self.last_input > 1.0:
            self.release()
        if controls is None:
            action = self.keyboard.action(dt)
        else:
            action = controls[:2]
            self.drift, self.boost = controls[2] > 0, controls[3] > 0
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
        if not defer_laps:
            self.score(race_laps, elapsed)

    def score(self, race_laps, elapsed):
        kart = self.racer.env.kart
        # Count net travelled distance, not repeated forward/backward crossings
        # of the start line. Respawn never gives progress.
        progress = self.racer.env.total_progress
        length = self.racer.env.track.length
        if progress >= (len(self.laps) + 1) * length:
            self.laps.append(self.racer.env.t - self.last_lap_at)
            if len(self.laps) > 1 and self.penalty == self.lap_penalty:
                self.best_flying = min(self.best_flying or math.inf, self.laps[-1])
            self.lap_penalty = self.penalty
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
            "avatar": self.avatar,
            "car": self.racer.env.kart.car,
            "tyre": self.racer.env.kart.tyre,
            "lap": len(self.laps),
            "boost": self.racer.env.kart.boosting,
            "parked": self.finish_time is not None,
            "length": self.racer.env.kart.p.length,
            "width": self.racer.env.kart.p.width,
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
            "level": self.level,
            "best_flying": self.best_flying,
            "best": min(self.laps) if self.laps else None,
            "penalty": self.penalty,
            "waiting": self.waiting,
            "ready": self.ready,
            "bot": self.bot,
            "avatar": self.avatar,
            "car": self.racer.env.kart.car,
            "tyre": self.racer.env.kart.tyre,
        }


class RaceRoom:
    def __init__(self, laps=3, max_players=5, results_path=None, *, legacy=False):
        self.legacy = legacy
        self.laps, self.max_players = laps, max_players
        self.players = {}
        self.counter = 0
        self.host_id = None
        self.phase, self.elapsed, self.countdown = "lobby", 0.0, 0.0
        self.dt = 1 / 30
        self.race_id, self.saved = "", False
        self.contacts = set()
        self.grid = []
        self.departed = []
        self.results = []
        self.results_path = Path(results_path) if results_path else ROOT / "runs/arcade/races.csv"

    def join(self, bot=False, profile=None):
        if len(self.players) >= self.max_players:
            raise ValueError("Die Lobby ist voll (kein freier Startplatz).")
        if profile is not None:
            if not isinstance(profile, dict):
                raise ValueError("Fahrerprofil fehlt.")
            name = profile.get("name")
            if (
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 24
                or not all(c.isalnum() or c in " _-" for c in name)
            ):
                raise ValueError("Name: 1–24 Buchstaben, Ziffern, Leerzeichen, - oder _.")
            if any(p.name.casefold() == name.strip().casefold() for p in self.players.values()):
                raise ValueError("Dieser Fahrername ist schon vergeben.")
            if (
                profile.get("avatar") not in AVATARS
                or profile.get("car") not in tuple(CARS)
                or profile.get("tyre") not in tuple(TYRES)
            ):
                raise ValueError("Unbekannte Auswahl.")
        self.counter += 1
        pid = f"p{self.counter}"
        racer = Racer(pid, {**GAME_ENV, "dt": self.dt, "wall_friction": 0.14})
        racer.env.record = False
        racer.env.kart = ArcadeKart(self.legacy)
        player = Player(pid, f"Spieler {self.counter}", COLORS[(self.counter - 1) % len(COLORS)], racer)
        player.reset()
        player.waiting = self.phase in ("countdown", "racing")
        player.bot = player.ready = bot
        if profile:
            player.name = profile["name"].strip()
            player.avatar = profile["avatar"]
            racer.env.kart.configure(profile["car"], profile["tyre"])
        if bot:
            player.avatar = random.choice(AVATARS)
            player.name = f"Bot {self.counter}"
        while any(p.name.casefold() == player.name.casefold() for p in self.players.values()):
            player.name += "_"
        self.players[pid] = player
        if self.host_id is None:
            self.host_id = pid
        return player

    def leave(self, pid):
        player = self.players.pop(pid, None)
        if player:
            if self.phase in ("countdown", "racing") and not player.waiting:
                self.departed.append(player.hud())
            player.release()
            player.racer.env.close()
        if self.host_id == pid:
            self.host_id = next((p.id for p in self.players.values() if not p.bot), None)
        if not self.players:
            self.phase, self.elapsed, self.countdown = "lobby", 0.0, 0.0
            self.counter = 0

    def start(self, pid):
        if (
            pid != self.host_id
            or self.phase not in ("lobby", "finished")
            or not self.players
            or not all(p.ready for p in self.players.values())
        ):
            return False
        self.contacts.clear()
        self.departed.clear()
        self.results.clear()
        order = list(self.players.values())
        random.shuffle(order)
        self.grid = []
        for index, player in enumerate(order):
            player.reset()
            env = player.racer.env
            # Roggwil: 22 m straight after the stripe. Pairs sit 3.8 m apart lengthwise.
            grid = 12 - 3.8 * (index // 2) - 0.85 * (index % 2)
            lateral = 1.55 if index % 2 == 0 else -1.55
            x, y, yaw = env.track.start_pose(grid, lateral)
            self.grid.append(
                {"position": index + 1, "id": player.id, "name": player.name, "x": x, "y": y, "yaw": yaw}
            )
            env.kart.reset(x, y, yaw, 0)
            env.s, env.lateral, env.idx = env.track.localize(x, y)
            env.total_progress = env.max_progress = env.s
            player.racer.obs = env._observe()
        self.phase, self.elapsed, self.countdown = "countdown", 0.0, 7.0
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
        elif kind in ("set_name", "setup", "ready", "bot"):
            if self.phase not in ("lobby", "finished"):
                return "Auswahl erst vor dem nächsten Start möglich."
            if kind == "set_name" or (kind == "ready" and "name" in message):
                raw = message.get("name")
                name = (
                    "".join(c for c in raw if c.isalnum() or c in " _-")[:24].strip()
                    if isinstance(raw, str)
                    else ""
                )
                if not name or any(
                    p.id != pid and p.name.casefold() == name.casefold() for p in self.players.values()
                ):
                    return "Bitte einen freien Spielernamen wählen."
                player.name, player.ready = name, False
            if kind == "ready":
                player.ready = message.get("value") is True
            elif kind == "setup":
                car, tyre = message.get("car"), message.get("tyre")
                if car not in tuple(CARS) or tyre not in tuple(TYRES):
                    return "Unbekannte Auswahl."
                player.racer.env.kart.configure(car, tyre)
                player.ready = False
            elif kind == "bot" and pid == self.host_id:
                if message.get("add") is True:
                    if len(self.players) >= self.max_players:
                        return "Alle Startplätze sind belegt."
                    level = message.get("level", "Sport")
                    if level not in tuple(BOT_LEVELS):
                        return "Unbekannte Bot-Stufe."
                    self.join(bot=True).level = level
                else:
                    bots = [p.id for p in self.players.values() if p.bot]
                    if bots:
                        self.leave(bots[-1])
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
                env = player.racer.env
                old = env.kart.state.copy()
                target = env.track.start_pose(env.s)
                env.kart.state[:3] = target
                occupied = any(
                    swept_contact(env.kart, p.racer.env.kart, env.kart.state, p.racer.env.kart.state)
                    for p in self.players.values()
                    if p is not player and not p.waiting and p.finish_time is None
                )
                env.kart.state[:] = old
                if occupied:
                    return "Rücksetzpunkt belegt."
                player.racer.respawn()
                player.release()
                player.keyboard.steer = player.keyboard.throttle = 0.0
                player.penalty += 3.0
                player.drift_charge = player.mini_boost = 0.0

    def tick(self, bot_policy=None):
        if self.phase == "countdown":
            self.countdown = max(0.0, self.countdown - self.dt)
            if self.countdown <= 1e-9:
                self.phase, self.countdown = "racing", 0.0
        elif self.phase == "racing":
            self.elapsed += self.dt
            active = [p for p in self.players.values() if not p.waiting]
            moving = [p for p in active if p.finish_time is None]
            actions = [
                None if not p.bot else bot_policy(p) if bot_policy else rule_action(p, moving) for p in moving
            ]
            advance_field(moving, actions, self.dt, self.laps, self.elapsed)
            if not active or all(p.finish_time is not None for p in active) or self.elapsed >= 600:
                self.phase = "finished"
                entrants = [p.hud() for p in active] + self.departed
                self.results = self.ranked(entrants)
                self.save_results()
                for player in self.players.values():
                    player.ready = player.bot
        return self.message()

    @staticmethod
    def ranked(board):
        board.sort(
            key=lambda p: (
                p["waiting"],
                p["finish_time"] is None,
                p["finish_time"] if p["finish_time"] is not None else -p["progress"],
            )
        )
        for i, row in enumerate(board):
            row["rank"] = i + 1
        return board

    def message(self):
        board = self.ranked([p.hud() for p in self.players.values()])
        return {
            "type": "states",
            "states": [p.state() for p in self.players.values() if not p.waiting],
            "game": {
                "arcade": True,
                "phase": self.phase,
                "race_id": self.race_id,
                "rules": RULES,
                "grid": self.grid,
                "host_id": self.host_id,
                "countdown": round(self.countdown, 2),
                "elapsed": round(self.elapsed, 2),
                "race_laps": self.laps,
                "players": board,
                "results": self.results,
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
            for row in self.results:
                writer.writerow(
                    [self.race_id, "arcade", row["name"], row["lap"], row["finish_time"], row["penalty"]]
                )


def advance_field(players, actions, dt, laps, elapsed):
    previous = [p.racer.env.kart.state.copy() for p in players]
    origins = [(p.racer.env.s, p.racer.env.total_progress, p.racer.env.max_progress) for p in players]
    for player, action in zip(players, actions):
        player.step(dt, laps, elapsed, action, defer_laps=True)
    changed = resolve(players, previous)
    for i, p in enumerate(players):
        env = p.racer.env
        if i in changed or env.off_track:
            env.s, env.lateral, env.idx = env.track.localize(env.kart.x, env.kart.y, hint=env.idx)
            s, progress, peak = origins[i]
            length = env.track.length
            env.total_progress = progress + (env.s - s + length / 2) % length - length / 2
            env.max_progress = max(peak, env.total_progress)
            p.racer.obs = env._observe()
        p.score(laps, elapsed)


def rule_action(player, opponents=()):
    """Preview curvature, cut toward the apex and choose a clear passing lane."""
    env = player.racer.env
    kart, track = env.kart, env.track
    skill = BOT_LEVELS[player.level]
    look = 3.0 + kart.vx * 0.27
    ahead = np.array([look, look + 4, look + 10, look + 18])
    bends = track.curvature_ahead(env.s, ahead)
    corner = bends[0]
    blockers = [
        p
        for p in opponents
        if p is not player
        and 0 < (p.racer.env.s - env.s) % track.length < max(7, kart.vx * 0.85)
        and abs(p.racer.env.lateral - env.lateral) < 1.8
    ]
    if blockers:
        closest = min(blockers, key=lambda p: (p.racer.env.s - env.s) % track.length)
        if not player.passing_lane:
            player.passing_lane = -1.45 if closest.racer.env.lateral >= 0 else 1.45
    elif not any(p is not player and abs(p.racer.env.s - env.s) < 5 for p in opponents):
        player.passing_lane = 0
    offset = player.passing_lane or float(np.clip(corner * 13, -0.95, 0.95))
    target = track.start_pose(env.s + look, offset)
    heading = math.atan2(target[1] - kart.y, target[0] - kart.x)
    error = (heading - kart.yaw + math.pi) % (2 * math.pi) - math.pi
    steer = np.clip(error * (2.4 + kart.vx * 0.035), -1, 1)
    # Lateral acceleration budget plus braking distance to each preview point.
    corner_speeds = np.sqrt(18 * skill * kart.grip / np.maximum(abs(bends), 0.015))
    limits = np.sqrt(corner_speeds**2 + 2 * 13 * np.maximum(ahead - look, 0))
    straight = max(abs(bends[:3])) < 0.035 and abs(error) < 0.16
    boost = straight and skill >= 0.88 and player.energy > 5
    target_speed = min((kart.limit + 6 * boost) * skill, min(limits), 26 / (1 + 1.7 * abs(error)))
    drift = skill >= 0.88 and abs(steer) > 0.48 and 8 < kart.vx < 18
    return np.array(
        [steer, np.clip((target_speed - kart.vx) * 1.7, -1, 1), 1 if drift else -1, 1 if boost else -1],
        dtype=np.float32,
    )
