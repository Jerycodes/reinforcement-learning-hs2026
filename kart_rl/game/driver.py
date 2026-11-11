"""Shared kart and keyboard adapters."""

import numpy as np

from ..envs import KartLapEnv, LapEnvConfig

GAME_ENV = {
    "random_start": False,
    "laps": 999,
    "max_time": 36000.0,
    "off_track_mode": "penalty",
    "wall_friction": 0.1,
    "stuck_time": 1e9,
    "wrong_way_distance": 1e9,
}


class KeyboardDriver:
    """Turns held keys into a smooth (steer, throttle) action; feels like a game controller."""

    def __init__(self, steer_rate: float = 3.5, steer_return: float = 6.0, throttle_rate: float = 5.0):
        self.keys = {"left": False, "right": False, "up": False, "down": False}
        self.steer = 0.0
        self.throttle = 0.0
        self.steer_rate, self.steer_return, self.throttle_rate = steer_rate, steer_return, throttle_rate

    def set_keys(self, keys: dict):
        for k in self.keys:
            if k in keys:
                self.keys[k] = bool(keys[k])

    def release(self):
        for k in self.keys:
            self.keys[k] = False

    def action(self, dt: float) -> np.ndarray:
        target = (1.0 if self.keys["left"] else 0.0) - (1.0 if self.keys["right"] else 0.0)  # + = left
        rate = self.steer_rate if target != 0 else self.steer_return
        self.steer += float(np.clip(target - self.steer, -rate * dt, rate * dt))
        target = (1.0 if self.keys["up"] else 0.0) - (1.0 if self.keys["down"] else 0.0)
        self.throttle += float(
            np.clip(target - self.throttle, -self.throttle_rate * dt, self.throttle_rate * dt)
        )
        return np.array([self.steer, self.throttle], dtype=np.float32)


class Racer:
    """One kart in the game: env + bookkeeping of laps and wall contacts."""

    def __init__(self, name: str, env_cfg: dict):
        self.name = name
        self.env = KartLapEnv(LapEnvConfig.from_dict(env_cfg), record=True)
        self.obs = None
        self.reset()

    def reset(self):
        self.obs, _ = self.env.reset()
        self.laps: list[dict] = []  # {"time", "valid"}
        self.contact_in_lap = False
        self.last_action = np.zeros(2, dtype=np.float32)

    def step(self, action: np.ndarray) -> dict | None:
        self.obs, _, _, _, info = self.env.step(action)
        self.last_action = np.asarray(action, dtype=np.float32)
        if info.get("wall_contact"):
            self.contact_in_lap = True
        done_lap = None
        if "lap_time" in info:
            done_lap = {"time": round(float(info["lap_time"]), 2), "valid": not self.contact_in_lap}
            self.laps.append(done_lap)
            self.contact_in_lap = False
        return done_lap

    def respawn(self):
        """Back onto the centreline at the current position, standing still (lap stays invalid)."""
        env = self.env
        x, y, yaw = env.track.start_pose(env.s, 0.0)
        env.kart.reset(x, y, yaw, 0.0)
        env.s, env.lateral, env.idx = env.track.localize(x, y)
        env.off_track = False
        self.contact_in_lap = True

    @property
    def best(self) -> float | None:
        valid = [lap["time"] for lap in self.laps if lap["valid"]]
        return min(valid) if valid else None

    def state(self, kart_id: str) -> dict:
        k = self.env.kart
        return {
            "id": kart_id,
            "t": round(self.env.t, 3),
            "x": round(float(k.x), 3),
            "y": round(float(k.y), 3),
            "yaw": round(float(k.yaw), 4),
            "vx": round(float(k.vx), 3),
            "vy": round(float(k.vy), 3),
            "steer": round(float(k.steer), 4),
            "throttle": round(float(self.last_action[1]), 3),
            "s": round(self.env.s, 2),
            "lap": self.env.laps_done,
            "off_track": bool(self.env.off_track),
        }

    def hud(self) -> dict:
        return {
            "lap": self.env.laps_done + 1,
            "current": round(self.env.t - self.env.lap_start_t, 2),
            "last": self.laps[-1]["time"] if self.laps else None,
            "last_valid": self.laps[-1]["valid"] if self.laps else None,
            "best": self.best,
            "laps": self.laps[-8:],
            "contact": self.contact_in_lap,
        }
