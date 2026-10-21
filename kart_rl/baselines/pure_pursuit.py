"""Scripted reference driver: pure-pursuit steering + speed profile.

Not a learning agent. It follows a given closed line (the track centreline by
default, or a computed racing line) and is used to sanity-check the physics,
to produce reference lap times that the RL agents should beat, and as a
"someone is driving" demo for the viewers.
"""
from __future__ import annotations

import numpy as np

from ..path import ClosedPath, wrap_angle


class PurePursuitDriver:
    def __init__(self, line: ClosedPath, speed: np.ndarray | None = None, lookahead_min: float = 3.0,
                 lookahead_gain: float = 0.25, a_lat_max: float = 6.5, a_brake: float = 4.5,
                 v_max: float = 18.0, k_speed: float = 1.0, preview: float = 60.0):
        """``line``: closed path to follow (e.g. ``Track`` or ``RaceLine``).
        ``speed``: target speed per sample; if None a conservative profile is
        derived from the curvature (``a_lat_max``, ``a_brake``)."""
        self.line = line
        self.lookahead_min = lookahead_min
        self.lookahead_gain = lookahead_gain
        self.a_brake = a_brake
        self.a_lat_cap = 9.0  # lateral acceleration at which no brake force is left [m/s^2]
        self.v_max = v_max
        self.k_speed = k_speed
        self.preview = preview
        if speed is None:
            kappa = np.abs(line.curvature)
            v = np.minimum(np.sqrt(a_lat_max / np.maximum(kappa, 1e-3)), v_max)
            for _ in range(2):  # backward pass (periodic) so that the kart brakes early enough
                for i in range(line.n - 1, -1, -1):
                    j = (i + 1) % line.n
                    v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_brake * line.spacing))
            speed = v
        self.v_profile = np.asarray(speed, float)
        self.hint: int | None = None
        self.s = 0.0

    def reset(self):
        self.hint = None

    def act(self, env) -> np.ndarray:
        k = env.kart
        line = self.line
        self.s, _, self.hint = line.localize(k.x, k.y, self.hint)
        speed = k.speed
        # steering: aim at a point L ahead on the line
        L = self.lookahead_min + self.lookahead_gain * speed
        target = line.point_at(self.s + L)
        alpha = wrap_angle(np.arctan2(target[1] - k.y, target[0] - k.x) - k.yaw)
        delta = np.arctan2(2.0 * k.p.wheelbase * np.sin(alpha), max(L, 1.0))
        steer = float(np.clip(delta / k.p.max_steer, -1.0, 1.0))
        # speed: lowest profile value over the preview, each raised by the braking distance
        idx = int(self.s / line.spacing) % line.n
        n_prev = int(self.preview / line.spacing)
        ids = (idx + np.arange(n_prev)) % line.n
        dist = np.arange(n_prev) * line.spacing
        allowed = np.sqrt(self.v_profile[ids] ** 2 + 2 * self.a_brake * dist)
        v_target = float(min(allowed.min(), self.v_max))
        throttle = float(np.clip(self.k_speed * (v_target - speed), -1.0, 1.0))
        # friction circle: do not brake hard while the tyres are busy cornering
        # (uses the measured lateral acceleration of the kart)
        a_lat = abs(k.vx * k.yaw_rate)
        brake_cap = float(np.sqrt(max(1.0 - (a_lat / self.a_lat_cap) ** 2, 0.0)))
        throttle = max(throttle, -max(brake_cap, 0.15))
        return np.array([steer, throttle], dtype=np.float32)
