"""Kart vehicle model.

A dynamic single-track ("bicycle") model with kart parameters: the two wheels
of each axle are lumped together, the front axle steers, the rear axle drives
and brakes (rental karts have rear brakes only). Lateral tyre forces saturate
smoothly at ``mu * Fz`` and the rear axle uses a friction ellipse, so the kart
can understeer, oversteer and spin when it is driven over the limit.

Simplifications (documented on purpose, this is a training simulator, not a
vehicle-dynamics tool): flat track, no load transfer, no engine/clutch
dynamics, no differential effects, no rolling of the chassis.

State vector: x, y [m], yaw [rad], vx, vy [m/s] (body frame), yaw rate [rad/s],
steer [rad] (actual wheel angle, rate limited).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

G = 9.81


@dataclass
class KartParams:
    mass: float = 180.0            # kart + driver [kg]
    inertia_z: float = 45.0        # yaw inertia [kg m^2]
    lf: float = 0.62               # centre of gravity -> front axle [m]
    lr: float = 0.43               # centre of gravity -> rear axle [m]
    width: float = 1.40            # overall width (rendering / off-track margin) [m]
    length: float = 1.90           # overall length (rendering) [m]
    max_steer: float = 0.40        # max wheel angle [rad] (~23 deg)
    steer_rate: float = 3.0        # max steering speed [rad/s]
    mu: float = 1.00               # peak tyre friction coefficient (baseline compound)
    cornering_stiffness_f: float = 12000.0  # [N/rad] per axle
    cornering_stiffness_r: float = 16000.0  # [N/rad] per axle
    drive_force_max: float = 720.0  # max drive force at the rear axle [N]
    power_max: float = 5300.0       # max drive power [W] (~7 hp)
    brake_force_max: float = 1000.0  # max brake force, rear axle only [N]
    drag_coeff: float = 0.48        # 0.5 * rho * Cd * A [kg/m]
    roll_resistance: float = 0.015  # rolling resistance coefficient
    speed_limit: float = 18.0       # electronic limiter [m/s] = 65 km/h (<= 0 disables it)
    v_blend: float = 3.0            # below this speed blend towards the kinematic model [m/s]

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @classmethod
    def from_dict(cls, d: dict | None) -> "KartParams":
        p = cls()
        for k, v in (d or {}).items():
            if not hasattr(p, k):
                raise KeyError(f"unknown kart parameter '{k}'")
            setattr(p, k, float(v))
        return p

    def to_dict(self) -> dict:
        return asdict(self)


class Kart:
    """Single kart, integrated with semi-implicit Euler at ``dt_sim``."""

    STATE_NAMES = ("x", "y", "yaw", "vx", "vy", "yaw_rate", "steer")

    def __init__(self, params: KartParams | None = None, dt_sim: float = 0.01):
        self.p = params or KartParams()
        self.dt_sim = dt_sim
        self.state = np.zeros(7)
        # per-step modifiers used by the race / boost phases
        self.grip_scale = 1.0   # multiplies mu (tyre compound and wear)
        self.power_scale = 1.0  # multiplies drive force and power (boost)
        self.last = {}          # diagnostics of the last sub-step

    # ------------------------------------------------------------------ helpers
    def reset(self, x: float, y: float, yaw: float, speed: float = 0.0):
        self.state[:] = (x, y, yaw, speed, 0.0, 0.0, 0.0)
        self.last = {}

    @property
    def x(self): return self.state[0]
    @property
    def y(self): return self.state[1]
    @property
    def yaw(self): return self.state[2]
    @property
    def vx(self): return self.state[3]
    @property
    def vy(self): return self.state[4]
    @property
    def yaw_rate(self): return self.state[5]
    @property
    def steer(self): return self.state[6]
    @property
    def speed(self) -> float:
        return float(np.hypot(self.state[3], self.state[4]))

    # --------------------------------------------------------------------- step
    def step(self, steer_cmd: float, throttle_cmd: float, dt: float):
        """Advance by ``dt`` seconds (several sub-steps of ``dt_sim``).

        steer_cmd in [-1, 1] (fraction of max steer, + = left),
        throttle_cmd in [-1, 1] (+ = drive, - = brake).
        """
        n = max(1, int(round(dt / self.dt_sim)))
        h = dt / n
        for _ in range(n):
            self._substep(float(np.clip(steer_cmd, -1, 1)), float(np.clip(throttle_cmd, -1, 1)), h)

    def _substep(self, steer_cmd: float, throttle_cmd: float, h: float):
        p = self.p
        x, y, yaw, vx, vy, r, delta = self.state
        mu = p.mu * self.grip_scale

        # steering actuator with rate limit
        target = steer_cmd * p.max_steer
        delta += np.clip(target - delta, -p.steer_rate * h, p.steer_rate * h)

        # axle loads (static)
        fz_f = p.mass * G * p.lr / p.wheelbase
        fz_r = p.mass * G * p.lf / p.wheelbase

        # longitudinal force at the rear axle
        if throttle_cmd >= 0.0:
            f_drive_cap = min(p.drive_force_max, p.power_max / max(vx, 1.0)) * self.power_scale
            if p.speed_limit > 0 and vx > p.speed_limit:
                f_drive_cap = 0.0
            fx_r = throttle_cmd * f_drive_cap
        else:
            fx_r = throttle_cmd * p.brake_force_max if vx > 0.05 else 0.0
        fx_r = float(np.clip(fx_r, -mu * fz_r, mu * fz_r))

        # slip angles (vx clamped to keep atan2 well behaved at low speed)
        vx_eff = max(vx, 0.5)
        alpha_f = np.arctan2(vy + p.lf * r, vx_eff) - delta
        alpha_r = np.arctan2(vy - p.lr * r, vx_eff)

        # lateral forces: smooth saturation, rear with friction ellipse
        fy_f_max = mu * fz_f
        fy_f = -fy_f_max * np.tanh(p.cornering_stiffness_f * alpha_f / fy_f_max)
        fy_r_max = mu * fz_r * np.sqrt(max(1.0 - (fx_r / (mu * fz_r)) ** 2, 0.04))
        fy_r = -fy_r_max * np.tanh(p.cornering_stiffness_r * alpha_r / fy_r_max)

        # resistances
        f_drag = p.drag_coeff * vx * vx
        f_roll = p.roll_resistance * p.mass * G if vx > 0.05 else 0.0

        # body-frame accelerations
        ax = (fx_r - fy_f * np.sin(delta) - f_drag - f_roll) / p.mass + vy * r
        ay = (fy_f * np.cos(delta) + fy_r) / p.mass - vx * r
        r_dot = (p.lf * fy_f * np.cos(delta) - p.lr * fy_r) / p.inertia_z

        vx = max(vx + ax * h, 0.0)
        if vx < 0.05 and throttle_cmd <= 0.0:
            vx = 0.0  # standstill (no creeping from numerical residuals)
        vy = vy + ay * h
        r = r + r_dot * h

        # blend towards the kinematic model at low speed (the dynamic model is
        # ill-conditioned when the kart is almost standing still)
        if vx < p.v_blend:
            w = vx / p.v_blend
            r_kin = vx * np.tan(delta) / p.wheelbase
            r = w * r + (1.0 - w) * r_kin
            vy = w * vy + (1.0 - w) * p.lr * r_kin

        # world-frame integration
        x += (vx * np.cos(yaw) - vy * np.sin(yaw)) * h
        y += (vx * np.sin(yaw) + vy * np.cos(yaw)) * h
        yaw = (yaw + r * h + np.pi) % (2 * np.pi) - np.pi

        self.state[:] = (x, y, yaw, vx, vy, r, delta)
        self.last = {"ax": ax, "ay": ay + vx * r, "fx_r": fx_r, "fy_f": fy_f, "fy_r": fy_r,
                     "alpha_f": alpha_f, "alpha_r": alpha_r, "slip": np.arctan2(vy, vx_eff)}

    # ------------------------------------------------------------------ geometry
    def footprint(self) -> np.ndarray:
        """Corners of the kart rectangle in world coordinates (4, 2)."""
        p = self.p
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        half_l, half_w = p.length / 2, p.width / 2
        local = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + (self.x, self.y)
