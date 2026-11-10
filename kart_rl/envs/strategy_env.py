"""Lap-level pit strategy with the existing kart physics and a scripted driver.

One action drives the next lap. Pit service is an abstract time loss at the
start/finish line, not a simulated drive through the pit lane.
"""

import math
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..baselines import PurePursuitDriver
from .lap_env import KartLapEnv, LapEnvConfig


@dataclass
class StrategyConfig:
    laps: int = 8
    pit_loss_s: float = 12.0
    wear_per_lap: float = 0.18
    min_grip: float = 0.55
    max_time_s: float = 1200.0

    def __post_init__(self):
        if isinstance(self.laps, bool) or not isinstance(self.laps, int) or self.laps < 1:
            raise ValueError("laps must be a positive integer")
        for key in ("pit_loss_s", "wear_per_lap", "min_grip", "max_time_s"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if self.pit_loss_s < 0 or self.wear_per_lap < 0:
            raise ValueError("pit loss and wear must be nonnegative")
        if not 0 < self.min_grip <= 1 or self.max_time_s <= 0:
            raise ValueError("min_grip must be in (0, 1]; max_time_s must be positive")


class KartStrategyEnv(gym.Env):
    """Discrete actions: 0 continue, 1 change tyres before the next lap.

    Observation: fraction of laps remaining, wear, grip, elapsed/budget,
    stop count/laps, and whether pitting is currently allowed. Reward is
    minus elapsed seconds; failures additionally pay the unused time budget.
    Thus a crash cannot beat a completed race by ending early.
    """

    def __init__(self, config: StrategyConfig | dict | None = None):
        super().__init__()
        self.config = config if isinstance(config, StrategyConfig) else StrategyConfig(**(config or {}))
        c = self.config
        self.driving = KartLapEnv(
            LapEnvConfig(
                laps=c.laps,
                random_start=False,
                max_time=c.max_time_s,
            )
        )
        self.driver = PurePursuitDriver(self.driving.track, a_lat_max=3.0, a_brake=2.0)
        self.fresh_profile = self.driver.v_profile.copy()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0.0, 1.0, (6,), dtype=np.float32)
        self.done = True

    @property
    def grip(self):
        return 1.0 - (1.0 - self.config.min_grip) * self.wear

    def _observation(self):
        return np.array(
            [
                (self.config.laps - self.driving.laps_done) / self.config.laps,
                self.wear,
                self.grip,
                min(self.elapsed / self.config.max_time_s, 1.0),
                self.stops / self.config.laps,
                float(self.driving.laps_done > 0 and not self.done),
            ],
            dtype=np.float32,
        )

    def _info(self):
        return {
            "elapsed_s": self.elapsed,
            "wear": self.wear,
            "grip": self.grip,
            "pit_stops": self.stops,
            "laps_done": self.driving.laps_done,
            "finished": self.driving.laps_done == self.config.laps and self.reason == "laps",
            "episode_done": self.reason,
            "history": [dict(row) for row in self.history],
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.driving.kart.grip_scale = 1.0
        self.driving.reset(seed=seed)
        self.driver.reset()
        self.wear, self.elapsed, self.stops = 0.0, 0.0, 0
        self.history, self.reason, self.done = [], "", False
        return self._observation(), self._info()

    def step(self, action):
        if self.done:
            raise RuntimeError("reset before stepping a finished race")
        if not self.action_space.contains(action):
            raise ValueError("action must be 0 (continue) or 1 (pit)")
        c, env = self.config, self.driving
        start = self.elapsed
        pit = bool(action == 1 and env.laps_done > 0)
        if pit:
            # Charge service to both the race clock and this lap's clock.
            service = min(c.pit_loss_s, c.max_time_s - self.elapsed)
            self.elapsed += service
            env.t += service
            self.stops += 1
            self.wear = 0.0
        wear_start = self.wear
        target_lap = env.laps_done + 1
        terminated, truncated = False, self.elapsed >= c.max_time_s
        info = {"episode_done": "time"} if truncated else {}
        while not (terminated or truncated) and env.laps_done < target_lap:
            env.kart.grip_scale = self.grip
            # Same controller for every strategy, slower as grip falls.
            self.driver.v_profile = self.fresh_profile * self.grip
            self.driver.a_brake = 2.0 * self.grip
            self.driver.a_lat_cap = 9.0 * self.grip
            progress = env.total_progress
            _, _, terminated, truncated, info = env.step(self.driver.act(env))
            distance = max(0.0, env.total_progress - progress)
            self.wear = min(1.0, self.wear + c.wear_per_lap * distance / env.track.length)
            self.elapsed = env.t
        self.reason = info.get("episode_done", "")
        self.done = bool(terminated or truncated)
        duration = self.elapsed - start
        self.history.append(
            {
                "lap": target_lap,
                "duration_s": duration,
                "pit": pit,
                "wear_start": wear_start,
                "wear_end": self.wear,
                "completed": env.laps_done >= target_lap,
            }
        )
        reward = -duration
        if self.done and self.reason != "laps":
            reward -= max(0.0, c.max_time_s - self.elapsed)
        return self._observation(), float(reward), bool(terminated), bool(truncated), self._info()

    def close(self):
        self.driving.close()

    def configuration(self):
        return asdict(self.config)
