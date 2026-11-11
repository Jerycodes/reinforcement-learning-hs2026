"""Solo training on the same track, with the same tick and finish rules as a room."""

import json
from zipfile import ZipFile

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..game.race import CARS, RULES, TYRES, RaceRoom, advance_field


def observation(player):
    kart = player.racer.env.kart
    extras = [
        player.energy / 100,
        player.drift_charge / 1.6,
        player.mini_boost / 1.2,
        kart.accel / 14,
        kart.limit / 24,
        kart.handling,
        kart.grip,
        kart.drive,
    ]
    return np.concatenate((player.racer.obs, extras)).astype(np.float32)


class ArcadeEnv(gym.Env):
    def __init__(self, laps=3, *, legacy=False, previous=False):
        self.room = RaceRoom(laps, legacy=legacy, previous=previous)
        self.player = self.room.join()
        self.observation_space = spaces.Box(-10, 10, shape=observation(self.player).shape, dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        kart = self.player.racer.env.kart
        kart.configure(
            options.get("car", self.np_random.choice(list(CARS))),
            options.get("tyre", self.np_random.choice(list(TYRES))),
        )
        self.player.reset()
        self.elapsed = 0.0
        return observation(self.player), {}

    def step(self, action):
        env = self.player.racer.env
        previous = env.total_progress
        self.elapsed += self.room.dt
        controls = np.clip(action, -1, 1)
        if self.room.legacy:
            self.player.step(self.room.dt, self.room.laps, self.elapsed, controls)
        else:
            advance_field([self.player], [controls], self.room.dt, self.room.laps, self.elapsed)
        finished = self.player.finish_time is not None
        reward = env.total_progress - previous - self.room.dt * 0.2 - float(env.off_track)
        return (
            observation(self.player),
            reward + 10 * finished,
            finished,
            self.elapsed >= 600 and not finished,
            {
                "progress": env.total_progress,
                "wall_contact": env.off_track,
                "laps": len(self.player.laps),
            },
        )

    def close(self):
        self.player.racer.env.close()


def load_arcade_policy(path):
    # Inspect metadata before loading any serialized policy objects.
    with ZipFile(path) as archive:
        metadata = json.loads(archive.read("data"))
    if (
        metadata.get("arcade_rules") != RULES
        or metadata.get("observation_space", {}).get("_shape") != [38]
        or metadata.get("action_space", {}).get("_shape") != [4]
    ):
        raise ValueError(
            "Inkompatibles Modell: KartArcade-v2 mit 38 Beobachtungen und 4 Aktionen erforderlich."
        )
    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")
    return lambda player: model.predict(observation(player), deterministic=True)[0]
