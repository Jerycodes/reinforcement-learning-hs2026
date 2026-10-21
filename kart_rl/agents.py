"""Load trained policies (SB3 .zip or scratch-PPO .pt) as plain ``obs -> action`` callables."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .evaluation import Policy


def load_policy(path: str | Path, deterministic: bool = True) -> Policy:
    path = Path(path)
    if path.suffix == ".zip":
        from stable_baselines3 import PPO

        model = PPO.load(path, device="cpu")

        def policy(obs: np.ndarray) -> np.ndarray:
            a, _ = model.predict(obs, deterministic=deterministic)
            return np.asarray(a, dtype=np.float32)

        policy.name = f"sb3:{path.stem}"
        return policy
    if path.suffix == ".pt":
        import sys

        import torch

        try:
            from train.ppo_scratch import ActorCritic
        except ImportError:  # not running from the repository root
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from train.ppo_scratch import ActorCritic

        ck = torch.load(path, map_location="cpu", weights_only=False)
        p = ck["config"]["ppo"]
        model = ActorCritic(ck["obs_dim"], ck["act_dim"], tuple(p["hidden_sizes"]), p.get("activation", "tanh"),
                            float(p.get("log_std_init", -0.5)))
        model.load_state_dict(ck["model"])
        model.eval()

        def policy(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                a, _, _ = model.act(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=deterministic)
            return a.squeeze(0).clamp(-1, 1).numpy()

        policy.name = f"scratch:{path.stem}"
        return policy
    raise ValueError(f"unknown model format: {path} (expected .zip from SB3 or .pt from ppo_scratch)")


def baseline_policy(name: str = "raceline"):
    """Scripted drivers as policies. They need the env (not just the observation), so the
    returned object exposes ``bind(env)``; ``run_episode`` handles that automatically."""
    from .baselines import PurePursuitDriver
    from .raceline import RaceLine
    from .track import Track

    track = Track()
    if name == "centerline":
        driver = PurePursuitDriver(track)
    elif name == "raceline":
        line = RaceLine.load()
        driver = PurePursuitDriver(line, speed=line.speed, a_brake=4.0)
    else:
        raise ValueError(name)

    class _Scripted:
        def __init__(self, label):
            self.name = f"baseline:{label}"
            self.env = None

        def bind(self, env):
            self.env = env
            driver.reset()

        def __call__(self, obs):
            return driver.act(self.env)

    return _Scripted(name)
