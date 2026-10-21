"""Kart RL: go-kart simulation and Gymnasium environments (Race-Inn Roggwil)."""
from gymnasium.envs.registration import register

from .kart import Kart, KartParams
from .track import Track

__version__ = "0.1.0"

register(id="KartLap-v0", entry_point="kart_rl.envs.lap_env:KartLapEnv")

__all__ = ["Kart", "KartParams", "Track"]
