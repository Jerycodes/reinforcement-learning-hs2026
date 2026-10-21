"""Evaluation helpers shared by the trainers, the evaluate script and the viewers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .envs import KartLapEnv, LapEnvConfig

Policy = Callable[[np.ndarray], np.ndarray]  # observation -> action in [-1, 1]^2


@dataclass
class EvalResult:
    lap_times: list[float] = field(default_factory=list)
    episode_done: str = ""
    total_reward: float = 0.0
    steps: int = 0
    distance: float = 0.0
    trajectory: dict = field(default_factory=dict)

    @property
    def best_lap(self) -> float | None:
        flying = self.lap_times[1:] if len(self.lap_times) > 1 else self.lap_times
        return min(flying) if flying else None

    @property
    def standing_lap(self) -> float | None:
        return self.lap_times[0] if self.lap_times else None


def run_episode(env: KartLapEnv, policy: Policy, seed: int | None = None, options: dict | None = None,
                frame_callback: Callable[[KartLapEnv], None] | None = None) -> EvalResult:
    """Run one episode with a deterministic policy; returns lap times etc."""
    obs, _ = env.reset(seed=seed, options=options)
    if hasattr(policy, "bind"):  # scripted drivers need the env state
        policy.bind(env)
    res = EvalResult()
    done = False
    while not done:
        action = policy(obs)
        obs, r, term, trunc, info = env.step(action)
        res.total_reward += r
        res.steps += 1
        done = term or trunc
        if frame_callback is not None:
            frame_callback(env)
    res.lap_times = list(info["lap_times"])
    res.episode_done = info.get("episode_done", "")
    res.distance = info["total_progress"]
    res.trajectory = env.get_trajectory() if env.record else {}
    return res


def evaluate_lap_time(policy: Policy, env_config: dict | LapEnvConfig | None = None, laps: int = 2,
                      max_time: float | None = None) -> EvalResult:
    """Standing start at the line, drive ``laps`` laps deterministically."""
    cfg = env_config if isinstance(env_config, LapEnvConfig) else LapEnvConfig.from_dict(env_config)
    cfg.random_start = False
    cfg.laps = laps
    cfg.max_time = max_time if max_time else 90.0 * laps
    env = KartLapEnv(cfg)
    return run_episode(env, policy)


def summarize(res: EvalResult, references: dict[str, float] | None = None) -> str:
    lines = [f"episode end: {res.episode_done}, steps {res.steps}, distance {res.distance:.0f} m, "
             f"return {res.total_reward:.1f}"]
    if res.lap_times:
        lines.append("lap times: " + ", ".join(f"{t:.2f} s" for t in res.lap_times)
                     + (f"  (best flying lap {res.best_lap:.2f} s)" if len(res.lap_times) > 1 else ""))
    else:
        lines.append("no complete lap")
    for name, t in (references or {}).items():
        if res.best_lap is not None:
            lines.append(f"vs {name} ({t:.1f} s): {res.best_lap - t:+.2f} s")
    return "\n".join(lines)


REFERENCE_LAP_TIMES = {"Mittellinien-Fahrer": 71.6, "Ideallinien-Fahrer": 54.1}


def load_reference_lap_times() -> dict[str, float]:
    """Reference lap times measured by tools/compute_raceline.py (falls back to constants)."""
    try:
        import json

        from .track import DATA_DIR

        d = json.loads((DATA_DIR / "race_inn_roggwil_raceline.json").read_text())
        meta = d.get("meta", {})
        out = {}
        if meta.get("centerline_measured_lap_time_s"):
            out["Mittellinien-Fahrer"] = float(meta["centerline_measured_lap_time_s"])
        if meta.get("measured_lap_time_s"):
            out["Ideallinien-Fahrer"] = float(meta["measured_lap_time_s"])
        if meta.get("theoretical_lap_time_s"):
            out["Modell-Optimum (Schätzung)"] = float(meta["theoretical_lap_time_s"])
        return out or dict(REFERENCE_LAP_TIMES)
    except Exception:  # pragma: no cover
        return dict(REFERENCE_LAP_TIMES)


def save_replay(path: str | Path, trajectory: dict, meta: dict | None = None):
    """Write a replay JSON for the 3-D viewer."""
    import json

    d = {"meta": meta or {}, "trajectory": trajectory}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d))
