"""Tests for the optional 'learn from the human' add-on and the game server."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kart_rl.baselines import PurePursuitDriver  # noqa: E402
from kart_rl.envs import KartLapEnv, LapEnvConfig  # noqa: E402
from kart_rl.game.server import GAME_ENV, GameServer, KeyboardDriver  # noqa: E402
from kart_rl.raceline import RaceLine  # noqa: E402
from train.learn_from_human import behaviour_cloning, load_demos  # noqa: E402


def _demo_from_scripted_driver(path: Path, laps: int = 1):
    """Write a demo file in the game's format, driven by the raceline driver (stands in for a human)."""
    env = KartLapEnv(LapEnvConfig(random_start=False, laps=laps + 1))
    line = RaceLine.load()
    driver = PurePursuitDriver(line, speed=line.speed, a_brake=4.0)
    obs, info = env.reset(seed=0)
    driver.reset()
    O, A, L = [], [], []
    while True:
        a = driver.act(env)
        O.append(np.asarray(obs, np.float32)); A.append(np.asarray(a, np.float32)); L.append(env.laps_done)
        obs, _, term, trunc, info = env.step(a)
        if "lap_time" in info:
            L[-1] = env.laps_done - 1
        if len(info["lap_times"]) >= laps or term or trunc:
            break
    demo = {"meta": {"lap_times": [round(t, 2) for t in info["lap_times"]], "valid": [True] * len(info["lap_times"]),
                     "ghost_best": 99.0, "driver": "test"},
            "obs": np.asarray(O).tolist(), "act": np.asarray(A).tolist(), "lap": L}
    path.write_text(json.dumps(demo))
    return demo


def test_load_demos_filters_laps(tmp_path):
    demo = _demo_from_scripted_driver(tmp_path / "x_demo.json")
    obs, act, rep = load_demos(str(tmp_path / "*_demo.json"), faster_than="ghost", verbose=False)
    assert len(obs) == len(act) > 500 and obs.shape[1] == 30 and act.shape[1] == 2
    assert rep[0]["used"] == [0]
    obs2, _, rep2 = load_demos(str(tmp_path / "*_demo.json"), faster_than=1.0, verbose=False)  # nobody is that fast
    assert len(obs2) == 0 and rep2[0]["used"] == []


def test_behaviour_cloning_fits_the_actor(tmp_path):
    from stable_baselines3 import PPO

    _demo_from_scripted_driver(tmp_path / "x_demo.json")
    obs, act, _ = load_demos(str(tmp_path / "*_demo.json"), verbose=False)
    env = KartLapEnv(LapEnvConfig())
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, policy_kwargs={"net_arch": [32, 32]}, device="cpu", seed=0)
    stats = behaviour_cloning(model, obs, act, epochs=15, verbose=False)
    assert stats["nll_after"] < stats["nll_before"] - 0.3
    assert stats["mse_after"] < 0.2
    model.save(tmp_path / "bc.zip")
    assert (tmp_path / "bc.zip").exists()


def test_keyboard_driver_ramps():
    kb = KeyboardDriver()
    kb.set_keys({"up": True, "left": True})
    a = kb.action(0.05)
    assert 0 < a[0] < 1 and 0 < a[1] < 1
    for _ in range(40):
        a = kb.action(0.05)
    assert a[0] == 1.0 and a[1] == 1.0
    kb.release()
    for _ in range(40):
        a = kb.action(0.05)
    assert a[0] == 0.0 and a[1] == 0.0


def test_game_server_runs_without_clients(tmp_path, monkeypatch):
    import kart_rl.game.server as gs

    monkeypatch.setattr(gs, "LAPS_CSV", tmp_path / "laps.csv")
    monkeypatch.setattr(gs, "REPLAY_DIR", tmp_path / "replays")
    g = GameServer("baseline:raceline", driver="tester", countdown=0.0)
    g.keyboard.set_keys({"up": True})
    for _ in range(60):
        msg, events = g.step()
    assert msg["type"] == "states" and [s["id"] for s in msg["states"]] == ["du", "KI"]
    assert msg["states"][1]["ghost"] is True
    assert g.human.env.s > 5 and g.ghost.env.s > 5          # both moved
    assert len(g.demo_obs) == 60 and g.demo_obs[0].shape == (30,)
    g.human.laps.append({"time": 60.0, "valid": True})      # pretend a lap was completed
    g.log_lap(g.human.laps[-1])
    g.save_run()
    assert (tmp_path / "laps.csv").exists()
    files = sorted(p.name for p in (tmp_path / "replays").glob("*.json"))
    assert any(f.endswith("_du.json") for f in files) and any(f.endswith("_demo.json") for f in files)
    assert GAME_ENV["off_track_mode"] == "penalty"
