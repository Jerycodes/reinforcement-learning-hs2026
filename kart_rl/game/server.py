#!/usr/bin/env python3
"""Game mode: drive the kart yourself with the keyboard - against a trained model.

Same simulation, same kart, same track limits as in training; the model drives as a
semi-transparent "ghost" next to you (no collisions between the two). Every lap is
timed, compared and logged, every run is recorded as a replay for the 3-D viewer.

    python -m kart_rl.game --policy runs/phase1_v2/best.zip --open
    python -m kart_rl.game --policy runs/phase1_v2/best.zip --name Jeremy

Keys (in the viewer): arrows or WASD = steer / throttle / brake, Backspace = back onto
the track (after a crash), R = restart, 1-4 = camera. Results: runs/human/laps.csv (every lap), replays/human/<name>_<time>_{du,KI}.json
(for the viewer) and ..._demo.json (observations + your actions, used only by the
optional "learn from the human" add-on, train/learn_from_human.py).

Protocol (JSON): like kart_rl.live.server, plus
    server -> client
        {"type": "hello", "mode": "game", "driver", "policy", "policies", "track"}
        {"type": "states", "states": [{id: "du", ...}, {id: "KI", ..., ghost: true}], "game": {...}}
        {"type": "lap", "who": "du" | "KI", "lap_time", "valid", "best_du", "best_ki"}
    client -> server
        {"type": "keys", "left": bool, "right": bool, "up": bool, "down": bool}
        {"type": "control", "action": "reset"}
        {"type": "select_policy", "path": "..."}
        {"type": "set_name", "name": "..."}
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import threading
import time
from pathlib import Path

import numpy as np

from ..agents import baseline_policy, load_policy
from ..envs import KartLapEnv, LapEnvConfig
from ..live.server import serve_http

ROOT = Path(__file__).resolve().parents[2]
LAPS_CSV = ROOT / "runs" / "human" / "laps.csv"
REPLAY_DIR = ROOT / "replays" / "human"

GAME_ENV = dict(random_start=False, laps=999, max_time=36000.0, off_track_mode="penalty", wall_friction=0.1,
                stuck_time=1e9, wrong_way_distance=1e9)


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
        self.throttle += float(np.clip(target - self.throttle, -self.throttle_rate * dt, self.throttle_rate * dt))
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
        self.laps: list[dict] = []       # {"time", "valid"}
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
        return {"id": kart_id, "t": round(self.env.t, 3), "x": round(float(k.x), 3), "y": round(float(k.y), 3),
                "yaw": round(float(k.yaw), 4), "vx": round(float(k.vx), 3), "vy": round(float(k.vy), 3),
                "steer": round(float(k.steer), 4), "throttle": round(float(self.last_action[1]), 3),
                "s": round(self.env.s, 2), "lap": self.env.laps_done, "off_track": bool(self.env.off_track)}

    def hud(self) -> dict:
        return {"lap": self.env.laps_done + 1, "current": round(self.env.t - self.env.lap_start_t, 2),
                "last": self.laps[-1]["time"] if self.laps else None,
                "last_valid": self.laps[-1]["valid"] if self.laps else None,
                "best": self.best, "laps": self.laps[-8:], "contact": self.contact_in_lap}


class GameServer:
    def __init__(self, policy: str, driver: str = "du", env_cfg: dict | None = None, countdown: float = 3.0):
        cfg = {**GAME_ENV, **(env_cfg or {})}
        self.driver = driver
        self.human = Racer("du", cfg)
        self.ghost = Racer("KI", cfg)
        self.keyboard = KeyboardDriver()
        self.policy_name = policy
        self.policy = self._load(policy)
        self.countdown_total = countdown
        self.clients: set = set()
        self.demo_obs: list[np.ndarray] = []
        self.demo_act: list[np.ndarray] = []
        self.demo_lap: list[int] = []
        self.reset()

    def _load(self, name: str):
        if name.startswith("baseline:"):
            return baseline_policy(name.split(":", 1)[1])
        return load_policy(name)

    # ---------------------------------------------------------------- control
    def reset(self, save: bool = True):
        if save:
            self.save_run()
        self.human.reset()
        self.ghost.reset()
        if hasattr(self.policy, "bind"):
            self.policy.bind(self.ghost.env)
        self.keyboard.steer = self.keyboard.throttle = 0.0
        self.countdown_end = time.perf_counter() + self.countdown_total
        self.started = time.strftime("%Y%m%d-%H%M%S")
        self.demo_obs, self.demo_act, self.demo_lap = [], [], []

    def select_policy(self, name: str):
        self.policy_name = name
        self.policy = self._load(name)
        self.reset()

    def countdown(self) -> float:
        return max(0.0, self.countdown_end - time.perf_counter())

    def step(self) -> tuple[dict, list[dict]]:
        dt = self.human.env.cfg.dt
        events = []
        if self.countdown() > 0:        # both karts wait on the grid
            return self.message(), events
        obs_before = np.array(self.human.obs, dtype=np.float32)
        a_h = self.keyboard.action(dt)
        lap_h = self.human.step(a_h)
        self.demo_obs.append(obs_before)
        self.demo_act.append(a_h.copy())
        self.demo_lap.append(self.human.env.laps_done if lap_h is None else self.human.env.laps_done - 1)
        a_g = self.policy(self.ghost.obs)
        lap_g = self.ghost.step(np.asarray(a_g, dtype=np.float32))
        if lap_h:
            events.append(self.lap_event("du", lap_h))
            self.log_lap(lap_h)
        if lap_g:
            events.append(self.lap_event("KI", lap_g))
        return self.message(), events

    def lap_event(self, who: str, lap: dict) -> dict:
        return {"type": "lap", "who": who, "lap_time": lap["time"], "valid": lap["valid"],
                "best_du": self.human.best, "best_ki": self.ghost.best}

    def message(self) -> dict:
        ghost_state = self.ghost.state("KI")
        ghost_state["ghost"] = True
        game = {"driver": self.driver, "policy": self.policy_name, "countdown": round(self.countdown(), 1),
                "du": self.human.hud(), "ki": self.ghost.hud(), "recording": True}
        if self.human.best is not None and self.ghost.best is not None:
            game["delta"] = round(self.human.best - self.ghost.best, 2)
        return {"type": "states", "states": [self.human.state("du"), ghost_state], "game": game}

    # ---------------------------------------------------------------- results
    def log_lap(self, lap: dict):
        LAPS_CSV.parent.mkdir(parents=True, exist_ok=True)
        new = not LAPS_CSV.exists()
        with open(LAPS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "driver", "lap_time_s", "valid", "ghost_policy", "ghost_best_s", "run"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.driver, lap["time"], int(lap["valid"]),
                        self.policy_name, self.ghost.best if self.ghost.best is not None else "", self.started])

    def save_run(self):
        """Replays of both karts + the demonstration file (only when at least one lap was completed)."""
        if not self.human.laps:
            return
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{self.driver}_{self.started}"
        meta_common = {"ghost_policy": self.policy_name, "driver": self.driver, "started": self.started}
        for racer, suffix in ((self.human, "du"), (self.ghost, "KI")):
            traj = racer.env.get_trajectory()
            if not traj:
                continue
            meta = {**meta_common, "policy": self.driver if suffix == "du" else self.policy_name,
                    "lap_times": [lap["time"] for lap in racer.laps], "valid": [lap["valid"] for lap in racer.laps]}
            (REPLAY_DIR / f"{stem}_{suffix}.json").write_text(json.dumps({"meta": meta, "trajectory": traj}))
        demo = {"meta": {**meta_common, "lap_times": [lap["time"] for lap in self.human.laps],
                         "valid": [lap["valid"] for lap in self.human.laps], "ghost_best": self.ghost.best,
                         "obs_names": self.human.env.obs_names},
                "obs": np.asarray(self.demo_obs).round(5).tolist(), "act": np.asarray(self.demo_act).round(4).tolist(),
                "lap": self.demo_lap}
        (REPLAY_DIR / f"{stem}_demo.json").write_text(json.dumps(demo))
        print(f"saved replays/human/{stem}_{{du,KI,demo}}.json  laps {[lap['time'] for lap in self.human.laps]}")

    # ------------------------------------------------------------- networking
    @staticmethod
    def available_policies() -> list[str]:
        files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "runs").glob("*/best.*") if p.suffix in (".zip", ".pt"))
        return files + ["baseline:raceline", "baseline:centerline"]

    def hello(self) -> dict:
        return {"type": "hello", "mode": "game", "track": self.human.env.track.name, "driver": self.driver,
                "policy": self.policy_name, "policies": self.available_policies(), "schema": []}

    async def broadcast(self, msg: dict):
        if not self.clients:
            return
        data = json.dumps(msg)
        await asyncio.gather(*(c.send(data) for c in list(self.clients)), return_exceptions=True)

    async def handler(self, ws):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps(self.hello()))
            async for raw in ws:
                try:
                    m = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = m.get("type")
                if t == "keys":
                    self.keyboard.set_keys(m)
                elif t == "control" and m.get("action") == "reset":
                    self.reset()
                    await self.broadcast({"type": "reset", "policy": self.policy_name})
                elif t == "control" and m.get("action") == "respawn":
                    if self.countdown() <= 0:
                        self.human.respawn()
                        self.keyboard.steer = self.keyboard.throttle = 0.0
                elif t == "select_policy":
                    try:
                        self.select_policy(m["path"])
                        await self.broadcast({"type": "reset", "policy": self.policy_name})
                        await self.broadcast(self.hello())
                    except Exception as e:  # noqa: BLE001
                        await ws.send(json.dumps({"type": "error", "message": str(e)}))
                elif t == "set_name":
                    self.driver = "".join(c for c in str(m.get("name", "du")).strip() if c.isalnum() or c in "-_") or "du"
        finally:
            self.clients.discard(ws)
            if not self.clients:
                self.keyboard.release()

    async def sim_loop(self):
        dt = self.human.env.cfg.dt
        next_t = time.perf_counter()
        while True:
            if not self.clients:
                await asyncio.sleep(0.05)
                next_t = time.perf_counter()
                continue
            msg, events = self.step()
            await self.broadcast(msg)
            for ev in events:
                await self.broadcast(ev)
            next_t += dt
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_t = time.perf_counter()


async def main_async(args):
    import websockets

    server = GameServer(args.policy, driver=args.name, countdown=args.countdown)
    if args.http:
        serve_http(ROOT, args.http)
        print(f"viewer: http://localhost:{args.http}/viewer/?game&ws={args.ws}")
    try:
        async with websockets.serve(server.handler, "127.0.0.1", args.ws, max_size=2**22):
            print(f"game: ws://localhost:{args.ws}  ghost policy: {args.policy}  driver: {args.name}  (Ctrl+C to stop)")
            await server.sim_loop()
    finally:
        server.save_run()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="baseline:raceline", help="ghost: model file (.zip/.pt) or baseline:raceline|centerline")
    ap.add_argument("--name", default="du", help="driver name for the leaderboard and replay files")
    ap.add_argument("--ws", type=int, default=8765, help="websocket port")
    ap.add_argument("--http", type=int, default=8000, help="static http port for the viewer (0 = off)")
    ap.add_argument("--countdown", type=float, default=3.0, help="seconds on the grid before the start")
    ap.add_argument("--open", action="store_true", help="open the viewer in the browser")
    args = ap.parse_args()
    if args.open and args.http:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.http}/viewer/?game&ws={args.ws}")).start()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
