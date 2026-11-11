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
from ..live.server import serve_http
from .driver import GAME_ENV, KeyboardDriver, Racer

ROOT = Path(__file__).resolve().parents[2]
LAPS_CSV = ROOT / "runs" / "human" / "laps.csv"
REPLAY_DIR = ROOT / "replays" / "human"

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
    ap.add_argument("--arcade-policy", help="Lokaler, mit train.train_arcade trainierter PPO-Checkpoint")
    ap.add_argument("--arcade", action="store_true", help="agile multiplayer arcade mode (separate physics)")
    ap.add_argument("--lan", action="store_true", help="with --arcade: allow players on the local network")
    ap.add_argument("--race-laps", type=int, default=3, help="arcade race length")
    args = ap.parse_args()
    if args.arcade:
        if args.race_laps < 1 or not args.http:
            ap.error("arcade requires a positive race-laps value and an HTTP port")
        from .arcade import serve
        try:
            asyncio.run(serve(args))
        except KeyboardInterrupt:
            pass
        return
    if args.open and args.http:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.http}/viewer/?game&ws={args.ws}")).start()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
