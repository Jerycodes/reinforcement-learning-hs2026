#!/usr/bin/env python3
"""Live server for the 3-D viewer: runs the simulation with a trained policy
(or a scripted driver) and streams the kart state over a WebSocket, while the
viewer's parameter panel can change kart / environment parameters on the fly.

    python -m kart_rl.live.server --policy runs/phase1_ppo/best.zip
    python -m kart_rl.live.server --policy baseline:raceline --http 8000 --ws 8765

Then open http://localhost:8000/viewer/?live  (the HTTP server serves the
repository root so the viewer finds the track data and replays).

Protocol (JSON messages)
    server -> client
        {"type": "hello", "track", "policy", "policies": [...], "schema": [...]}
        {"type": "state", "state": {t, x, y, yaw, vx, vy, steer, throttle, s, lap, ...},
         "lap_times": [...], "lidar": [[angle, dist], ...]}
        {"type": "episode_end", "reason", "lap_times"}
        {"type": "reset", "policy"}
        {"type": "params", "values": {...}}
    client -> server
        {"type": "control", "action": "start" | "stop" | "reset"}
        {"type": "select_policy", "path": "..."}
        {"type": "set_speed", "factor": 1 | 2 | 4 | 0 (max)}
        {"type": "set_params", "values": {"kart.mu": 0.9, ...}}
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import threading
import time
from pathlib import Path

import numpy as np

from ..agents import baseline_policy, load_policy
from ..envs import KartLapEnv, LapEnvConfig

# parameter schema shown in the viewer panel: key -> (label, min, max, step)
KART_SCHEMA = {
    "kart.mu": ("Reifengrip µ", 0.6, 1.3, 0.01),
    "kart.drive_force_max": ("Antriebskraft [N]", 300, 1200, 10),
    "kart.power_max": ("Motorleistung [W]", 2000, 10000, 100),
    "kart.brake_force_max": ("Bremskraft [N]", 400, 1600, 10),
    "kart.max_steer": ("max. Lenkwinkel [rad]", 0.2, 0.6, 0.01),
    "kart.speed_limit": ("Begrenzer [m/s]", 10, 25, 0.5),
    "kart.mass": ("Masse [kg]", 120, 260, 5),
}
ENV_SCHEMA = {
    "env.off_track_margin": ("Off-Track-Abstand [m] (ab Reset)", 0.0, 1.5, 0.05),
}


class LiveServer:
    def __init__(self, policy: str, env_cfg: dict | None = None, laps: int = 999, speed: float = 1.0):
        self.env_cfg = dict(env_cfg or {})
        self.env_cfg.update(random_start=False, laps=laps, max_time=3600.0)
        self.env = KartLapEnv(LapEnvConfig.from_dict(self.env_cfg))
        self.policy_name = policy
        self.policy = self._load(policy)
        self.speed = speed  # 0 = as fast as possible
        self.running = True
        self.clients: set = set()
        self.obs = None
        self.reset()

    def _load(self, name: str):
        if name.startswith("baseline:"):
            return baseline_policy(name.split(":", 1)[1])
        return load_policy(name)

    # ------------------------------------------------------------ parameters
    def schema(self) -> list[dict]:
        out = []
        for key, (label, lo, hi, step) in KART_SCHEMA.items():
            out.append({"key": key, "label": label, "min": lo, "max": hi, "step": step,
                        "value": getattr(self.env.kart.p, key.split(".")[1])})
        for key, (label, lo, hi, step) in ENV_SCHEMA.items():
            out.append({"key": key, "label": label, "min": lo, "max": hi, "step": step,
                        "value": getattr(self.env.cfg, key.split(".")[1])})
        out += self.extra_schema()
        return out

    def extra_schema(self) -> list[dict]:
        """Hook for later phases (tyres, pit stops, boost)."""
        return []

    def set_params(self, values: dict):
        for key, val in values.items():
            group, name = key.split(".", 1)
            if group == "kart" and hasattr(self.env.kart.p, name):
                setattr(self.env.kart.p, name, float(val))
            elif group == "env" and hasattr(self.env.cfg, name):
                setattr(self.env.cfg, name, type(getattr(self.env.cfg, name))(val))
            else:
                self.set_extra_param(key, val)

    def set_extra_param(self, key: str, val):
        pass

    def current_params(self) -> dict:
        return {s["key"]: s["value"] for s in self.schema()}

    # --------------------------------------------------------------- control
    def reset(self):
        self.obs, _ = self.env.reset()
        if hasattr(self.policy, "bind"):
            self.policy.bind(self.env)
        self.episode_end = None

    def select_policy(self, name: str):
        self.policy_name = name
        self.policy = self._load(name)
        self.reset()

    def step(self) -> dict:
        action = self.policy(self.obs)
        self.obs, r, term, trunc, info = self.env.step(action)
        k = self.env.kart
        state = {"t": round(self.env.t, 3), "x": round(float(k.x), 3), "y": round(float(k.y), 3),
                 "yaw": round(float(k.yaw), 4), "vx": round(float(k.vx), 3), "vy": round(float(k.vy), 3),
                 "steer": round(float(k.steer), 4), "throttle": round(float(action[1]), 3),
                 "s": round(self.env.s, 2), "lap": self.env.laps_done}
        state.update(self.extra_state())
        msg = {"type": "state", "state": state, "lap_times": [round(t, 3) for t in self.env.lap_times],
               "lidar": [[round(float(a), 3), round(float(d), 2)] for a, d in zip(self.env.lidar_angles, self.env.lidar_dist)]}
        if term or trunc:
            self.episode_end = {"type": "episode_end", "reason": info.get("episode_done", ""),
                                "lap_times": [round(t, 3) for t in self.env.lap_times]}
        return msg

    def extra_state(self) -> dict:
        return {}

    # ------------------------------------------------------------ networking
    async def broadcast(self, msg: dict):
        if not self.clients:
            return
        data = json.dumps(msg)
        await asyncio.gather(*(c.send(data) for c in list(self.clients)), return_exceptions=True)

    def hello(self) -> dict:
        return {"type": "hello", "track": self.env.track.name, "policy": self.policy_name,
                "policies": self.available_policies(), "schema": self.schema()}

    @staticmethod
    def available_policies() -> list[str]:
        files = sorted(str(p) for p in Path("runs").glob("*/best.*") if p.suffix in (".zip", ".pt"))
        return files + ["baseline:raceline", "baseline:centerline"]

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
                if t == "control":
                    a = m.get("action")
                    if a == "start":
                        self.running = True
                    elif a == "stop":
                        self.running = False
                    elif a == "reset":
                        self.reset()
                        await self.broadcast({"type": "reset", "policy": self.policy_name})
                elif t == "select_policy":
                    try:
                        self.select_policy(m["path"])
                        await self.broadcast({"type": "reset", "policy": self.policy_name})
                    except Exception as e:  # noqa: BLE001
                        await ws.send(json.dumps({"type": "error", "message": str(e)}))
                elif t == "set_speed":
                    self.speed = float(m.get("factor", 1.0))
                elif t == "set_params":
                    self.set_params(m.get("values", {}))
                    await self.broadcast({"type": "params", "values": self.current_params()})
        finally:
            self.clients.discard(ws)

    async def sim_loop(self):
        dt = self.env.cfg.dt
        next_t = time.perf_counter()
        while True:
            if not self.running or not self.clients:
                await asyncio.sleep(0.05)
                next_t = time.perf_counter()
                continue
            msg = self.step()
            await self.broadcast(msg)
            if self.episode_end is not None:
                await self.broadcast(self.episode_end)
                await asyncio.sleep(1.0)
                self.reset()
                await self.broadcast({"type": "reset", "policy": self.policy_name})
                next_t = time.perf_counter()
                continue
            if self.speed > 0:
                next_t += dt / self.speed
                delay = next_t - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_t = time.perf_counter()
            else:
                await asyncio.sleep(0)


def serve_http(root: Path, port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None  # quiet
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def main_async(args):
    import websockets

    server = LiveServer(args.policy, speed=args.speed)
    root = Path(__file__).resolve().parents[2]
    if args.http:
        serve_http(root, args.http)
        print(f"viewer: http://localhost:{args.http}/viewer/?live")
    async with websockets.serve(server.handler, "127.0.0.1", args.ws, max_size=2**22):
        print(f"websocket: ws://localhost:{args.ws}  policy: {args.policy}  (Ctrl+C to stop)")
        await server.sim_loop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="baseline:raceline", help="model file (.zip/.pt) or baseline:raceline|centerline")
    ap.add_argument("--ws", type=int, default=8765, help="websocket port")
    ap.add_argument("--http", type=int, default=8000, help="static http port for the viewer (0 = off)")
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed factor (0 = max)")
    ap.add_argument("--open", action="store_true", help="open the viewer in the browser")
    args = ap.parse_args()
    if args.open and args.http:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.http}/viewer/?live")).start()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
