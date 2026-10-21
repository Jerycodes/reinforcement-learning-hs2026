"""Broadcast training states to the 3-D viewer while a training run is going on.

``StateStreamer`` runs a WebSocket server in a background thread; the training
loop calls ``send(...)`` (thread-safe, non-blocking, drops messages when no
client is connected).  Used by the ``--stream`` option of train/train_sb3.py
and train/ppo_scratch.py.  Open the viewer with  http://localhost:8000/viewer/?live
(serve the repository root e.g. with  python scripts/serve_viewer.py --no-open).
"""
from __future__ import annotations

import asyncio
import json
import threading
import time


class StateStreamer:
    def __init__(self, port: int = 8765, hello: dict | None = None, max_rate: float = 25.0):
        self.port = port
        self.hello = hello or {}
        self.min_interval = 1.0 / max_rate
        self._last_sent = 0.0
        self.clients: set = set()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------- server
    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        import websockets

        async with websockets.serve(self._handler, "127.0.0.1", self.port, max_size=2**22):
            await asyncio.Future()  # run forever

    async def _handler(self, ws):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps({"type": "hello", "mode": "training", "schema": [], "policies": [],
                                      **self.hello}))
            async for _ in ws:  # ignore incoming messages (the panel controls nothing during training)
                pass
        finally:
            self.clients.discard(ws)

    async def _broadcast(self, data: str):
        if self.clients:
            await asyncio.gather(*(c.send(data) for c in list(self.clients)), return_exceptions=True)

    # ------------------------------------------------------------------ api
    @property
    def connected(self) -> bool:
        return bool(self.clients)

    def send(self, msg: dict, force: bool = False) -> bool:
        """Queue a message for all clients. Rate limited unless ``force``."""
        if not self.clients:
            return False
        now = time.perf_counter()
        if not force and now - self._last_sent < self.min_interval:
            return False
        self._last_sent = now
        self.loop.call_soon_threadsafe(asyncio.ensure_future, self._broadcast(json.dumps(msg)))
        return True


class Showcase(threading.Thread):
    """One kart that drives the *current* policy deterministically in real time while the
    training continues in the background. This is what ``--stream`` shows: a clean single
    kart whose driving improves with every policy update (the parallel training
    environments stay invisible unless ``--stream-all`` is used).

    ``predict_fn(obs) -> action`` must be safe to call from this thread (a plain forward
    pass under ``torch.no_grad()``); ``stats_fn() -> dict`` returns the training stats
    shown in the HUD.
    """

    def __init__(self, streamer: StateStreamer, env_cfg: dict, predict_fn, stats_fn, dt_wall: float | None = None,
                 lidar: bool = True):
        super().__init__(daemon=True)
        from ..envs import KartLapEnv, LapEnvConfig

        cfg = LapEnvConfig.from_dict(env_cfg)
        cfg.random_start = False
        cfg.laps = 10_000
        cfg.max_time = 1e9
        cfg.sector_length_m = 0.0
        cfg.domain_randomization = {}
        self.env = KartLapEnv(cfg)
        self.streamer = streamer
        self.predict_fn = predict_fn
        self.stats_fn = stats_fn
        self.dt_wall = dt_wall if dt_wall else cfg.dt
        self.lidar = lidar
        self.stop_flag = False

    def run(self):
        env = self.env
        obs, _ = env.reset()
        next_t = time.perf_counter()
        while not self.stop_flag:
            if not self.streamer.connected:
                time.sleep(0.2)
                next_t = time.perf_counter()
                continue
            action = self.predict_fn(obs)
            obs, _, term, trunc, info = env.step(action)
            st = env_state_from_info(info, 0)
            msg = {"type": "state", "state": st, "lap_times": [round(t, 2) for t in info["lap_times"]],
                   "training": self.stats_fn()}
            if self.lidar:
                msg["lidar"] = [[round(float(a), 3), round(float(d), 2)]
                                for a, d in zip(env.lidar_angles, env.lidar_dist)]
            self.streamer.send(msg, force=True)
            if term or trunc:
                self.streamer.send({"type": "episode_end", "reason": info.get("episode_done", ""),
                                    "lap_times": [round(t, 2) for t in info["lap_times"]]}, force=True)
                time.sleep(1.0)
                obs, _ = env.reset()
                self.streamer.send({"type": "reset"}, force=True)
                next_t = time.perf_counter()
                continue
            next_t += self.dt_wall
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()


def open_viewer(http_port: int = 8000, ws_port: int = 8765, open_browser: bool = True) -> str:
    """Serve the repository root over HTTP (background thread) and open the viewer in
    live mode. Returns the URL. If the port is taken (e.g. serve_viewer.py already runs)
    the existing server is used."""
    import functools
    import http.server
    import webbrowser
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if http_port <= 0:  # started from the dashboard, which serves the viewer itself
        return f"(viewer served by the dashboard) ws port {ws_port}"
    url = f"http://localhost:{http_port}/viewer/?live&ws={ws_port}"
    try:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
        handler.log_message = lambda *a, **k: None
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", http_port), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    except OSError:
        pass  # port busy: assume a viewer server is already running
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    return url


def env_state_from_info(info: dict, env_id: int) -> dict:
    """Compact state dict for the viewer from a KartLapEnv info dict."""
    return {"id": env_id, "t": round(info["t"], 2), "x": round(info["x"], 3), "y": round(info["y"], 3),
            "yaw": round(info["yaw"], 4), "vx": round(info.get("vx", info["speed"]), 2), "vy": round(info.get("vy", 0.0), 2),
            "steer": round(info.get("steer", 0.0), 3), "throttle": round(info.get("throttle", 0.0), 2),
            "s": round(info["s"], 1), "lap": info["laps_done"], "off_track": bool(info.get("off_track", False)),
            "done": info.get("episode_done", "")}
