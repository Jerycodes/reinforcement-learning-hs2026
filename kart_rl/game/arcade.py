"""Responsive arcade kart racing, with one authoritative kart per browser."""

import asyncio
import functools
import http.server
import json
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from websockets.exceptions import ConnectionClosed

from .race import RaceRoom
from .server import ROOT


class RoomService:
    def __init__(self, laps=3, bot_policy=None):
        self.rooms = {}
        self.clients = {}
        self.laps = laps
        self.bot_policy = bot_policy

    async def handler(self, ws):
        room = player = token = None
        try:
            request = json.loads(await asyncio.wait_for(ws.recv(), 15))
            if not isinstance(request, dict):
                raise TypeError("Ungültige Anfrage.")
            if request.get("type") == "create":
                if len(self.rooms) >= 32:
                    raise ValueError("Der Server ist voll.")
                token = secrets.token_urlsafe(18)
                room = self.rooms[token] = RaceRoom(self.laps)
            elif request.get("type") == "join":
                token = request.get("room")
                room = self.rooms.get(token) if isinstance(token, str) else None
            if room is None:
                raise ValueError("Rennraum unbekannt oder bereits geschlossen.")
            player = room.join(profile=request.get("profile"))
            await ws.send(
                json.dumps(
                    {
                        "type": "hello",
                        "mode": "game",
                        "arcade": True,
                        "room": token,
                        "player_id": player.id,
                        "host_id": room.host_id,
                        "driver": player.name,
                        "track": "roggwil_arcade",
                        "policy": "Arcade",
                        "policies": [],
                        "schema": [],
                    }
                )
            )
            self.clients[ws] = token
            async for raw in ws:
                try:
                    error = room.handle(player.id, json.loads(raw))
                except (ValueError, RecursionError):
                    error = "Ungültige Nachricht."
                if error:
                    await ws.send(json.dumps({"type": "error", "message": error}))
        except (TypeError, ValueError, asyncio.TimeoutError, RecursionError) as exc:
            await ws.send(json.dumps({"type": "error", "message": str(exc)}))
        except ConnectionClosed:
            pass
        finally:
            self.clients.pop(ws, None)
            if player:
                room.leave(player.id)
            if room is not None and room.host_id is None:
                for pid in list(room.players):
                    room.leave(pid)
                self.rooms.pop(token, None)

    async def run(self):
        import websockets

        while True:
            start = time.perf_counter()
            for token, room in self.rooms.items():
                payload = json.dumps(room.tick(self.bot_policy), allow_nan=False)
                websockets.broadcast([ws for ws, key in self.clients.items() if key == token], payload)
            await asyncio.sleep(max(0.001, 1 / 30 - (time.perf_counter() - start)))


class ArcadeAssets(http.server.SimpleHTTPRequestHandler):
    """Serve only the viewer and track assets when the game is shared over LAN."""

    def send_head(self):
        path = Path(self.translate_path(urlsplit(self.path).path)).resolve()
        if not any(path.is_relative_to((ROOT / prefix).resolve()) for prefix in ("viewer", "kart_rl/data")):
            self.send_error(404)
            return None
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404)


async def serve(args):
    import websockets

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    bot_policy = None
    if args.arcade_policy:
        from ..envs.arcade_env import load_arcade_policy

        bot_policy = load_arcade_policy(args.arcade_policy)
    server = RoomService(args.race_laps, bot_policy)
    handler = functools.partial(ArcadeAssets, directory=str(ROOT))
    httpd = http.server.ThreadingHTTPServer((host, args.http), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://localhost:{args.http}/viewer/?game&arcade&ws={args.ws}"
    try:
        async with websockets.serve(server.handler, host, args.ws, max_size=2048):
            print(f"Arcade ready: {url}", flush=True)
            if args.open:
                import webbrowser

                webbrowser.open(url)
            await server.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
