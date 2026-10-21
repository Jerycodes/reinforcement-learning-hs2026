#!/usr/bin/env python3
"""Render a video of the 3-D viewer without screen recording (headless Chromium via Playwright).

Replay mode renders frame by frame at a fixed simulation time step, so the video is
smooth no matter how fast the machine is:

    python scripts/record_viewer.py --replay replays/agent.json --replay replays/baseline_raceline.json \
        --camera chase --seconds 60 --out docs/video/agent_vs_baseline_3d.mp4
    python scripts/record_viewer.py --replay replays/agent.json --camera top --from 0 --seconds 30 --out top.mp4

Live mode captures a running training stream (train_sb3.py --stream) in wall-clock time:

    python scripts/record_viewer.py --live ws://localhost:8765 --camera top --seconds 30 --out training_live.mp4

Setup once:  pip install playwright && playwright install chromium
"""
from __future__ import annotations

import argparse
import functools
import http.server
import threading
import time
from pathlib import Path

import numpy as np


def serve(root: Path, port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="append", default=[], help="replay JSON (repo-relative), repeatable")
    ap.add_argument("--live", default=None, help="websocket URL of a training stream / live server")
    ap.add_argument("--camera", default="chase", choices=["chase", "onboard", "top", "free"])
    ap.add_argument("--hud-kart", type=int, default=0, help="which kart gets camera and telemetry")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--panel", action="store_true", help="keep the side panel visible")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--out", default="docs/video/viewer.mp4")
    ap.add_argument("--chromium-args", default="", help="extra Chromium flags (headless GL etc.)")
    args = ap.parse_args()
    if not args.replay and not args.live:
        ap.error("give --replay (one or more) or --live")

    from playwright.sync_api import sync_playwright

    root = Path(__file__).resolve().parents[1]
    httpd = serve(root, args.port)
    url = f"http://127.0.0.1:{args.port}/viewer/?panel={'1' if args.panel else '0'}&cam={args.camera}"
    for r in args.replay:
        url += f"&replay=../{Path(r).as_posix()}"
    frames = []
    with sync_playwright() as p:
        extra = args.chromium_args.split() if args.chromium_args else []
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                          "--ignore-gpu-blocklist", *extra])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(url, wait_until="load")
        page.wait_for_function("window.kartViewer && window.kartViewer.ready", timeout=30000)
        page.wait_for_timeout(500)
        page.evaluate(f"kartViewer.setHudKart({args.hud_kart})")
        n = int(args.seconds * args.fps)
        if args.live:
            page.evaluate("document.getElementById('modeLive').click()")
            page.evaluate(f"kartViewer.connect('{args.live}')")
            page.wait_for_timeout(1500)
            t_next = time.perf_counter()
            for i in range(n):
                frames.append(np.asarray(_png(page.screenshot(type="png"))))
                t_next += 1.0 / args.fps
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                if i % args.fps == 0:
                    print(f"  {i / args.fps:.0f}/{args.seconds:.0f} s", end="\r")
        else:
            print("karts:", page.evaluate("kartViewer.karts()"))
            # the chase camera smooths towards the kart: let it settle at the first frame
            page.evaluate(f"kartViewer.setTime({args.t_from})")
            for _ in range(30):
                page.evaluate(f"kartViewer.setTime({args.t_from})")
            for i in range(n):
                t = args.t_from + i / args.fps
                page.evaluate(f"kartViewer.setTime({t})")
                frames.append(np.asarray(_png(page.screenshot(type="png"))))
                if i % args.fps == 0:
                    print(f"  {i / args.fps:.0f}/{args.seconds:.0f} s", end="\r")
        browser.close()
    httpd.shutdown()

    import imageio.v3 as iio

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack([f[..., :3] for f in frames])
    if out.suffix.lower() == ".gif":
        iio.imwrite(out, arr, duration=1000 / args.fps, loop=0)
    else:
        iio.imwrite(out, arr, fps=args.fps, codec="libx264", macro_block_size=1)
    print(f"\nwrote {out}: {len(frames)} frames, {len(frames) / args.fps:.0f} s")


def _png(data: bytes):
    import io

    import imageio.v3 as iio

    return iio.imread(io.BytesIO(data), extension=".png")


if __name__ == "__main__":
    main()
