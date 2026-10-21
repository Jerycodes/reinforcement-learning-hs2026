#!/usr/bin/env python3
"""Serve the repository over HTTP and open the 3-D viewer (replay mode).

    python scripts/serve_viewer.py                 # http://localhost:8000/viewer/
    python scripts/serve_viewer.py --port 8080 --no-open
For the live mode use:  python -m kart_rl.live.server --policy runs/phase1_ppo/best.zip --open
"""
import argparse
import functools
import http.server
import threading
import webbrowser
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/viewer/"
    print(f"serving {root} at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
