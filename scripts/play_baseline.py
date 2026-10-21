#!/usr/bin/env python3
"""Watch the scripted reference driver (pure pursuit) in the 2-D viewer.

Examples::

    python scripts/play_baseline.py                     # live window, 2 laps
    python scripts/play_baseline.py --follow            # camera follows the kart
    python scripts/play_baseline.py --gif docs/img/baseline.gif --laps 1
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from kart_rl.baselines import PurePursuitDriver
from kart_rl.envs import KartLapEnv, LapEnvConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--gif", default=None, help="write a GIF instead of opening a window")
    ap.add_argument("--mp4", default=None, help="write an MP4 (needs imageio-ffmpeg)")
    ap.add_argument("--follow", action="store_true", help="camera follows the kart")
    ap.add_argument("--every", type=int, default=4, help="keep every n-th frame for video export")
    ap.add_argument("--realtime", action="store_true", help="slow the live window down to real time")
    args = ap.parse_args()

    video = args.gif or args.mp4
    viewer = {"follow": args.follow}
    if video:
        viewer.update(figsize=(8, 6.4), dpi=70)
    env = KartLapEnv(LapEnvConfig(laps=args.laps, random_start=False, max_time=120.0 * args.laps, viewer=viewer),
                     render_mode="rgb_array" if video else "human")
    driver = PurePursuitDriver(env.track)
    env.reset()
    frames = []
    done, step = False, 0
    t0 = time.time()
    while not done:
        _, _, term, trunc, info = env.step(driver.act(env))
        done = term or trunc
        step += 1
        if video:
            if step % args.every == 0:
                frames.append(env.render())
        elif args.realtime:
            lag = env.t - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)
    print(f"episode end: {info.get('episode_done')}  lap times: {[round(t, 2) for t in info['lap_times']]}")
    if video:
        import imageio.v3 as iio

        fps = max(1, round(env.metadata["render_fps"] / args.every))
        if args.gif:
            iio.imwrite(args.gif, np.stack(frames), duration=1000 / fps, loop=0)
            print(f"wrote {args.gif} ({len(frames)} frames)")
        if args.mp4:
            iio.imwrite(args.mp4, np.stack(frames), fps=fps, codec="libx264")
            print(f"wrote {args.mp4} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
