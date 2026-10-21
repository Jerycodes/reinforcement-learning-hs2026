#!/usr/bin/env python3
"""Evaluate trained agents (and the scripted baselines) from a standing start.

    python scripts/evaluate.py runs/phase1_ppo/best.zip                 # lap times vs references
    python scripts/evaluate.py runs/phase1_ppo/best.zip --laps 5
    python scripts/evaluate.py runs/phase1_ppo/best.zip --gif docs/img/agent.gif
    python scripts/evaluate.py runs/phase1_ppo/best.zip --replay replays/agent.json   # for the 3-D viewer
    python scripts/evaluate.py runs/phase1_ppo/best.zip --render         # live 2-D window
    python scripts/evaluate.py --baseline raceline --baseline centerline
    python scripts/evaluate.py runs/a/best.zip runs/b/best.pt            # compare several models
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kart_rl.agents import baseline_policy, load_policy
from kart_rl.envs import KartLapEnv, LapEnvConfig
from kart_rl.evaluation import load_reference_lap_times, run_episode, save_replay, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="model files (.zip = SB3, .pt = scratch PPO)")
    ap.add_argument("--baseline", action="append", default=[], choices=["centerline", "raceline"])
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    ap.add_argument("--render", action="store_true", help="live 2-D window")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--every", type=int, default=5, help="frame subsampling for the GIF")
    ap.add_argument("--replay", default=None, help="write replay JSON (first model/baseline)")
    args = ap.parse_args()

    policies = [load_policy(m, deterministic=not args.stochastic) for m in args.models]
    policies += [baseline_policy(b) for b in args.baseline]
    if not policies:
        ap.error("give at least one model file or --baseline")
    refs = load_reference_lap_times()

    results = {}
    for i, pol in enumerate(policies):
        cfg = LapEnvConfig(laps=args.laps, random_start=False, max_time=args.max_time or 90.0 * args.laps,
                           viewer={"follow": args.follow, **({"figsize": (8, 6.4), "dpi": 70} if args.gif else {})})
        render_mode = "human" if args.render else ("rgb_array" if (args.gif and i == 0) else None)
        env = KartLapEnv(cfg, render_mode=render_mode, record=bool(args.replay) and i == 0)
        frames = []
        counter = {"n": 0}

        def grab(e):
            counter["n"] += 1
            if render_mode == "rgb_array" and counter["n"] % args.every == 0:
                frames.append(e.render())

        res = run_episode(env, pol, frame_callback=grab if (args.gif and i == 0) else None)
        results[pol.name] = res
        print(f"=== {pol.name}")
        print(summarize(res, refs))
        if args.gif and i == 0 and frames:
            import imageio.v3 as iio

            fps = max(1, round(env.metadata["render_fps"] / args.every))
            iio.imwrite(args.gif, np.stack(frames), duration=1000 / fps, loop=0)
            print(f"wrote {args.gif} ({len(frames)} frames)")
        if args.replay and i == 0:
            save_replay(args.replay, res.trajectory, meta={"policy": pol.name, "lap_times": res.lap_times,
                                                           "episode_done": res.episode_done, "track": env.track.name})
            print(f"wrote {args.replay}")
        env.close()

    if len(results) > 1:
        print("\n=== summary (best flying lap)")
        for name, res in sorted(results.items(), key=lambda kv: kv[1].best_lap or 1e9):
            lap = f"{res.best_lap:.2f} s" if res.best_lap is not None else "no lap"
            print(f"{name:40s} {lap:>10s}   ({res.episode_done})")


if __name__ == "__main__":
    main()
