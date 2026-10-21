#!/usr/bin/env python3
"""Watch a training run live: opens the 2-D viewer and keeps driving with the
newest checkpoint of a run while training continues in another terminal.

    # terminal 1
    python train/train_sb3.py --config train/configs/phase1_ppo.yaml
    # terminal 2
    python scripts/watch_training.py --run runs/phase1_ppo
    python scripts/watch_training.py --run runs/phase1_ppo --best      # follow best.zip instead
    python scripts/watch_training.py --run runs/phase1_scratch         # works for the scratch PPO too
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from kart_rl.agents import load_policy
from kart_rl.envs import KartLapEnv, LapEnvConfig


def newest_checkpoint(run_dir: Path, best: bool) -> Path | None:
    if best:
        for name in ("best.zip", "best.pt"):
            if (run_dir / name).exists():
                return run_dir / name
        return None
    cands = list((run_dir / "checkpoints").glob("step_*.zip")) + list((run_dir / "checkpoints").glob("step_*.pt"))
    cands = [c for c in cands if "vecnormalize" not in c.name]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/phase1_ppo")
    ap.add_argument("--best", action="store_true", help="follow the best model instead of the latest checkpoint")
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--follow", action="store_true", help="camera follows the kart")
    ap.add_argument("--realtime", action="store_true", help="slow down to real time")
    ap.add_argument("--stochastic", action="store_true", help="sample actions like during training")
    args = ap.parse_args()

    run_dir = Path(args.run)
    env = KartLapEnv(LapEnvConfig(laps=args.laps, random_start=False, max_time=90.0 * args.laps,
                                  viewer={"follow": args.follow}), render_mode="human")
    seen = None
    policy = None
    print("waiting for the first checkpoint ...")
    while True:
        ck = newest_checkpoint(run_dir, args.best)
        if ck is None:
            time.sleep(3)
            continue
        if ck != seen or policy is None:
            try:
                policy = load_policy(ck, deterministic=not args.stochastic)
            except Exception as e:  # checkpoint may still be written
                print(f"could not load {ck.name} ({e}), retrying")
                time.sleep(2)
                continue
            seen = ck
            print(f"driving with {ck.name}")
        obs, _ = env.reset()
        done = False
        t0 = time.time()
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
            if args.realtime:
                lag = env.t - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
        print(f"  {ck.name}: {info.get('episode_done')}, laps {[round(t, 2) for t in info['lap_times']]}, "
              f"distance {info['total_progress']:.0f} m")


if __name__ == "__main__":
    main()
