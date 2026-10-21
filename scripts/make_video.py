#!/usr/bin/env python3
"""Make a "from the first steps to the final agent" video out of checkpoints.

    # sequential: one segment per checkpoint, labelled with step count and result
    python scripts/make_video.py --run runs/phase1_ppo --out docs/video/evolution.mp4
    # pick specific checkpoints / models with labels
    python scripts/make_video.py --model runs/phase1_ppo/checkpoints/step_200000_steps.zip:200k \
                                 --model runs/phase1_ppo/best.zip:best --out evolution.mp4
    # 2x2 grid, all models drive at the same time
    python scripts/make_video.py --run runs/phase1_ppo --grid --out docs/video/evolution_grid.mp4

Needs ffmpeg for .mp4 (pip install imageio-ffmpeg, already in the train extra); .gif works without.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from kart_rl.agents import load_policy
from kart_rl.envs import KartLapEnv, LapEnvConfig


def pick_checkpoints(run_dir: Path, n: int) -> list[tuple[Path, str]]:
    cks = sorted(run_dir.glob("checkpoints/step_*"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    cks = [c for c in cks if c.suffix in (".zip", ".pt")]
    if not cks:
        raise SystemExit(f"no checkpoints in {run_dir}/checkpoints")
    if len(cks) > n:
        idx = np.unique(np.round(np.linspace(0, len(cks) - 1, n)).astype(int))
        cks = [cks[i] for i in idx]
    out = [(c, f"{int(re.findall(r'\\d+', c.stem)[0]) // 1000}k Schritte") for c in cks]
    best = run_dir / "best.zip"
    if not best.exists():
        best = run_dir / "best.pt"
    if best.exists():
        out.append((best, "bestes Modell"))
    return out


def segment_frames(model: Path, label: str, laps: int, every: int, max_time: float, figsize, dpi) -> tuple[list, str]:
    env = KartLapEnv(LapEnvConfig(laps=laps, random_start=False, max_time=max_time,
                                  viewer={"figsize": figsize, "dpi": dpi}), render_mode="rgb_array")
    policy = load_policy(model)
    obs, _ = env.reset()
    env.render()
    env._viewer.title.set_text(label)
    frames, done, step = [], False, 0
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs))
        done = term or trunc
        step += 1
        if step % every == 0:
            frames.append(env.render())
    laps_t = info["lap_times"]
    result = (f"Rundenzeit {min(laps_t):.2f} s" if laps_t else f"{info['total_progress']:.0f} m, dann {info.get('episode_done')}")
    env._viewer.title.set_text(f"{label} – {result}")
    for _ in range(int(1.5 * env.metadata["render_fps"] / every)):  # hold the result for 1.5 s
        frames.append(env.render())
    env.close()
    return frames, result


def grid_frames(models: list[tuple[Path, str]], laps: int, every: int, max_time: float, figsize, dpi) -> list:
    envs, pols, obs = [], [], []
    for m, label in models:
        env = KartLapEnv(LapEnvConfig(laps=laps, random_start=False, max_time=max_time,
                                      viewer={"figsize": figsize, "dpi": dpi}), render_mode="rgb_array")
        o, _ = env.reset()
        env.render()
        env._viewer.title.set_text(label)
        envs.append(env)
        pols.append(load_policy(m))
        obs.append(o)
    done = [False] * len(envs)
    frames, step = [], 0
    while not all(done):
        for i, env in enumerate(envs):
            if done[i]:
                continue
            obs[i], _, term, trunc, info = env.step(pols[i](obs[i]))
            if term or trunc:
                done[i] = True
                laps_t = info["lap_times"]
                res = f"{min(laps_t):.2f} s" if laps_t else f"{info['total_progress']:.0f} m ({info.get('episode_done')})"
                env._viewer.title.set_text(f"{models[i][1]} – {res}")
        step += 1
        if step % every == 0:
            imgs = [env.render() for env in envs]
            frames.append(compose(imgs))
    for env in envs:
        env.close()
    return frames


def compose(imgs: list[np.ndarray]) -> np.ndarray:
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = imgs[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run directory; picks up to --n checkpoints + best")
    ap.add_argument("--n", type=int, default=4, help="number of checkpoints to pick from the run")
    ap.add_argument("--model", action="append", default=[], help="path[:label], can be repeated")
    ap.add_argument("--out", default="docs/video/evolution.mp4")
    ap.add_argument("--grid", action="store_true", help="all models at the same time in a grid")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=90.0)
    ap.add_argument("--every", type=int, default=2, help="render every n-th step (20 Hz / n = video fps)")
    ap.add_argument("--dpi", type=int, default=72)
    args = ap.parse_args()

    models: list[tuple[Path, str]] = []
    for m in args.model:
        path, _, label = m.partition(":")
        models.append((Path(path), label or Path(path).stem))
    if args.run:
        models += pick_checkpoints(Path(args.run), args.n)
    if not models:
        ap.error("give --run or --model")
    print("models:", [f"{m} ({l})" for m, l in models])

    fps = max(1, round(20 / args.every))
    if args.grid:
        figsize = (5.6, 4.6)
        frames = grid_frames(models, args.laps, args.every, args.max_time, figsize, args.dpi)
    else:
        figsize = (8, 6.4)
        frames = []
        for m, label in models:
            seg, result = segment_frames(m, label, args.laps, args.every, args.max_time, figsize, args.dpi)
            print(f"  {label}: {result} ({len(seg)} frames)")
            frames += seg
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    arr = np.stack(frames)
    if out.suffix.lower() == ".gif":
        iio.imwrite(out, arr, duration=1000 / fps, loop=0)
    else:
        iio.imwrite(out, arr, fps=fps, codec="libx264", macro_block_size=1)
    print(f"wrote {out}: {len(frames)} frames, {len(frames) / fps:.0f} s at {fps} fps")


if __name__ == "__main__":
    main()
