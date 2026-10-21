#!/usr/bin/env python3
"""Optional add-on: let the agent learn from your own driving (behaviour cloning).

The game (python -m kart_rl.game) records every run as replays/human/<name>_<time>_demo.json:
the observation the agent would have seen and the action you gave, step by step. This
script keeps only your *valid* laps (no barrier contact) that are faster than a threshold -
by default the ghost's best lap of the same run, i.e. "learn only where you beat the model" -
and fits the policy network of an existing Stable-Baselines3 model to those
(observation -> action) pairs. The result is an ordinary checkpoint: continue it with PPO,
and reinforcement learning polishes what it copied.

    python train/learn_from_human.py --model runs/phase1_v2/best.zip --out runs/phase1_v2_human
    python train/train_sb3.py --config train/configs/phase1_ppo.yaml --run-name phase1_v2_human \\
        --resume runs/phase1_v2_human/checkpoints/last.zip --more 3000000 --stream
    # or in the dashboard: "Weitertrainieren ab runs/phase1_v2_human/checkpoints/last.zip"

Pure add-on: nothing in the normal training pipeline depends on it.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_demos(pattern: str, faster_than: float | str = "ghost", all_laps: bool = False, verbose: bool = True):
    """Collect (obs, act) pairs from the demo files. Returns (obs, act, report)."""
    obs_all, act_all, report = [], [], []
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        meta = d["meta"]
        lap_times, valid = meta["lap_times"], meta.get("valid", [True] * len(meta["lap_times"]))
        if all_laps:
            threshold = float("inf")
        elif faster_than == "ghost":
            threshold = meta.get("ghost_best") or float("inf")
        else:
            threshold = float(faster_than)
        keep = {i for i, (t, ok) in enumerate(zip(lap_times, valid)) if ok and t < threshold}
        lap = np.asarray(d["lap"])
        mask = np.isin(lap, list(keep)) if keep else np.zeros(len(lap), bool)
        obs, act = np.asarray(d["obs"], np.float32)[mask], np.asarray(d["act"], np.float32)[mask]
        report.append({"file": f, "laps": len(lap_times), "used": sorted(keep),
                       "used_times": [lap_times[i] for i in sorted(keep)], "threshold": threshold, "steps": int(mask.sum())})
        if verbose:
            print(f"{f}: {len(lap_times)} laps, threshold {threshold if threshold != float('inf') else '-'} s "
                  f"-> using laps {sorted(keep)} ({[lap_times[i] for i in sorted(keep)]}), {int(mask.sum())} steps")
        if mask.any():
            obs_all.append(obs)
            act_all.append(act)
    if not obs_all:
        return np.zeros((0, 1), np.float32), np.zeros((0, 2), np.float32), report
    return np.concatenate(obs_all), np.concatenate(act_all), report


def behaviour_cloning(model, obs: np.ndarray, act: np.ndarray, epochs: int = 30, lr: float = 1e-3,
                      batch: int = 256, seed: int = 0, verbose: bool = True) -> dict:
    """Maximum-likelihood fit of the SB3 actor to the demonstrations (value net untouched)."""
    import torch

    torch.manual_seed(seed)
    policy = model.policy
    device = policy.device
    X = torch.as_tensor(obs, dtype=torch.float32, device=device)
    Y = torch.as_tensor(np.clip(act, -1.0, 1.0), dtype=torch.float32, device=device)
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters()) + [policy.log_std]
    opt = torch.optim.Adam(params, lr=lr)

    def nll(xb, yb):
        return -policy.get_distribution(xb).log_prob(yb).mean()

    with torch.no_grad():
        before = float(nll(X, Y))
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = nll(X[idx], Y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (ep == 0 or (ep + 1) % 5 == 0):
            print(f"  epoch {ep + 1:3d}  NLL {total / n:.3f}")
    with torch.no_grad():
        policy.log_std.data.clamp_(min=-1.5)   # keep some exploration noise for the PPO that follows
        after = float(nll(X, Y))
        mse = float(((policy.get_distribution(X).mode() - Y) ** 2).mean())
    return {"nll_before": before, "nll_after": after, "mse_after": mse, "steps": int(n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", default="replays/human/*_demo.json", help="glob of demo files from the game")
    ap.add_argument("--model", required=True, help="SB3 model to start from (runs/<name>/best.zip)")
    ap.add_argument("--out", required=True, help="new run directory (checkpoints/last.zip to continue from)")
    ap.add_argument("--faster-than", default="ghost",
                    help="use laps faster than this [s]; 'ghost' = the model's best lap in the same run (default)")
    ap.add_argument("--all-laps", action="store_true", help="use every valid lap regardless of its time")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--no-eval", action="store_true", help="skip the lap-time check of the result")
    args = ap.parse_args()

    from stable_baselines3 import PPO

    obs, act, report = load_demos(args.demos, args.faster_than, args.all_laps)
    if len(obs) == 0:
        print("no usable laps: drive valid laps (no barrier contact) that beat the ghost, or use --all-laps / --faster-than")
        return 1
    model = PPO.load(args.model, device="cpu")
    if obs.shape[1] != model.observation_space.shape[0]:
        print(f"observation size {obs.shape[1]} of the demos does not match the model ({model.observation_space.shape[0]})")
        return 1
    print(f"behaviour cloning on {len(obs)} steps from {sum(1 for r in report if r['steps'])} run(s) ...")
    stats = behaviour_cloning(model, obs, act, epochs=args.epochs, lr=args.lr, batch=args.batch)
    print(f"NLL {stats['nll_before']:.3f} -> {stats['nll_after']:.3f}, action MSE {stats['mse_after']:.4f}")

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    model.save(out / "checkpoints" / "last.zip")
    src = Path(args.model)
    for cand in (src.parent / "checkpoints" / "vecnormalize.pkl", src.parent / "best_vecnormalize.pkl",
                 src.with_name("vecnormalize.pkl")):
        if cand.exists():
            shutil.copy(cand, out / "checkpoints" / "vecnormalize.pkl")
            break
    (out / "human_bc.json").write_text(json.dumps({"model": str(src), "demos": report, **stats}, indent=1))
    print(f"wrote {out / 'checkpoints' / 'last.zip'}  (continue: --resume {out / 'checkpoints' / 'last.zip'})")

    if not args.no_eval:
        from kart_rl.agents import load_policy
        from kart_rl.envs import KartLapEnv, LapEnvConfig
        from kart_rl.evaluation import run_episode

        env = KartLapEnv(LapEnvConfig(random_start=False, laps=2))
        res = run_episode(env, load_policy(out / "checkpoints" / "last.zip"))
        print(f"cloned policy, standing start: laps {[round(t, 2) for t in res.lap_times]}  end {res.episode_done}  "
              f"(it usually needs the PPO continuation to drive clean laps again)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
