"""Search a small actor-output calibration, with unchanged track and kart physics."""

import argparse
import copy
import json
import math
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kart_rl.checkpoints import load_portable_ppo
from scripts.compare_driving_penalty import evaluate


def apply_calibration(model, weight, bias, steering_gain, throttle_bias, throttle_gain=1.0):
    """Always start from the original head, never accumulate candidate changes."""
    import torch

    if (
        not math.isfinite(steering_gain)
        or steering_gain <= 0
        or not math.isfinite(throttle_bias)
        or not math.isfinite(throttle_gain)
        or throttle_gain <= 0
    ):
        raise ValueError("gains must be positive and all calibration values finite")

    with torch.no_grad():
        model.policy.action_net.weight.copy_(weight)
        model.policy.action_net.bias.copy_(bias)
        model.policy.action_net.weight[0].mul_(steering_gain)
        model.policy.action_net.bias[0].mul_(steering_gain)
        model.policy.action_net.weight[1].mul_(throttle_gain)
        model.policy.action_net.bias[1].mul_(throttle_gain)
        model.policy.action_net.bias[1].add_(throttle_bias)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "runs/phase1_v2/best.zip")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/driver_calibration_v1")
    ap.add_argument("--gains", type=float, nargs="+", default=[0.96, 0.98, 1.0, 1.02, 1.04])
    ap.add_argument("--biases", type=float, nargs="+", default=[-0.06, -0.03, 0.0, 0.03, 0.06])
    ap.add_argument("--throttle-gains", type=float, nargs="+", default=[1.0])
    args = ap.parse_args()
    import torch

    torch.set_num_threads(1)
    out = args.out.resolve()
    if out.exists():
        ap.error("choose a fresh --out")
    out.mkdir(parents=True)
    cfg = yaml.safe_load(args.model.with_name("config.yaml").read_text(encoding="utf-8"))
    model = load_portable_ppo(args.model, cfg["ppo"], device="cpu", tensorboard_log=None)
    # Jery's snapshot normalizes reward only. Reject other setups rather than
    # silently evaluating the wrong observation distribution.
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from kart_rl.envs import KartLapEnv

    vec = DummyVecEnv([lambda: KartLapEnv(cfg.get("env", {}))])
    vec = VecNormalize.load(str(args.model.with_name("best_vecnormalize.pkl")), vec)
    if vec.norm_obs:
        vec.close()
        ap.error("this calibration currently requires norm_obs=False")
    weight = model.policy.action_net.weight.detach().clone()
    bias = model.policy.action_net.bias.detach().clone()
    original = evaluate(model, cfg.get("env", {}), 10)
    rows = []
    best = None
    try:
        for gain, offset, throttle_gain in product(args.gains, args.biases, args.throttle_gains):
            apply_calibration(model, weight, bias, gain, offset, throttle_gain)
            result = evaluate(model, cfg.get("env", {}), 3)
            result.update(steering_gain=gain, throttle_bias=offset, throttle_gain=throttle_gain)
            rows.append(result)
            if result["finished"] and (best is None or result["mean_flying_s"] < best["mean_flying_s"]):
                best = result
            (out / "search.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(
                f"gain={gain:g}, bias={offset:g}, throttle_gain={throttle_gain:g}: "
                f"{result['episode_done']}, mean={result['mean_flying_s']}",
                flush=True,
            )
        if best is None:
            raise RuntimeError("no valid candidate")
        apply_calibration(
            model, weight, bias, best["steering_gain"], best["throttle_bias"], best["throttle_gain"]
        )
        validation = evaluate(model, cfg.get("env", {}), 10)
        if not validation["finished"]:
            raise RuntimeError("selected calibration failed the ten-lap validation; no model promoted")
        model.save(out / "best.zip")
        vec.save(str(out / "best_vecnormalize.pkl"))
        effective = copy.deepcopy(cfg)
        effective["calibration"] = {
            "steering_gain": best["steering_gain"],
            "throttle_bias": best["throttle_bias"],
            "throttle_gain": best["throttle_gain"],
        }
        (out / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        result = {
            "original": original,
            "selection": best,
            "validation": validation,
            "method": "actor head calibration; no physics or track changes; not PPO training",
            "source_model": str(args.model),
            "trials": len(rows),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
