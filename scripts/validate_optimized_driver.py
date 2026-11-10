"""Validate an optimized policy against the original on long runs and held-out starts."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kart_rl.checkpoints import load_portable_ppo
from kart_rl.envs import KartLapEnv
from kart_rl.evaluation import run_episode
from scripts.compare_driving_penalty import evaluate


def qualifies(original, candidate):
    return (
        original["long_run"]["finished"]
        and candidate["long_run"]["finished"]
        and candidate["long_run"]["mean_flying_s"] < original["long_run"]["mean_flying_s"]
        and candidate["random_finished"] >= original["random_finished"]
    )


def validate(path, env_config, laps, seeds):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = yaml.safe_load(path.with_name("config.yaml").read_text(encoding="utf-8"))
    vec = DummyVecEnv([lambda: KartLapEnv(env_config)])
    vec = VecNormalize.load(str(path.with_name("best_vecnormalize.pkl")), vec)
    vec.training = False
    try:
        model = load_portable_ppo(path, cfg["ppo"], device="cpu")
        long_run = evaluate(model, env_config, laps, vec)
        starts = []

        def policy(obs):
            return model.predict(vec.normalize_obs(obs), deterministic=True)[0]

        for seed in seeds:
            env = KartLapEnv({**env_config, "random_start": True, "laps": 2, "max_time": 180.0})
            try:
                result = run_episode(env, policy, seed=seed)
                starts.append(
                    {
                        "seed": seed,
                        "finished": result.episode_done == "laps",
                        "episode_done": result.episode_done,
                        "distance_m": result.distance,
                    }
                )
            finally:
                env.close()
        return {
            "model": str(path),
            "long_run": long_run,
            "random_starts": starts,
            "random_finished": sum(r["finished"] for r in starts),
        }
    finally:
        vec.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--original", type=Path, default=ROOT / "runs/phase1_v2/best.zip")
    ap.add_argument("--laps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/driver_validation_v1")
    args = ap.parse_args()
    if args.laps < 2 or args.out.exists():
        ap.error("use at least two laps and a fresh --out")
    args.out.mkdir(parents=True)
    import torch

    torch.set_num_threads(1)
    config = yaml.safe_load(args.original.with_name("config.yaml").read_text(encoding="utf-8"))
    seeds = list(range(100, 110))
    results = {}
    for name, path in (("original", args.original), ("candidate", args.candidate)):
        results[name] = validate(path, config["env"], args.laps, seeds)
        print(
            f"{name}: mean {results[name]['long_run']['mean_flying_s']}, "
            f"random starts {results[name]['random_finished']}/{len(seeds)}",
            flush=True,
        )
        (args.out / f"{name}.json").write_text(json.dumps(results[name], indent=2), encoding="utf-8")
    results["improved"] = qualifies(results["original"], results["candidate"])
    results["laps"] = args.laps
    (args.out / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if results["improved"]:
        winner = args.out / "winner"
        winner.mkdir()
        for name in ("best.zip", "best_vecnormalize.pkl", "config.yaml"):
            shutil.copyfile(args.candidate.with_name(name), winner / name)
        print(f"Validated faster model: {winner / 'best.zip'}", flush=True)
    else:
        print("Candidate not promoted: speed or reliability criterion failed.", flush=True)


if __name__ == "__main__":
    main()
