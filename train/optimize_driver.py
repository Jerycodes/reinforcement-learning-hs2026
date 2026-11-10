"""Fine-tune a driver, retaining the fastest safe intermediate checkpoint."""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kart_rl.checkpoints import load_portable_ppo
from kart_rl.envs import KartLapEnv
from scripts.compare_driving_penalty import evaluate

PROFILES = {
    "gentle": {"learning_rate": 3e-5, "ent_coef": 0.0},
    "long_horizon": {"learning_rate": 3e-5, "ent_coef": 0.0, "gamma": 0.997, "gae_lambda": 0.97},
    "time_pressure": {"learning_rate": 3e-5, "ent_coef": 0.0, "reward_time": -0.05},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "runs/phase1_v2/best.zip")
    ap.add_argument("--steps", type=int, default=102400)
    ap.add_argument("--eval-every", type=int, default=20480)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    source = yaml.safe_load(args.model.with_name("config.yaml").read_text(encoding="utf-8"))
    rollout = int(source["n_envs"]) * int(source["ppo"]["n_steps"])
    if args.steps <= 0 or args.eval_every <= 0 or args.steps % rollout or args.eval_every % rollout:
        ap.error(f"steps and eval-every must be positive multiples of {rollout}")
    if args.out.exists():
        ap.error("choose a fresh --out")
    args.out.mkdir(parents=True)
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    torch.set_num_threads(1)
    options = PROFILES[args.profile]
    config = copy.deepcopy(source)
    config["env"].update({k: v for k, v in options.items() if k.startswith("reward_")})
    config["ppo"].update({k: v for k, v in options.items() if not k.startswith("reward_")})
    config["ppo"]["lr_schedule"] = "constant"
    config.update(run_name=args.out.name, total_timesteps=args.steps, seed=args.seed)
    (args.out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    vec = DummyVecEnv([lambda: KartLapEnv(config["env"]) for _ in range(int(source["n_envs"]))])
    vec = VecMonitor(vec, filename=str(args.out / "monitor.csv"))
    vec = VecNormalize.load(str(args.model.with_name("best_vecnormalize.pkl")), vec)
    vec.gamma = config["ppo"]["gamma"]
    vec.training = True
    changes = {k: v for k, v in options.items() if not k.startswith("reward_") and k != "learning_rate"}
    model = load_portable_ppo(
        args.model,
        source["ppo"],
        env=vec,
        device="cpu",
        learning_rate=options["learning_rate"],
        seed=args.seed,
        tensorboard_log=None,
        verbose=1,
        **changes,
    )
    initial = evaluate(model, source["env"], 3, vec)
    model.save(args.out / "best.zip")
    vec.save(str(args.out / "best_vecnormalize.pkl"))
    history = [{"steps": 0, **initial}]
    best_score = initial["mean_flying_s"] if initial["finished"] else float("inf")
    best_steps = 0

    class Selection(BaseCallback):
        def _on_step(self):
            nonlocal best_score, best_steps
            if self.num_timesteps % args.eval_every:
                return True
            result = evaluate(self.model, source["env"], 3, vec)
            history.append({"steps": self.num_timesteps, **result})
            score = result["mean_flying_s"] if result["finished"] else float("inf")
            if score < best_score:
                best_score, best_steps = score, self.num_timesteps
                self.model.save(args.out / "best.zip")
                vec.save(str(args.out / "best_vecnormalize.pkl"))
            (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(
                f"[selection {self.num_timesteps}] {result['episode_done']}, "
                f"mean={result['mean_flying_s']}, best={best_score}",
                flush=True,
            )
            return True

    try:
        model.learn(total_timesteps=args.steps, callback=Selection(), reset_num_timesteps=True)
        # Callback at a rollout boundary runs before that rollout's update.
        # Explicitly evaluate the final updated weights as well.
        final = evaluate(model, source["env"], 3, vec)
        history.append({"steps": model.num_timesteps, "after_final_update": True, **final})
        if final["finished"] and final["mean_flying_s"] < best_score:
            best_score, best_steps = final["mean_flying_s"], model.num_timesteps
            model.save(args.out / "best.zip")
            vec.save(str(args.out / "best_vecnormalize.pkl"))
        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        best = load_portable_ppo(args.out / "best.zip", config["ppo"], device="cpu")
        validation_vec = DummyVecEnv([lambda: KartLapEnv(source["env"])])
        validation_vec = VecNormalize.load(str(args.out / "best_vecnormalize.pkl"), validation_vec)
        validation_vec.training = False
        try:
            validation = evaluate(best, source["env"], 10, validation_vec)
        finally:
            validation_vec.close()
        (args.out / "result.json").write_text(
            json.dumps(
                {
                    "profile": args.profile,
                    "source_model": str(args.model),
                    "seed": args.seed,
                    "additional_steps": model.num_timesteps,
                    "selected_steps": best_steps,
                    "selection_mean_s": best_score,
                    "validation": validation,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(validation), flush=True)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
