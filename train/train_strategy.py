"""Train lap-level PPO pit decisions, independently of the driving policy."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kart_rl.envs.strategy_env import KartStrategyEnv, StrategyConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2048, help="strategy decisions, not physics steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--laps", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("runs/strategy_ppo"))
    args = ap.parse_args()
    if args.steps <= 0:
        ap.error("--steps must be positive")
    if args.rollout_steps < 2:
        ap.error("--rollout-steps must be at least 2")
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    config = StrategyConfig(laps=args.laps)
    env = Monitor(KartStrategyEnv(config))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "strategy_config.json").write_text(
        json.dumps(
            {
                "env": config.__dict__,
                "seed": args.seed,
                "steps": args.steps,
                "rollout_steps": args.rollout_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=args.rollout_steps,
            batch_size=args.rollout_steps,
            gamma=1.0,
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
        model.learn(total_timesteps=args.steps)
        model.save(args.out / "strategy.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
