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
    ap.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    ap.add_argument("--policy-only-eval", action="store_true", help="skip scripted reference evaluations")
    ap.add_argument("--laps", type=int, default=8)
    ap.add_argument("--reward-time-scale", type=float, default=100.0)
    ap.add_argument("--reward-progress", type=float, default=2.0)
    ap.add_argument("--reward-finish", type=float, default=15.0)
    ap.add_argument("--reward-failure", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path("runs/strategy_ppo"))
    args = ap.parse_args()
    if args.steps <= 0:
        ap.error("--steps must be positive")
    if args.rollout_steps < 2:
        ap.error("--rollout-steps must be at least 2")
    if args.threads < 1:
        ap.error("--threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    config = StrategyConfig(
        laps=args.laps,
        reward_time_scale_s=args.reward_time_scale,
        reward_progress=args.reward_progress,
        reward_finish=args.reward_finish,
        reward_failure=args.reward_failure,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    env = Monitor(
        KartStrategyEnv(config),
        filename=str(args.out / "monitor.csv"),
        info_keywords=("elapsed_s", "pit_stops", "finished"),
    )
    (args.out / "strategy_config.json").write_text(
        json.dumps(
            {
                "env": config.__dict__,
                "seed": args.seed,
                "steps": args.steps,
                "rollout_steps": args.rollout_steps,
                "threads": args.threads,
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
        from scripts.evaluate_strategy import run_strategy, write_report

        results = {
            "PPO-Strategie": run_strategy(
                config, lambda obs, info: model.predict(obs, deterministic=True)[0]
            ),
        }
        if not args.policy_only_eval:
            results["Ohne Stopp"] = run_strategy(config, lambda obs, info: 0)
        if not args.policy_only_eval and config.laps > 1:
            stop_lap = config.laps // 2
            results[f"Stopp nach Runde {stop_lap}"] = run_strategy(
                config, lambda obs, info: info["laps_done"] == stop_lap
            )
        write_report(args.out / "evaluation", config, results)
        for name, result in results.items():
            print(
                f"{name}: {result['elapsed_s']:.2f} s, return {result['return']:.3f}, "
                f"{result['pit_stops']} stop(s), {result['episode_done']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
