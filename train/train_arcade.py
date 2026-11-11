"""Train an arcade policy separately from the original lap benchmark."""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO

from kart_rl.envs.arcade_env import ArcadeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250000)
    parser.add_argument("--output", type=Path, default=Path("runs/arcade_v1/policy.zip"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    env = ArcadeEnv()
    try:
        model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, device="cpu", n_steps=512)
        model.arcade_rules = "KartArcade-v1"
        model.learn(args.steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
