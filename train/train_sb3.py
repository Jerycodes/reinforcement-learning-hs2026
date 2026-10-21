#!/usr/bin/env python3
"""Train a PPO agent with Stable-Baselines3 on KartLap-v0.

    python train/train_sb3.py --config train/configs/phase1_ppo.yaml
    python train/train_sb3.py --config ... --total-timesteps 500000 --n-envs 4
    python train/train_sb3.py --config ... --resume runs/phase1_ppo/checkpoints/last.zip

Outputs go to runs/<run_name>/:
    tensorboard/            TensorBoard logs      (tensorboard --logdir runs)
    checkpoints/last.zip    latest model (+ vecnormalize.pkl)
    checkpoints/step_*.zip  periodic checkpoints
    best.zip                best model by standing-start lap time
    config.yaml             the config that was used
    eval.csv                lap-time evaluation history
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from kart_rl.envs import KartLapEnv, LapEnvConfig
from kart_rl.evaluation import evaluate_lap_time, load_reference_lap_times

ACTIVATIONS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}


def make_env_fn(env_cfg: dict, seed: int, rank: int):
    def _init():
        env = KartLapEnv(LapEnvConfig.from_dict(env_cfg))
        env.reset(seed=seed + rank)
        return env
    return _init


class LapTimeEval(BaseCallback):
    """Every ``every`` env steps: deterministic standing-start run, log lap time, keep best model."""

    def __init__(self, env_cfg: dict, every: int, laps: int, run_dir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.env_cfg, self.every, self.laps, self.run_dir = env_cfg, every, laps, run_dir
        self.next_eval = every
        self.best = float("inf")
        self.csv_path = run_dir / "eval.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "wall_s", "episode_done", "standing_lap", "best_flying_lap",
                                    "distance_m"])
        self.t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            self.evaluate()
        return True

    def evaluate(self):
        model = self.model
        vec = model.get_env()

        def policy(obs):
            o = obs
            if isinstance(vec, VecNormalize):
                o = vec.normalize_obs(obs)
            a, _ = model.predict(o, deterministic=True)
            return a

        res = evaluate_lap_time(policy, self.env_cfg, laps=self.laps)
        lap = res.best_lap if res.best_lap is not None else float("inf")
        standing = res.standing_lap if res.standing_lap is not None else float("nan")
        self.logger.record("eval/best_flying_lap_s", lap if np.isfinite(lap) else float("nan"))
        self.logger.record("eval/standing_lap_s", standing)
        self.logger.record("eval/distance_m", res.distance)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, round(time.time() - self.t0, 1), res.episode_done,
                                    round(standing, 2), round(lap, 2) if np.isfinite(lap) else "",
                                    round(res.distance, 1)])
        if self.verbose:
            print(f"[eval @ {self.num_timesteps:,} steps] {res.episode_done}: laps "
                  f"{[round(t, 2) for t in res.lap_times]}  distance {res.distance:.0f} m")
        if lap < self.best:
            self.best = lap
            model.save(self.run_dir / "best.zip")
            if isinstance(vec, VecNormalize):
                vec.save(str(self.run_dir / "best_vecnormalize.pkl"))
            print(f"    new best lap {lap:.2f} s -> saved {self.run_dir / 'best.zip'}")


class LiveStream(BaseCallback):
    """Stream the state of every training environment to the 3-D viewer (``--stream``)."""

    def __init__(self, port: int, run_name: str, eval_cb: "LapTimeEval", verbose: int = 0):
        super().__init__(verbose)
        from kart_rl.live.stream import StateStreamer

        self.streamer = StateStreamer(port=port, hello={"track": "Race-Inn Roggwil", "policy": f"training:{run_name}"})
        self.eval_cb = eval_cb
        self.episodes = 0
        self.t0 = time.time()
        print(f"live stream on ws://localhost:{port}  ->  open viewer with ?live")

    def _on_step(self) -> bool:
        from kart_rl.live.stream import env_state_from_info

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        self.episodes += int(np.sum(dones)) if len(dones) else 0
        if not self.streamer.connected:
            return True
        states = [env_state_from_info(info, i) for i, info in enumerate(infos)]
        buf = self.model.ep_info_buffer
        training = {
            "steps": int(self.num_timesteps), "episodes": int(self.episodes),
            "ep_rew_mean": float(np.mean([e["r"] for e in buf])) if len(buf) else None,
            "ep_len_mean": float(np.mean([e["l"] for e in buf])) if len(buf) else None,
            "best_lap": None if not np.isfinite(self.eval_cb.best) else float(self.eval_cb.best),
            "elapsed_s": round(time.time() - self.t0, 1),
            "steps_per_s": round(self.num_timesteps / max(time.time() - self.t0, 1e-6), 1),
        }
        self.streamer.send({"type": "states", "states": states, "training": training}, force=bool(np.any(dones)))
        return True


class Curriculum(BaseCallback):
    """Apply env config changes at given step counts, e.g. sector lengths that grow over training.

    config.yaml:
        curriculum:
          - {at: 0,       sector_length_m: 80}
          - {at: 400000,  sector_length_m: 200}
          - {at: 1000000, sector_length_m: 0}      # 0 = full laps
    """

    def __init__(self, stages: list[dict], verbose: int = 1):
        super().__init__(verbose)
        self.stages = sorted(stages, key=lambda st: int(st.get("at", 0)))
        self.applied = -1

    def _on_step(self) -> bool:
        idx = -1
        for i, st in enumerate(self.stages):
            if self.num_timesteps >= int(st.get("at", 0)):
                idx = i
        if idx != self.applied:
            self.applied = idx
            st = {k: v for k, v in self.stages[idx].items() if k != "at"}
            for key, val in st.items():
                self.model.get_env().env_method("set_config", key, val)
            if self.verbose:
                print(f"[curriculum @ {self.num_timesteps:,} steps] {st}")
        return True


def linear_schedule(initial: float):
    return lambda progress_remaining: progress_remaining * initial


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="train/configs/phase1_ppo.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None, help="model .zip to continue training from")
    ap.add_argument("--subproc", action="store_true", default=None, help="force SubprocVecEnv")
    ap.add_argument("--no-subproc", dest="subproc", action="store_false")
    ap.add_argument("--stream", action="store_true", help="stream all training envs to the 3-D viewer (ws://localhost:8765)")
    ap.add_argument("--stream-port", type=int, default=8765)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.total_timesteps:
        cfg["total_timesteps"] = args.total_timesteps
    if args.n_envs:
        cfg["n_envs"] = args.n_envs
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_dir = Path("runs") / cfg["run_name"]
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    env_cfg = cfg.get("env", {})
    n_envs = int(cfg["n_envs"])
    use_subproc = args.subproc if args.subproc is not None else n_envs > 1
    fns = [make_env_fn(env_cfg, int(cfg["seed"]), i) for i in range(n_envs)]
    vec = SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns)
    vec = VecMonitor(vec)
    if cfg.get("normalize_reward", True):
        if args.resume and Path(args.resume).with_name("vecnormalize.pkl").exists():
            vec = VecNormalize.load(str(Path(args.resume).with_name("vecnormalize.pkl")), vec)
        else:
            vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=float(cfg["ppo"]["gamma"]))

    algo = cfg.get("algo", "ppo").lower()
    p = cfg["ppo"]
    if algo == "ppo":
        policy_kwargs = dict(net_arch=dict(pi=list(p["hidden_sizes"]), vf=list(p["hidden_sizes"])),
                             activation_fn=ACTIVATIONS[p.get("activation", "tanh")],
                             log_std_init=float(p.get("log_std_init", 0.0)))
        lr = float(p["learning_rate"])
        lr = linear_schedule(lr) if p.get("lr_schedule", "constant") == "linear" else lr
        if args.resume:
            model = PPO.load(args.resume, env=vec, device="cpu")
            print(f"resumed from {args.resume} at {model.num_timesteps:,} steps")
        else:
            model = PPO("MlpPolicy", vec, learning_rate=lr, n_steps=int(p["n_steps"]), batch_size=int(p["batch_size"]),
                        n_epochs=int(p["n_epochs"]), gamma=float(p["gamma"]), gae_lambda=float(p["gae_lambda"]),
                        clip_range=float(p["clip_range"]), ent_coef=float(p["ent_coef"]), vf_coef=float(p["vf_coef"]),
                        max_grad_norm=float(p["max_grad_norm"]), policy_kwargs=policy_kwargs, seed=int(cfg["seed"]),
                        tensorboard_log=str(run_dir / "tensorboard"), device="cpu", verbose=0)
    elif algo == "sac":
        from stable_baselines3 import SAC

        q = cfg.get("sac", {})
        policy_kwargs = dict(net_arch=list(q.get("hidden_sizes", p["hidden_sizes"])))
        if args.resume:
            model = SAC.load(args.resume, env=vec, device="cpu")
            print(f"resumed from {args.resume} at {model.num_timesteps:,} steps")
        else:
            model = SAC("MlpPolicy", vec, learning_rate=float(q.get("learning_rate", 3e-4)),
                        buffer_size=int(q.get("buffer_size", 300_000)), learning_starts=int(q.get("learning_starts", 10_000)),
                        batch_size=int(q.get("batch_size", 256)), tau=float(q.get("tau", 0.005)),
                        gamma=float(q.get("gamma", p["gamma"])), train_freq=int(q.get("train_freq", 1)),
                        gradient_steps=int(q.get("gradient_steps", 1)), ent_coef=q.get("ent_coef", "auto"),
                        policy_kwargs=policy_kwargs, seed=int(cfg["seed"]),
                        tensorboard_log=str(run_dir / "tensorboard"), device="cpu", verbose=0)
    else:
        raise SystemExit(f"unknown algo '{algo}' (ppo | sac)")
    print(model.policy)

    eval_cb = LapTimeEval(env_cfg, every=int(cfg["eval_every"]), laps=int(cfg.get("eval_laps", 2)), run_dir=run_dir)
    callbacks = [
        CheckpointCallback(save_freq=max(int(cfg["checkpoint_every"]) // n_envs, 1), save_path=str(run_dir / "checkpoints"),
                           name_prefix="step", save_vecnormalize=True),
        eval_cb,
    ]
    if args.stream:
        callbacks.append(LiveStream(args.stream_port, cfg["run_name"], eval_cb))
    if cfg.get("curriculum"):
        callbacks.append(Curriculum(cfg["curriculum"]))
    refs = load_reference_lap_times()
    print("reference lap times:", {k: round(v, 1) for k, v in refs.items()})
    t0 = time.time()
    try:
        model.learn(total_timesteps=int(cfg["total_timesteps"]), callback=callbacks, progress_bar=False,
                    reset_num_timesteps=not args.resume, tb_log_name=algo)
    except KeyboardInterrupt:
        print("interrupted - saving last model")
    model.save(run_dir / "checkpoints" / "last.zip")
    if isinstance(vec, VecNormalize):
        vec.save(str(run_dir / "checkpoints" / "vecnormalize.pkl"))
    dt = time.time() - t0
    print(f"done: {model.num_timesteps:,} steps in {dt / 60:.1f} min ({model.num_timesteps / max(dt, 1):.0f} steps/s)")
    print(f"models: {run_dir / 'checkpoints' / 'last.zip'}  best: {run_dir / 'best.zip'}")
    vec.close()


if __name__ == "__main__":
    main()
