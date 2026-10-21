#!/usr/bin/env python3
"""Population Based Training (PBT) for KartLap-v0.

Several PPO agents ("workers") train in parallel with different hyper-parameters.
Every ``interval`` steps the whole population is evaluated (standing-start lap
time). The worst workers then copy the weights and hyper-parameters of the best
ones (exploit) and perturb the hyper-parameters a little (explore) - the
population "talks" through the coordinator and the best policy is always kept
as runs/<name>/best.zip and shown in the 3-D viewer.
Reference: Jaderberg et al., "Population Based Training of Neural Networks", 2017.

    python train/pbt.py --config train/configs/phase1_ppo.yaml --workers 4 --n-envs 4 --stream
    python train/pbt.py --workers 6 --n-envs 2 --interval 200000 --total 4000000 --run-name pbt_big

Each worker is one process with ``n_envs`` in-process environments, so
``workers * n_envs`` should be about the number of CPU cores.
"""
from __future__ import annotations

import argparse
import copy
import csv
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np
import yaml

# hyper-parameters that PBT may change, with bounds
PERTURB = {
    "learning_rate": (1e-5, 1e-3),
    "ent_coef": (0.0, 0.03),
    "clip_range": (0.05, 0.4),
    "gamma": (0.95, 0.999),
}


# --------------------------------------------------------------------------- #
# worker process
# --------------------------------------------------------------------------- #

def worker_main(wid: int, cfg: dict, hparams: dict, run_dir: str, interval: int, seed: int, cmd_q, res_q):
    import torch
    from stable_baselines3 import PPO
    try:
        from stable_baselines3.common.utils import FloatSchedule as _sched
    except ImportError:  # older SB3
        from stable_baselines3.common.utils import get_schedule_fn as _sched
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    from kart_rl.envs import KartLapEnv, LapEnvConfig
    from kart_rl.evaluation import evaluate_lap_time

    torch.set_num_threads(1)
    env_cfg = cfg.get("env", {})
    p = cfg["ppo"]
    n_envs = int(cfg["n_envs"])

    def make(i):
        def _init():
            env = KartLapEnv(LapEnvConfig.from_dict(env_cfg))
            env.reset(seed=seed + i)
            return env
        return _init

    vec = VecMonitor(DummyVecEnv([make(i) for i in range(n_envs)]))
    if cfg.get("normalize_reward", True):
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=float(hparams["gamma"]))
    policy_kwargs = dict(net_arch=dict(pi=list(p["hidden_sizes"]), vf=list(p["hidden_sizes"])),
                         activation_fn={"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[p.get("activation", "tanh")],
                         log_std_init=float(p.get("log_std_init", 0.0)))
    model = PPO("MlpPolicy", vec, learning_rate=float(hparams["learning_rate"]), n_steps=int(p["n_steps"]),
                batch_size=int(p["batch_size"]), n_epochs=int(p["n_epochs"]), gamma=float(hparams["gamma"]),
                gae_lambda=float(p["gae_lambda"]), clip_range=float(hparams["clip_range"]),
                ent_coef=float(hparams["ent_coef"]), vf_coef=float(p["vf_coef"]), max_grad_norm=float(p["max_grad_norm"]),
                policy_kwargs=policy_kwargs, seed=seed, device="cpu", verbose=0)
    wdir = Path(run_dir) / "workers" / f"w{wid}"
    wdir.mkdir(parents=True, exist_ok=True)

    def apply_hparams(h):
        model.learning_rate = float(h["learning_rate"])
        model._setup_lr_schedule()
        model.ent_coef = float(h["ent_coef"])
        model.clip_range = _sched(float(h["clip_range"]))
        model.gamma = float(h["gamma"])
        model.rollout_buffer.gamma = float(h["gamma"])

    def policy(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    while True:
        cmd = cmd_q.get()
        if cmd["action"] == "stop":
            break
        if cmd["action"] == "exploit":
            model.set_parameters(cmd["from_path"], exact_match=True, device="cpu")
            hparams = cmd["hparams"]
            apply_hparams(hparams)
        model.learn(total_timesteps=interval, reset_num_timesteps=False, progress_bar=False)
        res = evaluate_lap_time(policy, env_cfg, laps=2)
        lap = res.best_lap
        score = lap if lap is not None else 200.0 + (1280.0 - res.distance) / 10.0
        path = wdir / "last.zip"
        model.save(path)
        buf = model.ep_info_buffer
        res_q.put({"wid": wid, "steps": int(model.num_timesteps), "lap": lap, "score": float(score),
                   "laps": list(res.lap_times), "distance": float(res.distance), "hparams": dict(hparams),
                   "path": str(path), "ep_rew_mean": float(np.mean([e["r"] for e in buf])) if len(buf) else None})
    vec.close()


# --------------------------------------------------------------------------- #
# coordinator
# --------------------------------------------------------------------------- #

def spread(h: dict, rng: np.random.Generator) -> dict:
    """Initial diversity: random factors between 0.4x and 2.5x for the first generation."""
    out = dict(h)
    for key, (lo, hi) in PERTURB.items():
        factor = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))
        if key == "gamma":
            out[key] = 1.0 - 1.0 / (1.0 / (1.0 - out[key]) * factor)
        else:
            out[key] = out[key] * factor
        out[key] = float(np.clip(out[key], lo, hi))
    return out


def perturb(h: dict, rng: np.random.Generator) -> dict:
    out = dict(h)
    for key, (lo, hi) in PERTURB.items():
        factor = rng.choice([0.8, 1.25])
        if key == "gamma":  # perturb the "horizon" 1/(1-gamma) instead of gamma itself
            horizon = 1.0 / (1.0 - out[key]) * factor
            out[key] = 1.0 - 1.0 / horizon
        else:
            out[key] = out[key] * factor
        out[key] = float(np.clip(out[key], lo, hi))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="train/configs/phase1_ppo.yaml")
    ap.add_argument("--run-name", default="pbt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-envs", type=int, default=4, help="environments per worker (in-process)")
    ap.add_argument("--interval", type=int, default=250_000, help="steps per worker between evaluations")
    ap.add_argument("--total", type=int, default=5_000_000, help="steps per worker in total")
    ap.add_argument("--exploit-frac", type=float, default=0.25, help="fraction of the population replaced per generation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream", action="store_true", help="3-D viewer: the current best policy drives live")
    ap.add_argument("--stream-port", type=int, default=8765)
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["n_envs"] = args.n_envs
    cfg["ppo"]["lr_schedule"] = "constant"  # PBT adapts the learning rate itself
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "pbt": vars(args)}, sort_keys=False))
    rng = np.random.default_rng(args.seed)
    base = {"learning_rate": float(cfg["ppo"]["learning_rate"]), "ent_coef": float(cfg["ppo"]["ent_coef"]),
            "clip_range": float(cfg["ppo"]["clip_range"]), "gamma": float(cfg["ppo"]["gamma"])}
    hparams = [base if i == 0 else spread(base, rng) for i in range(args.workers)]

    ctx = mp.get_context("spawn")
    cmd_qs, res_q, procs = [], ctx.Queue(), []
    for i in range(args.workers):
        q = ctx.Queue()
        pr = ctx.Process(target=worker_main, args=(i, cfg, hparams[i], str(run_dir), args.interval,
                                                   args.seed * 1000 + i * 100, q, res_q), daemon=True)
        pr.start()
        procs.append(pr)
        cmd_qs.append(q)
        q.put({"action": "continue"})
    print(f"PBT: {args.workers} workers x {args.n_envs} envs, interval {args.interval:,}, total {args.total:,} per worker")
    for i, h in enumerate(hparams):
        print(f"  w{i}: lr {h['learning_rate']:.2e} ent {h['ent_coef']:.4f} clip {h['clip_range']:.2f} gamma {h['gamma']:.4f}")

    # --- showcase / viewer -------------------------------------------------
    holder = {"model": None, "stats": {"steps": 0, "episodes": 0, "generation": 0, "best_lap": None, "population": []}}
    t0 = time.time()
    if args.stream:
        from kart_rl.live.stream import Showcase, StateStreamer, open_viewer

        streamer = StateStreamer(port=args.stream_port, hello={"track": "Race-Inn Roggwil", "policy": f"pbt:{args.run_name}"})
        print(f"3D viewer: {open_viewer(args.http_port, args.stream_port, open_browser=not args.no_open)}")

        def predict(obs):
            m = holder["model"]
            if m is None:
                return np.zeros(2, np.float32)
            a, _ = m.predict(obs, deterministic=True)
            return a

        def stats():
            st = dict(holder["stats"])
            st["elapsed_s"] = round(time.time() - t0, 1)
            st["steps_per_s"] = round(st["steps"] / max(time.time() - t0, 1e-6), 1)
            return st

        Showcase(streamer, cfg.get("env", {}), predict, stats).start()

    log_path = run_dir / "pbt_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["generation", "worker", "steps", "lap", "score", "distance", "learning_rate", "ent_coef",
                                "clip_range", "gamma", "exploited_from"])
    eval_path = run_dir / "eval.csv"
    with open(eval_path, "w", newline="") as f:
        csv.writer(f).writerow(["timesteps", "wall_s", "episode_done", "standing_lap", "best_flying_lap", "distance_m"])

    best_score, generation = float("inf"), 0
    generations = max(1, args.total // args.interval)
    try:
        for generation in range(1, generations + 1):
            results = {}
            while len(results) < args.workers:
                r = res_q.get()
                results[r["wid"]] = r
            ranked = sorted(results.values(), key=lambda r: r["score"])
            total_steps = sum(r["steps"] for r in results.values())
            best = ranked[0]
            if best["score"] < best_score:
                best_score = best["score"]
                import shutil

                shutil.copy(best["path"], run_dir / "best.zip")
                if args.stream:
                    from stable_baselines3 import PPO

                    holder["model"] = PPO.load(run_dir / "best.zip", device="cpu")
            n_exploit = max(1, int(round(args.exploit_frac * args.workers))) if args.workers >= 2 else 0
            losers = ranked[len(ranked) - n_exploit:] if n_exploit > 0 else []
            winners = ranked[:max(1, len(ranked) - n_exploit)]
            commands = {r["wid"]: {"action": "continue"} for r in ranked}
            exploited = {}
            for loser in losers:
                src = winners[rng.integers(len(winners))]
                if src["wid"] == loser["wid"]:
                    continue
                commands[loser["wid"]] = {"action": "exploit", "from_path": src["path"],
                                          "hparams": perturb(src["hparams"], rng)}
                exploited[loser["wid"]] = src["wid"]
            # logging
            lap_str = lambda r: f"{r['lap']:.2f} s" if r["lap"] is not None else f"{r['distance']:.0f} m"
            best_str = f"{best_score:.2f} s" if best_score < 190 else "no lap yet"
            print(f"\n=== generation {generation}  ({total_steps:,} steps total, {(time.time() - t0) / 60:.1f} min)  "
                  f"best ever {best_str}")
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                for r in ranked:
                    h = r["hparams"]
                    tag = f"  <- copies w{exploited[r['wid']]}" if r["wid"] in exploited else ""
                    print(f"  w{r['wid']}: {lap_str(r):>9}  lr {h['learning_rate']:.2e} ent {h['ent_coef']:.4f} "
                          f"clip {h['clip_range']:.2f} gamma {h['gamma']:.4f}{tag}")
                    w.writerow([generation, r["wid"], r["steps"], r["lap"] if r["lap"] is not None else "",
                                round(r["score"], 3), round(r["distance"], 1), h["learning_rate"], h["ent_coef"],
                                h["clip_range"], h["gamma"], exploited.get(r["wid"], "")])
            with open(eval_path, "a", newline="") as f:
                csv.writer(f).writerow([total_steps, round(time.time() - t0, 1), "laps" if best["lap"] else "off_track",
                                        round(best["laps"][0], 2) if best["laps"] else "nan",
                                        round(best["lap"], 2) if best["lap"] is not None else "", round(best["distance"], 1)])
            holder["stats"] = {
                "steps": total_steps, "episodes": 0, "generation": generation,
                "best_lap": None if not np.isfinite(best_score) or best_score > 190 else float(best_score),
                "ep_rew_mean": best.get("ep_rew_mean"),
                "population": [{"id": r["wid"], "lap": r["lap"], "distance": r["distance"],
                                "lr": r["hparams"]["learning_rate"], "ent": r["hparams"]["ent_coef"],
                                "clip": r["hparams"]["clip_range"]} for r in ranked],
            }
            if generation < generations:
                for wid, cmd in commands.items():
                    cmd_qs[wid].put(cmd)
    except KeyboardInterrupt:
        print("interrupted")
    for q in cmd_qs:
        q.put({"action": "stop"})
    print(f"\nPBT done: best lap {best_score:.2f} s -> {run_dir / 'best.zip'}   log: {log_path}"
          if best_score < 190 else f"\nPBT done: no complete lap yet   log: {log_path}")
    if args.stream:
        holder["stats"]["finished"] = True
        print("the viewer keeps driving the best policy (Ctrl+C to quit)")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    for pr in procs:
        pr.join(timeout=10)


if __name__ == "__main__":
    main()
