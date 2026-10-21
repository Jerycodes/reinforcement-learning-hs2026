#!/usr/bin/env python3
"""PPO implemented from scratch in PyTorch (the "we understand the algorithm" part).

Same environment, same config file and same outputs as train/train_sb3.py, so
both can be compared directly:

    python train/ppo_scratch.py --config train/configs/phase1_ppo.yaml --run-name phase1_scratch
    python train/ppo_scratch.py --config ... --resume runs/phase1_scratch/checkpoints/last.pt

Algorithm (Schulman et al. 2017, "Proximal Policy Optimization Algorithms"):
    1. collect a rollout of n_steps per environment with the current policy
    2. compute advantages with GAE(lambda) and bootstrapped returns
    3. for n_epochs: shuffle the rollout into minibatches and maximise the
       clipped surrogate objective  L = min(r*A, clip(r, 1-eps, 1+eps)*A)
       minus a value loss, plus an entropy bonus
    4. repeat

Implementation notes
    - Gaussian policy with a state-independent log-std, actions clipped to [-1, 1]
    - orthogonal initialisation, advantage normalisation, gradient clipping
    - truncated episodes (time limit) bootstrap with V(s_T); terminated ones don't
    - our own tiny vector-env wrapper (resets finished envs immediately) so the
      rollout logic stays easy to read and independent of gymnasium's autoreset modes
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from kart_rl.envs import KartLapEnv, LapEnvConfig
from kart_rl.evaluation import evaluate_lap_time, load_reference_lap_times

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


# --------------------------------------------------------------------------- #
# environments
# --------------------------------------------------------------------------- #

class VecEnv:
    """Minimal synchronous vector env: step all, auto-reset the finished ones."""

    def __init__(self, env_cfg: dict, n: int, seed: int):
        self.envs = [KartLapEnv(LapEnvConfig.from_dict(env_cfg)) for _ in range(n)]
        self.obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])
        self.ep_return = np.zeros(n)
        self.ep_len = np.zeros(n, int)

    def step(self, actions: np.ndarray):
        obs, rew, term, trunc, final_obs, finished = [], [], [], [], [], []
        self.infos = []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, te, tr, info = env.step(a)
            self.infos.append(info)
            self.ep_return[i] += r
            self.ep_len[i] += 1
            fo = None
            if te or tr:
                fo = o
                finished.append({"return": self.ep_return[i], "length": self.ep_len[i],
                                 "done": info.get("episode_done", ""), "laps": len(info["lap_times"])})
                self.ep_return[i], self.ep_len[i] = 0.0, 0
                o, _ = env.reset()
            obs.append(o)
            rew.append(r)
            term.append(te)
            trunc.append(tr)
            final_obs.append(fo)
        self.obs = np.stack(obs)
        return self.obs, np.array(rew), np.array(term), np.array(trunc), final_obs, finished


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(sizes, act):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [layer_init(nn.Linear(sizes[i], sizes[i + 1])), act()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), activation="tanh", log_std_init=-0.5):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.actor_body = mlp([obs_dim, *hidden], act)
        self.actor_mean = layer_init(nn.Linear(hidden[-1], act_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.critic_body = mlp([obs_dim, *hidden], act)
        self.critic_head = layer_init(nn.Linear(hidden[-1], 1), std=1.0)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_body(obs)).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor_body(obs))
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        d = self.dist(obs)
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1), self.value(obs)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def compute_gae(rewards, values, terms, truncs, final_values, last_values, gamma, lam):
    """rewards/values/terms/truncs: (T, N); final_values: (T, N) value of the final obs
    where an episode ended (0 elsewhere); last_values: (N,) for the step after the rollout."""
    T, N = rewards.shape
    adv = np.zeros((T, N), np.float32)
    last_gae = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        done = terms[t] | truncs[t]
        next_val = last_values if t == T - 1 else values[t + 1]
        # bootstrap: terminated -> 0; truncated -> V(final obs); otherwise V(s_{t+1})
        next_val = np.where(done, np.where(truncs[t] & ~terms[t], final_values[t], 0.0), next_val)
        not_done = 1.0 - done.astype(np.float32)
        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        adv[t] = last_gae
    return adv, adv + values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="train/configs/phase1_ppo.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None, help="checkpoint .pt to continue from")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--stream", action="store_true", help="stream all training envs to the 3-D viewer (ws://localhost:8765)")
    ap.add_argument("--stream-port", type=int, default=8765)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["run_name"] = args.run_name or cfg["run_name"].replace("_ppo", "") + "_scratch"
    if args.total_timesteps:
        cfg["total_timesteps"] = args.total_timesteps
    if args.n_envs:
        cfg["n_envs"] = args.n_envs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.threads:
        torch.set_num_threads(args.threads)
    p = cfg["ppo"]
    run_dir = Path("runs") / cfg["run_name"]
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    writer = SummaryWriter(str(run_dir / "tensorboard"))

    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_envs, n_steps = int(cfg["n_envs"]), int(p["n_steps"])
    venv = VecEnv(cfg.get("env", {}), n_envs, seed)
    obs_dim = venv.envs[0].observation_space.shape[0]
    act_dim = venv.envs[0].action_space.shape[0]
    model = ActorCritic(obs_dim, act_dim, tuple(p["hidden_sizes"]), p.get("activation", "tanh"),
                        float(p.get("log_std_init", -0.5)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(p["learning_rate"]), eps=1e-5)
    global_step, update = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        global_step, update = ck["global_step"], ck["update"]
        print(f"resumed from {args.resume} at {global_step:,} steps")

    total = int(cfg["total_timesteps"])
    batch_steps = n_envs * n_steps
    n_updates = max(1, (total - global_step) // batch_steps)
    gamma, lam = float(p["gamma"]), float(p["gae_lambda"])
    clip, ent_coef, vf_coef = float(p["clip_range"]), float(p["ent_coef"]), float(p["vf_coef"])
    max_grad, n_epochs, batch_size = float(p["max_grad_norm"]), int(p["n_epochs"]), int(p["batch_size"])
    lr0 = float(p["learning_rate"])
    linear_lr = p.get("lr_schedule", "constant") == "linear"
    eval_every, ckpt_every = int(cfg["eval_every"]), int(cfg["checkpoint_every"])
    next_eval, next_ckpt = (global_step // eval_every + 1) * eval_every, (global_step // ckpt_every + 1) * ckpt_every
    best_lap = float("inf")
    refs = load_reference_lap_times()
    print(f"scratch PPO: {n_envs} envs x {n_steps} steps = {batch_steps} per update, {n_updates} updates; "
          f"references {({k: round(v, 1) for k, v in refs.items()})}")
    eval_csv = run_dir / "eval.csv"
    if not args.resume:
        with open(eval_csv, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "wall_s", "episode_done", "standing_lap", "best_flying_lap",
                                    "distance_m"])

    def policy_fn(o):
        with torch.no_grad():
            a, _, _ = model.act(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0), deterministic=True)
        return a.squeeze(0).clamp(-1, 1).numpy()

    def save(path: Path):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": global_step,
                    "update": update, "config": cfg, "obs_dim": obs_dim, "act_dim": act_dim}, path)

    # rollout storage
    obs_buf = np.zeros((n_steps, n_envs, obs_dim), np.float32)
    act_buf = np.zeros((n_steps, n_envs, act_dim), np.float32)
    logp_buf = np.zeros((n_steps, n_envs), np.float32)
    rew_buf = np.zeros((n_steps, n_envs), np.float32)
    val_buf = np.zeros((n_steps, n_envs), np.float32)
    term_buf = np.zeros((n_steps, n_envs), bool)
    trunc_buf = np.zeros((n_steps, n_envs), bool)
    final_val_buf = np.zeros((n_steps, n_envs), np.float32)

    streamer = None
    if args.stream:
        from kart_rl.live.stream import StateStreamer

        streamer = StateStreamer(port=args.stream_port, hello={"track": "Race-Inn Roggwil",
                                                               "policy": f"training:{cfg['run_name']}"})
        print(f"live stream on ws://localhost:{args.stream_port}  ->  open viewer with ?live")
    n_episodes, recent_returns = 0, []
    t0 = time.time()
    recent = []
    obs = venv.obs
    for _ in range(n_updates):
        update += 1
        if linear_lr:
            frac = max(1.0 - global_step / total, 0.0)
            for g in optimizer.param_groups:
                g["lr"] = frac * lr0
        # ---------------------------------------------------------- rollout
        model.eval()
        for t in range(n_steps):
            with torch.no_grad():
                a, logp, v = model.act(torch.as_tensor(obs, dtype=torch.float32))
            a_np = a.numpy()
            next_obs, r, term, trunc, final_obs, finished = venv.step(np.clip(a_np, -1, 1))
            obs_buf[t], act_buf[t], logp_buf[t], rew_buf[t], val_buf[t] = obs, a_np, logp.numpy(), r, v.numpy()
            term_buf[t], trunc_buf[t] = term, trunc
            final_val_buf[t] = 0.0
            for i, fo in enumerate(final_obs):
                if fo is not None and trunc[i] and not term[i]:
                    with torch.no_grad():
                        final_val_buf[t, i] = model.value(torch.as_tensor(fo, dtype=torch.float32)).item()
            recent.extend(finished)
            n_episodes += len(finished)
            recent_returns = (recent_returns + [e["return"] for e in finished])[-50:]
            obs = next_obs
            global_step += n_envs
            if streamer is not None and streamer.connected:
                from kart_rl.live.stream import env_state_from_info

                streamer.send({"type": "states",
                               "states": [env_state_from_info(info, i) for i, info in enumerate(venv.infos)],
                               "training": {"steps": global_step, "episodes": n_episodes,
                                            "ep_rew_mean": float(np.mean(recent_returns)) if recent_returns else None,
                                            "best_lap": None if not np.isfinite(best_lap) else float(best_lap),
                                            "elapsed_s": round(time.time() - t0, 1),
                                            "steps_per_s": round(global_step / max(time.time() - t0, 1e-6), 1)}},
                              force=bool(finished))
        with torch.no_grad():
            last_values = model.value(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        adv, ret = compute_gae(rew_buf, val_buf, term_buf, trunc_buf, final_val_buf, last_values, gamma, lam)

        # ----------------------------------------------------------- update
        model.train()
        b_obs = torch.as_tensor(obs_buf.reshape(-1, obs_dim))
        b_act = torch.as_tensor(act_buf.reshape(-1, act_dim))
        b_logp = torch.as_tensor(logp_buf.reshape(-1))
        b_adv = torch.as_tensor(adv.reshape(-1))
        b_ret = torch.as_tensor(ret.reshape(-1))
        b_val = torch.as_tensor(val_buf.reshape(-1))
        idx = np.arange(batch_steps)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        n_mb = 0
        for _ in range(n_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_steps, batch_size):
                mb = idx[start:start + batch_size]
                logp, entropy, value = model.evaluate(b_obs[mb], b_act[mb])
                ratio = (logp - b_logp[mb]).exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - clip, 1 + clip)).mean()
                v_clipped = b_val[mb] + (value - b_val[mb]).clamp(-clip, clip)
                v_loss = 0.5 * torch.max((value - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimizer.step()
                with torch.no_grad():
                    stats["policy_loss"] += pg_loss.item()
                    stats["value_loss"] += v_loss.item()
                    stats["entropy"] += ent.item()
                    stats["approx_kl"] += ((ratio - 1) - (logp - b_logp[mb])).mean().item()
                    stats["clip_frac"] += ((ratio - 1).abs() > clip).float().mean().item()
                n_mb += 1
        for k in stats:
            writer.add_scalar(f"train/{k}", stats[k] / n_mb, global_step)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("train/log_std", model.log_std.mean().item(), global_step)
        if recent:
            ep_ret = np.mean([e["return"] for e in recent])
            ep_len = np.mean([e["length"] for e in recent])
            laps = np.mean([e["laps"] for e in recent])
            writer.add_scalar("rollout/ep_rew_mean", ep_ret, global_step)
            writer.add_scalar("rollout/ep_len_mean", ep_len, global_step)
            writer.add_scalar("rollout/laps_per_episode", laps, global_step)
            sps = global_step / (time.time() - t0)
            print(f"update {update:4d}  step {global_step:>9,}  ep_return {ep_ret:8.1f}  ep_len {ep_len:6.0f}  "
                  f"laps/ep {laps:4.2f}  kl {stats['approx_kl'] / n_mb:.4f}  {sps:5.0f} steps/s")
            recent = []
        # ----------------------------------------------------- eval / ckpt
        if global_step >= next_eval:
            next_eval += eval_every
            res = evaluate_lap_time(policy_fn, cfg.get("env", {}), laps=int(cfg.get("eval_laps", 2)))
            lap = res.best_lap if res.best_lap is not None else float("inf")
            standing = res.standing_lap if res.standing_lap is not None else float("nan")
            writer.add_scalar("eval/best_flying_lap_s", lap if np.isfinite(lap) else float("nan"), global_step)
            writer.add_scalar("eval/standing_lap_s", standing, global_step)
            writer.add_scalar("eval/distance_m", res.distance, global_step)
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow([global_step, round(time.time() - t0, 1), res.episode_done, round(standing, 2),
                                        round(lap, 2) if np.isfinite(lap) else "", round(res.distance, 1)])
            print(f"[eval @ {global_step:,} steps] {res.episode_done}: laps {[round(t, 2) for t in res.lap_times]}  "
                  f"distance {res.distance:.0f} m")
            if lap < best_lap:
                best_lap = lap
                save(run_dir / "best.pt")
                print(f"    new best lap {lap:.2f} s -> saved {run_dir / 'best.pt'}")
        if global_step >= next_ckpt:
            next_ckpt += ckpt_every
            save(run_dir / "checkpoints" / f"step_{global_step}.pt")
            save(run_dir / "checkpoints" / "last.pt")
    save(run_dir / "checkpoints" / "last.pt")
    dt = time.time() - t0
    print(f"done: {global_step:,} steps in {dt / 60:.1f} min; models in {run_dir}")
    writer.close()


if __name__ == "__main__":
    main()
