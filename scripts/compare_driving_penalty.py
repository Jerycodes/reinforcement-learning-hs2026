"""Fine-tune Jery's driving PPO: matched control vs time penalty, without pit stops."""

import argparse
import copy
import csv
import hashlib
import html
import io
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kart_rl.checkpoints import load_portable_ppo
from kart_rl.envs import KartLapEnv, LapEnvConfig
from kart_rl.evaluation import run_episode


def penalty_config(source, scale):
    """Convert seconds per negative point into the existing per-control-step reward."""
    config = copy.deepcopy(source)
    if scale is not None:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("time scale must be positive and finite")
        config["reward_time"] = float(config.get("reward_time", 0)) - float(config.get("dt", 0.05)) / scale
    return config


def evaluate(model, config, laps, normalizer=None):
    cfg = LapEnvConfig.from_dict(config)
    cfg.random_start, cfg.laps, cfg.max_time = False, laps, 90.0 * laps
    cfg.sector_length_m, cfg.domain_randomization = 0.0, {}
    env = KartLapEnv(cfg)

    def policy(obs):
        if normalizer is not None:
            obs = normalizer.normalize_obs(obs)
        return model.predict(obs, deterministic=True)[0]

    try:
        result = run_episode(env, policy, seed=0)
        flying = result.lap_times[1:]
        return {
            "finished": result.episode_done == "laps" and len(result.lap_times) == laps,
            "episode_done": result.episode_done,
            "lap_times_s": result.lap_times,
            "standing_s": result.standing_lap,
            "best_flying_s": min(flying) if flying else None,
            "mean_flying_s": statistics.mean(flying) if flying else None,
            "distance_m": result.distance,
        }
    finally:
        env.close()


def summarize_trials(trials):
    groups = {}
    for row in trials:
        groups.setdefault(row["arm"], []).append(row)
    summary = []
    for arm, rows in groups.items():
        times = [r["mean_flying_s"] for r in rows if r["finished"] and r["mean_flying_s"] is not None]
        summary.append(
            {
                "arm": arm,
                "runs": len(rows),
                "finished": sum(r["finished"] for r in rows),
                "mean_flying_s": statistics.mean(times) if times else None,
                "std_s": statistics.stdev(times) if len(times) > 1 else None,
            }
        )
    return summary


def worker(args, source):
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    torch.set_num_threads(1)
    cfg = penalty_config(source.get("env", {}), args.scale if args.worker_arm == "time_penalty" else None)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    vec = DummyVecEnv([lambda: KartLapEnv(cfg) for _ in range(int(source["n_envs"]))])
    vec = VecMonitor(vec, filename=str(out / "monitor.csv"))
    norm_path = args.model.with_name("best_vecnormalize.pkl")
    if source.get("normalize_reward", True):
        vec = VecNormalize.load(str(norm_path), vec)
        vec.training = True
    try:
        model = load_portable_ppo(
            args.model,
            source["ppo"],
            env=vec,
            device="cpu",
            learning_rate=args.learning_rate,
            seed=args.worker_seed,
            tensorboard_log=None,
            verbose=1,
        )
        source_steps = model.num_timesteps
        model.learn(total_timesteps=args.steps, reset_num_timesteps=True)
        model.save(out / "last.zip")
        if isinstance(vec, VecNormalize):
            vec.save(str(out / "vecnormalize.pkl"))
        effective = {
            **copy.deepcopy(source),
            "run_name": out.name,
            "seed": args.worker_seed,
            "total_timesteps": args.steps,
            "env": cfg,
        }
        effective["ppo"].update(learning_rate=args.learning_rate, lr_schedule="constant")
        (out / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        result = evaluate(
            model, source.get("env", {}), args.eval_laps, vec if isinstance(vec, VecNormalize) else None
        )
        result.update(
            arm=args.worker_arm,
            seed=args.worker_seed,
            additional_steps=model.num_timesteps,
            source_steps=source_steps,
            reward_time_per_step=cfg.get("reward_time", 0),
            reward_off_track=cfg.get("reward_off_track", -20),
        )
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    finally:
        vec.close()


def write_report(out, manifest, original, trials):
    summary = summarize_trials(trials)
    data = {"experiment": manifest, "original": original, "summary": summary, "trials": trials}
    (out / "comparison.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)

    def number(value):
        return "–" if value is None else f"{value:.2f}"

    names = {"control": "Weitertraining ohne Zeitstrafe", "time_penalty": "Weitertraining mit Zeitstrafe"}
    rows = "".join(
        f"<tr><td>{names[r['arm']]}</td><td>{r['finished']}/{r['runs']}</td>"
        f"<td>{number(r['mean_flying_s'])}</td><td>{number(r['std_s'])}</td></tr>"
        for r in summary
    )
    details = "".join(
        f"<tr><td>{names[r['arm']]}</td><td>{r['seed']}</td>"
        f"<td>{html.escape(r['episode_done'])}</td><td>{number(r['standing_s'])}</td>"
        f"<td>{number(r['mean_flying_s'])}</td><td>{number(r['best_flying_s'])}</td></tr>"
        for r in trials
    )
    page = f"""<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kart-RL · Jerys Fahrmodell</title><style>body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px system-ui}}
main{{max-width:1050px;margin:auto;padding:40px 20px}}h1{{font-size:clamp(28px,5vw,44px)}}p,li{{color:#b5c4d8;line-height:1.65}}
section{{background:#131e30;border:1px solid #26354d;border-radius:14px;padding:24px;margin:24px 0;overflow:auto}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #26354d}}a,.tag{{color:#38bdf8}}</style>
<main><div class="tag">KART-RL / FAHRTRAINING OHNE BOXENSTOPP</div><h1>Hilft die Zeitstrafe Jerys Fahrer?</h1>
<p>Gleicher Start-Checkpoint · {manifest["steps"]:,} zusätzliche Fahr-Schritte je Lauf ·
Seeds {", ".join(map(str, manifest["seeds"]))} · {manifest["eval_laps"]} Runden pro Evaluation.</p>
<section><h2>Originalmodell</h2><p>Status: {html.escape(original["episode_done"])} · stehende Runde:
{number(original["standing_s"])} s · fliegende Runden im Mittel: {number(original["mean_flying_s"])} s ·
beste fliegende Runde: {number(original["best_flying_s"])} s.</p></section>
<section><h2>Kontrollierter Vergleich</h2><p>Die Kontrollgruppe wird genauso lange weitertrainiert.
Dadurch wird zusätzliches Training nicht mit dem Effekt der Strafe verwechselt. Mittelwerte unten verwenden nur vollständig beendete Evaluationen.
Streuung = Standardabweichung zwischen Trainings-Seeds, nicht zwischen Runden.</p>
<table><tr><th>Variante</th><th>Im Ziel</th><th>Ø fliegende Runde (s)</th><th>Streuung (s)</th></tr>{rows}</table></section>
<section><h2>Was ist anders?</h2><p>Nur die Versuchsgruppe zahlt zusätzlich −1 Reward-Punkt pro {manifest["scale"]:g} Sekunden Fahrzeit.
Bei einem Kontrollschritt von {manifest["dt"]:g} s entspricht das {-(manifest["dt"] / manifest["scale"]):g} Punkten je Schritt.
Fortschrittsreward, Kartphysik und Kollisionsregel bleiben gleich. Bandenkontakt kostet {manifest["reward_off_track"]:g} Punkte und beendet die Episode.</p>
<p>Kein Reifenverschleiss, kein Reifenwechsel, kein Boxenstopp. Der PPO-Agent selbst steuert Gas, Bremse und Lenkung.
Beide Gruppen verwenden eine konstante Lernrate von {manifest["learning_rate"]:g} fürs Weitertraining. Die ursprünglichen Gewichte und die passenden Normalisierungsstatistiken werden geladen.</p></section>
<section><h2>Einzelne Läufe</h2><table><tr><th>Variante</th><th>Seed</th><th>Status</th><th>Stehend (s)</th><th>Ø fliegend (s)</th><th>Beste fliegend (s)</th></tr>{details}</table></section>
<p>Kurzes Weitertraining ist ein Pilotversuch, keine belastbare Aussage über die beste Reward-Funktion.
Identische deterministische Evaluation; die Seeds variieren das Lernen. Grössere negative Rewards allein garantieren keine schnellere Runde.</p>
<p><a href="comparison.json">JSON</a> · <a href="comparison.csv">CSV</a></p></main></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "runs/phase1_v2/best.zip")
    ap.add_argument("--steps", type=int, default=40960)
    ap.add_argument("--scale", type=float, default=25)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--eval-laps", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/jery_driving_penalty")
    ap.add_argument("--worker-arm", choices=["control", "time_penalty"])
    ap.add_argument("--worker-seed", type=int, default=0)
    args = ap.parse_args()
    args.model = args.model.resolve()
    source = yaml.safe_load(args.model.with_name("config.yaml").read_text(encoding="utf-8"))
    rollout = int(source["n_envs"]) * int(source["ppo"]["n_steps"])
    if args.steps <= 0 or args.steps % rollout:
        ap.error(f"steps must be positive and divisible by the original rollout size {rollout}")
    if args.eval_laps < 2 or args.jobs < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        ap.error("use at least 2 eval laps, positive jobs and unique seeds")
    if any(s < 0 for s in args.seeds) or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        ap.error("seeds must be nonnegative and learning rate positive and finite")
    penalty_config(source.get("env", {}), args.scale)
    if args.worker_arm:
        worker(args, source)
        return
    out = args.out.resolve()
    if out.exists():
        ap.error("output already exists; choose a fresh --out")
    out.mkdir(parents=True)
    import torch

    torch.set_num_threads(1)
    model = load_portable_ppo(args.model, source["ppo"], device="cpu")
    with ZipFile(args.model) as archive:
        weights = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
    assert all(torch.equal(value, model.policy.state_dict()[key]) for key, value in weights.items())
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    original_vec = DummyVecEnv([lambda: KartLapEnv(source.get("env", {}))])
    if source.get("normalize_reward", True):
        original_vec = VecNormalize.load(str(args.model.with_name("best_vecnormalize.pkl")), original_vec)
        original_vec.training = False
    try:
        original = evaluate(
            model,
            source.get("env", {}),
            args.eval_laps,
            original_vec if isinstance(original_vec, VecNormalize) else None,
        )
    finally:
        original_vec.close()
    print(f"Original: {original}", flush=True)
    manifest = {
        "model": str(args.model.relative_to(ROOT)) if args.model.is_relative_to(ROOT) else str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "steps": args.steps,
        "seeds": args.seeds,
        "scale": args.scale,
        "eval_laps": args.eval_laps,
        "learning_rate": args.learning_rate,
        "dt": source.get("env", {}).get("dt", 0.05),
        "reward_off_track": source.get("env", {}).get("reward_off_track", -20),
        "initial_weights_verified": True,
        "source_config": source,
    }
    (out / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def trial(arm, seed):
        run = out / f"{arm}_seed_{seed}"
        run.mkdir()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-arm",
            arm,
            "--worker-seed",
            str(seed),
            "--model",
            str(args.model),
            "--steps",
            str(args.steps),
            "--scale",
            str(args.scale),
            "--learning-rate",
            str(args.learning_rate),
            "--eval-laps",
            str(args.eval_laps),
            "--out",
            str(run),
        ]
        with (run / "train.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        print(f"{arm}, seed {seed}: {result['episode_done']}, laps {result['lap_times_s']}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(trial, arm, seed) for arm in ("control", "time_penalty") for seed in args.seeds
        ]
        trials = [f.result() for f in as_completed(futures)]
    trials.sort(key=lambda r: (r["arm"], r["seed"]))
    write_report(out, manifest, original, trials)
    print(f"Report: {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
