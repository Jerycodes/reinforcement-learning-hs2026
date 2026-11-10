"""Controlled PPO reward-scale experiment, ranked by race time, not raw return."""

import argparse
import csv
import html
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kart_rl.envs.strategy_env import StrategyConfig
from scripts.evaluate_strategy import run_strategy


def aggregate(trials):
    """DNFs stay in the finish rate; time statistics use finishers only."""
    groups = {}
    for trial in trials:
        groups.setdefault(trial["time_scale_s"], []).append(trial)
    summary = []
    for scale, rows in groups.items():
        times = [r["elapsed_s"] for r in rows if r["finished"]]
        summary.append(
            {
                "time_scale_s": scale,
                "runs": len(rows),
                "finished": len(times),
                "mean_s": statistics.mean(times) if times else None,
                "std_s": statistics.stdev(times) if len(times) > 1 else None,
                "mean_stops": statistics.mean(r["pit_stops"] for r in rows if r["finished"])
                if times
                else None,
            }
        )
    return sorted(
        summary,
        key=lambda r: (-r["finished"] / r["runs"], r["mean_s"] if r["mean_s"] is not None else math.inf),
    )


def write_comparison(out, manifest, trials, references):
    summary = aggregate(trials)
    data = {"experiment": manifest, "summary": summary, "trials": trials, "references": references}
    (out / "comparison.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)

    def number(value):
        return "–" if value is None else f"{value:.2f}"

    rows = "".join(
        f"<tr><td>−1 / {r['time_scale_s']:g} s</td><td>{r['finished']}/{r['runs']}</td>"
        f"<td>{number(r['mean_s'])}</td><td>{number(r['std_s'])}</td>"
        f"<td>{number(r['mean_stops'])}</td></tr>"
        for r in summary
    )
    details = "".join(
        f"<tr><td>{t['time_scale_s']:g} s</td><td>{t['seed']}</td>"
        f"<td>{'Ziel' if t['finished'] else html.escape(t['episode_done'])}</td>"
        f"<td>{t['elapsed_s']:.2f}</td><td>{t['pit_stops']}</td>"
        f'<td><a href="{html.escape(t["report"])}">Lauf öffnen</a></td></tr>'
        for t in trials
    )
    refs = "".join(
        f"<li>{html.escape(name)}: {result['elapsed_s']:.2f} s "
        f"({'Ziel' if result['finished'] else 'Ausfall'})</li>"
        for name, result in references.items()
    )
    penalties = "".join(
        f"<li>Zeitskala {scale:g} s: Ausfall insgesamt −{1200 / scale + 5:g} Reward-Punkte</li>"
        for scale in manifest["scales"]
    )
    page = f"""<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kart-RL · Reward-Vergleich</title><style>
body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px system-ui}}main{{max-width:1050px;margin:auto;padding:40px 20px}}
h1{{font-size:clamp(28px,5vw,44px)}}p,li{{line-height:1.65;color:#b5c4d8}}section{{background:#131e30;padding:24px;border:1px solid #26354d;border-radius:14px;margin:24px 0;overflow:auto}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #26354d}}a,.tag{{color:#38bdf8}}
</style><main><div class="tag">KART-RL / REWARD-LAB</div><h1>Hilft eine stärkere Zeitstrafe?</h1>
<p>{manifest["steps"]} Strategieentscheidungen je Training · {manifest["laps"]} Runden je Rennen ·
Seeds {", ".join(map(str, manifest["seeds"]))} · gleiche PPO-Einstellungen und Physik.</p>
<section><h2>Rennzeit entscheidet</h2><p>Sortiert nach Zielankunftsquote, danach mittlerer Zeit der erfolgreichen Rennen.
Reward-Summen verschiedener Skalen sind nicht vergleichbar. Streuung = Stichproben-Standardabweichung zwischen Trainings-Seeds.</p>
<table><tr><th>Zeitstrafe</th><th>Im Ziel</th><th>Mittlere Zeit (s)</th><th>Streuung (s)</th><th>Stopps (Mittel)</th></tr>{rows}</table></section>
<section><h2>Was wird bestraft?</h2><p>Jede verlorene Sekunde zählt negativ, auch beim Boxenstopp.
Bandenkontakt, falsche Fahrtrichtung, Stillstand oder Rennzeitlimit beenden das Rennen mit dem verbleibenden Zeitbudget und einer zusätzlichen Strafe von 5 Punkten.
Zielbonus +15; Fortschrittsfeedback netto null. Keine zusätzliche Strafe für einen sinnvollen Reifenwechsel.</p><ul>{penalties}</ul></section>
<section><h2>Referenzstrategien</h2><ul>{refs}</ul></section>
<section><h2>Einzelne Läufe</h2><table><tr><th>Zeitskala</th><th>Seed</th><th>Status</th><th>Zeit (s)</th><th>Stopps</th><th>Details</th></tr>{details}</table></section>
<p>Kurze Trainingsläufe sind ein Pilotversuch, kein Nachweis einer optimalen Reward-Funktion.
Die Seeds variieren das Lernen; die Evaluationsstrecke selbst ist deterministisch. Der Fahrer ist fest programmiert:
Dieser Vergleich untersucht Boxenentscheidungen, nicht das Lernen von Lenkung und Bremspunkten.</p>
<p><a href="comparison.json">JSON</a> · <a href="comparison.csv">CSV</a></p></main></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def validate(scales, seeds, steps, rollout, laps, jobs):
    if not scales or len(set(scales)) != len(scales) or any(not math.isfinite(x) or x <= 0 for x in scales):
        raise ValueError("scales must be unique finite positive values")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if steps <= 0 or rollout < 2 or steps % rollout:
        raise ValueError("steps must be positive and divisible by rollout-steps (>=2)")
    if laps < 2 or jobs < 1:
        raise ValueError("laps must be >=2 and jobs >=1")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", nargs="+", type=float, default=[100, 50, 25])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--rollout-steps", type=int, default=32)
    ap.add_argument("--laps", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("runs/reward_comparison"))
    args = ap.parse_args()
    try:
        validate(args.scales, args.seeds, args.steps, args.rollout_steps, args.laps, args.jobs)
    except ValueError as e:
        ap.error(str(e))
    out = args.out.resolve()
    if out.exists():
        ap.error("output directory already exists; use a new --out to preserve earlier experiments")
    out.mkdir(parents=True)
    manifest = {
        "scales": args.scales,
        "seeds": args.seeds,
        "steps": args.steps,
        "rollout_steps": args.rollout_steps,
        "laps": args.laps,
        "jobs": args.jobs,
    }
    (out / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def trial(scale, seed):
        run = out / f"scale_{scale:g}_seed_{seed}"
        run.mkdir()
        cmd = [
            sys.executable,
            str(ROOT / "train/train_strategy.py"),
            "--steps",
            str(args.steps),
            "--rollout-steps",
            str(args.rollout_steps),
            "--laps",
            str(args.laps),
            "--seed",
            str(seed),
            "--reward-time-scale",
            str(scale),
            "--threads",
            "1",
            "--policy-only-eval",
            "--out",
            str(run),
        ]
        with (run / "train.log").open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads((run / "evaluation/results.json").read_text(encoding="utf-8"))
        result = data["results"]["PPO-Strategie"]
        return {
            "time_scale_s": scale,
            "seed": seed,
            "elapsed_s": result["elapsed_s"],
            "finished": result["finished"],
            "episode_done": result["episode_done"],
            "pit_stops": result["pit_stops"],
            "return": result["return"],
            "report": f"{run.name}/evaluation/index.html",
        }

    trials = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(trial, scale, seed) for scale in args.scales for seed in args.seeds]
        for future in as_completed(futures):
            result = future.result()
            trials.append(result)
            print(
                f"Scale {result['time_scale_s']:g}, seed {result['seed']}: "
                f"{result['elapsed_s']:.2f} s, {result['pit_stops']} stops, "
                f"{result['episode_done']}",
                flush=True,
            )
    config = StrategyConfig(laps=args.laps)
    references = {
        "Ohne Stopp": run_strategy(config, lambda obs, info: 0),
        f"Stopp nach Runde {args.laps // 2}": run_strategy(
            config, lambda obs, info: info["laps_done"] == args.laps // 2
        ),
    }
    trials.sort(key=lambda r: (-r["time_scale_s"], r["seed"]))
    write_comparison(out, manifest, trials, references)
    print(f"Report: {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
