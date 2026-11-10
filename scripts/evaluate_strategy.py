"""Compare pit strategies; write a standalone HTML report, JSON and CSV.

python scripts/evaluate_strategy.py --laps 8 --fixed-lap 4
"""

import argparse
import csv
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kart_rl.envs.strategy_env import KartStrategyEnv, StrategyConfig


def run_strategy(config, choose, seed=0):
    env = KartStrategyEnv(config)
    try:
        obs, info = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        while not done:
            obs, reward, term, trunc, info = env.step(int(choose(obs, info)))
            total_reward += reward
            done = term or trunc
        return {**info, "return": total_reward}
    finally:
        env.close()


def write_report(out, config, results):
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"config": config.__dict__, "results": results}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (out / "laps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["strategy", "lap", "duration_s", "pit", "wear_start", "wear_end", "completed"]
        )
        writer.writeheader()
        for name, result in results.items():
            writer.writerows({"strategy": name, **lap} for lap in result["history"])
    colors = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24"]
    rows, bars, details = [], [], []
    scale = max(r["elapsed_s"] for r in results.values()) or 1
    for i, (name, r) in enumerate(results.items()):
        label = html.escape(name)
        status = "Ziel" if r["finished"] else "Ausfall: " + html.escape(r["episode_done"])
        rows.append(
            f"<tr><td>{label}</td><td>{status}</td><td>{r['elapsed_s']:.2f} s</td>"
            f"<td>{r['pit_stops']}</td><td>{r['laps_done']}/{config.laps}</td></tr>"
        )
        bars.append(
            f'<div class="barrow"><span>{label}</span><div class="bar" '
            f'style="width:{r["elapsed_s"] / scale * 100:.2f}%;background:{colors[i % 4]}">'
            f"{r['elapsed_s']:.1f} s · {status}</div></div>"
        )
        laps = "".join(
            f"<tr><td>{x['lap']}</td><td>{x['duration_s']:.2f} s</td>"
            f"<td>{'Ja' if x['pit'] else '–'}</td><td>{x['wear_end']:.0%}</td>"
            f"<td>{'Ja' if x['completed'] else 'Nein'}</td></tr>"
            for x in r["history"]
        )
        details.append(
            f"<details><summary>{label}: Runden und Reifen</summary><table><tr>"
            "<th>Runde</th><th>Zeit inkl. Stopp</th><th>Stopp</th><th>Verschleiss</th>"
            f"<th>Abgeschlossen</th></tr>{laps}</table></details>"
        )
    finishers = [(name, r) for name, r in results.items() if r["finished"]]
    winner = min(finishers, key=lambda x: x[1]["elapsed_s"])[0] if finishers else "Kein Rennen beendet"
    page = f"""<!doctype html><html lang="de"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Kart-RL · Boxenstrategie</title>
<style>body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px system-ui}}main{{max-width:1000px;margin:auto;padding:40px 20px}}
h1{{font-size:clamp(28px,5vw,48px);margin:12px 0}}p{{color:#aebdd0;line-height:1.6}}.tag{{color:#38bdf8;letter-spacing:.15em}}
section,details{{background:#131e30;border:1px solid #26354d;border-radius:14px;padding:22px;margin:20px 0;overflow:auto}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #26354d}}
.barrow{{margin:20px 0}}.barrow span{{display:block;margin-bottom:8px}}.bar{{box-sizing:border-box;min-width:150px;padding:10px;border-radius:6px;color:#07111f;font-weight:650}}
summary{{cursor:pointer;font-weight:650}}a{{color:#38bdf8}}</style><main><div class="tag">KART-RL / STRATEGIE-LAB</div>
<h1>Wann lohnt sich der Boxenstopp?</h1><p>{config.laps} Runden · {config.pit_loss_s:g} s pro Stopp ·
Verschleiss {config.wear_per_lap:.0%} pro Runde · minimaler Grip {config.min_grip:.0%}</p>
<section><div class="tag">SCHNELLSTE ERFOLGREICHE STRATEGIE</div><h2>{html.escape(winner)}</h2>
<p>Gleiche Physik und gleicher Referenzfahrer für alle Strategien. Ausfälle zählen nicht als Sieger.
Ein deterministischer Vergleichslauf; keine statistische Aussage über trainierte Modelle.</p></section>
<section><h2>Gesamte Rennzeit</h2>{"".join(bars)}<table><tr><th>Strategie</th><th>Status</th><th>Zeit</th>
<th>Stopps</th><th>Runden</th></tr>{"".join(rows)}</table></section>{"".join(details)}
<p>Eine Reifenmischung. Boxenservice als Zeitverlust an der Start-/Ziellinie; keine animierte Boxengasse.
Die Rundenzeit enthält den Service vor der jeweiligen Runde.</p>
<p><a href="results.json">Ergebnisse als JSON</a> · <a href="laps.csv">Runden als CSV</a></p></main></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--laps", type=int, default=8)
    ap.add_argument("--fixed-lap", type=int, default=4, help="pit after this completed lap")
    ap.add_argument("--pit-loss", type=float, default=12)
    ap.add_argument("--wear", type=float, default=0.18)
    ap.add_argument("--model", help="optional PPO strategy model (not a driving model)")
    ap.add_argument("--out", type=Path, default=Path("runs/strategy_report"))
    args = ap.parse_args()
    if not 1 <= args.fixed_lap < args.laps:
        ap.error("--fixed-lap must be at least 1 and less than --laps")
    config = StrategyConfig(laps=args.laps, pit_loss_s=args.pit_loss, wear_per_lap=args.wear)
    strategies = {
        "Ohne Stopp": lambda obs, info: 0,
        f"Stopp nach Runde {args.fixed_lap}": lambda obs, info: info["laps_done"] == args.fixed_lap,
        "Stopp ab 65 % Verschleiss": lambda obs, info: info["wear"] >= 0.65,
    }
    if args.model:
        metadata = Path(args.model).parent / "strategy_config.json"
        if not metadata.exists():
            ap.error("strategy_config.json must be next to the strategy model")
        trained_config = json.loads(metadata.read_text(encoding="utf-8"))["env"]
        if trained_config != config.__dict__:
            ap.error("evaluation settings differ from strategy_config.json; use the training settings")
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")
        strategies["PPO-Strategie"] = lambda obs, info: model.predict(obs, deterministic=True)[0]
    results = {name: run_strategy(config, choose) for name, choose in strategies.items()}
    write_report(args.out, config, results)
    for name, r in results.items():
        print(f"{name}: {r['elapsed_s']:.2f} s, {r['pit_stops']} stop(s), {r['episode_done']}")
    print(f"Report: {(args.out / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
