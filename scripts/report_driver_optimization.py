"""Render the recorded driver search and final validation as a local report."""

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/driver_validation_v1")
    args = ap.parse_args()
    out = args.out
    validation = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    experiments = [
        ("PPO: vorsichtig", "driver_optimization_v1/gentle"),
        ("PPO: längere Voraussicht", "driver_optimization_v1/long_horizon"),
        ("PPO: stärkere Zeitstrafe", "driver_optimization_v1/time_pressure"),
        ("Lenkung: Grobsuche", "driver_calibration_v1"),
        ("Lenkung: Feinsuche", "driver_calibration_refine_v1"),
        ("Gas-/Bremsverstärkung", "driver_calibration_throttle_v1"),
        ("PPO + Policy-Feinabstimmung", "driver_optimization_combined_v1"),
    ]
    comparisons = []
    for label, directory in experiments:
        result = json.loads((ROOT / "runs" / directory / "result.json").read_text(encoding="utf-8"))
        comparisons.append({"label": label, "directory": directory, **result["validation"]})
    original10 = json.loads((ROOT / "runs/driver_calibration_v1/result.json").read_text())["original"]
    original = validation["original"]
    candidate = validation["candidate"]
    gain = original["long_run"]["mean_flying_s"] - candidate["long_run"]["mean_flying_s"]
    data = {"validation": validation, "original_10_laps": original10, "search": comparisons}
    (out / "optimization_summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    initial = None
    for label, directory in experiments[:3]:
        history = json.loads((ROOT / "runs" / directory / "history.json").read_text())
        initial = history[0]["mean_flying_s"]
        ax.plot(
            [r["steps"] / 1000 for r in history],
            [r["mean_flying_s"] if r["finished"] else float("nan") for r in history],
            marker="o",
            label=label,
        )
    ax.axhline(initial, linestyle="--", color="gray", label="Original (gleiche 3-Runden-Prüfung)")
    ax.set(
        xlabel="Gesammelte Trainingsschritte (Tsd.)",
        ylabel="Mittlere fliegende Runde (s)",
        title="Zwischenstände im PPO-Weitertraining",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=140)
    plt.close(fig)

    def number(value):
        return "–" if value is None else f"{value:.3f}"

    rows = "".join(
        f"<tr><td>{html.escape(r['label'])}</td><td>{'10/10' if r['finished'] else html.escape(r['episode_done'])}</td>"
        f"<td>{number(r['mean_flying_s'])}</td><td>{number(r['best_flying_s'])}</td></tr>"
        for r in comparisons
    )
    safety = "".join(
        f"<tr><td>{a['seed']}</td><td>{html.escape(a['episode_done'])}</td>"
        f"<td>{html.escape(b['episode_done'])}</td></tr>"
        for a, b in zip(original["random_starts"], candidate["random_starts"], strict=True)
    )
    status = "Schneller und geprüft" if validation["improved"] else "Kandidat nicht übernommen"
    page = f"""<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kart-RL · Optimierter Fahrer</title><style>body{{margin:0;background:#0b1220;color:#e2e8f0;font:16px system-ui}}
main{{max-width:1050px;margin:auto;padding:40px 20px}}h1{{font-size:clamp(28px,5vw,44px)}}p,li{{line-height:1.65;color:#b5c4d8}}
section{{background:#131e30;border:1px solid #26354d;border-radius:14px;padding:24px;margin:24px 0;overflow:auto}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #26354d}}
img{{width:100%;border-radius:8px}}a,.tag{{color:#38bdf8}}.gain{{font-size:40px;font-weight:700;color:#34d399}}
</style><main><div class="tag">KART-RL / FAHRER-OPTIMIERUNG</div><h1>{status}</h1>
<section><div class="gain">{gain:.3f} s pro Runde</div><p>Differenz der mittleren fliegenden Runden im abschliessenden
{validation["laps"]}-Runden-Vergleich. Identische Strecke, Kartphysik, Sensoren und Kart-Kontur-Kollisionsregel. Keine Boxenstopps.</p>
<table><tr><th>Modell</th><th>Ø fliegend (s)</th><th>Beste fliegend (s)</th><th>Neue Zufallsstarts</th></tr>
<tr><td>Jerys Original</td><td>{number(original["long_run"]["mean_flying_s"])}</td><td>{number(original["long_run"]["best_flying_s"])}</td><td>{original["random_finished"]}/10</td></tr>
<tr><td>Optimierte Policy</td><td>{number(candidate["long_run"]["mean_flying_s"])}</td><td>{number(candidate["long_run"]["best_flying_s"])}</td><td>{candidate["random_finished"]}/10</td></tr></table></section>
<section><h2>Was wurde optimiert?</h2><p>Drei PPO-Varianten erhielten je 102'400 zusätzliche Schritte: vorsichtige Lernrate,
längere Voraussicht und eine stärkere Zeitstrafe. Die schnellsten sicheren Zwischenstände wurden gespeichert.
Daneben wurde die Ausgabeschicht der Policy mit einer kleinen Parametersuche feinabgestimmt; diese Kalibrierung ist kein zusätzliches PPO-Training.</p>
<p>Der endgültige Kandidat kombiniert längere Voraussicht mit einer Lenkausgabe-Verstärkung von 0.94 und einem Gas-/Bremsbias von −0.01.
Die Umwelt begrenzt alle Aktionen weiterhin auf die normalen Werte. Leistung, Grip, Bande und Strecke wurden nicht erleichtert.</p></section>
<section><h2>Erkundete Kandidaten: jeweils zehn Runden</h2><p>Original in dieser kürzeren Prüfung: {number(original10["mean_flying_s"])} s im Mittel.
Diese Tabelle diente der Auswahl. Die abschliessende längere Prüfung oben verwendet ein anderes Rundenbudget.</p>
<table><tr><th>Kandidat</th><th>Runden</th><th>Ø fliegend (s)</th><th>Beste fliegend (s)</th></tr>{rows}</table></section>
<section><h2>Warum den besten Zwischenstand behalten?</h2><img src="learning_curves.png" alt="Rundenzeiten während des PPO-Trainings">
<p>Die Kurven verwenden kurze Prüfungen mit drei Runden. Mehr Training verbessert nicht jeden späteren Stand.
Die letzte Aktualisierung wird separat geprüft; sie muss nicht das schnellste Modell liefern.</p></section>
<section><h2>Zusätzliche Startzustände</h2><p>Beide Modelle fahren dieselben zehn neuen Zufallsstarts (Seeds 100–109), jeweils bis zu zwei Ziellinienüberquerungen.
Die optimierte Version wird nur übernommen, wenn sie den langen Lauf beendet, schneller ist und mindestens so viele Zufallsstarts beendet wie das Original.</p>
<table><tr><th>Seed</th><th>Original</th><th>Kandidat</th></tr>{safety}</table></section>
<p>Ergebnis gilt für diese deterministische Strecke und die geprüften Starts. Kein Nachweis allgemeiner Überlegenheit auf anderen Strecken.
Das Original bleibt unverändert; die geprüfte Kopie liegt im Unterordner <code>winner/</code>.</p>
<p><a href="optimization_summary.json">Alle Vergleiche als JSON</a> · <a href="comparison.json">Lange Prüfung</a></p></main></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"Report: {out / 'index.html'}")


if __name__ == "__main__":
    main()
