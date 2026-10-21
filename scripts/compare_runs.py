#!/usr/bin/env python3
"""Compare training runs: lap time vs. steps from each run's eval.csv, plus a table.

    python scripts/compare_runs.py                      # all runs in runs/
    python scripts/compare_runs.py runs/phase1_ppo runs/exp_sector_curriculum --out docs/img/compare.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_eval(run_dir: Path):
    f = run_dir / "eval.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    steps = np.array([int(r["timesteps"]) for r in rows])
    lap = np.array([float(r["best_flying_lap"]) if r["best_flying_lap"] not in ("", "inf") else np.nan for r in rows])
    dist = np.array([float(r["distance_m"]) for r in rows])
    wall = np.array([float(r["wall_s"]) for r in rows])
    return steps, lap, dist, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out", default="docs/img/compare.png")
    args = ap.parse_args()
    run_dirs = [Path(r) for r in args.runs] or sorted(p for p in Path("runs").iterdir() if p.is_dir())
    data = {r.name: read_eval(r) for r in run_dirs}
    data = {k: v for k, v in data.items() if v is not None and len(v[0])}
    if not data:
        raise SystemExit("no eval.csv found")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rows = []
    for name, (steps, lap, dist, wall) in data.items():
        ok = ~np.isnan(lap)
        ax1.plot(steps[ok] / 1e6, lap[ok], marker="o", ms=3, lw=1, label=name)
        best_running = np.fmin.accumulate(np.where(ok, lap, np.inf))
        ax2.plot(steps / 1e6, np.where(np.isfinite(best_running), best_running, np.nan), lw=1.5, label=name)
        first = steps[ok][0] if ok.any() else None
        rows.append((name, int(steps[-1]), float(np.nanmin(lap)) if ok.any() else None, first, float(wall[-1]) / 60))
    for ax, title in ((ax1, "Rundenzeit der Evaluation (stehender Start)"), (ax2, "beste Rundenzeit bisher")):
        ax.set_xlabel("Trainingsschritte [Mio]")
        ax.set_ylabel("Rundenzeit [s]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axhline(54.1, color="k", ls="--", lw=0.8)
        ax.text(0.01, 54.3, "Ideallinien-Fahrer 54.1 s", fontsize=8)
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}\n")
    print("| Lauf | Schritte | beste Rundenzeit | erste Runde ab Schritt | Trainingszeit |")
    print("|---|---|---|---|---|")
    for name, steps, best, first, mins in sorted(rows, key=lambda r: r[2] if r[2] else 1e9):
        print(f"| {name} | {steps:,} | {best:.2f} s | {first:,} |" if best else f"| {name} | {steps:,} | – | – |",
              f" {mins:.0f} min |")


if __name__ == "__main__":
    main()
