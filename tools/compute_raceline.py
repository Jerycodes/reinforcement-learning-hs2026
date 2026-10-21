#!/usr/bin/env python3
"""Compute the minimum-curvature racing line, drive it with the scripted
driver in the simulator and write line + plot.

    python tools/compute_raceline.py                 # defaults (Race-Inn Roggwil)
    python tools/compute_raceline.py --margin 0.9 --safety 0.95
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kart_rl.baselines import PurePursuitDriver
from kart_rl.envs import KartLapEnv, LapEnvConfig
from kart_rl.raceline import RaceLine, speed_profile
from kart_rl.track import Track


def drive(env: KartLapEnv, driver: PurePursuitDriver, record: bool = True):
    env.reset()
    driver.reset()
    done, xs, vs = False, [], []
    while not done:
        _, _, term, trunc, info = env.step(driver.act(env))
        done = term or trunc
        if record:
            xs.append((env.kart.x, env.kart.y))
            vs.append(env.kart.speed)
    return info, np.array(xs), np.array(vs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="race_inn_roggwil")
    ap.add_argument("--margin", type=float, default=1.0, help="distance kept from the track edge [m]")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--safety", type=float, default=0.80,
                    help="fraction of the grip used by the driven speed profile (0.92 = theoretical optimum)")
    ap.add_argument("--out", default=None, help="output JSON (default: kart_rl/data/<track>_raceline.json)")
    ap.add_argument("--plot", default="docs/img/raceline.png")
    ap.add_argument("--laps", type=int, default=3, help="laps to drive for the measured lap time")
    args = ap.parse_args()

    track = Track(args.track)
    print("optimising racing line ...")
    line = RaceLine.compute(track, margin=args.margin, iterations=args.iterations, safety=args.safety, verbose=True)
    centre = RaceLine.from_track_centerline(track)
    print(f"racing line: {line.length:.1f} m (centreline {track.length:.1f} m), "
          f"max |curvature| {np.abs(line.curvature).max():.3f} vs {np.abs(track.curvature).max():.3f} 1/m")
    theoretical = RaceLine(line.points, speed_profile(line, safety=0.92))
    print(f"estimated lap times: racing line {line.lap_time_estimate:.2f} s (driven profile), "
          f"{theoretical.lap_time_estimate:.2f} s (theoretical optimum of the model), "
          f"centreline {centre.lap_time_estimate:.2f} s")
    line.meta["theoretical_lap_time_s"] = round(theoretical.lap_time_estimate, 2)

    # drive both lines in the simulator
    env = KartLapEnv(LapEnvConfig(laps=args.laps, random_start=False, max_time=120.0 * args.laps))
    results = {}
    for name, drv in (("Mittellinie", PurePursuitDriver(track)),
                      ("Ideallinie", PurePursuitDriver(line, speed=line.speed, a_brake=4.0))):
        info, xy, v = drive(env, drv)
        laps = info["lap_times"]
        best = min(laps[1:]) if len(laps) > 1 else (laps[0] if laps else None)
        results[name] = (info.get("episode_done"), laps, xy, v, best)
        print(f"{name:12s}: {info.get('episode_done')}, lap times {[round(t, 2) for t in laps]}")
    line.meta["measured_lap_time_s"] = results["Ideallinie"][4]
    line.meta["centerline_measured_lap_time_s"] = results["Mittellinie"][4]

    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "kart_rl" / "data" / f"{args.track}_raceline.json"
    line.save(out)
    print(f"wrote {out}")

    # ------------------------------------------------------------------ plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3])
    ax = fig.add_subplot(gs[0, :])
    poly = track.polygon()
    ax.fill(poly[:, 0], poly[:, 1], color="0.75", label="Strecke")
    ax.plot(*np.vstack([track.centerline, track.centerline[:1]]).T, color="0.3", lw=1, ls="--", label="Mittellinie")
    xy = results["Ideallinie"][2]
    ax.plot(xy[:, 0], xy[:, 1], color="black", lw=0.6, alpha=0.6, label="gefahren (Ideallinie)")
    sc = ax.scatter(line.points[:, 0], line.points[:, 1], c=line.speed * 3.6, cmap="viridis", s=10, zorder=5,
                    label="Ideallinie (Farbe = Geschwindigkeit)")
    plt.colorbar(sc, ax=ax, label="Sollgeschwindigkeit [km/h]", fraction=0.03)
    ax.scatter(*track.centerline[0], s=80, marker="s", color="lime", zorder=6, label="Start/Ziel")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fmt = lambda v: f"{v:.1f} s" if v else "–"
    ax.set_title(f"Ideallinie (Minimum Curvature, Rand-Abstand {args.margin} m): {line.length:.0f} m, "
                 f"gefahren {fmt(results['Ideallinie'][4])} (Modell-Optimum {theoretical.lap_time_estimate:.1f} s)"
                 f"  |  Mittellinie gefahren {fmt(results['Mittellinie'][4])}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(track.s, track.curvature, color="0.4", lw=1, label="Mittellinie")
    ax2.plot(line.s * track.length / line.length, line.curvature, color="C0", lw=1, label="Ideallinie")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Streckenmeter (auf 640 m normiert)")
    ax2.set_ylabel("Krümmung [1/m]")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(line.s, line.speed * 3.6, color="C0", lw=1.2, label="Sollprofil Ideallinie")
    ax3.plot(centre.s, centre.speed * 3.6, color="0.4", lw=1, ls="--", label="Profil Mittellinie")
    ax3.set_xlabel("Meter entlang der Linie")
    ax3.set_ylabel("Geschwindigkeit [km/h]")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=110)
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
