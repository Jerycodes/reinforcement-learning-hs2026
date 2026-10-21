#!/usr/bin/env python3
"""Experiment dashboard: configure, start, watch and compare training runs in the browser.

    python -m kart_rl.dashboard            # http://localhost:8080

- Parameter form with explanations and recommended ranges (learning rate, network,
  reward weights, lidar, curriculum, ...). "Start" writes a reproducible config to
  train/configs/<name>.yaml and launches the trainer as a separate process.
- Running experiments: live steps / best lap / steps per second, log tail, stop
  button, link to the 3-D viewer (each experiment gets its own websocket port).
- Results: every run in runs/ with its config and lap-time curve; select runs to
  compare them in one chart.
- Everything is file based (train/configs, runs/<name>/eval.csv, best.zip), so after
  `git push` the team sees the same experiments and can start their own.

Standard library only (no Flask).
"""
from __future__ import annotations

import csv
import functools
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "train" / "configs"
RUNS_DIR = ROOT / "runs"
BASE_CONFIG = CONFIG_DIR / "phase1_ppo.yaml"

# ---------------------------------------------------------------------------
# parameter schema shown in the form: key -> (label, help, type, min, max, step)
# help texts are the "advice" built in; keep them short and concrete
# ---------------------------------------------------------------------------
SCHEMA = [
    {"group": "Allgemein", "items": [
        {"key": "total_timesteps", "label": "Trainingsschritte", "type": "int", "min": 100000, "max": 200000000, "step": 100000,
         "help": "Wie lange trainiert wird. 3 Mio = Kurztest (~7 min auf einem M-Mac), 30 Mio = ausgelernt (~1 h). Mehr als 30 Mio brachte in Phase 1 nur noch Zehntel."},
        {"key": "n_envs", "label": "Parallele Umgebungen", "type": "int", "min": 1, "max": 32, "step": 1,
         "help": "Etwa so viele wie CPU-Kerne. Mehr = schneller sammeln, ändert das Lernen kaum."},
        {"key": "seed", "label": "Seed", "type": "int", "min": 0, "max": 9999, "step": 1,
         "help": "Zufalls-Startwert. Gleiche Config mit 2–3 Seeds laufen lassen zeigt, wie stark Ergebnisse streuen."},
    ]},
    {"group": "PPO (Lernalgorithmus)", "items": [
        {"key": "ppo.learning_rate", "label": "Lernrate", "type": "float", "min": 1e-5, "max": 3e-3, "step": 1e-5,
         "help": "Schrittweite der Updates. 3e-4 ist Standard; 1e-4 lernt langsamer aber stabiler, 1e-3 schneller mit Risiko von Einbrüchen."},
        {"key": "ppo.lr_schedule", "label": "Lernraten-Verlauf", "type": "choice", "choices": ["linear", "constant"],
         "help": "linear = sinkt bis zum Ende auf 0 (feinere Endphase); constant = bleibt gleich (gut für Weitertrainieren)."},
        {"key": "ppo.n_steps", "label": "Rollout-Länge pro Env", "type": "int", "min": 128, "max": 8192, "step": 128,
         "help": "Schritte pro Umgebung zwischen zwei Updates. Länger = stabilere Schätzungen, seltenere Updates. 1024–2048 üblich."},
        {"key": "ppo.batch_size", "label": "Minibatch", "type": "int", "min": 64, "max": 4096, "step": 64,
         "help": "Grösse der Teilpakete beim Update. 256–1024; sollte n_steps × n_envs teilen."},
        {"key": "ppo.n_epochs", "label": "Epochen pro Update", "type": "int", "min": 1, "max": 30, "step": 1,
         "help": "Wie oft ein Rollout durchgearbeitet wird. 10 Standard; mehr = stärkere Updates (Überanpassung möglich)."},
        {"key": "ppo.gamma", "label": "Gamma (Diskont)", "type": "float", "min": 0.9, "max": 0.999, "step": 0.001,
         "help": "Weitsicht. 0.99 = Horizont ~5 s. 0.995 für längere Planung (z. B. Boxenstopps), 0.98 für schnelleres Lernen."},
        {"key": "ppo.gae_lambda", "label": "GAE Lambda", "type": "float", "min": 0.8, "max": 1.0, "step": 0.01,
         "help": "Bias/Varianz des Advantage. 0.95 Standard; selten ändern."},
        {"key": "ppo.clip_range", "label": "Clip-Bereich", "type": "float", "min": 0.05, "max": 0.5, "step": 0.01,
         "help": "Wie weit sich die Policy pro Update ändern darf. 0.2 Standard; 0.1 vorsichtiger."},
        {"key": "ppo.ent_coef", "label": "Entropie-Bonus", "type": "float", "min": 0.0, "max": 0.05, "step": 0.0005,
         "help": "Exploration. 0.001 Standard; 0.005–0.01 wenn der Agent früh in einer schlechten Linie hängen bleibt."},
        {"key": "ppo.hidden_sizes", "label": "Netzgrösse", "type": "choice", "choices": ["[64, 64]", "[128, 128]", "[256, 256]", "[512, 512]", "[256, 256, 256]"],
         "help": "Neuronen pro Schicht. [256, 256] reicht für 30 Eingänge; grösser = langsamer, selten besser."},
        {"key": "ppo.log_std_init", "label": "Start-Streuung (log)", "type": "float", "min": -2.0, "max": 0.5, "step": 0.1,
         "help": "Zufall in den Aktionen am Anfang. -0.5 ≈ 0.6 Streuung; -1 = ruhiger, 0 = wilder."},
    ]},
    {"group": "Umgebung / Reward", "items": [
        {"key": "env.reward_progress", "label": "Reward pro Meter", "type": "float", "min": 0.1, "max": 5.0, "step": 0.1,
         "help": "Hauptreward. Skaliert alles andere mit; meist bei 1.0 lassen."},
        {"key": "env.reward_off_track", "label": "Strafe Abflug", "type": "float", "min": -100.0, "max": 0.0, "step": 1.0,
         "help": "-20 Standard. Härter (-50) = vorsichtiger fahren; milder (-5) = mehr Risiko."},
        {"key": "env.off_track_mode", "label": "Bandenkontakt", "type": "choice", "choices": ["terminate", "penalty"],
         "help": "terminate = Kontakt beendet die Episode mit Strafe (Standard). penalty = Bande ist ein Hindernis: Kart wird zurückgedrückt, verliert Tempo, Strafe pro Kontakt-Schritt, Fahrt geht weiter."},
        {"key": "env.reward_steer_change", "label": "Strafe Zickzack", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01,
         "help": "Gewicht auf (Änderung des Lenkbefehls)². 0.05 Standard; höher = ruhigere Lenkung."},
        {"key": "env.reward_speed", "label": "Tempo-Bonus", "type": "float", "min": 0.0, "max": 2.0, "step": 0.05,
         "help": "Zusätzlicher Reward für Geschwindigkeit (0 = aus). Experiment: schneller, aber mehr Abflüge?"},
        {"key": "env.reward_drift", "label": "Drift-Strafe (pro Grad und Schritt)", "type": "float", "min": -0.2, "max": 0.0, "step": 0.005,
         "help": "0 = aus (Driften erlaubt). Negativ, z. B. -0.02: pro Schritt und Grad Schwimmwinkel über der Drift-Schwelle. -0.02 bei 10° drüber = -0.2/Schritt = -4/s, etwa ein Viertel des Fortschritts-Rewards bei Vollgas."},
        {"key": "env.drift_threshold_deg", "label": "Drift-Schwelle [°]", "type": "float", "min": 0.0, "max": 20.0, "step": 0.5,
         "help": "Schwimmwinkel (Winkel zwischen Fahrtrichtung und Kart-Längsachse), ab dem es als Drift zählt. Normales Kurvenfahren liegt bei 2-5°."},
        {"key": "env.off_track_margin", "label": "Abstand Kart–Bande [m]", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
         "help": "Mindestabstand der ganzen Kart-Kontur (Ecken + Seiten) zur Bande. Darunter = Bandenkontakt. 0.1 = Berührung zählt; grösser = Sicherheitsabstand erzwingen."},
        {"key": "env.n_lidar", "label": "Lidar-Strahlen", "type": "int", "min": 5, "max": 61, "step": 2,
         "help": "15 Standard. 31 sieht feiner, braucht etwas länger zum Lernen."},
        {"key": "env.lidar_fov_deg", "label": "Lidar-Öffnung [°]", "type": "float", "min": 90, "max": 360, "step": 10,
         "help": "200° Standard (sieht schräg nach hinten). 240–270° für mehr Rundumsicht."},
        {"key": "env.lidar_range", "label": "Lidar-Reichweite [m]", "type": "float", "min": 10, "max": 60, "step": 5,
         "help": "30 m Standard."},
        {"key": "env.random_start", "label": "Zufallsstart im Training", "type": "bool",
         "help": "An = überall auf der Strecke starten (empfohlen). Aus = immer stehend an der Ziellinie."},
        {"key": "env.laps", "label": "Runden pro Episode", "type": "int", "min": 1, "max": 20, "step": 1,
         "help": "Nach so vielen Zieldurchfahrten endet die Episode. 2 Standard."},
        {"key": "env.max_time", "label": "Zeitlimit Episode [s]", "type": "float", "min": 30, "max": 1200, "step": 10,
         "help": "150 s Standard."},
        {"key": "env.sector_length_m", "label": "Sektor-Länge [m] (0 = ganze Runde)", "type": "float", "min": 0, "max": 640, "step": 10,
         "help": "Episode = ein Sektor dieser Länge ab zufälligem Start (Curriculum-Experiment). 0 = normale Runden."},
        {"key": "env.domain_randomization", "label": "Grip/Masse zufällig", "type": "choice", "choices": ["aus", "leicht", "stark"],
         "help": "Pro Episode zufällige Kart-Parameter: leicht = µ 0.9–1.1, Masse 160–200; stark = µ 0.8–1.2, Masse 150–220. Robustere Policy."},
    ]},
    {"group": "Kart-Physik", "items": [
        {"key": "env.kart.mu", "label": "Reifengrip µ", "type": "float", "min": 0.6, "max": 1.3, "step": 0.01,
         "help": "1.0 Standard. Weniger Grip = langsamer, mehr Rutschen."},
        {"key": "env.kart.drive_force_max", "label": "Antriebskraft [N]", "type": "float", "min": 300, "max": 1500, "step": 10,
         "help": "720 N Standard (0–50 km/h in ~5 s)."},
        {"key": "env.kart.power_max", "label": "Motorleistung [W]", "type": "float", "min": 2000, "max": 15000, "step": 100,
         "help": "5300 W Standard."},
        {"key": "env.kart.speed_limit", "label": "Begrenzer [m/s]", "type": "float", "min": 10, "max": 30, "step": 0.5,
         "help": "18 m/s = 65 km/h Standard."},
    ]},
]

DOMAIN_RANDOM = {"aus": {}, "leicht": {"mu": [0.9, 1.1], "mass": [160.0, 200.0]}, "stark": {"mu": [0.8, 1.2], "mass": [150.0, 220.0]}}

TIPS = [
    "Nur einen Parameter pro Experiment ändern – sonst weiss man nicht, was gewirkt hat.",
    "Aussagekräftiger Vergleich: gleiche Schrittzahl, am besten 2 Seeds pro Config.",
    "Erst 3 Mio Schritte als Schnelltest; nur vielversprechende Configs auf 30 Mio verlängern.",
    "Lernrate ist der wirksamste Hebel: 1e-4 / 3e-4 / 1e-3 vergleichen.",
    "Entropie 0.005 oder Zufallsstart aus, wenn der Agent lange keine Runde schafft.",
    "Gamma 0.995 testen, wenn der Agent zu kurzfristig fährt (zu spät bremst).",
    "Sektor-Curriculum: 80 m, dann 200 m, dann ganze Runde – siehe docs/experimente.md.",
    "PBT (Population) sucht Lernrate/Entropie/Clip automatisch – gut für 'mehrere gleichzeitig'.",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_path(d: dict, key: str, default=None):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(d: dict, key: str, value):
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def load_configs() -> list[dict]:
    out = []
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            out.append({"name": p.stem, "config": yaml.safe_load(p.read_text())})
        except Exception as e:  # noqa: BLE001
            out.append({"name": p.stem, "error": str(e)})
    return out


def read_eval(run_dir: Path) -> list[dict]:
    f = run_dir / "eval.csv"
    if not f.exists():
        return []
    rows = []
    for r in csv.DictReader(open(f)):
        lap = r.get("best_flying_lap") or ""
        rows.append({"steps": int(float(r["timesteps"])), "wall_s": float(r.get("wall_s") or 0),
                     "lap": float(lap) if lap not in ("", "inf") else None,
                     "distance": float(r.get("distance_m") or 0)})
    return rows


def human_results() -> dict:
    """Leaderboard from runs/human/laps.csv (written by the game): per driver and ghost model."""
    f = RUNS_DIR / "human" / "laps.csv"
    rows = list(csv.DictReader(open(f))) if f.exists() else []
    table: dict[tuple, dict] = {}
    for r in rows:
        key = (r["driver"], r["ghost_policy"])
        e = table.setdefault(key, {"driver": r["driver"], "ghost": r["ghost_policy"], "laps": 0, "valid": 0,
                                   "best": None, "ghost_best": None, "last": r["timestamp"]})
        e["laps"] += 1
        t = float(r["lap_time_s"])
        if r.get("valid", "1") == "1":
            e["valid"] += 1
            e["best"] = t if e["best"] is None else min(e["best"], t)
        if r.get("ghost_best_s"):
            g = float(r["ghost_best_s"])
            e["ghost_best"] = g if e["ghost_best"] is None else min(e["ghost_best"], g)
        e["last"] = max(e["last"], r["timestamp"])
    out = sorted(table.values(), key=lambda e: (e["best"] is None, e["best"] or 0))
    demo_dir = ROOT / "replays" / "human"
    demos = sorted(p.name for p in demo_dir.glob("*_demo.json")) if demo_dir.is_dir() else []
    return {"rows": out, "n_laps": len(rows), "demos": demos}


def schema_defaults() -> dict:
    """Default value of every schema key that a base config may not mention (env/kart from the
    dataclass defaults, so the form shows the real value and only true changes are sent)."""
    from dataclasses import asdict

    from kart_rl.envs.lap_env import LapEnvConfig
    from kart_rl.kart import KartParams

    env_d, kart_d = LapEnvConfig().to_dict(), asdict(KartParams())
    out = {}
    for group in SCHEMA:
        for it in group["items"]:
            key = it["key"]
            if key.startswith("env.kart."):
                out[key] = kart_d.get(key[len("env.kart."):])
            elif key.startswith("env.") and key != "env.domain_randomization":
                out[key] = env_d.get(key[len("env."):])
    return {k: v for k, v in out.items() if v is not None}


class Experiments:
    """Processes started from the dashboard (other runs are detected from runs/)."""

    def __init__(self):
        self.procs: dict[str, dict] = {}
        self.next_ws = 8765

    def _take_port(self) -> int:
        """Next websocket port that is not in use (skips e.g. a live server started by hand)."""
        import socket

        while True:
            port = self.next_ws
            self.next_ws += 1
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue

    def start(self, name: str, cfg: dict, trainer: str, stream: bool, extra: dict) -> dict:
        run_dir = RUNS_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = CONFIG_DIR / f"{name}.yaml"
        cfg["run_name"] = name
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        ws_port = self._take_port() if stream else None
        py = sys.executable
        if trainer == "pbt":
            cmd = [py, "train/pbt.py", "--config", str(cfg_path.relative_to(ROOT)), "--run-name", name,
                   "--workers", str(extra.get("workers", 4)), "--n-envs", str(cfg.get("n_envs", 3)),
                   "--interval", str(extra.get("interval", 250000)), "--total", str(cfg.get("total_timesteps", 5000000))]
        else:
            script = "train/ppo_scratch.py" if trainer == "scratch" else "train/train_sb3.py"
            cmd = [py, script, "--config", str(cfg_path.relative_to(ROOT)), "--run-name", name]
            if extra.get("resume"):
                cmd += ["--resume", extra["resume"]]
                if extra.get("more"):
                    cmd += ["--more", str(extra["more"])]
        if stream:
            cmd += ["--stream", "--stream-port", str(ws_port), "--http-port", "0", "--no-open"]
        log = open(run_dir / "train.log", "a")
        log.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.procs[name] = {"proc": proc, "cmd": " ".join(cmd), "ws_port": ws_port,
                            "started": time.time(), "trainer": trainer}
        return {"name": name, "pid": proc.pid, "ws_port": ws_port, "cmd": " ".join(cmd)}

    def view(self, name: str) -> dict:
        """Start the live server for a finished model (replaces a previous one)."""
        prev = self.procs.pop("__view__", None)
        if prev and prev["proc"].poll() is None:
            os.killpg(os.getpgid(prev["proc"].pid), signal.SIGINT)
        model = next((p for p in (RUNS_DIR / name / "best.zip", RUNS_DIR / name / "best.pt") if p.exists()), None)
        if model is None:
            raise FileNotFoundError(f"kein best.zip in runs/{name}")
        ws_port = self._take_port()
        cmd = [sys.executable, "-m", "kart_rl.live.server", "--policy", str(model.relative_to(ROOT)),
               "--ws", str(ws_port), "--http", "0"]
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        self.procs["__view__"] = {"proc": proc, "cmd": " ".join(cmd), "ws_port": ws_port, "started": time.time(),
                                  "trainer": "view"}
        return {"ws_port": ws_port, "model": str(model.relative_to(ROOT))}

    def game(self, name: str, driver: str = "du") -> dict:
        """Start the game server (keyboard vs. this run's best model as ghost)."""
        prev = self.procs.pop("__game__", None)
        if prev and prev["proc"].poll() is None:
            os.killpg(os.getpgid(prev["proc"].pid), signal.SIGINT)
        model = next((p for p in (RUNS_DIR / name / "best.zip", RUNS_DIR / name / "best.pt") if p.exists()), None)
        if model is None:
            raise FileNotFoundError(f"kein best.zip in runs/{name}")
        ws_port = self._take_port()
        cmd = [sys.executable, "-m", "kart_rl.game", "--policy", str(model.relative_to(ROOT)), "--name", driver,
               "--ws", str(ws_port), "--http", "0"]
        (RUNS_DIR / "human").mkdir(parents=True, exist_ok=True)
        log = open(RUNS_DIR / "human" / "game.log", "a")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.procs["__game__"] = {"proc": proc, "cmd": " ".join(cmd), "ws_port": ws_port, "started": time.time(),
                                  "trainer": "game"}
        return {"ws_port": ws_port, "model": str(model.relative_to(ROOT))}

    def stop(self, name: str) -> bool:
        info = self.procs.get(name)
        if not info or info["proc"].poll() is not None:
            return False
        os.killpg(os.getpgid(info["proc"].pid), signal.SIGINT)  # like Ctrl+C: saves last.zip
        return True

    def status(self, name: str) -> dict | None:
        info = self.procs.get(name)
        if not info:
            return None
        code = info["proc"].poll()
        return {"running": code is None, "exit_code": code, "ws_port": info["ws_port"], "cmd": info["cmd"],
                "started": info["started"], "trainer": info["trainer"]}


def list_runs(exps: Experiments) -> list[dict]:
    out = []
    if not RUNS_DIR.exists():
        return out
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or d.name == "human":   # runs/human = game results, not a training run
            continue
        ev = read_eval(d)
        cfg = {}
        if (d / "config.yaml").exists():
            try:
                cfg = yaml.safe_load((d / "config.yaml").read_text()) or {}
            except Exception:  # noqa: BLE001
                cfg = {}
        laps = [r["lap"] for r in ev if r["lap"] is not None]
        st = exps.status(d.name) if d.name not in ("__view__", "__game__") else None
        best_file = next((str(p.name) for p in (d / "best.zip", d / "best.pt") if p.exists()), None)
        out.append({
            "name": d.name, "steps": ev[-1]["steps"] if ev else 0, "evals": len(ev),
            "best_lap": min(laps) if laps else None, "wall_min": round(ev[-1]["wall_s"] / 60, 1) if ev else 0,
            "first_lap_steps": next((r["steps"] for r in ev if r["lap"] is not None), None),
            "best_file": best_file, "running": bool(st and st["running"]), "ws_port": st["ws_port"] if st else None,
            "trainer": st["trainer"] if st else ("pbt" if (d / "pbt_log.csv").exists() else "sb3"),
            "summary": {"lr": get_path(cfg, "ppo.learning_rate"), "n_envs": cfg.get("n_envs"),
                        "total": cfg.get("total_timesteps"), "n_lidar": get_path(cfg, "env.n_lidar"),
                        "gamma": get_path(cfg, "ppo.gamma"), "ent": get_path(cfg, "ppo.ent_coef"),
                        "reward_speed": get_path(cfg, "env.reward_speed"), "sector": get_path(cfg, "env.sector_length_m"),
                        "curriculum": bool(cfg.get("curriculum"))},
            "modified": d.stat().st_mtime,
        })
    return out


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(http.server.SimpleHTTPRequestHandler):
    exps: Experiments = None  # set in main
    dashboard_html: Path = Path(__file__).with_name("index.html")

    def log_message(self, *a, **k):  # quiet
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/dashboard", "/dashboard/"):
            data = self.dashboard_html.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if u.path == "/api/schema":
            base = yaml.safe_load(BASE_CONFIG.read_text())
            from kart_rl.evaluation import load_reference_lap_times

            return self._json({"schema": SCHEMA, "base": base, "tips": TIPS, "configs": load_configs(),
                               "cpu_count": os.cpu_count(), "references": load_reference_lap_times(),
                               "defaults": schema_defaults()})
        if u.path == "/api/runs":
            return self._json(list_runs(self.exps))
        if u.path == "/api/human":
            return self._json(human_results())
        if u.path.startswith("/api/eval/"):
            name = u.path.split("/")[-1]
            return self._json(read_eval(RUNS_DIR / name))
        if u.path.startswith("/api/log/"):
            name = u.path.split("/")[-1]
            f = RUNS_DIR / name / "train.log"
            text = f.read_text(errors="replace")[-6000:] if f.exists() else "(kein Log – Lauf wurde nicht vom Dashboard gestartet)"
            return self._json({"log": text})
        if u.path.startswith("/api/config/"):
            name = u.path.split("/")[-1]
            f = RUNS_DIR / name / "config.yaml"
            return self._json({"yaml": f.read_text() if f.exists() else ""})
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/start":
            try:
                name = "".join(c for c in body.get("name", "").strip() if c.isalnum() or c in "-_")
                if not name:
                    return self._json({"error": "Name fehlt"}, 400)
                if (RUNS_DIR / name).exists() and not body.get("overwrite"):
                    return self._json({"error": f"Lauf '{name}' existiert schon – anderen Namen wählen"}, 400)
                base_name = body.get("base", "phase1_ppo")
                cfg = yaml.safe_load((CONFIG_DIR / f"{base_name}.yaml").read_text())
                for key, val in (body.get("values") or {}).items():
                    if val is None or (isinstance(val, float) and val != val):  # empty field / NaN: keep base
                        continue
                    if key == "env.domain_randomization":
                        set_path(cfg, key, DOMAIN_RANDOM.get(val, {}))
                    elif key == "ppo.hidden_sizes":
                        set_path(cfg, key, json.loads(val) if isinstance(val, str) else val)
                    else:
                        set_path(cfg, key, val)
                if body.get("curriculum"):
                    cfg["curriculum"] = body["curriculum"]
                cfg["algo"] = body.get("algo", "ppo")
                res = self.exps.start(name, cfg, body.get("trainer", "sb3"), bool(body.get("stream", True)),
                                      body.get("extra") or {})
                return self._json(res)
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 500)
        if u.path == "/api/stop":
            return self._json({"stopped": self.exps.stop(body.get("name", ""))})
        if u.path == "/api/game":
            try:
                return self._json(self.exps.game(body.get("name", ""), body.get("driver") or "du"))
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)
        if u.path == "/api/view":
            try:
                return self._json(self.exps.view(body.get("name", "")))
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)
        return self._json({"error": "unknown endpoint"}, 404)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    Handler.exps = Experiments()
    handler = functools.partial(Handler, directory=str(ROOT))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/"
    print(f"Kart-RL Dashboard: {url}   (Ctrl+C beendet nur das Dashboard, laufende Trainings laufen weiter)")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
