# Kart-RL – Reinforcement Learning auf der Kartbahn Race-Inn Roggwil

Gruppenprojekt Reinforcement Learning (HS 2026). Ein Go-Kart lernt per Reinforcement Learning,
die Indoor-Strecke Race-Inn Roggwil (640 m) so schnell wie möglich zu fahren – und später,
ein Rennen mit Reifenverschleiss, Boxenstopps und Boost zu gewinnen.

| Phase | Ziel | Status |
|---|---|---|
| 1 | Fahren lernen, schnellste Rundenzeit (`KartLap-v0`) | Env fertig, Training läuft |
| 2 | Rennen: Reifenmischungen Soft/Medium/Hard, Verschleiss, Boxenstopps, Strategie | geplant |
| 3 | Begrenzter Boost: wann, wo, wie viel | geplant |

Die Strecke wurde aus dem offiziellen Streckenplan extrahiert (siehe `docs/strecke.md`), die
Physik ist ein Einspurmodell mit Kart-Parametern (`docs/simulation.md`). Der Agent bekommt
**keine** Ideallinie vorgegeben – er sieht nur Lidar-Strahlen, Geschwindigkeit und die Kurven
voraus und wird für Fortschritt belohnt. Eine klassisch berechnete Ideallinie dient nur als
Vergleichsmassstab (`docs/ideallinie.md`).

![Strecke](docs/img/track_extraction.png)

## Setup (Mac/Linux, Python ≥ 3.10)

```bash
git clone https://github.com/Jerycodes/reinforcement-learning-hs2026.git
cd reinforcement-learning-hs2026
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"
pytest -q                      # 28 Tests, alles grün = Setup ok
```

GPU ist nicht nötig: Das Netz ist klein, die Simulation läuft auf der CPU. Mehr CPU-Kerne =
schnelleres Training (`n_envs` in der Config anpassen).

## Training starten und live zuschauen

```bash
# Training (Stable-Baselines3 PPO); --stream öffnet den 3D-Viewer im Browser und streamt alle Trainings-Karts
python train/train_sb3.py --config train/configs/phase1_ppo.yaml --stream

# optional in einem zweiten Terminal, 2D: fährt automatisch mit dem jeweils neuesten Checkpoint
python scripts/watch_training.py --run runs/phase1_ppo --follow

# Lernkurven
tensorboard --logdir runs
```

Im 3D-Viewer fährt ein Kart in Echtzeit den aktuellen Trainingsstand – man sieht zu, wie der Agent
von Update zu Update besser wird (Schritte, Episoden, beste Rundenzeit im HUD). Das Training selbst
läuft im Hintergrund mit 8 parallelen Umgebungen (nur fürs Tempo; es ist ein einzelner lernender
Agent) – `--stream-all` zeigt auch diese. Öffnet sich der Browser nicht von selbst:
http://localhost:8000/viewer/?live. Details, Replay-Modus und Parameter-Panel: `docs/viewer.md`.

Eigene PPO-Implementierung (gleiche Config, gleiche Umgebung, zum Vergleichen und Verstehen):

```bash
python train/ppo_scratch.py --config train/configs/phase1_ppo.yaml      # -> runs/phase1_scratch
```

Weitertrainieren ab Checkpoint (z. B. wenn jemand anderes weitermacht):

```bash
python train/train_sb3.py --config train/configs/phase1_ppo.yaml --resume runs/phase1_ppo/checkpoints/last.zip
python train/ppo_scratch.py --config train/configs/phase1_ppo.yaml --resume runs/phase1_scratch/checkpoints/last.pt
```

Wichtige Config-Werte (`train/configs/phase1_ppo.yaml`): `total_timesteps`, `n_envs`
(parallele Umgebungen ≈ CPU-Kerne), `ppo.*` (Hyperparameter), `env.*` (Reward, Episodenlänge).
Jeder Lauf schreibt nach `runs/<run_name>/`: `best.zip` (bestes Modell nach Rundenzeit),
`checkpoints/`, `eval.csv`, `tensorboard/`, `config.yaml`.

## Auswerten und anschauen

```bash
python scripts/evaluate.py runs/phase1_ppo/best.zip                   # Rundenzeiten vs. Referenzen
python scripts/evaluate.py runs/phase1_ppo/best.zip --render --follow # live im 2D-Fenster
python scripts/evaluate.py runs/phase1_ppo/best.zip --gif docs/img/agent.gif
python scripts/evaluate.py runs/phase1_ppo/best.zip --replay replays/agent.json   # für den 3D-Viewer
python scripts/evaluate.py --baseline raceline --baseline centerline  # die geskripteten Referenzfahrer
python scripts/play_baseline.py --follow                              # Referenzfahrer live
```

Referenz-Rundenzeiten (geskriptete Fahrer, gleiches Kart-Modell): Mittellinie 71.6 s,
berechnete Ideallinie 54.1 s, theoretisches Modell-Optimum ≈ 49.8 s.

## Mehr, länger und schneller trainieren

```bash
# länger: 3 Mio Schritte sind auf einem M-Series-Mac ~7 Minuten – für die besten Runden 20-50 Mio laufen lassen
python train/train_sb3.py --config train/configs/phase1_ppo.yaml --run-name phase1_long --total-timesteps 30000000 --n-envs 10 --stream

# jeden Config-Wert von der Kommandozeile überschreiben (--set key=value, beliebig oft)
python train/train_sb3.py --config train/configs/phase1_ppo.yaml --set ppo.learning_rate=1e-4 --set ppo.n_steps=2048 --set env.n_lidar=31

# weitertrainieren ab Checkpoint (auch nach Stunden / auf einem anderen Rechner)
python train/train_sb3.py --config train/configs/phase1_ppo.yaml --run-name phase1_long --resume runs/phase1_long/checkpoints/last.zip --stream

# Population Based Training: mehrere Agenten gleichzeitig, die Schlechten übernehmen Gewichte + Hyperparameter der Besten
python train/pbt.py --config train/configs/phase1_ppo.yaml --workers 4 --n-envs 3 --interval 250000 --total 5000000 --stream
```

- `n_envs` ≈ Anzahl CPU-Kerne; mehr bringt nichts mehr. Die Schritte/s stehen im Terminal und im Viewer-HUD.
- Mit `--stream` fährt das Kart nach dem Training mit dem fertigen Modell weiter, bis du Ctrl+C drückst.
- Alle Hyperparameter stehen in `train/configs/phase1_ppo.yaml` (Lernrate, Netzgrösse, Rollout-Länge, Reward-Gewichte, Episodenlänge …); Kopie anlegen = eigenes Experiment.
- PBT (`train/pbt.py`): jeder Worker ist ein eigener Prozess mit eigener Lernrate/Entropie/Clip/Gamma; nach jedem Intervall werden alle bewertet, die schlechtesten kopieren die besten und variieren die Hyperparameter (Jaderberg et al. 2017). Ergebnis in `runs/<name>/best.zip`, Verlauf in `pbt_log.csv`, Population im Viewer-HUD.

## 3D-Viewer, Video und Experimente

```bash
python scripts/serve_viewer.py                                   # Replays in 3D (mehrere gleichzeitig)
python -m kart_rl.live.server --policy runs/phase1_ppo/best.zip --open   # Modell live + Parameter-Regler
python scripts/make_video.py --run runs/phase1_ppo --grid --out docs/video/evolution_grid.mp4  # Anfang vs. Ende
python train/train_sb3.py --config train/configs/exp_sector_curriculum.yaml --stream   # Experiment: Episode = Sektor
python scripts/compare_runs.py                                   # Lernkurven aller Läufe + Tabelle
```

Alle Experimente (Sektor-Curriculum, Reward-Varianten, Lidar, Domain Randomization, SAC) sind in
`docs/experimente.md` beschrieben, inkl. Kurzrecherche, was bei Racing-RL üblich ist.

## Projektstruktur

```
kart_rl/                 Python-Paket
  track.py               Strecke: Mittellinie, Ränder, Fortschritt, Raster, Lidar
  kart.py                Kart-Physik (Einspurmodell)
  path.py                gemeinsame Geometrie für geschlossene Linien
  raceline.py            Ideallinie (Minimum Curvature) + Geschwindigkeitsprofil (nur Vergleich)
  envs/lap_env.py        Phase 1: KartLap-v0 (Gymnasium)
  baselines/             geskripteter Referenzfahrer (Pure Pursuit)
  render/viewer2d.py     2D-Viewer (matplotlib)
  evaluation.py, agents.py   Auswertung, Laden von Modellen
  live/                  Websocket-Streaming (Training) und Live-Server mit Parameter-Panel
  data/                  Streckendaten (JSON + Maske), Ideallinie
viewer/                  3D-Viewer (Three.js, Browser)
train/
  train_sb3.py           Training mit Stable-Baselines3 (PPO oder SAC), Curriculum, Streaming
  ppo_scratch.py         eigener PPO in PyTorch
  configs/*.yaml         Baseline und Experimente
scripts/                 evaluate.py, watch_training.py, play_baseline.py, serve_viewer.py, make_video.py, compare_runs.py
tools/                   extract_track.py (Bild → Strecke), compute_raceline.py
tests/                   pytest
docs/                    Dokumentation (Deutsch) und Bilder
assets/                  Streckenplan
runs/                    Trainingsläufe (Checkpoints/TensorBoard sind git-ignoriert, best.zip nicht)
```

## Dokumentation

- `docs/strecke.md` – Streckenextraktion, Koordinatensystem, Datenformat
- `docs/simulation.md` – Kart-Modell, Beobachtung, Aktion, Reward, Episoden
- `docs/ideallinie.md` – berechnete Ideallinie als Vergleich
- `docs/viewer.md` – 3D-Viewer (Replay, Training live, Live-Server), Video
- `docs/experimente.md` – Stand der Technik und Experimentplan
