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
pytest -q                      # ~30 Tests, alles grün = Setup ok
```

GPU ist nicht nötig: Das Netz ist klein, die Simulation läuft auf der CPU. Mehr CPU-Kerne =
schnelleres Training (`n_envs` in der Config anpassen).

## Training starten und live zuschauen

```bash
# Terminal 1: Training (Stable-Baselines3 PPO)
python train/train_sb3.py --config train/configs/phase1_ppo.yaml

# Terminal 2: live zuschauen – lädt automatisch immer den neuesten Checkpoint
python scripts/watch_training.py --run runs/phase1_ppo
python scripts/watch_training.py --run runs/phase1_ppo --best --follow   # bestes Modell, Verfolgerkamera

# Terminal 3: Lernkurven
tensorboard --logdir runs
```

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
  data/                  Streckendaten (JSON + Maske), Ideallinie
train/
  train_sb3.py           Training mit Stable-Baselines3 PPO
  ppo_scratch.py         eigener PPO in PyTorch
  configs/*.yaml         Trainingskonfigurationen
scripts/                 evaluate.py, watch_training.py, play_baseline.py
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
