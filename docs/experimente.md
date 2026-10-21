# Experimente und Stand der Technik

## Was bei Racing-RL üblich ist (Kurzrecherche)

Unser Aufbau folgt dem, was sich in der Literatur und in bekannten Projekten bewährt hat:

- **Beobachtung:** Abstände zu den Streckenrändern (Lidar-Strahlen), eigene Geschwindigkeit/Gierrate
  und die Streckenkrümmung voraus. Genau so arbeitet Sonys *GT Sophy* bzw. die Vorarbeit "Super-Human
  Performance in Gran Turismo Sport" (Fuchs et al. 2021): Geschwindigkeit, Beschleunigung, Abstands-
  messungen zum Rand und die Kurve der Strecke in den nächsten Metern als Eingabe. Auch die F1TENTH-
  Racing-Benchmarks (Lidar + Geschwindigkeit) und Unity-ML-Agents-Karts (Ray-Perception-Sensoren)
  nutzen dieses Muster.
- **Reward:** Fortschritt entlang der Strecke pro Schritt ("course progress") ist der Standard
  (GT Sophy, F1TENTH), dazu eine Strafe für Wand-/Randkontakt. AWS DeepRacer stellt dafür die
  Grössen `progress`, `speed`, `distance_from_center`, `all_wheels_on_track` bereit; in der Praxis
  gewinnen Reward-Funktionen, die Fortschritt/Geschwindigkeit belohnen und Zickzack bestrafen,
  nicht solche, die das Auto an die Mittellinie zwingen.
- **Episoden:** Start an zufälligen Positionen auf der Strecke (Diversität), Abbruch bei Verlassen
  der Strecke; Evaluation aus stehendem Start.
- **Algorithmen:** PPO (Unity ML-Agents, SB3-Standard) und SAC (GT Sophy, viele Racing-Arbeiten)
  sind die beiden üblichen Wahlen für kontinuierliche Lenk-/Gas-Aktionen.
- **Curriculum:** kürzere Abschnitte oder einfachere Strecken zuerst, dann ganze Runden; ausserdem
  Domain Randomization (Grip, Masse) für robustere Policies.

Quellen:
[Super-Human Performance in Gran Turismo Sport Using Deep RL (Fuchs et al.)](https://arxiv.org/abs/2008.07971),
[Sony AI – GT Sophy](https://ai.sony/publications/super-human-performance-in-gran-turismo-sport-using-deep-reinforcement-learning),
[Outracing champion Gran Turismo drivers with deep RL (Nature 2022)](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf),
[Unifying F1TENTH Autonomous Racing: Survey, Methods and Benchmarks](https://arxiv.org/pdf/2402.18558),
[AWS DeepRacer – Input parameters of the reward function](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html),
[AWS DeepRacer – reward function examples](https://docs.aws.amazon.com/zh_cn/deepracer/latest/developerguide/deepracer-reward-function-examples.html).

## Experimente (je eine Config in `train/configs/`)

Alle Läufe schreiben `runs/<name>/eval.csv`; `python scripts/compare_runs.py` zeichnet die
Lernkurven (Rundenzeit vs. Schritte) und druckt eine Tabelle.

| Config | Idee | Erwartung / Frage |
|---|---|---|
| `phase1_ppo.yaml` | Baseline: PPO, Fortschritts-Reward, Zufallsstart | Referenz |
| `exp_sector_curriculum.yaml` | **Episode = ein Sektor**: erst 80 m lange Abschnitte, dann 200 m, 400 m, dann ganze Runden | lernt am Anfang schneller, weil jede Episode kurz und dicht an Reward ist; findet er danach die ganze Runde? |
| `exp_reward_speed.yaml` | zusätzlicher Geschwindigkeits-Bonus | schneller, aber riskanter? mehr Abflüge? |
| `exp_off_track_penalty.yaml` | Off-Track beendet die Episode nicht, kostet nur Reward | längere Episoden, mehr Exploration – oder lernt er "Abkürzungen"? |
| `exp_lidar31.yaml` | 31 statt 15 Lidar-Strahlen, 240° | bessere Linienwahl in engen Kurven? |
| `exp_domain_random.yaml` | Grip und Masse pro Episode zufällig | robustere Policy (wichtig für Phase 2 mit Reifenverschleiss) |
| `exp_sac.yaml` | SAC statt PPO | sample-effizienter? Rundenzeit nach 1 Mio Schritten |
| `train/ppo_scratch.py` | eigener PPO, gleiche Config | reproduziert er SB3? (Lernkurve vergleichen) |

Weitere Ideen: Hyperparameter (Lernrate, `n_steps`, Netzgrösse), Seeds vergleichen (gleiche Config,
`--seed 1 2 3` – die Streuung ist bei RL gross), Aktion als Lenkwinkel-*Änderung* statt Absolutwert,
Frame-Stacking (letzte 3 Beobachtungen), Beobachtung mit Punkten der Mittellinie voraus statt Krümmung.

```bash
python train/train_sb3.py --config train/configs/exp_sector_curriculum.yaml --stream
python train/train_sb3.py --config train/configs/exp_sac.yaml
python scripts/compare_runs.py runs/phase1_ppo runs/exp_sector_curriculum runs/exp_sac
```

## Ergebnisse (laufend ergänzen)

| Lauf | Maschine | Schritte | beste Rundenzeit | Bemerkung |
|---|---|---|---|---|
| phase1_ppo | Cloud, 2 Kerne, 4 Envs | 1.4 Mio | 57.2 s | erste Runde nach 400k Schritten; Referenz Ideallinie 54.1 s |
