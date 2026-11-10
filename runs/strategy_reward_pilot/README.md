# Reward-Pilot: Pipeline-Test, kein ausgelerntes Modell

```powershell
python train/train_strategy.py --steps 16 --rollout-steps 8 --out runs/strategy_reward_pilot
```

Seed 0, acht Runden, Standard-Rewards. Nur zwei Trainingsepisoden.
`strategy.zip` ist der letzte Stand, nicht der beste mögliche Agent.

| Deterministische Evaluation | Rennzeit | Stopps | Gesamt-Reward |
|---|---:|---:|---:|
| Ohne Stopp | 1142.05 s | 0 | 3.5795 |
| Stopp nach Runde 4 | 952.80 s | 1 | 5.4720 |
| PPO-Pilot | 905.75 s | 7 | 5.9425 |

Alle Rennen wurden beendet. Der Pilot wechselt vor jeder weiteren Runde die
Reifen. Im vereinfachten Verschleissmodell kann sich das gegenüber dem einzelnen
Stopp rechnen. Das ist weder eine realistische Boxenempfehlung noch ein Nachweis
einer optimalen Strategie oder eines Vorteils gegenüber den bisherigen Rewards.
Die beiden zufälligen Trainingsepisoden waren sogar schneller als die abschliessende
deterministische Policy; siehe `monitor.csv`. Für einen belastbaren Lernvergleich
braucht es längere Trainingsläufe mit mehreren Seeds und identischen Bedingungen.

Vollständige Parameter: `strategy_config.json`. Aufgeschlüsselte Ergebnisse:
`evaluation/index.html`, `evaluation/results.json` und `evaluation/laps.csv`.
