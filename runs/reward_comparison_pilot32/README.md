# Reward-Vergleich: kurzer Pilot

32 Strategieentscheidungen je Lauf, Rollout 16, acht Runden pro Rennen.
Drei identische Trainings-Seeds (0, 1, 2) pro Zeitskala; insgesamt neun Trainings.
Vier Rennen pro Training sind zu wenig fuer eine belastbare Optimierung.

| Sekunden pro Strafpunkt | Im Ziel | Mittlere Rennzeit | Streuung zwischen Seeds |
|---|---:|---:|---:|
| 25 | 3/3 | 906.93 s | 2.05 s |
| 100 | 3/3 | 913.62 s | 13.63 s |
| 50 | 3/3 | 913.62 s | 13.63 s |

Die Skala 25 verbessert bei Seed 1 die Zeit von 929.35 auf 909.30 s.
Bei Seed 0 und 2 sind die Ergebnisse aller drei Skalen identisch (905.75 s).
Die Skala 50 hat hier keinen Vorteil gegenueber 100. Roh-Rewards werden zwischen
den Skalen nicht verglichen; massgeblich sind Zielankunft und Rennzeit.
Die Standardgewichte bleiben unveraendert. Fuer eine Empfehlung braucht es
laengere Trainings und weitere Seeds. Die Strecke ist deterministisch; die
Seeds variieren das Lernen, nicht Strecken- oder Wetterbedingungen.

Der Fahrer ist fest programmiert. Dieser Versuch optimiert die Boxenentscheidungen,
nicht Lenkung oder Bremspunkte. Stopps kosten Zeit; ein Ausfall zahlt das
restliche Zeitbudget und eine zusaetzliche Strafe von fuenf Reward-Punkten.

Bericht: `index.html`. Einzelwerte: `comparison.json`, `comparison.csv`.
Modelle, Konfigurationen und Logs liegen in den jeweiligen Seed-Verzeichnissen.

Laengerer Folgeversuch in einem neuen Verzeichnis:

```powershell
python scripts/compare_strategy_rewards.py --steps 256 --rollout-steps 32 --seeds 0 1 2 --out runs/reward_comparison_256
```
