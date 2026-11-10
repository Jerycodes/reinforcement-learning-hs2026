# Fahrer optimieren: Ergebnisse und Grenzen

## Ergebnis auf derselben Strecke

Der geprüfte Kandidat liegt im langen Vergleich knapp vor Jerys gespeichertem
`phase1_v2/best.zip`. Beide fahren ohne Stopps und ohne Reifenverschleiss mit
identischer Physik, Leistung, Grip, Sensorik und Kart-Kontur-Kollisionsregel.

| Gleiche Prüfung mit 30 Runden | Original | Optimierte Policy |
|---|---:|---:|
| Mittlere fliegende Runde (29 Runden) | 51.8966 s | 51.8500 s |
| Beste fliegende Runde | 51.80 s | 51.70 s |
| Stehende erste Runde | 54.15 s | 54.20 s |
| Vollständiger langer Lauf | 30/30 | 30/30 |
| Neue Zufallsstarts (Seeds 100–109) | 9/10 | 9/10 |

Der mittlere Vorteil beträgt rund **0.0466 Sekunden pro fliegender Runde**, also
etwa 0.09 %. Das ist ein kleiner Vorsprung in dieser Prüfung, kein Nachweis
allgemeiner Überlegenheit. Der stehende Start ist sogar 0.05 s langsamer.
Das Ergebnis darf nicht mit den früheren 5-Runden-Mittelwerten verglichen werden;
oben werden exakt dieselben 30 Runden gegenübergestellt. Die Zeitauflösung
einzelner Runden beträgt durch `dt=0.05` 0.05 Sekunden; der Mittelwert über
29 Runden kann feinere Unterschiede zeigen.

Das Original wurde nicht ersetzt. Die geprüfte Kopie liegt hier:

```text
runs/driver_validation_v1/winner/best.zip
runs/driver_validation_v1/winner/best_vecnormalize.pkl
runs/driver_validation_v1/winner/config.yaml
```

Auswertung im bestehenden Evaluator:

```powershell
python scripts/evaluate.py runs/driver_validation_v1/winner/best.zip --laps 30
```

## Wie der Kandidat entstand

1. Drei PPO-Profile erhielten je 102'400 zusätzliche Schritte ab demselben
   Originalmodell, Seed 2. Alle verwenden die vorsichtige konstante Lernrate
   3e-5 und keinen Entropiebonus. Ein Profil behält Gamma 0.99, eines erhöht
   Gamma auf 0.997 und GAE-Lambda auf 0.97, eines testet `reward_time=-0.05`.
2. Alle 20'480 gesammelten Schritte werden drei Runden geprüft und der schnellste
   vollständig erfolgreiche Stand behalten. Die letzte PPO-Aktualisierung wird
   zusätzlich evaluiert. Dadurch wird ein guter früher Stand nicht durch einen
   langsameren späteren ersetzt. Der Callback läuft vor der Aktualisierung des
   gerade gesammelten Rollouts; seine Schrittzahl bezeichnet gesammelte Daten.
3. Unabhängig davon wird die Ausgabeschicht der Policy feinabgestimmt. Das ist
   eine kleine Parametersuche, kein weiterer PPO-Lernlauf: Lenkausgabe skalieren,
   Gas-/Bremsausgabe skalieren oder leicht verschieben. Die normalen Aktionsgrenzen
   der Umgebung bleiben bestehen. Jeder Versuch startet von unveränderten
   Ausgangsgewichten; Änderungen werden nicht aufeinander aufaddiert.
4. Der beste kombinierte Kandidat stammt vom frühen Zwischenstand des Profils
   mit längerer Voraussicht, mit Lenkausgabe-Faktor **0.94**, Gas-/Bremsbias
   **-0.01** und unveränderter Gas-/Bremsverstärkung **1.0**. Seine mittlere
   fliegende Runde in der 10-Runden-Auswahl beträgt 51.8056 s.
5. Erst danach folgt der längere 30-Runden-Lauf mit zehn weiteren Startzuständen.
   Eine Kopie wird nur als Gewinner abgelegt, wenn sie den langen Lauf beendet,
   schneller ist und mindestens dieselbe Anzahl neuer Startzustände beendet.

Das längere Training und die stärkere Zeitstrafe allein lieferten keinen grossen
Gewinn. Die Verbesserung des finalen Modells kommt aus der Kombination mit der
Feinabstimmung der Policy. Es gibt keinen Leistungsboost und keine gelockerte Bande.

## Dateien und Wiederholung

```powershell
python train/optimize_driver.py --profile gentle --out runs/new_gentle
python train/optimize_driver.py --profile long_horizon --out runs/new_long
python train/optimize_driver.py --profile time_pressure --out runs/new_time

python scripts/calibrate_driver.py --model runs/new_long/best.zip --out runs/new_combined --gains 0.94 0.96 0.98 1 --biases -0.01 0 0.01
python scripts/validate_optimized_driver.py --candidate runs/new_combined/best.zip --out runs/new_validation
```

Die Verzeichnisse müssen neu sein. `history.json` und `monitor.csv` dokumentieren
die Trainingsläufe, `search.json` alle Kalibrierungsversuche, `result.json` die
jeweilige Zehn-Runden-Prüfung. Der Gesamtbericht liegt unter
`runs/driver_validation_v1/index.html`; die Rohdaten der abschliessenden Prüfung
in `comparison.json`. `scripts/report_driver_optimization.py` rendert den Bericht
aus den gespeicherten Ergebnissen dieser Versuchsreihe.

Es wurden ein Trainings-Seed pro Profil und dieselbe feste Strecke verwendet.
Das erlaubt die Auswahl eines konkreten schnelleren Modells, aber keine Aussage,
dass diese Hyperparameter generell oder auf anderen Strecken besser lernen.
