# Boxenstrategie: erste Version

`KartStrategy-v0` trennt die strategische Entscheidung vom Lenken: Ein fester
Pure-Pursuit-Fahrer fährt mit der bestehenden Kartphysik. Ein Strategie-Agent
entscheidet vor jeder Runde zwischen **Weiterfahren** und **Reifen wechseln**.
Vor der ersten Runde ist ein Stopp wirkungslos. Nach dem Ziel gibt es keine Aktion.

## Vergleich und Auswertung

```powershell
python scripts/evaluate_strategy.py --laps 8 --fixed-lap 4
start runs/strategy_report/index.html
```

Der eigenständige HTML-Report vergleicht keinen Stopp, einen Stopp nach Runde 4
und Stopps ab 65 % Verschleiss. Er zeigt Gesamtzeit, Zielankunft/Ausfall,
Stopps sowie Rundenzeiten und Verschleiss. JSON und CSV dienen der weiteren
Auswertung. Ausfälle werden nicht als schnellste Strategie gewertet.

Alle Strategien nutzen dieselben Bedingungen und denselben Fahrer. Der Vergleich
ist deterministisch; mehrere Seeds allein schaffen hier keine unabhängigen
Versuche. Die Parameter sind ein didaktisches Modell, keine Messdaten von Roggwil.

Mit den Standardparametern beenden alle drei Referenzstrategien das Rennen:
ohne Stopp 1142.05 s, mit einem Stopp nach Runde 4 952.80 s. Die 65-%-Regel
wählt denselben Zeitpunkt und erzielt dieselbe Zeit. Dieser Vergleich zeigt
den modellierten Zielkonflikt; er ist noch kein Ergebnis eines gelernten Agenten.

## Modell

- Eine Reifenmischung, Verschleiss linear zur gefahrenen Strecke, maximal 100 %.
- Standard: 18 Prozentpunkte Verschleiss pro Runde. Grip sinkt von 1 auf 0.55.
- Der Grip wirkt auf die vorhandene Physik (`kart.grip_scale`). Der Referenzfahrer
  reduziert sein Geschwindigkeitsprofil und die Bremsvorschau proportional zum Grip.
- Ein Reifenwechsel setzt den Verschleiss zurück und kostet 12 Sekunden.
  Der Service wird an der Start-/Ziellinie als Zeitverlust modelliert. Es gibt
  noch keine Fahrt durch die Boxengasse, Animation oder zusätzliche Wiederanfahrt.
- Zeiten pro Runde enthalten einen etwaigen Service vor dieser Runde.
- Reward: skalierte Zeitkosten, Zielbonus und neutrale Fortschrittsformung
  (Details unten). Ein Ausfall wird als echtes Episodenende behandelt, auch
  beim Rennzeitlimit: PPO soll nach einem verlorenen Rennen keinen zukünftigen
  Wert mehr anrechnen.
- Beobachtung: verbleibende Runden, Verschleiss, Grip, verbrauchtes Zeitbudget,
  Stoppzahl und Verfügbarkeit eines Stopps (jeweils normalisiert).

## Strategie lernen

```powershell
pip install -e ".[train,dev]"
python train/train_strategy.py --steps 2048 --seed 0
python scripts/evaluate_strategy.py --model runs/strategy_ppo/strategy.zip
```

Ein Trainingsschritt entspricht einer ganzen simulierten Runde und ist daher
wesentlich teurer als ein Fahrtrainingsschritt. Der Trainer nutzt PPO mit Gamma 1,
um die ungekürzte gesamte Rennzeit zu optimieren. Der Strategie-Checkpoint ist
kein Fahrmodell und wird nur mit `evaluate_strategy.py` geladen. Beim Vergleich
müssen dieselbe Rundenzahl und dieselben Modellparameter wie im Training gelten.
Ein vorhandener Fahr-PPO wird in dieser Version noch nicht als Fahrer verwendet.

## Rewards einstellen

Die Standardwerte sind ein erster, überprüfter Ausgangspunkt, keine bereits
empirisch optimierten Hyperparameter:

| Parameter | Standard | Wirkung |
|---|---|---|
| `reward_time_scale_s` | 100 | 100 Sekunden kosten einen Reward-Punkt |
| `reward_progress` | 2 | Zwischenfeedback für abgeschlossene Runden |
| `reward_finish` | 15 | Einmaliger Bonus bei erfolgreicher Zielankunft |
| `reward_failure` | 5 | Zusätzliche Strafe bei Ausfall oder Zeitlimit |

Pro Entscheidung gilt:

```text
reward = -verstrichene_sekunden / zeitskala
         + potential_nachher - potential_vorher
         + zielbonus (nur bei Zielankunft)
         - ausfallkosten (nur bei Ausfall)

potential = fortschrittsgewicht * abgeschlossene_runden / rennrunden
potential_am_episodenende = 0
ausfallkosten = restliches_zeitbudget / zeitskala + ausfallstrafe
```

Das Fortschrittsfeedback summiert sich über jedes ganze Rennen zu null,
auch bei Ausfall. Bei Gamma 1 bleibt die Rangfolge abgeschlossener Rennen
daher genau dieselbe wie bei der Gesamtzeit: Gesamt-Reward = Zielbonus minus
Rennzeit / Zeitskala. Im Standardbeispiel erhält die Strategie ohne Stopp
3.5795 Punkte, der Stopp nach Runde 4 5.4720 Punkte. Ein Ausfall innerhalb
des Zeitbudgets erhält insgesamt -17 Punkte, unabhängig vom Abbruchzeitpunkt.

Frische Reifen, Boxenstopps und blosse Anzahl gefahrener Runden erhalten
keinen zusätzlichen Gesamtbonus. Sonst könnte eine langsame Strategie durch
viele Reifenwechsel mehr Reward bekommen. Der Stopp kostet bereits seine
zwölf Sekunden; eine weitere Stoppstrafe würde das Rennziel verändern.

Alle vier Gewichte lassen sich beim Training und bei der Evaluation setzen:

```powershell
python train/train_strategy.py --steps 2048 --reward-time-scale 100 --reward-progress 2 --reward-finish 15 --reward-failure 5
python scripts/evaluate_strategy.py --model runs/strategy_ppo/strategy.zip --reward-time-scale 100 --reward-progress 2 --reward-finish 15 --reward-failure 5
```

Training speichert die vollständige Konfiguration, den Episodenverlauf in
`monitor.csv`, das Modell und einen anschliessenden Vergleich unter `evaluation/`.
Der Vergleich enthält die gelernte Strategie, keinen Stopp und einen Stopp zur
Rennmitte. Der Report zeigt die Reward-Anteile insgesamt und pro Runde;
CSV und JSON enthalten die gleichen Werte. Höherer Gesamt-Reward ist besser.
Kurze Pilotläufe prüfen die Pipeline; sie belegen noch keinen Lernvorteil.

## Nächste Erweiterungen

Reale Boxengassengeometrie und Darstellung im 3D-Viewer, Reifenmischungen,
trainierter Fahrer, zufällige Rennbedingungen und eine Auswertung über mehrere
unabhängige Rennen. Das bestehende Fahrtraining und Dashboard bleiben separat.
