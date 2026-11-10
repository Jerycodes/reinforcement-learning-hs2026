# Jerys Fahrmodell mit Zeitstrafe: ohne Boxenstopp

```powershell
python scripts/compare_driving_penalty.py
start runs/jery_driving_penalty/index.html
```

Dieser Versuch verwendet **KartLap-v0** und `runs/phase1_v2/best.zip` als
Startpunkt. Der PPO-Agent selbst lenkt, bremst und gibt Gas. Es gibt weder
Reifenverschleiss noch Boxenstopps oder den festen Fahrer aus dem Strategie-Test.

## Fairer Vergleich

- Unverändertes Originalmodell als Referenz.
- Kontrollgruppe: Weitertraining mit Jerys bisherigem Reward.
- Versuchsgruppe: gleiches Weitertraining plus Zeitstrafe von einem Punkt
  pro 25 Sekunden. Bei `dt=0.05` sind das **-0.002 Punkte je Schritt**.
- Fortschritt, Zickzack-Strafe, Bandenkontakt (-20 und Episodenende), Physik
  und Beobachtungen sind in beiden Gruppen identisch.
- Pro Gruppe Seeds 0, 1 und 2, jeweils 40'960 zusätzliche Fahr-Schritte.
  Die ursprüngliche Rollout-Grösse bleibt erhalten: 10 Envs mal 1024 Schritte.
- Beide Gruppen bekommen fürs Weitertraining dieselbe konstante Lernrate 1e-4.
  Das ist ein kontrolliertes Fine-Tuning, keine Fortsetzung der ursprünglichen
  linearen Lernratensenkung bis zu deren konfiguriertem Endpunkt.
- Die zum besten Modell gehörenden `best_vecnormalize.pkl`-Statistiken werden
  geladen und anschliessend je Lauf separat weitergeführt.

Die Evaluation fährt fünf Runden aus stehendem Start. Stehende und fliegende
Runden werden getrennt ausgewiesen. Die Gruppenstatistik mittelt die fliegenden
Runden pro Modell und danach über die Trainings-Seeds. Ausfälle bleiben in der
Zielankunftsquote sichtbar und werden nicht als besonders schnelle Fahrer
gewertet. Die Streuung zwischen Seeds ist keine Streuung zwischen Runden.

Das Originalmodell fährt hier im Mittel **51.925 s** in den vier fliegenden
Runden, beste Runde **51.90 s**, stehende Runde **54.15 s**. Alle fünf Runden
werden unter der aktuellen Kart-Kontur-Kollisionsregel beendet.

Der ausgeführte Pilot ergibt folgende Mittelwerte:

| Variante | Evaluationen im Ziel | Mittlere fliegende Runde |
|---|---:|---:|
| Original, ohne Weitertraining | 1/1 | 51.925 s |
| Weitertraining ohne Zeitstrafe | 3/3 | 52.033 s |
| Weitertraining mit Zeitstrafe | 3/3 | 52.054 s |

Die Zusatzstrafe hilft in diesem Versuch nicht; gegenüber der Kontrollgruppe
liegt sie im Mittel rund 0.021 Sekunden zurück. Beide weitertrainierten Gruppen
sind hier langsamer als das Original. Die Unterschiede sind klein und bei drei
Seeds kein belastbarer Nachweis eines generellen Nachteils. Das Original bleibt
deshalb unverändert; es wird kein Pilotmodell automatisch übernommen.

## Übertragbarkeit der Strafe

Der bisherige Fahr-Reward belohnt gefahrene Fortschrittsmeter. Eine Zeitstrafe
von 1/25 Punkt pro Sekunde ist gegenüber rund 640 Fortschrittspunkten pro Runde
klein (bei 52 Sekunden etwa 2.08 Strafpunkte). Die Gewichtung ist deshalb nicht
direkt mit dem reinen Zeit-Reward des Boxenstrategie-Agenten vergleichbar.
Ein kurzer Versuch kann zeigen, ob sich die konkrete Ergänzung lohnt, aber
keine allgemeine Aussage über die beste Reward-Funktion beweisen.

## Checkpoint unter Windows laden

Jerys Modell wurde unter Python 3.14.4 auf einem Mac gespeichert. Die serialisierte
Python-Funktion für die Lernrate führt beim Aufruf unter dem hiesigen Python 3.12
zu einer Access Violation. `kart_rl/checkpoints.py` ersetzt die gespeicherten
Schedule-Objekte vor dem Deserialisieren durch explizite Werte aus der Konfiguration.
Die Policy-Gewichte werden unverändert geladen; der Vergleich prüft diese vor
dem Training bitgenau gegen `policy.pth` aus dem Originalarchiv.

## Ergebnisse und Wiederholung

`experiment.json` dokumentiert Original-Checkpoint, SHA-256, Konfiguration und
Trainingsbudget. Pro Kombination werden Gewichte, Normalisierung, effektive
Config, Trainingslog und Evaluation gespeichert. Der HTML-Bericht, JSON und CSV
fassen die Ergebnisse zusammen. Vorhandene Versuchsverzeichnisse werden nicht
überschrieben, das Originalmodell bleibt unverändert.

Ein längerer Folgeversuch kann beispielsweise mit `--steps 102400` und einem
frischen `--out` gestartet werden. Die Schrittzahl muss ein Vielfaches von
10'240 sein, damit die Budgets tatsächlich gleich sind. Die Seeds ändern das
Lernen; die deterministische Strecke in der Evaluation bleibt identisch.
