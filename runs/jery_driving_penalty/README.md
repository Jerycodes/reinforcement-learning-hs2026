# Jerys Fahrmodell ohne Boxenstopp: Pilot mit Zeitstrafe

Original: `runs/phase1_v2/best.zip`, gespeicherter Stand bei 22.3 Mio. Schritten.
Je Variante drei Seeds (0, 1, 2) und 40’960 weitere Fahr-Schritte.
Jeder Lauf wird ueber fuenf Runden aus stehendem Start ausgewertet.

| Variante | Im Ziel | Mittlere fliegende Runde | Streuung zwischen Seeds |
|---|---:|---:|---:|
| Original, kein Weitertraining | 1/1 | 51.925 s | - |
| Weitertraining ohne Zeitstrafe | 3/3 | 52.033 s | 0.109 s |
| Weitertraining mit Zeitstrafe | 3/3 | 52.054 s | 0.038 s |

Die Zeitstrafe verbessert diesen kurzen Test nicht: im Gruppenmittel ist sie
etwa 0.021 Sekunden pro fliegender Runde langsamer als die Kontrollgruppe.
Beide weitertrainierten Gruppen sind hier langsamer als das Originalmodell.
Alle Evaluationen beenden fuenf Runden. Die Unterschiede sind klein und bei
drei Seeds kein belastbarer Nachweis fuer einen generellen Nachteil.

Das Original bleibt unveraendert und wird nicht durch ein Pilotmodell ersetzt.
Keine Reifenabnutzung und keine Stopps; PPO steuert tatsaechlich das Kart.
Zusaetzliche Strafe: -0.002 Punkte je 0.05-s-Schritt, also -1 Punkt pro 25 s.
Fortschrittsreward, Bandenkontaktstrafe (-20), Physik und Sensorik sind gleich.
Die Zusatzstrafe ist klein gegenueber rund 640 Fortschrittspunkten pro Runde.

Beide Gruppen nutzen fuer das Fine-Tuning eine konstante Lernrate 1e-4,
die gespeicherten Originalgewichte und die passende VecNormalize-Datei.
Fuer das Laden des Mac-/Python-3.14-Checkpoints unter Python 3.12 werden nur
die serialisierten Schedules rekonstruiert; die Policy-Gewichte wurden zuvor
bitgenau mit dem Archiv verglichen.

Bericht: `index.html`. Daten: `comparison.json`, `comparison.csv`.
Je Seed-Verzeichnis: `last.zip`, `vecnormalize.pkl`, `config.yaml`,
`result.json`, `monitor.csv`, `train.log`.
