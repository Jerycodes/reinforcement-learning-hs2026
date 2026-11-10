# Validierte optimierte Policy

Originalmodell und optimierte Kopie fahren unter identischen Bedingungen, ohne Boxenstopps.
30-Runden-Pruefung: Original 51.896552 s, Kandidat 51.850000 s im Mittel der fliegenden Runden.
Vorsprung: 0.046552 s pro fliegender Runde.
Beste Runde: Original 51.80 s, Kandidat 51.70 s.
Beide beenden alle 30 Runden und 9 von 10 neuen Zufallsstarts (Seeds 100-109).

Dies ist ein kleiner Testvorsprung auf dieser Strecke, kein allgemeiner Ueberlegenheitsnachweis.
Der stehende Start ist beim Kandidaten 0.05 s langsamer.

Die finale Kopie unter winner/ kombiniert das PPO-Profil long_horizon
(gamma=0.997, gae_lambda=0.97, konstante Lernrate 3e-5, ent_coef=0)
mit Steuerfaktor 0.94 und Gas-/Bremsbias -0.01. Leistung, Grip und Regeln bleiben gleich.
Die Policy-Ausgabekalibrierung ist eine Parametersuche, kein weiterer PPO-Lernlauf.

Der beste Zwischenstand wurde aus 102400 zusaetzlichen Trainingsschritten ausgewaehlt;
es wurde nicht automatisch der letzte Stand verwendet. Das Original bleibt unveraendert.

Bericht: index.html. Rohdaten: comparison.json, original.json, candidate.json.
Modelldateien: winner/best.zip, winner/best_vecnormalize.pkl, winner/config.yaml.
