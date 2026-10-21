# Ideallinie als Vergleichsmassstab

**Wichtig:** Die hier berechnete Ideallinie wird **nicht** zum Training benutzt. Der RL-Agent
bekommt sie nie zu sehen – er muss die schnelle Linie selbst finden (Lidar + Kurvenvorschau,
Reward nur für Fortschritt). Die berechnete Linie dient als klassischer Vergleichsmassstab
("gelernt vs. berechnet") und als Overlay im Viewer.

![Ideallinie](img/raceline.png)

## Methode

`kart_rl/raceline.py`, Werkzeug `tools/compute_raceline.py`:

1. **Minimum-Curvature-Optimierung**: Die Linie wird als seitlicher Versatz zur Mittellinie
   beschrieben, der innerhalb der Streckenränder bleiben muss (Rand-Abstand 1.0 m, damit das
   1.4 m breite Kart Platz hat). Die Krümmung wird um die Referenzlinie linearisiert, so dass
   "minimale Summe der quadrierten Krümmungen" ein beschränktes Least-Squares-Problem ist
   (Ansatz nach Heilmeier et al. 2019). Vier Re-Linearisierungen reichen zur Konvergenz.
2. **Geschwindigkeitsprofil** mit Reibungskreis: Kurvenlimit aus der Krümmung, Beschleunigung
   aus Motorleistung und Restgrip, Bremsen mit der Hinterradbremse (75 % nutzbar) – Vorwärts- und
   Rückwärtsdurchlauf.
3. **Fahren**: Der geskriptete Fahrer (`PurePursuitDriver`) folgt der Linie und dem Profil, mit
   Reibungskreis-Logik (bremst nicht hart, während die Reifen in der Kurve ausgelastet sind).

## Ergebnis (Race-Inn Roggwil, Kart-Modell Phase 1)

| Linie | Länge | Rundenzeit gefahren | Bemerkung |
|---|---|---|---|
| Mittellinie | 640 m | 71.6 s | einfacher Referenzfahrer |
| Ideallinie | 591 m | **54.1 s** | geskripteter Fahrer mit 80 % Grip-Nutzung |
| Ideallinie, Modell-Optimum | 591 m | ≈ 49.8 s | Profil mit 92 % Grip, nur Schätzung |

Zielmarke für den RL-Agenten: unter 54 s, ideal Richtung 50 s.

```bash
python tools/compute_raceline.py            # berechnet, fährt, plottet (ca. 35 s)
python tools/compute_raceline.py --margin 0.8 --safety 0.85
```

Ausgabe: `kart_rl/data/race_inn_roggwil_raceline.json` (Punkte, Krümmung, Sollgeschwindigkeit,
Rundenzeiten) und `docs/img/raceline.png`.
