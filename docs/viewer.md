# 3D-Viewer (Browser, Three.js)

`viewer/index.html` – läuft ohne Installation im Browser (three.js liegt im Repo unter
`viewer/vendor/`, kein CDN nötig). Der Viewer braucht einen HTTP-Server aus dem Repo-Root, damit er
Strecke und Replays laden kann.

```bash
python scripts/serve_viewer.py              # öffnet http://localhost:8000/viewer/
```

## Drei Betriebsarten

### 1. Replay
Aufgezeichnete Fahrten (`scripts/evaluate.py ... --replay replays/name.json`) laden – per
Dateiauswahl, URL-Feld oder `?replay=../replays/a.json&replay=../replays/b.json`. Mehrere Replays
fahren gleichzeitig (Agent nach 200k vs. 2 Mio Schritten, Agent vs. Referenzfahrer), mit Zeitleiste,
Tempo, Telemetrie und Fahrspur (Farbe = Geschwindigkeit).

### 2. Training live
Training mit `--stream` starten, Viewer verbinden – alle parallelen Trainings-Karts fahren live auf
der Strecke (roter Punkt = Abflug), dazu Schritte, Episoden, Ø Return, beste Rundenzeit und
Schritte/s:

```bash
python train/train_sb3.py --config train/configs/phase1_ppo.yaml --stream     # ws://localhost:8765
python scripts/serve_viewer.py --no-open                                      # zweites Terminal
# Browser: http://localhost:8000/viewer/?live  -> "verbinden"
```
Das Training läuft schneller als Echtzeit; der Viewer zeigt jeweils den aktuellen Stand (max. 25 Bilder/s).

### 3. Live-Server mit Parameter-Panel
Ein trainiertes Modell (oder Referenzfahrer) fährt in Echtzeit, Parameter lassen sich per Regler
ändern (Reifengrip, Antrieb, Bremse, Lenkwinkel, Begrenzer, Masse, Off-Track-Abstand; in Phase 2/3
kommen Reifen, Boxenstopp und Boost dazu):

```bash
python -m kart_rl.live.server --policy runs/phase1_ppo/best.zip --open
python -m kart_rl.live.server --policy baseline:raceline --speed 2
```

## Bedienung

- Kameras: Verfolger, Onboard, Draufsicht, Frei (Maus) – Tasten 1–4
- Leertaste: Play/Pause (Replay); Anzeige: Mittellinie, berechnete Ideallinie, Fahrspur, Lidar (Live-Server)
- HUD-Kart: welches Kart Kamera und Telemetrie bekommt

## Eigenes Kart-Modell
Liegt eine Datei `viewer/assets/kart.glb` (glTF) vor, wird sie statt des eingebauten Low-Poly-Karts
geladen (Vorwärtsrichtung +X, Bodenhöhe y = 0, Masse in Metern).

## GitHub Pages
Repo-Root als Pages-Quelle wählen, dann ist der Viewer unter `https://<user>.github.io/<repo>/viewer/`
erreichbar (Replay-Modus; Live braucht den lokalen Python-Server).

## Video "Anfang vs. Ende"

```bash
python scripts/make_video.py --run runs/phase1_ppo --out docs/video/evolution.mp4          # nacheinander
python scripts/make_video.py --run runs/phase1_ppo --grid --out docs/video/evolution_grid.mp4  # gleichzeitig im Raster
```
Nimmt automatisch vier Checkpoints über das Training verteilt plus das beste Modell. Für 3D-Aufnahmen
den Viewer im Replay-Modus mit mehreren Replays laden und den Bildschirm aufnehmen (macOS: ⌘⇧5).
