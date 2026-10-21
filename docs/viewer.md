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
Training mit `--stream` starten – der Browser öffnet sich, und **ein Kart** fährt in Echtzeit den
aktuellen Trainingsstand (deterministisch, ohne Erkundungsrauschen; nach jedem Policy-Update fährt
es mit der neuen Policy weiter). Dazu Schritte, Episoden, Ø Return, beste Rundenzeit und Schritte/s.
Das Training selbst läuft im Hintergrund mit mehreren parallelen Umgebungen (nur für die
Geschwindigkeit – es bleibt ein einzelner lernender Agent). Mit `--stream-all` werden diese
Trainings-Umgebungen zusätzlich als eigene Karts angezeigt (roter Punkt = Abflug).

```bash
python train/train_sb3.py --config train/configs/phase1_ppo.yaml --stream     # öffnet den Browser automatisch
python train/ppo_scratch.py --config train/configs/phase1_ppo.yaml --stream   # geht auch mit dem eigenen PPO
```
`--stream` startet den Websocket (Port 8765) und einen HTTP-Server für den Viewer (Port 8000) und
öffnet http://localhost:8000/viewer/?live – mit `--no-open` ohne Browser, `--http-port`/`--stream-port`
für andere Ports.
Das Training läuft schneller als Echtzeit; der Viewer zeigt jeweils den aktuellen Stand (max. 25 Bilder/s).

### 3. Live-Server mit Parameter-Panel
Ein trainiertes Modell (oder Referenzfahrer) fährt in Echtzeit, Parameter lassen sich per Regler
ändern (Reifengrip, Antrieb, Bremse, Lenkwinkel, Begrenzer, Masse, Off-Track-Abstand; in Phase 2/3
kommen Reifen, Boxenstopp und Boost dazu):

```bash
python -m kart_rl.live.server --policy runs/phase1_ppo/best.zip --open
python -m kart_rl.live.server --policy baseline:raceline --speed 2
```

### 4. Experiment-Dashboard (Parameter wählen, Experimente starten und vergleichen)

```bash
python -m kart_rl.dashboard            # öffnet http://localhost:8080/
```

Das Dashboard ist eine lokale Webseite (nur Python-Standardbibliothek, kein Login), über die man
Experimente ohne Terminal startet und vergleicht:

- **Neues Experiment**: Name, Basis-Config (z. B. `phase1_ppo` oder ein früheres Experiment),
  Trainer (Stable-Baselines3, eigener PPO, PBT), Algorithmus (PPO/SAC), Live-3D ja/nein,
  optional Sektor-Curriculum oder Weitertrainieren ab Checkpoint.
- **Parameter** in vier Gruppen (Allgemein, PPO, Umgebung/Reward, Kart-Physik), jeder mit kurzer
  Erklärung und dem Basiswert; geänderte Felder werden orange markiert. Nur die Änderungen werden
  übernommen, der Rest kommt aus der Basis-Config.
- Beim Start wird `train/configs/<name>.yaml` geschrieben (= reproduzierbar, kann gepusht werden)
  und der Trainer als eigener Prozess gestartet; das Log landet in `runs/<name>/train.log`.
  Das Dashboard darf man schliessen, die Trainings laufen weiter; *Stop* beendet sauber (Ctrl+C,
  `checkpoints/last.zip` wird geschrieben).
- **Läufe**: alle Ordner unter `runs/` mit Status, Schritten, bester Rundenzeit, wann die erste
  volle Runde geschafft wurde, Dauer und den wichtigsten Config-Werten. Pro Lauf: *3D live*
  (Training zuschauen), *Modell ansehen* (startet den Live-Server mit Policy-Auswahl und Reglern),
  *Log*, *Config*, *Stop*.
- **Vergleich**: Häkchen bei mehreren Läufen → Lernkurven (beste Rundenzeit über Schritte) mit den
  Referenzlinien Ideallinien-Fahrer und Modell-Optimum (aus `kart_rl/data/race_inn_roggwil_raceline.json`). Dieselbe Tabelle gibt es als Skript
  (`python scripts/compare_runs.py`).

Für die Gruppe: Jeder startet das Dashboard lokal auf seinem Rechner. Ergebnisse teilen =
`train/configs/<name>.yaml`, `runs/<name>/eval.csv`, `runs/<name>/best.zip` und `config.yaml`
committen (Checkpoints und TensorBoard-Logs sind git-ignoriert). Nach `git pull` erscheinen die
Läufe der anderen in der Tabelle und im Vergleichs-Chart, und ihre Configs stehen als Basis für
neue Experimente zur Auswahl.

Ports: Dashboard 8080 (`--port` ändert das), Live-Streams ab 8765 aufwärts (pro gestartetem
Lauf einer). Ein separat gestarteter `kart_rl.live.server` oder ein Training mit `--stream`
belegt 8765 ebenfalls – dann vorher beenden.

### 5. Spiel-Modus: selbst fahren gegen das Modell

```bash
python -m kart_rl.game --policy runs/phase1_v2/best.zip --name Jeremy --open
```
oder im Dashboard bei einem Lauf «Selbst fahren» klicken. Du steuerst dasselbe Kart in derselben
Simulation mit der Tastatur (Pfeiltasten oder WASD: lenken, Gas, Bremse), das Modell fährt als
halbtransparenter Geist mit – gleicher Start nach einem 3-s-Countdown, keine Kollision zwischen
beiden. Gleiche Regeln wie im Training: Bandenkontakt kostet Tempo (Kart wird zurückgedrückt) und
macht die Runde ungültig (`*`). Backspace setzt dich nach einem Crash zurück auf die Strecke,
R startet neu, 1–4 wechseln die Kamera.

- HUD und Panel zeigen Runde, aktuelle, letzte und beste Rundenzeit für dich und die KI sowie die
  Differenz der besten Runden.
- Jede Runde wird in `runs/human/laps.csv` geloggt (Fahrer, Zeit, gültig, Gegner-Modell, KI-Bestzeit);
  das Dashboard zeigt daraus die Tabelle «Mensch gegen KI».
- Jeder Lauf (Neustart/Beenden) wird als Replay gespeichert: `replays/human/<Name>_<Zeit>_du.json`
  und `…_KI.json` – im Replay-Modus zusammen laden, dann fährt dein Lauf gegen den der KI.
- Zusätzlich entsteht `…_demo.json` (Beobachtungen + deine Aktionen) für den optionalen Zusatz
  unten.

**Zusatz (optional): Lernen vom Menschen.** `train/learn_from_human.py` nimmt deine gültigen Runden,
die schneller waren als die KI im selben Lauf, und passt das Policy-Netz eines bestehenden Modells per
Behaviour Cloning (überwachtes Lernen: Beobachtung → deine Aktion) an. Das Ergebnis ist ein normaler
Checkpoint, der danach mit PPO weitertrainiert wird – RL verfeinert, was es kopiert hat:

```bash
python train/learn_from_human.py --model runs/phase1_v2/best.zip --out runs/phase1_v2_human
python train/train_sb3.py --config train/configs/phase1_ppo.yaml --run-name phase1_v2_human \
    --resume runs/phase1_v2_human/checkpoints/last.zip --more 3000000 --stream
```
(`--all-laps` nimmt alle gültigen Runden, `--faster-than 52` eine feste Schwelle.) Der normale
Trainingsablauf benutzt davon nichts – es ist ein reiner Zusatz.

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
