# Privates Arcade-Rennen

## Lokal spielen

Im Repository auf `feature/pitstop-strategy-analysis` unter Windows genügt mit der vorhandenen Umgebung:

```powershell
.\.venv\Scripts\python.exe -m kart_rl.game --arcade --http 8031 --ws 8796 --open
```

Auf einem frischen Checkout zuerst `py -3 -m venv .venv` und `.\.venv\Scripts\python.exe -m pip install -e .` ausführen. Für PPO oder Tests zusätzlich `.\.venv\Scripts\python.exe -m pip install -e ".[train,dev]"`.

1. [Fahrerprofil öffnen](http://localhost:8031/viewer/?game&arcade&ws=8796). Namen eingeben und Fuchs, Panda, Luchs oder Drache auswählen. **Zur Garage** zeigt das eigene 3D-Kart. Kart und Reifen verändern die Vorschau sowie die Werte für Tempo, Antritt und Handling. **In die Lobby** erstellt den Raum oder tritt einem eingeladenen Raum bei.
2. Als Host **Einladungslink kopieren**. Die zweite Person öffnet diesen Link, wählt ihr eigenes Profil und tritt bei. Fünf Fahrerkarten zeigen belegte und freie Plätze. Der Host kann mit **+ Bot** und **− Bot** das Feld ergänzen.
3. Alle Menschen klicken **Bereit melden**, der Host **Rennen starten**. Vier Sekunden lang erscheinen Aufstellung und Startpositionen. Danach leuchten im Sekundentakt drei rote Lampen auf. Bei Grün starten Fahrt und Uhr gemeinsam.
4. **WASD / Pfeile** lenken, beschleunigen oder bremsen. Beim Lenken **Shift** halten zum Driften. Loslassen nach mindestens 0,7 Sekunden Drift gibt einen Mini-Boost. **Leertaste** nutzt den Boostvorrat. **R / Backspace** setzt auf die Mittellinie zurück und kostet drei Sekunden. Kamera und Streckenoverlays liegen unter **Analyse**. Die LiDAR-Strahlen gehören zur bestehenden Live-Analyse.
5. Nach drei Runden erscheint die Zielzeit. Wenn das Feld gewertet ist, zeigt die Endrangliste alle Zeiten inklusive Strafen. Für eine Revanche melden sich die Menschen erneut bereit und der Host wählt **Neues Rennen**.

### Aufstellung und Zeitmessung

Die Roggwiler Mittellinie verläuft nach der Startlinie zunächst westwärts. Die erste deutliche Richtungsänderung liegt bei etwa 22 m. Auf der rund 7,2 m breiten Geraden lost der Server für jedes Rennen neu. Die Spuren liegen je 1,55 m von der Mittellinie entfernt, die Reihen 3,8 m auseinander, die rechte Spur ist 0,85 m nach hinten versetzt. Platz eins steht bei 12 m, Platz fünf bei 4,4 m. Alle Karts stehen vollständig jenseits der Ziellinie und vor der Kurve.

Jedes Kart startet bei null abgeschlossenen Runden. Der Rennfortschritt beginnt an seiner tatsächlichen Projektion auf die Mittellinie. Beim ersten regulären Umlauf über die Linie zählt Runde eins, beim dritten endet das Rennen. Die Auslosung wird zusammen mit Positionen und Ausrichtung an beide Browser gesendet. Die Solo-Gymnasium-Umgebung startet weiterhin an ihrer bisherigen Position. Beobachtungen, Aktionen, Reward und Modellformat bleiben kompatibel.

### Browsertest wiederholen

Bei laufendem lokalen Server kann `node scripts/test_paddock.cjs` den vollständigen Ablauf mit zwei getrennten Chromium-Sitzungen und drei Bots fahren. Dazu muss Node das Paket `playwright` finden und dessen Chromium installiert sein (alternativ `CHROME_BIN` auf ein installiertes Chrome setzen). Der Test erzeugt Tastaturereignisse, prüft die gerenderten Startpositionen und vergleicht Ergebnisse und Revanche beider Browser. Screenshots landen im ausgegebenen temporären Verzeichnis. Ein vollständiges Rennen benötigt einige Minuten. Die normale Ergebnis-CSV enthält auch diese Testrennen.

### Karts und Reifen

| Kart | Antritt | Spitze ohne Boost | Lenkung |
|---|---:|---:|---|
| Club | 12 m/s² | 79 km/h | ausgewogen |
| Sprint | 14 m/s² | 72 km/h | agil |
| GT | 10 m/s² | 86 km/h | ruhiger |

Standardreifen sind die Referenz. Grip-Reifen erhöhen die Kurvenreaktion um 16 % und reduzieren den Antritt um 8 %. Speed-Reifen erhöhen den Antritt um 8 % und reduzieren die Kurvenreaktion um 11 %. Die Auswahl beeinflusst auch den seitlichen Schlupf beim Driften.

### Rennwertung und Verbindungsabbruch

Der Server simuliert feste Schritte von 1/30 Sekunde. Der Browser übermittelt Tasten und Lobbywünsche. Das Rennen wertet Nettodistanz, abgeschlossene Runden und Zielzeit inklusive Strafen. Bandenkontakt kostet Geschwindigkeit. Bei Kartkontakt verliert das schnellere Kart einmal je neuem Kontakt Tempo. Nach zehn simulierten Minuten werden offene Ergebnisse als DNF gewertet. Die Endrangliste und `runs/arcade/races.csv` berücksichtigen auch ausgestiegene Teilnehmer, solange der Raum weiterbesteht.

Beim Schliessen eines Fensters wird der Platz nach Erkennung des Abbruchs frei. Der nächste menschliche Spieler übernimmt die Hostrolle. Während des Rennens warten neu verbundene Spieler bis zum nächsten Start. Ein Neuladen erzeugt einen neuen Teilnehmer. Sobald der letzte Mensch geht, verfällt der Raumlink. Laufende Räume liegen im Arbeitsspeicher.

## Gemeinsame Simulation und Training

Die Roggwiler Strecke, ihre Mittellinie, LiDAR, die Kartkontur und Bandenkollisionen kommen aus KartLapEnv. Der Three.js-Viewer zeigt autoritative Zustände und verfolgt das eigene Kart. Die klassischen Fahrversuche verwenden weiter ihr eigenes Fahrmodell.

- `kart_rl/game/driver.py` enthält die gemeinsame Tastatur- und Fahrzeuganbindung.
- `kart_rl/game/race.py` enthält die browserlose Arcade-Simulation, Auswahl, Rennregeln und den regelbasierten Mittellinienfahrer.
- `kart_rl/game/arcade.py` verwaltet Verbindungen, Raumlinks und Verteilung der Zustände.
- `kart_rl/envs/arcade_env.py` nutzt denselben Fahrerschritt in einer Solo-Gymnasium-Umgebung. Die 38 Beobachtungswerte bestehen aus 30 Strecken-/Fahrzustandswerten sowie Boostvorrat, Driftladung, Mini-Boost-Restzeit und fünf Setupwerten. Die vier Aktionen in [-1,1] sind Lenkung, Gas/Bremse, Drift und Boost. Positive Werte aktivieren die letzten beiden. Reward pro Schritt ist der vorzeichenbehaftete Fortschritt in Metern minus 0,2 × Schrittzeit und minus 1 bei Bandenkontakt. Das Ziel gibt zusätzlich 10 Punkte. Zielankunft beendet die Episode, Zeitablauf kürzt sie ab. Beim Reset werden Kart und Reifen variiert.

Jerys Fahr-PPO und der optimierte Lap-Checkpoint verwenden 30 Beobachtungswerte und zwei Aktionen. Der Arcade-Server verwendet standardmässig den regelbasierten Fahrer. Ein explizit gewähltes Modell wird erst nach Prüfung der gespeicherten Raumdimensionen und der Arcade-Kennung geladen. Nur eigene, vertrauenswürdige lokale Modellarchive angeben.

```powershell
.\.venv\Scripts\python.exe -m train.train_arcade --steps 250000 --output runs/arcade_ppo/policy.zip
.\.venv\Scripts\python.exe -m kart_rl.game --arcade --arcade-policy runs/arcade_ppo/policy.zip --http 8031 --ws 8796 --open
```

Den vorherigen Server vor einem Neustart auf denselben Ports beenden. Das Training übt Solorunden. Vor einem Einsatz als Renngegner Zielquote und Rundenzeiten über alle Kombinationen prüfen. Mit `.\.venv\Scripts\python.exe -m pytest -q` laufen auch die Raum-, Rennregel- und Gymnasium-Tests.

## Mit anderen Geräten und über das Internet

**Im WLAN** den Server mit zusätzlichem `--lan` starten und bei Bedarf beide TCP-Ports in der Firewall für das private Netzwerk erlauben. Auf dem Host die Startadresse mit seiner LAN-IP statt localhost öffnen, beispielsweise `http://192.168.1.20:8031/viewer/?game&arcade&ws=8796`. Der kopierte Raumlink verwendet denselben Hostnamen. Mitspielende benötigen einen Browser mit WebGL und eine Tastatur.

**Öffentlicher Link** auf einem eigenen oder bereits freigegebenen Linux-Server:

1. Den geprüften Feature-Branch auf den Server übertragen. Python ab 3.10 installieren, im Projekt `python3 -m venv .venv` und `.venv/bin/python -m pip install -e .` ausführen. Das Dienstkonto braucht Schreibrechte für `runs/arcade`.
2. Aus dem Projektverzeichnis `.venv/bin/python -m kart_rl.game --arcade --http 8031 --ws 8796` starten. Für dauerhaften Betrieb diesen Befehl als Dienst mit Projektverzeichnis als WorkingDirectory und automatischem Neustart betreiben. Ein einzelner Prozess hält die Räume. Neustarts schliessen laufende Räume.
3. Eine eigene Subdomain per DNS auf diesen Host zeigen lassen. TCP 80 und 443 zum Reverse Proxy freigeben. Die beiden Python-Ports bleiben auf Loopback. Caddy installieren, im folgenden Caddyfile die Beispieldomain ersetzen und `caddy run --config Caddyfile` starten (bei bestehendem Caddy-Dienst die Konfiguration dort integrieren und neu laden).

```caddyfile
kart.example.org {
    reverse_proxy /socket 127.0.0.1:8796
    reverse_proxy 127.0.0.1:8031
}
```

4. Im Browser `https://kart.example.org/viewer/?game&arcade` öffnen und ein Fahrerprofil erstellen. Ohne ws-Port im Link verwendet der Viewer dieselbe Domain mit WSS auf /socket. Caddy übernimmt TLS und den WebSocket-Upgrade. [Caddy-Dokumentation](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy)
5. Den erzeugten Link inklusive Raumparameter an vier weitere Personen senden. Wer den vollständigen Link kennt, kann einen freien Platz belegen. Von einem zweiten Netz Beitritt, Countdown, Steuerung, Hostwechsel und Zielwertung testen. Die Spielversion ist lokal geprüft. Ein öffentlicher Fünf-Personen-Test und die Messung der Netzwerklatenz stehen aus.
