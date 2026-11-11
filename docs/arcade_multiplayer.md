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

In Fahrtrichtung liegt die Linie vor dem gesamten Feld. Die Projektion des ersten Startplatzes liegt 8 m vor der Überfahrt, die des fünften 15,6 m. Die Spuren liegen 1,55 m links und rechts der Mittellinie. Reihenabstand 3,8 m, Versatz innerhalb der Reihe 0,85 m. Der Server lost aus und sendet Position und Gierwinkel jedes Platzes. Der Viewer verwendet diese Werte für Kart, Box und Startnummer. Die physische Ampel steht an der Linie.

Die erste Überfahrt öffnet die Runde. Erst nach den Kontrollpunkten bei 25 %, 50 % und 75 %, einer vollen Nettorundendistanz und der nächsten Vorwärtsüberfahrt zählt eine Runde. Kontakte werden vor der Wertung aufgelöst. Für die fliegende Bestzeit zählen weiterhin nur Folgerunden ohne Rücksetzen.

Nach der Aufstellungsansicht ist die Fahrersicht aktiv. Der Augenpunkt liegt 72 cm über dem Asphalt und 35 cm hinter der Kartmitte. Das vertikale Sichtfeld beträgt 54°. Mit **1 / 2 / 3 / 4** wechselst du zu Verfolger-, Fahrer-, Drauf- und freier Sicht.

[Startfeld mit fünf Karts](img/roggwil-startfeld.png) · [Fahrersicht während der Startampel](img/roggwil-fahrersicht.png)

### Browsertest wiederholen

Bei laufendem lokalen Server kann `node scripts/test_paddock.cjs` den vollständigen Ablauf mit zwei getrennten Chromium-Sitzungen und drei Bots fahren. Dazu muss Node das Paket `playwright` finden und dessen Chromium installiert sein (alternativ `CHROME_BIN` auf ein installiertes Chrome setzen). Der Test erzeugt Tastaturereignisse, prüft die gerenderten Startpositionen und vergleicht Ergebnisse und Revanche beider Browser. Screenshots landen im ausgegebenen temporären Verzeichnis. Ein vollständiges Rennen benötigt einige Minuten. Die normale Ergebnis-CSV enthält auch diese Testrennen.

### Karts und Reifen

| Kart | Antritt | Spitze ohne Boost | Lenkung |
|---|---:|---:|---|
| Club | 12 m/s² | 79 km/h | ausgewogen |
| Sprint | 14 m/s² | 72 km/h | agil |
| GT | 10 m/s² | 86 km/h | ruhiger |

Standardreifen sind die Referenz. Grip-Reifen erhöhen die Kurvenreaktion um 16 % und reduzieren den Antritt um 8 %. Speed-Reifen erhöhen den Antritt um 8 % und reduzieren die Kurvenreaktion um 11 %. Die Auswahl beeinflusst auch den seitlichen Schlupf beim Driften.

### Maßstab, Bot-Tempo und persönliche Referenz

Für die Arcade-Variante wurde das gesamte Roggwiler Layout um Faktor 1,35 skaliert. So bleiben auch die Abstände zwischen benachbarten Kurven erhalten. Asphalt, Banden und physische Kollisionsmaske beruhen auf denselben Streckenquerschnitten. Die Karts messen weiterhin 1,55 × 1,02 m. Das klassische Layout bleibt separat verfügbar.

| Messgröße | v1 | v2 |
|---|---:|---:|
| Rundenlänge | 639,64 m | 863,51 m |
| Asphaltbreite, Minimum / Median | 4,94 / 6,91 m | 6,66 / 9,32 m |
| Erlaubter seitlicher Bereich für die Kartmitte, Minimum / Median | 3,60 / 5,60 m | 5,52 / 8,28 m |
| Richtungsänderung bei 100 ms Tastendruck, 36 km/h | 10,72° | 1,65° |
| Richtungsänderung bei 100 ms Tastendruck, 79,2 km/h | 7,52° | 1,12° |

Messung mit `python -m scripts.measure_arcade_handling`. An 640 Querschnitten tastet das Skript mit dem tangential ausgerichteten Kart die Kollisionsmaske in 2-cm-Schritten ab, einschließlich 10 cm Randreserve. Schräg stehende Karts benötigen mehr Platz. Der Lenktest verwendet Club/Standard und misst einschließlich der Rückstellung. Die Tastatur baut den normierten Einschlag mit 3,0 pro Sekunde auf und mit 6,4 pro Sekunde ab. Kleine Ausschläge werden zusätzlich abgeflacht, die Drehgeschwindigkeit sinkt bei zunehmendem Tempo. Kart- und Reifeneinflüsse bleiben aktiv.

Historischer Solo-Vergleich auf der kürzeren Strecke in Simulationssekunden, Roggwil, Club-Kart, Standardreifen, stehender Start bei s = 0. Die erste Runde enthält den Start. **Fliegend** ist das Mittel der beiden folgenden vollen Runden. Die Messung ist deterministisch und beschreibt einen Fahrer ohne Verkehr.

| Fahrer | Runde 1 | Runde 2 | Runde 3 | Fliegend |
|---|---:|---:|---:|---:|
| Regelbot aus 60aad25, v0 | 60,10 | 59,60 | 59,60 | 59,60 |
| Club, v1 | 47,27 | 46,67 | 46,63 | 46,65 |
| Sport, v1 | 42,03 | 41,27 | 41,40 | 41,33 |
| Pro, v1 | 40,30 | 39,67 | 39,90 | 39,78 |

Mit `python -m scripts.benchmark_arcade` misst man das aktuelle v2-Regelwerk. Früher betrug das Zieltempo maximal 12 m/s und es gab keine Boost- oder Driftanforderung. Der neue Fahrer liest vier vorausliegende Krümmungen, plant seine Bremsung, zielt zum Scheitelpunkt und hält beim Überholen eine seitlich versetzte Linie. Der Host wählt die Stufe beim Hinzufügen. Sport und Pro nutzen den regulären Boostvorrat und die gemeinsame Driftmechanik. Kart, Reifen, Beschleunigung und Geschwindigkeitsbegrenzer gelten für alle gleich. Bei einem explizit geladenen PPO-Modell steuert dessen Policy den Bot.

Eine persönlich gefahrene Vergleichszeit liegt noch nicht vor. Nach einem Rennen zeigt die Auswertung die schnellste fliegende Runde des eigenen Fahrers ohne Rücksetzen auf dieser Runde. Über **Als persönliche Bestzeit merken** speicherst du sie bewusst in diesem Browser. In der Lobby und im Ergebnis siehst du den Rekord für denselben Namen, Kart, Reifen, Ursprung und dieselbe Regelversion. Eine langsamere Zeit ersetzt keine schnellere. Private Fenster oder gelöschte Browserdaten können den lokalen Rekord verlieren.

### Rennwertung und Verbindungsabbruch

Der Server simuliert feste Schritte von 1/30 Sekunde. Der Browser übermittelt Tasten und Lobbywünsche. Das Rennen wertet Nettodistanz, abgeschlossene Runden und Zielzeit inklusive Strafen. Bandenkontakt kostet Geschwindigkeit. Der Server prüft die über den Tick bewegten, gedrehten Rechtecke auf trennenden Achsen. Ein Stoß überträgt Geschwindigkeit in Normalenrichtung. Positionskorrekturen werden auf Bandenspielraum geprüft. Bei einem unauflösbaren Gedränge fällt die betroffene Gruppe auf ihren letzten sicheren Zustand zurück. Erst danach werden Fortschritt, Runden und Zielzeit gewertet. Gewertete Fahrer verlassen für andere das aktive Feld. Nach zehn simulierten Minuten werden offene Ergebnisse als DNF gewertet. Die Endrangliste und `runs/arcade/races.csv` berücksichtigen auch ausgestiegene Teilnehmer, solange der Raum weiterbesteht.

Beim Schliessen eines Fensters wird der Platz nach Erkennung des Abbruchs frei. Der nächste menschliche Spieler übernimmt die Hostrolle. Während des Rennens warten neu verbundene Spieler bis zum nächsten Start. Ein Neuladen erzeugt einen neuen Teilnehmer. Sobald der letzte Mensch geht, verfällt der Raumlink. Laufende Räume liegen im Arbeitsspeicher.

## Gemeinsame Simulation und Training

Die Roggwiler Strecke, ihre Mittellinie, LiDAR, die Kartkontur und Bandenkollisionen kommen aus KartLapEnv. Der Three.js-Viewer zeigt autoritative Zustände und verfolgt das eigene Kart. Die klassischen Fahrversuche verwenden weiter ihr eigenes Fahrmodell.

- `kart_rl/game/driver.py` enthält die gemeinsame Tastatur- und Fahrzeuganbindung.
- `kart_rl/game/race.py` enthält die browserlose Arcade-Simulation, Auswahl, Rennregeln und die regelbasierten Fahrer.
- `kart_rl/game/arcade.py` verwaltet Verbindungen, Raumlinks und Verteilung der Zustände.
- `kart_rl/envs/arcade_env.py` nutzt denselben Fahrerschritt in einer Solo-Gymnasium-Umgebung. Die 38 Beobachtungswerte bestehen aus 30 Strecken-/Fahrzustandswerten sowie Boostvorrat, Driftladung, Mini-Boost-Restzeit und fünf Setupwerten. Die vier Aktionen in [-1,1] sind Lenkung, Gas/Bremse, Drift und Boost. Positive Werte aktivieren die letzten beiden. Reward pro Schritt ist der vorzeichenbehaftete Fortschritt in Metern minus 0,2 × Schrittzeit und minus 1 bei Bandenkontakt. Das Ziel gibt zusätzlich 10 Punkte. Zielankunft beendet die Episode, Zeitablauf kürzt sie ab. Beim Reset werden Kart und Reifen variiert.

Die bisherigen Arcade-PPO-Archive können weiterhin mit SB3 und Gymnasium `KartArcade-v0` bzw. `KartArcade-v1` in ihrer Originalgeometrie ausgewertet werden. Für neue Rennen erwartet der Lader die Kennung `KartArcade-v2`. Beobachtungs- und Aktionsdimensionen sowie die Reward-Formel sind gleich geblieben. Lenkung, Rundenerkennung und Strecke ändern die Dynamik. Alte Arcade-Policies für v2 neu trainieren und evaluieren. Das Solotraining enthält keine Gegnerbeobachtungen. Die Regelbots brauchen kein Training.

Jerys Fahr-PPO und der optimierte Lap-Checkpoint verwenden 30 Beobachtungswerte und zwei Aktionen. Der Arcade-Server verwendet standardmässig den regelbasierten Fahrer. Ein explizit gewähltes Modell wird erst nach Prüfung der gespeicherten Raumdimensionen und der Arcade-Kennung geladen. Nur eigene, vertrauenswürdige lokale Modellarchive angeben.

```powershell
.\.venv\Scripts\python.exe -m train.train_arcade --steps 250000 --output runs/arcade_v2/policy.zip
.\.venv\Scripts\python.exe -m kart_rl.game --arcade --arcade-policy runs/arcade_v2/policy.zip --http 8031 --ws 8796 --open
```

Den vorherigen Server vor einem Neustart auf denselben Ports beenden. Das Training übt Solorunden. Vor einem Einsatz als Renngegner Zielquote und Rundenzeiten über alle Kombinationen prüfen. Mit `.\.venv\Scripts\python.exe -m pytest -q` laufen auch die Raum-, Rennregel- und Gymnasium-Tests.

## Mit anderen Geräten und über das Internet

**Im WLAN** den Server mit zusätzlichem `--lan` starten und bei Bedarf beide TCP-Ports in der Firewall für das private Netzwerk erlauben. Auf dem Host die Startadresse mit seiner LAN-IP statt localhost öffnen, beispielsweise `http://192.168.1.20:8031/viewer/?game&arcade&ws=8796`. Der kopierte Raumlink verwendet denselben Hostnamen. Mitspielende benötigen einen Browser mit WebGL und eine Tastatur.

### Öffentlicher Dauerbetrieb mit Docker Compose

Für einen Raum mit fünf Fahrern ist eine kleine Linux-VM mit öffentlicher IP, 1 vCPU und 2 GiB RAM ein sinnvoller Ausgangspunkt. Beim lokalen Fünfer-Rennen unter Docker belegte der Spielprozess in einer Stichprobe 289 MiB und 41 % eines CPU-Kerns, Caddy 15 MiB und 4 %. Das ist eine Momentaufnahme und kein Lasttest mehrerer Räume. Als Preisbeispiel kostet ein regulärer DigitalOcean Basic Droplet mit diesen Eckdaten 12 USD pro Monat, zuzüglich allfälliger Steuern und optionaler Backups. Eine bestehende Subdomain kann verwendet werden. Für eine neue .ch-Domain sind rund 10 CHF jährlich ein Richtwert. Tarif und Verlängerungspreis vor einer Bestellung prüfen. [Serverpreis](https://www.digitalocean.com/pricing/droplets) · [Domainpreise](https://www.infomaniak.com/fr/domaines/tarifs/ch)

1. Linux-Server bereitstellen, Docker Engine mit Compose v2 installieren und den Docker-Dienst beim Booten aktivieren. SSH-Zugang mit Schlüssel einrichten. DNS-A-Eintrag der gewünschten Domain auf die öffentliche IPv4-Adresse setzen. AAAA nur setzen, wenn die VM auch über diese IPv6-Adresse erreichbar ist.
2. Eingehend TCP 80/443 zu Caddy öffnen, SSH auf den eigenen Verwaltungszugang beschränken. Die beiden Spielports werden nur im internen Compose-Netz verwendet. Die VM braucht ausgehenden Zugriff für Images, Python-Abhängigkeiten und Zertifikate.
3. Genau diesen lokalen Branch auf die VM übertragen. Dazu im Repository auf dem Entwicklungsrechner nach dem Commit `git archive --format=tar.gz -o kart-release.tar.gz HEAD` ausführen und das Archiv per SCP auf die VM kopieren. Auf der VM in ein eigenes Verzeichnis entpacken. Alternativ einen bereits freigegebenen Remote-Branch auschecken.
4. Im entpackten Repository die Domain einsetzen und die zwei Dienste starten:

```sh
printf 'KART_DOMAIN=%s\n' 'DEINE-SUBDOMAIN' > deploy/.env
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --build
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
docker compose --env-file deploy/.env -f deploy/compose.yaml logs --tail 60 web game
```

5. Die Domain über HTTPS öffnen. Der Einstieg leitet zum Fahrerprofil weiter. Caddy beschafft und erneuert das Zertifikat. Aus derselben öffentlichen Browseradresse wird der WSS-Endpunkt `/socket` gebildet. Nach dem Beitritt kopiert der Host den privaten Raumlink. Freunde brauchen Browser, WebGL und Tastatur. [Automatisches HTTPS](https://caddyserver.com/docs/automatic-https) · [WebSocket-Proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy)
6. Zur Abnahme den Link von einem zweiten Internetanschluss öffnen, mit bis zu fünf Personen bereitmelden, drei Runden und eine Revanche fahren. Der automatisierte Test kann zusätzlich mit `KART_URL` auf die echte HTTPS-Startadresse gesetzt werden. Bei öffentlichen Domains prüft Chromium das Zertifikat regulär.

`restart: unless-stopped` startet die Dienste nach Prozess- und Hostneustarts wieder. Ergebnis-CSV und Caddy-Zertifikate liegen in benannten Volumes. Laufende Räume leben im Arbeitsspeicher und verfallen beim Serverneustart. Für ein Update den Branch erneut übertragen und denselben Compose-Startbefehl ausführen.

Der Stack wurde lokal mit Caddys eigenem Zertifikat auf Loopback getestet. Zwei Browser fuhren über den kopierten HTTPS-Link mit drei Bots über drei Runden und starteten eine Revanche. Zusätzlich traten fünf WSS-Clients mit geprüfter CA und geprüftem Hostnamen bei. Alle erhielten dieselbe Startaufstellung, der sechste Client wurde abgewiesen. Eine Internet-Domain und ein öffentlicher Server sind noch nicht eingerichtet.
