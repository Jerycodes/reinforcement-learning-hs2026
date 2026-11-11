# Arcade Kart: agiler fahren und gemeinsam spielen

Der zusätzliche Arcade-Modus verwendet direkte Lenkung, kräftigere Beschleunigung
und ein einfacheres Fahrmodell. Er ist getrennt von der RL-Simulation und ihren
Bestzeiten. Jerys Modell und die bestehenden Trainingsumgebungen bleiben unverändert.

## Spielen

```powershell
python -m kart_rl.game --arcade --lan --http 8021 --ws 8786 --open
```

Auf dem Host öffnet sich:

```text
http://localhost:8021/viewer/?game&arcade&ws=8786
```

Jeder Browser beziehungsweise jeder PC bekommt ein eigenes Kart. Bis zu acht
Spieler können derselben Lobby beitreten. Der erste verbundene Spieler ist Host
und startet das gemeinsame Rennen. Verlässt er die Lobby, übernimmt der nächste
Spieler. Wer während des Rennens beitritt, wartet bis zum nächsten Start.
Alle können auch allein ein Zeitrennen starten.

Mit **Einladungslink kopieren** erhält der Host einen Link mit seiner lokalen
Netzwerkadresse. Mitspieler im selben Netzwerk öffnen diesen in ihrem Browser;
sie müssen Python oder das Repository nicht installieren. `--lan` bindet den
Server ans lokale Netzwerk. Ohne dieses Flag ist er nur auf dem Host erreichbar.
Das ist kein öffentlich gehosteter Internet-Multiplayer. Zwei getrennte
Browser-Clients wurden automatisch getestet; ein zweiter physischer PC wurde
in dieser Sitzung nicht getestet.

## Steuerung

| Taste | Aktion |
|---|---|
| Pfeiltasten oder WASD | Lenken, Gas und Bremsen |
| Shift beim Lenken | Driften |
| Shift nach mindestens 0.7 s Drift loslassen | Mini-Boost |
| Leertaste | Boost aus dem wiederaufladbaren Vorrat |
| R oder Backspace | Auf die Strecke zurücksetzen, +3 s Rennzeit |
| 1–4 | Kamera wechseln |

Ein Klick in die 3D-Ansicht löst den Fokus vom Namensfeld. Jeder Browser folgt
mit seiner Kamera dem eigenen Kart. Boost-Flamme, Vorrat und Drift-Fortschritt
werden live angezeigt. Beim Fokusverlust werden die Tasten freigegeben;
ausbleibende Eingaben werden zusätzlich auf dem Server nach einer Sekunde gelöst.

## Rennregeln

- Standard: drei Runden, gemeinsamer Countdown und Live-Rangliste.
- Der Server simuliert alle Karts mit 30 Steuerungsschritten pro Sekunde;
  Browser senden nur Tasten, keine vertrauenswürdigen Positionen oder Runden.
- Runden zählen nach tatsächlich zurückgelegter Nettodistanz. Wiederholtes
  Vor-/Zurückfahren über die Startlinie bringt keine zusätzlichen Runden.
- Banden verlangsamen das Kart. Zurücksetzen gibt keinen Distanzvorteil und
  kostet drei Sekunden. Der Rennabschluss enthält diese Strafe.
- Nach zehn Minuten endet ein noch nicht abgeschlossenes Rennen. Der Host
  kann jederzeit ein neues Rennen starten.
- Ergebnisse werden separat in `runs/arcade/races.csv` mit dem Profil `arcade`
  gespeichert. Sie sind nicht mit RL-Rundenzeiten vergleichbar.

Die erste Version fährt ohne gegenseitige Kart-Kollisionen, Waffen oder Items.
Es gibt weder Split-Screen noch Gamepad-Steuerung; mehrere Spieler verwenden
jeweils einen Browser mit Tastatur. Der bisherige Modus Mensch gegen trainierte
KI bleibt ohne `--arcade` verfügbar und nutzt weiterhin die bisherige Physik.

## Umsetzung und Prüfung

`kart_rl/game/arcade.py` enthält das Arcade-Fahrmodell, pro Verbindung ein eigenes
Kart und Eingabegerät, Lobby, Countdown, Boost, Drift und Rennwertung.
Die Freigabe im Netzwerk liefert nur Viewer- und Streckenassets aus, keine
Repository-Konfigurationen oder Modellarchive.

```powershell
pytest -q
```

Die neuen Tests prüfen Fahrverhalten, getrennte Eingaben, Hostwechsel,
späten Einstieg, Drift/Boost, JSON-Übertragung, Rücksetzstrafe, Renndistanz und
WebSocket-Verbindungen. Zusätzlich wurden zwei echte Chrome-Kontexte mit
eigenen Karts, eigener Kamera, Boost und Hostberechtigungen geprüft.
