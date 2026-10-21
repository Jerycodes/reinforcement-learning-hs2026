# Reinforcement Learning verstehen – am Kart-Projekt erklärt

Diese Seite erklärt, **was** der Agent lernt, **wie** er lernt und **wo** das im Code steht.
Sie ist für alle im Team gedacht, auch ohne RL-Vorwissen. Begriffe aus der Vorlesung (MDP,
Policy, Reward, Return, Value, Policy Gradient) werden direkt auf unser Projekt abgebildet.

## 1. Die Idee in einem Satz

Ein Kart fährt 20-mal pro Sekunde einen Schritt: es *sieht* die Strecke (Lidar, Geschwindigkeit,
Kurven voraus), *entscheidet* Lenkung und Gas, *bekommt* dafür Punkte (Meter Fortschritt) oder
Strafe (Abflug) – und verändert sein Verhalten so, dass es über die Zeit mehr Punkte bekommt.
Niemand sagt ihm, wie man Kart fährt. Er findet die Linie, das Bremsen und das Gasgeben selbst.

## 2. Der Markov-Entscheidungsprozess (MDP)

Ein MDP besteht aus Zustand, Aktion, Übergang, Reward und Diskontfaktor. Bei uns:

| Begriff | Allgemein | Bei uns | Im Code |
|---|---|---|---|
| Zustand s_t | was der Agent über die Welt weiss | 30 Zahlen: Lidar, Geschwindigkeiten, Lage zur Strecke, Krümmung voraus, letzte Aktion (Tabelle in Abschnitt 4) | `KartLapEnv._observe()` in `kart_rl/envs/lap_env.py` |
| Aktion a_t | Entscheidung des Agenten | 2 stetige Werte in [−1, 1]: Lenken (+ = links), Gas/Bremse (+ = Gas) | `action_space`, `KartLapEnv.step()` |
| Übergang P(s'∣s,a) | wie die Welt auf die Aktion reagiert | die Kart-Physik: 50 ms Fahrt mit dem Einspurmodell | `Kart.step()` in `kart_rl/kart.py` |
| Reward r_t | Punkte für den letzten Schritt | + Meter Fortschritt, − Zickzack, − 20 bei Abflug (Abschnitt 3) | `KartLapEnv.step()` |
| Diskontfaktor γ | wie stark zukünftige Rewards zählen | 0.99 → Horizont ≈ 100 Schritte = 5 s | `ppo.gamma` in `train/configs/phase1_ppo.yaml` |
| Episode | ein Versuch von Start bis Ende | Start an zufälliger Stelle (Training) oder stehend an der Ziellinie (Auswertung); Ende nach 2 Zieldurchfahrten, 150 s oder Abflug | `KartLapEnv.reset()`, `step()` |

**Markov-Eigenschaft:** Die nächste Beobachtung hängt nur vom jetzigen Zustand und der Aktion
ab, nicht von der Vorgeschichte. Der Agent sieht nicht den kompletten Physik-Zustand (z. B. nicht
seine absolute Position), streng genommen ist es ein *teilweise beobachtbarer* MDP (POMDP). Wir
geben ihm deshalb alles mit, was zum Fahren nötig ist (Abstände, Tempo, Kurvenvorschau, letzte
Aktion) und behandeln es wie einen MDP – so machen es praktisch alle Racing-RL-Projekte.

**Ziel:** Der Agent maximiert den erwarteten *Return*, also die diskontierte Summe der Rewards:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …
```

## 3. Der Reward im Detail

Pro Zeitschritt (50 ms) bekommt der Agent:

| Anteil | Wert | Warum |
|---|---|---|
| Fortschritt | + 1.0 pro Meter entlang der Strecke (`reward_progress`) | das eigentliche Ziel: vorwärts kommen; eine Runde = 640 Punkte |
| Zickzack | − 0.05 · (Änderung des Lenkbefehls)² (`reward_steer_change`) | ruhige Lenkung, realistischeres Fahren |
| Bandenkontakt | − 20 und Episode zu Ende (`reward_off_track`) | kein Teil des Karts darf die Bande berühren (ganze Kontur wird geprüft) |
| Stillstand > 5 s, > 10 m rückwärts | − 20 und Episode zu Ende | sonst könnte "stehen bleiben" eine sichere Strategie sein |
| Nähe zur Mittellinie | **nichts** | die Linie soll er selbst finden – die Mittellinie ist nur der Meterzähler |

**Warum fährt er schnell, wenn eine Runde immer 640 Punkte gibt?**
Zwei Gründe: Durch den Diskontfaktor γ zählt ein Meter *jetzt* mehr als ein Meter in fünf Sekunden
– die Punkte lohnen sich also früh. Und in eine Episode mit Zeitlimit passen mehr Meter, wenn er
schnell ist. Beides zusammen macht "so schnell wie möglich ohne Abflug" zur besten Strategie.

Der Fortschritt wird gemessen, indem die Kart-Position auf die Mittellinie projiziert wird
(`Track.localize()` in `kart_rl/track.py`): Innen eine Kurve schneiden bringt dieselben Meter mit
weniger Weg – darum lernt er die Ideallinie, ohne dass wir sie ihm zeigen.

## 4. Die Beobachtung (was der Agent sieht)

| Werte | Anzahl | Beschreibung | Normierung |
|---|---|---|---|
| Lidar | 15 | Abstand zum Streckenrand entlang 15 Strahlen über 200° | / 30 m |
| vx, vy, Gierrate | 3 | Längs-/Quergeschwindigkeit, Drehrate | / 20, / 5, / 3 |
| Richtungsfehler | 1 | Winkel zwischen Kart und Streckenrichtung | / 90° |
| seitliche Position | 1 | links/rechts der Mittellinie | / halbe Breite |
| Krümmung voraus | 6 | in 5, 10, 20, 35, 50, 70 m (+ = Linkskurve) | × 4 |
| Randabstand | 1 | Abstand der Kart-Kontur zur nächsten Bande | / 4 m |
| Lenkwinkel | 1 | tatsächliche Radstellung | / max |
| letzte Aktion | 2 | Lenken, Gas des Vorschritts | – |

Die Normierung bringt alles grob in den Bereich [−1, 1] – Netze lernen so deutlich besser.
Die Reihenfolge steht in `KartLapEnv.obs_names`.

## 5. Die Policy (das "Gehirn")

Die Policy π(a∣s) ist ein neuronales Netz, das aus dem Zustand die Aktion macht:

```
30 Eingänge → 256 Neuronen (tanh) → 256 Neuronen (tanh) → 2 Ausgänge (Mittelwert Lenken, Mittelwert Gas)
```

- Die Ausgabe ist keine feste Zahl, sondern eine **Gauss-Verteilung** pro Aktion: Mittelwert aus
  dem Netz, Streuung σ wird mitgelernt (`log_std_init: -0.5` = Startstreuung ≈ 0.6).
- **Training:** Aktionen werden aus der Verteilung gezogen → Exploration (ausprobieren).
- **Auswertung / Viewer:** der Mittelwert wird genommen → deterministisch, das "echte Können".
- Dazu ein zweites Netz, die **Value-Funktion V(s)** (Critic): schätzt, wie viel Return ab diesem
  Zustand noch zu erwarten ist. Zusammen: **Actor-Critic**.

Im Code: `ActorCritic` in `train/ppo_scratch.py` (lesbar, ~30 Zeilen) bzw. `MlpPolicy` von
Stable-Baselines3 in `train/train_sb3.py`.

## 6. Der Lernalgorithmus: PPO in fünf Schritten

PPO (Proximal Policy Optimization, Schulman et al. 2017) ist ein **modellfreies Policy-Gradient-
Verfahren**: Der Agent kennt die Physik nicht, er lernt nur aus erlebten Übergängen (s, a, r, s').

1. **Sammeln (Rollout):** 8 Umgebungen fahren parallel je 1024 Schritte mit der aktuellen Policy
   (8 × 1024 = 8192 Übergänge pro Update). Parallel nur fürs Tempo – es ist ein Agent.
2. **Bewerten (Advantage):** Für jeden Schritt: War die Aktion besser oder schlechter als erwartet?
   A(s,a) = erzielter Return − V(s), berechnet mit GAE (λ = 0.95), das Bias und Varianz ausgleicht.
   Hier steckt die **Bellman-Gleichung** aus der Vorlesung: V(s) ≈ r + γ·V(s').
3. **Policy verbessern:** Aktionen mit positivem Advantage werden wahrscheinlicher gemacht, mit
   negativem unwahrscheinlicher. PPO "clippt" die Änderung pro Update (ε = 0.2), damit die Policy
   nicht in einem Schritt zu weit springt – das macht das Training stabil.
4. **Value-Funktion verbessern:** V(s) wird an die tatsächlich erzielten Returns angepasst
   (Gewicht `vf_coef: 0.5`).
5. **Exploration halten:** ein kleiner Entropie-Bonus (`ent_coef: 0.001`) verhindert, dass die
   Streuung zu früh auf null geht.

Die Schritte 3–5 werden pro Rollout 10-mal (`n_epochs`) in Minibatches von 512 wiederholt, dann
geht es zurück zu Schritt 1. Lernrate 3·10⁻⁴, linear abnehmend bis zum Ende des Trainings.

Alles ausgeschrieben und kommentiert: `train/ppo_scratch.py` (`compute_gae()` = Schritt 2,
die Schleife in `main()` = Schritte 3–5). `train/train_sb3.py` macht dasselbe mit der Bibliothek
Stable-Baselines3 – beide liefern vergleichbare Ergebnisse und lassen sich gegeneinander prüfen.

**Warum nicht Q-Learning / Value Iteration wie in den ersten Vorlesungen?** Die klassischen
Verfahren brauchen eine Tabelle über alle Zustände und Aktionen. Unser Zustand hat 30 stetige
Werte, die Aktion 2 stetige Werte – unendlich viele Kombinationen. Deshalb Funktionsapproximation
mit neuronalen Netzen und Policy Gradient; die Bellman-Idee lebt in der Value-Funktion weiter.

## 7. Was die Zahlen beim Training bedeuten

| Anzeige | Bedeutung |
|---|---|
| Schritte (timesteps) | Anzahl Umgebungsschritte insgesamt (alle 8 Umgebungen zusammen); 3 Mio ≈ 42 Stunden Fahrzeit |
| Episoden | abgeschlossene Versuche (Abflug, Zeitlimit oder 2 Runden) |
| Ø Return (`ep_rew_mean`) | mittlere Punkte pro Episode; steigt, wenn er weiter kommt und schneller fährt (max. ≈ 1280 für 2 Runden) |
| Eval / beste Rundenzeit | alle 100k Schritte: stehender Start, 2 Runden, deterministisch; `best.zip` = bestes Modell nach Rundenzeit |
| Schritte/s | Trainingsgeschwindigkeit (abhängig von CPU-Kernen) |

Typischer Verlauf: zuerst nur Abflüge (Ø Return negativ) → erste Meter → erste ganze Runde (bei
uns nach ≈ 300–400k Schritten) → Rundenzeit sinkt von 65 s auf ≈ 56 s. Die Evaluation schwankt
von Checkpoint zu Checkpoint – das ist bei Policy-Gradient-Verfahren normal; deshalb wird das
beste Modell separat gespeichert.

Referenzen zum Einordnen: geskripteter Mittellinien-Fahrer 71.6 s, geskriptete Ideallinie 54.6 s,
theoretisches Optimum des Kart-Modells ≈ 49.6 s (`docs/ideallinie.md`).

## 8. Hyperparameter (train/configs/phase1_ppo.yaml) – was sie bewirken

| Parameter | Wert | Wirkung |
|---|---|---|
| `gamma` | 0.99 | Weitsicht: höher = längerer Horizont, aber langsameres Lernen |
| `learning_rate` | 3e-4, linear → 0 | Schrittweite der Updates; zu hoch = instabil, zu tief = langsam |
| `n_steps` | 1024 pro Env | Rollout-Länge; länger = stabilere Schätzungen, seltenere Updates |
| `batch_size`, `n_epochs` | 512, 10 | wie intensiv ein Rollout ausgewertet wird |
| `clip_range` | 0.2 | wie weit sich die Policy pro Update ändern darf |
| `ent_coef` | 0.001 | Exploration (mehr = mehr Zufall in den Aktionen) |
| `gae_lambda` | 0.95 | Bias/Varianz-Abwägung beim Advantage |
| `hidden_sizes` | [256, 256] | Netzgrösse |
| `n_envs` | 8 | parallele Umgebungen ≈ CPU-Kerne |

Alles lässt sich mit `--set key=value` überschreiben; `train/pbt.py` sucht Lernrate, Entropie,
Clip und Gamma automatisch (Population Based Training).

## 9. Lesepfad durch den Code

1. `kart_rl/envs/lap_env.py` – der MDP: `reset()`, `step()` (Reward, Episodenende), `_observe()`
2. `kart_rl/kart.py` – die Physik (`_substep()`: Reifenkräfte, Beschleunigung, Integration)
3. `kart_rl/track.py` – Strecke: Fortschritt (`localize()`), Rand (`distance_to_edge()`), Lidar (`lidar()`)
4. `train/ppo_scratch.py` – PPO von Hand: `ActorCritic`, `compute_gae()`, Update-Schleife
5. `train/train_sb3.py` – dasselbe mit Stable-Baselines3, plus Checkpoints, Eval, Live-Viewer
6. `scripts/evaluate.py` – Modell laden, fahren lassen, Rundenzeiten vergleichen

## 10. Glossar

- **Episode** – ein Versuch von Reset bis Ende. **Rollout** – die gesammelten Schritte zwischen zwei Updates.
- **Return** – diskontierte Summe der Rewards ab einem Zeitpunkt. **Value V(s)** – erwarteter Return ab Zustand s.
- **Advantage A(s,a)** – wie viel besser eine Aktion war als der Durchschnitt in diesem Zustand.
- **Policy** – Abbildung Zustand → Aktion(sverteilung). **Deterministisch** – Mittelwert statt Ziehen.
- **On-policy** – PPO lernt nur aus Daten der aktuellen Policy (im Gegensatz zu Q-Learning/SAC mit Replay-Buffer).
- **Exploration / Exploitation** – Neues ausprobieren vs. Bekanntes ausnutzen; bei uns gesteuert über die Streuung σ und `ent_coef`.
- **Checkpoint** – gespeicherte Netzgewichte zu einem Zeitpunkt; `best.zip` = bestes Modell, `last.zip` = Endstand.
- **Vektorisierte Umgebung** – mehrere Kopien der Umgebung parallel (nur Tempo, kein Multi-Agent).
- **Seed** – Startwert des Zufalls; gleicher Seed = reproduzierbarer Lauf.

## 11. Häufige Fragen

- **Sieht der Agent die berechnete Ideallinie?** Nein. Sie dient nur zum Vergleich und als Overlay im Viewer.
- **Warum Zufallsstart im Training?** Damit alle Streckenteile gleich oft geübt werden; sonst würde er nur den Start lernen.
- **Warum ist die Evaluation manchmal schlechter als vorher?** Policy-Gradient-Updates sind verrauscht; ein Update kann kurzzeitig verschlechtern. `best.zip` behält den besten Stand.
- **Warum 8 Karts im Hintergrund?** Nur zum schnelleren Sammeln von Erfahrung – es ist ein Agent, der aus allen 8 lernt.
- **Was ändert sich in Phase 2 und 3?** Der MDP wird erweitert: mehr Zustand (Reifenzustand, Rundenstand), mehr Aktionen (Boxenstopp, Reifenwahl, Boost), Reward = Gesamtrennzeit. Der Lernalgorithmus bleibt derselbe.
