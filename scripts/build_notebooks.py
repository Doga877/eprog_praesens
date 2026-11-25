"""Build synchronized solution and student notebooks from a single source."""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG_NB = ROOT / "bace_drug_discovery.ipynb"
IMG_DIR = ROOT / "images"
OUT_SOLUTION = ROOT / "bace_drug_discovery_loesung.ipynb"
OUT_STUDENT = ROOT / "bace_drug_discovery_studierende.ipynb"


def load_embedded_daten_cell() -> str:
    """Reuse Doga's exact embedded gzip+base64 CSV cell."""
    orig = json.loads(ORIG_NB.read_text(encoding="utf-8"))
    for cell in orig["cells"]:
        src = "".join(cell["source"])
        if 'DATEN = "' in src and "bace.csv" in src:
            return src
    raise RuntimeError("Could not find embedded DATEN cell in original notebook")


def img_attachment(filename: str) -> dict:
    raw = (IMG_DIR / filename).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return {filename: {"image/png": b64}}


def md(text: str, image: str | None = None) -> dict:
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n"),
    }
    if image:
        cell["attachments"] = img_attachment(image)
    return cell


def code(solution_src: str, student_src: str | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": solution_src.strip("\n"),
        "_student_source": student_src.strip("\n") if student_src is not None else None,
    }


def build_cells(daten_cell_src: str) -> list[dict]:
    return [
        # --- INTRO ---
        md("""
# Drug Discovery mit Machine Learning

**EIDP Woche 13: Scientific Python**

Wir sagen voraus, ob ein Molekül das Enzym **BACE-1** blockiert. Dieses Enzym
spielt in der Alzheimer-Forschung eine wichtige Rolle. Der Datensatz stammt aus
[MoleculeNet](https://moleculenet.org) und enthält 1513 Moleküle, bei denen aus
Laborexperimenten bekannt ist, ob sie wirken oder nicht.

Die fünf Merkmale pro Molekül stammen aus der medizinischen Chemie (bekannt aus
**Lipinskis Rule of Five**). Dasselbe Grundmuster gilt aber in jedem Fach: Eine
Tabelle mit Messwerten (`X`) und einer Zielgrösse (`y`) findet man genauso in
der Physik oder in der Wirtschaft.

**Was wir heute üben:** eigene Klassen (`Datensatz`, `Neuron`), NumPy-Arrays,
Slicing, Boolean Masking, `axis=0`, Broadcasting, Matplotlib und eine fremde
Bibliothek (`scikit-learn`).

---

## Teil 0: Vorbereitung

Zuerst laden wir die Bibliotheken. `numpy` rechnet mit Arrays, `matplotlib`
zeichnet Diagramme, `sklearn` liefert Hilfsmittel und das neuronale Netz.
"""),
        code("""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

print("Alles bereit. NumPy-Version:", np.__version__)
"""),
        md("""
Die Daten stecken direkt in diesem Notebook, damit ihr nichts herunterladen
müsst. Diese Zelle einfach ausführen: Sie schreibt die Datei `bace.csv` neben
das Notebook.
"""),
        code(daten_cell_src),

        # --- TEIL 1: DATENSATZ KLASSE (DOGA) ---
        md("""
---

## Teil 1: Die Klasse `Datensatz`

Unsere erste Klasse verwaltet die Moleküldaten. Sie liest die CSV-Datei mit
`np.genfromtxt` ein und speichert zwei Arrays:

- `X`: die fünf Eigenschaften, eine Zeile pro Molekül, eine Spalte pro Eigenschaft
- `y`: die Antwort (`1` = wirkt, `0` = wirkt nicht)

| Spalte | Bedeutung |
|---|---|
| `MW` | Molekülmasse (g/mol) |
| `AlogP` | Fettlöslichkeit |
| `HBA` | Wasserstoffbrücken-Akzeptoren |
| `HBD` | Wasserstoffbrücken-Donoren |
| `RB` | drehbare Bindungen |
| `Class` | **Zielgrösse**: 1 = wirkt, 0 = wirkt nicht |
"""),
        code("""
class Datensatz:
    \"\"\"Hält die Molekül-Eigenschaften (X) und die Antwort (y).\"\"\"

    def __init__(self, dateiname):
        tabelle = np.genfromtxt(dateiname, delimiter=",", names=True)
        self.namen = ["MW", "AlogP", "HBA", "HBD", "RB"]
        self.X = np.column_stack([tabelle[name] for name in self.namen])
        self.y = tabelle["Class"]

    def anzahl(self):
        return len(self.y)

    def anzahl_aktiv(self):
        return int(self.y.sum())

    def nur_aktive(self):
        return self.X[self.y == 1]

    def nur_inaktive(self):
        return self.X[self.y == 0]

    def plot(self, name="MW"):
        i = self.namen.index(name)
        plt.figure(figsize=(6.5, 4))
        plt.hist(self.nur_aktive()[:, i], bins=30, alpha=0.6, label="aktiv")
        plt.hist(self.nur_inaktive()[:, i], bins=30, alpha=0.6, label="inaktiv")
        plt.xlabel(name)
        plt.ylabel("Anzahl Moleküle")
        plt.legend()
        plt.show()
"""),
        md("""
Jetzt erzeugen wir ein Objekt `daten` dieser Klasse und schauen uns Form,
Klassenverteilung und Boolean Masking an.
"""),
        code("""
daten = Datensatz("bace.csv")

print("Moleküle insgesamt:", daten.anzahl())
print("Davon aktiv:       ", daten.anzahl_aktiv(), f"({daten.y.mean():.1%})")
print("Form von X:        ", daten.X.shape)
print()
print("Die ersten drei Moleküle:")
print(daten.X[:3])
"""),
        md("""
### Boolean Masking von innen

`daten.y == 1` erzeugt ein Array aus `True` und `False`. Setzen wir diese Maske
in die eckigen Klammern von `daten.X`, bekommen wir nur die passenden Zeilen.
Genau das steckt hinter `daten.nur_aktive()` und `daten.nur_inaktive()`.
"""),
        code("""
maske = (daten.y == 1)
print("Die ersten zehn Einträge der Maske:", maske[:10])
print("Form von nur_aktive():  ", daten.nur_aktive().shape)
print("Form von nur_inaktive():", daten.nur_inaktive().shape)
"""),
        md("""
### Aufgabe 1: Gruppenmittelwerte mit `axis=0`

Unterscheiden sich aktive und inaktive Moleküle in ihren fünf Eigenschaften?
Mit `axis=0` rechnet NumPy spaltenweise, liefert also einen Mittelwert pro
Eigenschaft.

Berechnet `mittel_aktiv` und `mittel_inaktiv` mit `.mean(axis=0)` auf
`daten.nur_aktive()` und `daten.nur_inaktive()`.
"""),
        code(
            solution_src="""
mittel_aktiv = daten.nur_aktive().mean(axis=0)
mittel_inaktiv = daten.nur_inaktive().mean(axis=0)

print(f"{'Eigenschaft':<12}{'aktiv':>10}{'inaktiv':>10}")
print("-" * 32)
for i, name in enumerate(daten.namen):
    print(f"{name:<12}{mittel_aktiv[i]:>10.2f}{mittel_inaktiv[i]:>10.2f}")
""",
            student_src="""
# TODO: Spaltenweise Mittelwerte mit .mean(axis=0) berechnen
mittel_aktiv = ...
mittel_inaktiv = ...

print(f"{'Eigenschaft':<12}{'aktiv':>10}{'inaktiv':>10}")
print("-" * 32)
for i, name in enumerate(daten.namen):
    print(f"{name:<12}{mittel_aktiv[i]:>10.2f}{mittel_inaktiv[i]:>10.2f}")
""",
        ),
        md("""
Aktive Moleküle sind im Schnitt schwerer und haben mehr Akzeptoren, Donoren und
drehbare Bindungen. Mit `daten.plot("MW")` prüfen wir, ob eine einzelne
Eigenschaft zur Trennung ausreicht.
"""),
        code("""
daten.plot("MW")
"""),
        md("""
### Aufgabe 2: Streudiagramm für `MW` und `AlogP`

Das Histogramm überlappt stark. Vielleicht trennen zwei Eigenschaften zusammen
die beiden Gruppen besser?

Zeichnet ein Streudiagramm (`plt.scatter`) mit `MW` (Spalte 0) auf der x-Achse
und `AlogP` (Spalte 1) auf der y-Achse für `inaktive` und `aktive`.
"""),
        code(
            solution_src="""
aktive = daten.nur_aktive()
inaktive = daten.nur_inaktive()

plt.figure(figsize=(7, 4.5))
plt.scatter(inaktive[:, 0], inaktive[:, 1], alpha=0.4, s=18, label="inaktiv")
plt.scatter(aktive[:, 0], aktive[:, 1], alpha=0.4, s=18, label="aktiv")
plt.xlabel("MW")
plt.ylabel("AlogP")
plt.title("Molekülmasse gegen Fettlöslichkeit")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
""",
            student_src="""
aktive = daten.nur_aktive()
inaktive = daten.nur_inaktive()

plt.figure(figsize=(7, 4.5))
# TODO: Zeichnet inaktive und aktive mit plt.scatter (Spalte 0 = MW, Spalte 1 = AlogP)
# Tipp: alpha=0.4 und s=18 machen überlappende Punkte besser lesbar.
...
...

plt.xlabel("MW")
plt.ylabel("AlogP")
plt.title("Molekülmasse gegen Fettlöslichkeit")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
""",
        ),
        md("""
Auch mit zwei Eigenschaften lässt sich keine saubere Trennlinie ziehen. Deshalb
brauchen wir ein Modell, das alle fünf Merkmale gleichzeitig kombiniert.

---

## Teil 2: Aufteilen in Training, Validierung und Test

Wir prüfen ein Modell nie mit den Daten, aus denen es gelernt hat. Warum können
wir das Array nicht einfach mit Slicing zerschneiden? Unsere CSV-Datei ist nach
`Class` sortiert: zuerst alle aktiven, danach alle inaktiven Moleküle. Ein
naiver Schnitt produziert völlig unbrauchbare Teilmengen:
"""),
        code("""
y_naiv_train = daten.y[:1200]
y_naiv_test = daten.y[1200:]

print(f"Naiver Schnitt Training: {y_naiv_train.mean():.1%} aktiv")
print(f"Naiver Schnitt Test:     {y_naiv_test.mean():.1%} aktiv")
"""),
        md("""
`train_test_split` mischt die Daten zuerst zufällig (`shuffle=True` ist
Standard). Mit `stratify` bleibt zusätzlich der Anteil aktiver Moleküle in allen
Teilen gleich, und `random_state=42` sorgt dafür, dass alle dieselbe Aufteilung
sehen.

Wir teilen die Daten in drei Teile auf:
- **Training (60%):** Hier lernt das Modell seine Gewichte.
- **Validierung (20%):** Hier vergleichen wir verschiedene Einstellungen.
- **Test (20%):** Bleibt bis ganz am Schluss unberührt für das ehrliche Endergebnis.
"""),
        code("""
X_rest, X_test, y_rest, y_test = train_test_split(
    daten.X, daten.y, test_size=0.20, random_state=42, stratify=daten.y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=0.25, random_state=42, stratify=y_rest
)

print(f"Training:    {len(y_train):>4} Moleküle ({y_train.mean():.1%} aktiv)")
print(f"Validierung: {len(y_val):>4} Moleküle ({y_val.mean():.1%} aktiv)")
print(f"Test:        {len(y_test):>4} Moleküle ({y_test.mean():.1%} aktiv)")
"""),

        # --- TEIL 3: STANDARDISIEREN ---
        md("""
---

## Teil 3: Standardisieren mit Broadcasting und `StandardScaler`

Die Molekülmasse liegt um 500, die Anzahl Donoren um 2. Neuronale Netze
reagieren empfindlich auf so unterschiedliche Grössenordnungen. Deshalb
standardisieren wir jede Spalte auf Mittelwert 0 und Standardabweichung 1:

$$x_{\\text{neu}} = \\frac{x - \\text{Mittelwert}}{\\text{Standardabweichung}}$$

In NumPy geht das in zwei Zeilen dank **Broadcasting**: `mittel` und `streuung`
haben die Form `(5,)`, `X_train` hat die Form `(907, 5)`. NumPy wendet die fünf
Werte automatisch auf jede Zeile an.

**Wichtig:** Mittelwert und Streuung werden nur aus `X_train` berechnet.
Validierungs- und Testdaten werden mit genau denselben Werten umgerechnet,
damit keine Information aus den Prüfdaten ins Training sickert.
"""),
        code("""
mittel = X_train.mean(axis=0)
streuung = X_train.std(axis=0)

X_train_manuell = (X_train - mittel) / streuung
X_val_manuell = (X_val - mittel) / streuung

print("Form von X_train:", X_train.shape, " | Form von mittel:", mittel.shape)
print("Mittelwerte nachher (Training):", X_train_manuell.mean(axis=0).round(3))
print("Streuungen nachher (Training): ", X_train_manuell.std(axis=0).round(3))
"""),
        md("""
### Aufgabe 3: `StandardScaler` aus `scikit-learn`

`StandardScaler` ist eine fertige Klasse aus `scikit-learn`. Das Objekt
`skalierer` merkt sich Mittelwert und Streuung beim Aufruf von
`.fit_transform(X_train)` und wendet genau dieselben Zahlen mit `.transform(...)`
auf Validierung und Test an.

Vervollständigt die drei Zeilen unten.
"""),
        code(
            solution_src="""
skalierer = StandardScaler()

X_train_skaliert = skalierer.fit_transform(X_train)
X_val_skaliert = skalierer.transform(X_val)
X_test_skaliert = skalierer.transform(X_test)

print("Gelernte Mittelwerte im Objekt:", skalierer.mean_.round(2))
print("Erste Zeile vorher: ", X_train[0].round(2))
print("Erste Zeile nachher:", X_train_skaliert[0].round(2))
""",
            student_src="""
skalierer = StandardScaler()

# TODO: Auf X_train lernen und anwenden (fit_transform), auf X_val und X_test nur anwenden (transform)
X_train_skaliert = ...
X_val_skaliert = ...
X_test_skaliert = ...

print("Gelernte Mittelwerte im Objekt:", skalierer.mean_.round(2))
print("Erste Zeile vorher: ", X_train[0].round(2))
print("Erste Zeile nachher:", X_train_skaliert[0].round(2))
""",
        ),

        # --- PAUSE & CHECKPOINT ---
        md("""
---

## Pause und Checkpoint

Falls vor der Pause eine Zelle nicht geklappt hat, führt einfach die nächste
Zelle aus. Sie stellt exakt dieselben Variablen her, damit alle sauber in den
zweiten Teil starten können.
"""),
        code("""
# Checkpoint: Daten laden, aufteilen und standardisieren
daten = Datensatz("bace.csv")
namen = daten.namen

_X_rest, X_test, _y_rest, y_test = train_test_split(
    daten.X, daten.y, test_size=0.20, random_state=42, stratify=daten.y
)
X_train, X_val, y_train, y_val = train_test_split(
    _X_rest, _y_rest, test_size=0.25, random_state=42, stratify=_y_rest
)

skalierer = StandardScaler()
X_train_skaliert = skalierer.fit_transform(X_train)
X_val_skaliert = skalierer.transform(X_val)
X_test_skaliert = skalierer.transform(X_test)

print("Checkpoint bereit:", X_train_skaliert.shape, X_val_skaliert.shape, X_test_skaliert.shape)
"""),

        # --- TEIL 4: NN INTUITION + GLASS-BOX NEURON CLASS (JAN) ---
        md("""
---

## Teil 4: Wie funktioniert ein neuronales Netz?

*Hinweis: Dieser Abschnitt erklärt die Grundidee hinter neuronalen Netzen. Die
mathematischen Details kommen nicht an der Prüfung.*

### 4.1 Ein einzelnes Neuron

Ein künstliches Neuron bekommt unsere fünf standardisierten Molekülwerte
$x_1, \\dots, x_5$. Jede Eingabe wird mit einem **Gewicht** $w_i$ multipliziert.
Dazu kommt ein **Bias** $b$ (ein Grundwert):

$$z = w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 + w_5 x_5 + b$$

Danach läuft $z$ durch eine **Aktivierungsfunktion**. Für ein einzelnes Neuron,
das direkt eine Wahrscheinlichkeit zwischen 0 und 1 ausgeben soll, nehmen wir
**Sigmoid**. In den verborgenen Schichten eines grösseren Netzes nimmt man
meistens **ReLU**.

![Ein einzelnes Neuron](attachment:01_einzelnes_neuron.png)
""", image="01_einzelnes_neuron.png"),

        md("""
![Aktivierungsfunktionen ReLU und Sigmoid](attachment:03_aktivierungsfunktionen.png)
""", image="03_aktivierungsfunktionen.png"),

        md("""
### 4.2 Ein Neuron selbst bauen (`class Neuron`)

Damit das Training später nicht wie eine schwarze Box wirkt, bauen wir zuerst
ein einzelnes Neuron als kleine Klasse mit nur 14 Zeilen.

Ein Objekt der Klasse `Neuron` speichert seine eigenen Gewichte `self.w` und
seinen Bias `self.b`. Es kann zwei Dinge:
- `vorhersage(x)`: berechnet `z = np.sum(self.w * x) + self.b` und drückt `z` mit Sigmoid zwischen 0 und 1.
- `lernschritt(x, y, lernrate)`: vergleicht Vorhersage `p` mit der echten Antwort `y` (`fehler = p - y`) und korrigiert `self.w` und `self.b` ein kleines Stück in Richtung kleinerer Fehler.

### Aufgabe 4: `vorhersage` und `lernschritt` vervollständigen

Ergänzt die vier fehlenden Zeilen in der Klasse `Neuron`.
"""),
        code(
            solution_src="""
class Neuron:
    \"\"\"Ein einzelnes künstliches Neuron mit eigenen Gewichten und Bias.\"\"\"

    def __init__(self, n_eingaben):
        self.w = np.array([0.4, -0.3, 0.2, 0.1, 0.5])[:n_eingaben]
        self.b = 0.0

    def vorhersage(self, x):
        z = np.sum(self.w * x) + self.b
        return 1 / (1 + np.exp(-z))

    def lernschritt(self, x, y, lernrate=0.3):
        p = self.vorhersage(x)
        fehler = p - y
        self.w = self.w - lernrate * fehler * x
        self.b = self.b - lernrate * fehler
""",
            student_src="""
class Neuron:
    \"\"\"Ein einzelnes künstliches Neuron mit eigenen Gewichten und Bias.\"\"\"

    def __init__(self, n_eingaben):
        self.w = np.array([0.4, -0.3, 0.2, 0.1, 0.5])[:n_eingaben]
        self.b = 0.0

    def vorhersage(self, x):
        # TODO: Gewichtete Summe z = np.sum(self.w * x) + self.b und Sigmoid 1 / (1 + np.exp(-z))
        z = ...
        return ...

    def lernschritt(self, x, y, lernrate=0.3):
        p = self.vorhersage(x)
        fehler = p - y
        # TODO: Gewichte self.w und Bias self.b mit lernrate * fehler anpassen
        self.w = ...
        self.b = ...
""",
        ),
        md("""
Jetzt schauen wir unserem Neuron beim Lernen zu. Wir geben ihm ein echtes
aktives Trainingsmolekül (`y = 1`) und lassen es fünfmal `lernschritt()`
ausführen. Die Verlustfunktion (binäre Kreuzentropie) zeigt, wie falsch die
Vorhersage noch liegt.
"""),
        code("""
# Ein echtes aktives Molekül aus den Trainingsdaten wählen
idx = np.where(y_train == 1)[0][0]
x_bsp = X_train_skaliert[idx]
y_bsp = y_train[idx]

neuron = Neuron(n_eingaben=5)

print(f"Echte Klasse dieses Moleküls: y = {int(y_bsp)}\\n")
for schritt in range(6):
    p = neuron.vorhersage(x_bsp)
    kosten = -(y_bsp * np.log(p) + (1 - y_bsp) * np.log(1 - p))
    print(f"Schritt {schritt}: Vorhersage = {p:>5.1%} aktiv  |  Verlust = {kosten:.3f}")
    neuron.lernschritt(x_bsp, y_bsp, lernrate=0.3)
"""),
        md("""
Nach fünf Schritten steigt die Vorhersage für dieses Molekül deutlich an und
der Verlust fällt. Genau dieses schrittweise Bergabgehen heisst
**Gradientenabstieg**.

![Gradientenabstieg](attachment:04_gradientenabstieg.png)
""", image="04_gradientenabstieg.png"),

        md("""
### 4.3 Vom einzelnen Neuron zum neuronalen Netz

Ein einzelnes Neuron hat nur 5 Gewichte und 1 Bias. Für kompliziertere Muster
schalten wir mehrere Neuronen zu **Schichten** zusammen. Unser Netz aus
`scikit-learn` (`MLPClassifier(hidden_layer_sizes=(8,))`) macht im Kern genau
denselben Lernschritt wie unsere Klasse `Neuron`, nur mit 8 Neuronen in einer
verborgenen Schicht und automatisch über alle 907 Trainingsmoleküle.

![Architektur des neuronalen Netzes](attachment:02_netzarchitektur.png)
""", image="02_netzarchitektur.png"),

        # --- TEIL 5: MLP TRAINIEREN & BEWERTEN ---
        md("""
---

## Teil 5: Das ganze Netz mit `MLPClassifier` trainieren

Die wichtigsten Einstellungen von `MLPClassifier`:

| Einstellung | Bedeutung | Beispiele |
|---|---|---|
| `hidden_layer_sizes` | Schichten und Neuronen | `(8,)`, `(16, 8)`, `(64, 64, 32)` |
| `activation` | Aktivierungsfunktion in den verborgenen Schichten | `"relu"`, `"tanh"` |
| `learning_rate_init` | Schrittweite beim Lernen | `0.001` bis `0.1` |
| `alpha` | Strafe für zu grosse Gewichte (Regularisierung) | `0.0001` bis `0.5` |
| `max_iter` | maximale Anzahl Epochen | `1000` |

### Aufgabe 5: Modell erstellen und trainieren

Erstellt einen `MLPClassifier` mit `hidden_layer_sizes=(8,)`, `activation="relu"`,
`solver="adam"`, `learning_rate_init=0.01`, `max_iter=1000` und `random_state=42`.
Trainiert ihn mit `.fit(X_train_skaliert, y_train)`.
"""),
        code(
            solution_src="""
modell = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42,
)
modell.fit(X_train_skaliert, y_train)

print(f"Fertig nach {modell.n_iter_} Epochen")
print(f"Verlust am Anfang: {modell.loss_curve_[0]:.3f}")
print(f"Verlust am Ende:   {modell.loss_curve_[-1]:.3f}")
print("Form der gelernten Gewichte (Schicht 1):", modell.coefs_[0].shape)
""",
            student_src="""
# TODO: MLPClassifier mit hidden_layer_sizes=(8,), activation="relu", solver="adam",
# learning_rate_init=0.01, max_iter=1000, random_state=42 erstellen
modell = ...

# TODO: Modell auf X_train_skaliert und y_train trainieren
...

print(f"Fertig nach {modell.n_iter_} Epochen")
print(f"Verlust am Anfang: {modell.loss_curve_[0]:.3f}")
print(f"Verlust am Ende:   {modell.loss_curve_[-1]:.3f}")
print("Form der gelernten Gewichte (Schicht 1):", modell.coefs_[0].shape)
""",
        ),
        md("""
### Verlustkurve und Kreuzentropie

`modell.loss_curve_` speichert den Fehler nach jeder Epoche. Das Netz hört vor
den 1000 Epochen auf, sobald sich der Verlust kaum noch verbessert.
"""),
        code("""
plt.figure(figsize=(7, 4))
plt.plot(modell.loss_curve_)
plt.xlabel("Epoche")
plt.ylabel("Verlust")
plt.title("Trainingsverlauf (Verlustkurve)")
plt.grid(alpha=0.3)
plt.show()
"""),
        md("""
Die binäre Kreuzentropie bestraft falsche Vorhersagen umso stärker, je sicherer
sich das Modell war:
"""),
        code("""
def verlust(modell, X, y):
    p = modell.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


for p in [0.99, 0.9, 0.6, 0.5, 0.4, 0.1, 0.01]:
    print(f"Modell sagt {p:>5.0%} aktiv, Molekül ist aktiv  -> Kosten {-np.log(p):.2f}")
"""),
        md("""
### Genauigkeit, Basislinie und Konfusionsmatrix

Eine Genauigkeit allein sagt wenig aus. Wenn wir immer die häufigere Klasse
(`inaktiv`) raten, erreichen wir schon `54.5%`.
"""),
        code("""
def genauigkeit(modell, X, y):
    return (modell.predict(X) == y).mean()


print("                  Training   Validierung")
print(f"Genauigkeit:         {genauigkeit(modell, X_train_skaliert, y_train):.3f}      "
      f"{genauigkeit(modell, X_val_skaliert, y_val):.3f}")
print(f"Verlust:             {verlust(modell, X_train_skaliert, y_train):.3f}      "
      f"{verlust(modell, X_val_skaliert, y_val):.3f}")

raten = max(y_val.mean(), 1 - y_val.mean())
print(f"\\nImmer die häufigere Klasse raten: {raten:.3f}")
"""),
        md("""
Die Konfusionsmatrix zeigt auf den Validierungsdaten die vier Fälle. Ein
**Falsch-Positiv** kostet teure Laborzeit für ein unwirksames Molekül, ein
**Falsch-Negativ** übersieht einen möglichen Wirkstoff.
"""),
        code("""
vorhersage = modell.predict(X_val_skaliert)

print("                   vorhergesagt")
print("                 inaktiv   aktiv")
for echt, bezeichnung in [(0, "echt inaktiv"), (1, "echt aktiv  ")]:
    zeile = [np.sum((y_val == echt) & (vorhersage == p)) for p in (0, 1)]
    print(f"{bezeichnung}  {zeile[0]:>7}{zeile[1]:>8}")
"""),

        # --- TEIL 6: OVERFITTING, FINAL TEST, LIMITS ---
        md("""
---

## Teil 6: Überanpassung und Grenzen des Modells

### 6.1 Mehr Neuronen sind nicht automatisch besser

Wir vergleichen verschiedene Netzgrössen auf **Training** und **Validierung**.
Die Testdaten bleiben dabei weiterhin unberührt.
"""),
        code("""
def vergleiche(varianten):
    print(f"{'Neuronen':<14}{'Aktiv.':<8}{'alpha':<8}{'Epochen':>8}{'Training':>10}{'Valid.':>8}")
    print("-" * 56)

    for v in varianten:
        m = MLPClassifier(
            hidden_layer_sizes=v.get("neuronen", (8,)),
            activation=v.get("aktivierung", "relu"),
            solver="adam",
            learning_rate_init=v.get("lernrate", 0.01),
            alpha=v.get("alpha", 0.0001),
            max_iter=1500,
            random_state=42,
        )
        m.fit(X_train_skaliert, y_train)

        print(f"{str(v.get('neuronen', (8,))):<14}"
              f"{v.get('aktivierung', 'relu'):<8}"
              f"{v.get('alpha', 0.0001):<8.4f}"
              f"{m.n_iter_:>8}"
              f"{genauigkeit(m, X_train_skaliert, y_train):>10.3f}"
              f"{genauigkeit(m, X_val_skaliert, y_val):>8.3f}")


vergleiche([
    {"neuronen": (2,)},
    {"neuronen": (8,)},
    {"neuronen": (16, 8)},
    {"neuronen": (32, 16)},
    {"neuronen": (64, 64, 32), "aktivierung": "tanh"},
    {"neuronen": (64, 64, 32), "alpha": 0.5},
])
"""),
        md("""
Schaut euch die vorletzte Zeile (`(64, 64, 32)` mit `tanh`) an: `96.9%` auf den
Trainingsdaten, aber nur `70.6%` auf den Validierungsdaten. Das Netz hat die
Trainingsbeispiele auswendig gelernt, statt allgemeine Muster zu finden. Das
heisst **Überanpassung (Overfitting)**.

Die letzte Zeile zeigt ein Gegenmittel: Mit grösserem `alpha=0.5` bestraft das
Netz zu grosse Gewichte. Die Trainingsgenauigkeit sinkt, aber die
Validierungsgenauigkeit steigt wieder auf `73.9%`.

### 6.2 Ehrliche Endprüfung auf den Testdaten

Auf den Validierungsdaten hat `(16, 8)` mit `relu` am besten abgeschnitten
(`74.9%`). Jetzt prüfen wir dieses gewählte Modell genau einmal auf dem bisher
unberührten **Testdatensatz**:
"""),
        code("""
bestes_modell = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1500,
    random_state=42,
)
bestes_modell.fit(X_train_skaliert, y_train)

test_genauigkeit = genauigkeit(bestes_modell, X_test_skaliert, y_test)
test_raten = max(y_test.mean(), 1 - y_test.mean())

print(f"Test-Genauigkeit: {test_genauigkeit:.1%}  (Basislinie Raten: {test_raten:.1%})")
"""),
        md("""
### 6.3 Eigene Moleküle testen und Grenzen erkennen

Jetzt füttern wir bekannte Alltagsmoleküle und einen echten BACE-Inhibitor aus
dem Datensatz in unser Modell. Jedes neue Molekül muss mit demselben `skalierer`
umgerechnet werden wie die Trainingsdaten.
"""),
        code("""
def teste_molekuel(mw, alogp, hba, hbd, rb, name="Neues Molekül"):
    roh = np.array([[mw, alogp, hba, hbd, rb]])
    neu = skalierer.transform(roh)

    sicherheit = modell.predict_proba(neu)[0][1]
    antwort = "wirkt vermutlich" if sicherheit > 0.5 else "wirkt vermutlich nicht"

    unter = roh[0] < X_train.min(axis=0)
    ueber = roh[0] > X_train.max(axis=0)
    ausserhalb = [namen[i] for i in range(len(namen)) if unter[i] or ueber[i]]
    hinweis = f"  [ausserhalb Trainingsbereich: {', '.join(ausserhalb)}]" if ausserhalb else ""

    print(f"{name:<18} {antwort:<24} ({sicherheit:>3.0%}){hinweis}")


#               MW     AlogP  HBA HBD RB
teste_molekuel(180.2,   1.31,  3,  1,  2, name="Aspirin")
teste_molekuel(206.3,   3.07,  1,  1,  4, name="Ibuprofen")
teste_molekuel(657.8,   2.64,  5,  4, 16, name="BACE-Inhibitor")
teste_molekuel(194.2,  -1.03,  3,  0,  0, name="Koffein")
"""),
        md("""
**Schaut euch das Ergebnis kritisch an.** Warum bekommt **Koffein** eine hohe
Wahrscheinlichkeit (`82%`), obwohl es kein Wirkstoff gegen Alzheimer ist?

1. **Ausserhalb des Trainingsbereichs (Extrapolation):** Kein einziges Molekül
   in den Trainingsdaten hat `HBD = 0`, und auch `AlogP = -1.03` sowie
   `MW = 194.2` liegen ganz am unteren Rand. In Bereichen, die das Netz nie
   gesehen hat, rät es im Grunde.
2. **Zu grobe Merkmale:** Das Modell kennt von jedem Molekül nur fünf
   Summenzahlen (Lipinski-Eigenschaften), nicht seine dreidimensionale Struktur.
   Echte Modelle in der Wirkstoffforschung nutzen strukturelle Fingerabdrücke
   oder Molekülgraphen.

---

## Teil 7 (Optional): Top-10-Kandidaten fürs Labor

In der Praxis testet man im Labor nicht alle Moleküle, sondern nur die
vielversprechendsten Kandidaten. Statt nur bei `50%` zu schneiden, sortieren
wir die Testmoleküle nach ihrer vorhergesagten Wahrscheinlichkeit.

### Aufgabe 6 (Optional): Die 10 besten Kandidaten auswählen

1. Berechnet `wahrscheinlichkeiten = bestes_modell.predict_proba(X_test_skaliert)[:, 1]`.
2. Findet mit `np.argsort(wahrscheinlichkeiten)[-10:][::-1]` die Indizes der 10
   höchsten Werte.
3. Zählt, wie viele dieser 10 Moleküle in `y_test` tatsächlich aktiv (`1`) sind.
"""),
        code(
            solution_src="""
wahrscheinlichkeiten = bestes_modell.predict_proba(X_test_skaliert)[:, 1]
top10_idx = np.argsort(wahrscheinlichkeiten)[-10:][::-1]

treffer = int(y_test[top10_idx].sum())
print(f"Unter den Top-10-Kandidaten sind {treffer} von 10 wirklich aktiv ({treffer * 10}%).")
print()
for rang, idx in enumerate(top10_idx, start=1):
    status = "aktiv" if y_test[idx] == 1 else "inaktiv"
    print(f"Rang {rang:>2}: vorhergesagt {wahrscheinlichkeiten[idx]:>5.1%}  |  echt: {status}")
""",
            student_src="""
# TODO: Wahrscheinlichkeiten für Klasse 1 auf X_test_skaliert berechnen
wahrscheinlichkeiten = ...

# TODO: Indizes der 10 grössten Werte mit np.argsort finden
top10_idx = ...

treffer = int(y_test[top10_idx].sum())
print(f"Unter den Top-10-Kandidaten sind {treffer} von 10 wirklich aktiv ({treffer * 10}%).")
""",
        ),
        md("""
### Weitere Ideen zum Ausprobieren

- Ergänzt die Klasse `Datensatz` um eine Methode `groesstes_molekuel()`, die die höchste Molekülmasse (`MW`) zurückgibt.
- Setzt `learning_rate_init` auf `0.5` und schaut euch die neue Verlustkurve an.
- Ergänzt in `vergleiche` eine Spalte für die Lernrate und testet `0.001`, `0.01` und `0.1`.
- Sucht auf Wikipedia ein eigenes Medikament heraus und lasst es von `teste_molekuel` bewerten.
"""),
    ]


def write_notebook(cells: list[dict], path: Path, student: bool) -> None:
    out_cells = []
    for idx, c in enumerate(cells):
        cell = {k: v for k, v in c.items() if not k.startswith("_")}
        cell["id"] = f"cell-{idx:02d}"
        if student and c.get("_student_source") is not None:
            cell["source"] = c["_student_source"]
        cell["source"] = cell["source"].splitlines(keepends=True)
        out_cells.append(cell)

    nb = {
        "cells": out_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    daten_src = load_embedded_daten_cell()
    cells = build_cells(daten_src)
    write_notebook(cells, OUT_SOLUTION, student=False)
    write_notebook(cells, OUT_STUDENT, student=True)
    print(f"Built {OUT_SOLUTION.name} ({len(cells)} cells)")
    print(f"Built {OUT_STUDENT.name} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
