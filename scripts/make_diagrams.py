"""Generate clean teaching diagrams for the Neural Network theory section."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

C_IN = "#2563EB"      # blue
C_HID = "#0D9488"     # teal
C_OUT = "#D97706"     # amber
C_DARK = "#1E293B"    # slate-800
C_MUTED = "#64748B"   # slate-500
C_RED = "#DC2626"     # red
C_LIGHT = "#F8FAFC"


def make_single_neuron():
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    features = ["MW", "AlogP", "HBA", "HBD", "RB"]
    ys = np.linspace(4.5, 0.9, 5)
    weights = ["w₁", "w₂", "w₃", "w₄", "w₅"]

    # Input boxes
    for y, feat, w in zip(ys, features, weights):
        box = FancyBboxPatch(
            (0.4, y - 0.28), 1.5, 0.56,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor="#EFF6FF", edgecolor=C_IN, linewidth=1.8
        )
        ax.add_patch(box)
        ax.text(1.15, y, feat, ha="center", va="center", fontsize=12, fontweight="bold", color=C_DARK)

        # Arrow to sum node
        arrow = FancyArrowPatch(
            (1.98, y), (4.35, 2.7 + (y - 2.7) * 0.22),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.5, color=C_MUTED
        )
        ax.add_patch(arrow)

        # Weight label positioned cleanly along the arrow
        lx = 2.95
        ly = y + (2.7 - y) * 0.36 + (0.25 if y >= 2.7 else -0.26)
        ax.text(lx, ly, w, ha="center", va="center", fontsize=11, color=C_IN, fontweight="bold")

    # Bias arrow from top
    ax.text(5.1, 4.75, "Bias b", ha="center", va="center", fontsize=12, fontweight="bold", color=C_OUT)
    ax.add_patch(FancyArrowPatch(
        (5.1, 4.45), (5.1, 3.45),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.8, color=C_OUT
    ))

    # Sum circle
    sum_circle = Circle((5.1, 2.7), 0.72, facecolor="#F0FDFA", edgecolor=C_HID, linewidth=2.2)
    ax.add_patch(sum_circle)
    ax.text(5.1, 2.86, "Σ", ha="center", va="center", fontsize=22, fontweight="bold", color=C_DARK)
    ax.text(5.1, 2.34, "z = w·x + b", ha="center", va="center", fontsize=9.5, color=C_MUTED)

    # Arrow sum -> activation
    ax.add_patch(FancyArrowPatch(
        (5.85, 2.7), (6.95, 2.7),
        arrowstyle="-|>", mutation_scale=16, linewidth=2.0, color=C_DARK
    ))
    ax.text(6.4, 3.0, "z", ha="center", va="center", fontsize=12, fontweight="bold", color=C_DARK)

    # Activation box
    act_box = FancyBboxPatch(
        (7.0, 2.08), 1.85, 1.24,
        boxstyle="round,pad=0.06,rounding_size=0.14",
        facecolor="#FFFBEB", edgecolor=C_OUT, linewidth=2.0
    )
    ax.add_patch(act_box)
    ax.text(7.925, 2.90, "Aktivierung", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_DARK)
    ax.text(7.925, 2.45, "ReLU(z) = max(0, z)", ha="center", va="center", fontsize=10, color=C_OUT, fontweight="bold")

    # Arrow activation -> output
    ax.add_patch(FancyArrowPatch(
        (8.92, 2.7), (9.72, 2.7),
        arrowstyle="-|>", mutation_scale=16, linewidth=2.0, color=C_DARK
    ))
    ax.text(10.25, 2.7, "Ausgabe\na", ha="center", va="center", fontsize=12.5, fontweight="bold", color=C_DARK)

    # Section labels at bottom
    ax.text(1.15, 0.22, "Eingaben (x)", ha="center", va="center", fontsize=11, color=C_MUTED)
    ax.text(3.05, 0.22, "Gewichte (w)", ha="center", va="center", fontsize=11, color=C_MUTED)
    ax.text(6.5, 0.22, "Ein einzelnes Neuron", ha="center", va="center", fontsize=11, color=C_MUTED)

    ax.set_title("Wie ein einzelnes Neuron aus 5 Molekülwerten eine Zahl berechnet", fontsize=14, fontweight="bold", color=C_DARK, pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_einzelnes_neuron.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_mlp_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=200)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    in_labels = ["MW", "AlogP", "HBA", "HBD", "RB"]
    x_in, x_hid, x_out = 2.0, 5.4, 8.6
    y_in = np.linspace(4.8, 1.4, 5)
    y_hid = np.linspace(5.3, 0.9, 8)
    y_out = [3.1]

    # Connections: input -> hidden
    for yi in y_in:
        for yh in y_hid:
            ax.plot([x_in, x_hid], [yi, yh], color="#CBD5E1", linewidth=0.9, zorder=1)

    # Connections: hidden -> output
    for yh in y_hid:
        ax.plot([x_hid, x_out], [yh, y_out[0]], color="#94A3B8", linewidth=1.2, zorder=1)

    # Input nodes
    for yi, lbl in zip(y_in, in_labels):
        ax.add_patch(Circle((x_in, yi), 0.27, facecolor="#EFF6FF", edgecolor=C_IN, linewidth=2.0, zorder=3))
        ax.text(x_in - 0.50, yi, lbl, ha="right", va="center", fontsize=11.5, fontweight="bold", color=C_DARK)

    # Hidden nodes
    for idx, yh in enumerate(y_hid, start=1):
        ax.add_patch(Circle((x_hid, yh), 0.23, facecolor="#F0FDFA", edgecolor=C_HID, linewidth=2.0, zorder=3))
        ax.text(x_hid, yh, str(idx), ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_DARK, zorder=4)

    # Output node
    ax.add_patch(Circle((x_out, y_out[0]), 0.34, facecolor="#FFFBEB", edgecolor=C_OUT, linewidth=2.2, zorder=3))
    ax.text(x_out, y_out[0], "p", ha="center", va="center", fontsize=14, fontweight="bold", color=C_DARK, zorder=4)
    ax.text(x_out + 0.52, y_out[0], "P(aktiv)\n0 bis 100%", ha="left", va="center", fontsize=11, fontweight="bold", color=C_DARK)

    # Column headers
    ax.text(x_in, 5.85, "Eingabeschicht\n(5 Eigenschaften)", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_IN)
    ax.text(x_hid, 5.85, "Verborgene Schicht\n(8 Neuronen, ReLU)", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_HID)
    ax.text(x_out, 5.85, "Ausgabeschicht\n(1 Wert, Sigmoid)", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_OUT)

    # Parameter count footer
    ax.text(
        5.25, 0.25,
        "Lernbare Zahlen (Parameter):  5×8 + 8 = 48  (Schicht 1)   +   8×1 + 1 = 9  (Ausgabe)   =   57 Parameter",
        ha="center", va="center", fontsize=10.5, color=C_MUTED,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=C_LIGHT, edgecolor="#CBD5E1")
    )

    ax.set_title("Unser neuronales Netz: MLPClassifier(hidden_layer_sizes=(8,))", fontsize=13.5, fontweight="bold", color=C_DARK, pad=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_netzarchitektur.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_activations():
    z = np.linspace(-5, 5, 400)
    relu = np.maximum(0, z)
    sig = 1 / (1 + np.exp(-z))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200)

    # ReLU
    ax1.axhline(0, color="#CBD5E1", linewidth=1)
    ax1.axvline(0, color="#CBD5E1", linewidth=1)
    ax1.plot(z, relu, color=C_HID, linewidth=2.8)
    ax1.set_title("ReLU (verborgene Schicht)", fontsize=12.5, fontweight="bold", color=C_DARK)
    ax1.set_xlabel("Eingabe z", fontsize=11)
    ax1.set_ylabel("Ausgabe max(0, z)", fontsize=11)
    ax1.set_ylim(-0.6, 5.4)
    ax1.grid(alpha=0.25)
    ax1.annotate("Negative Werte\nwerden auf 0 gesetzt", xy=(-2.5, 0.05), xytext=(-4.6, 2.3),
                 arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.3), fontsize=10, color=C_DARK)
    ax1.annotate("Positive Werte\nbleiben unverändert", xy=(2.8, 2.8), xytext=(0.4, 4.2),
                 arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.3), fontsize=10, color=C_DARK)

    # Sigmoid
    ax2.axhline(0, color="#CBD5E1", linewidth=1)
    ax2.axhline(1, color="#CBD5E1", linewidth=1, linestyle="--")
    ax2.axhline(0.5, color="#CBD5E1", linewidth=1, linestyle=":")
    ax2.axvline(0, color="#CBD5E1", linewidth=1)
    ax2.plot(z, sig, color=C_OUT, linewidth=2.8)
    ax2.set_title("Sigmoid (Ausgabeschicht)", fontsize=12.5, fontweight="bold", color=C_DARK)
    ax2.set_xlabel("Eingabe z", fontsize=11)
    ax2.set_ylabel("Wahrscheinlichkeit p", fontsize=11)
    ax2.set_ylim(-0.08, 1.12)
    ax2.grid(alpha=0.25)
    ax2.annotate("Drückt jede Zahl\nzwischen 0 und 1", xy=(2.2, 0.90), xytext=(-4.5, 0.78),
                 arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.3), fontsize=10, color=C_DARK)
    ax2.scatter([0], [0.5], color=C_DARK, s=40, zorder=5)
    ax2.text(0.3, 0.45, "z = 0 → 50%", fontsize=10, color=C_DARK)

    fig.suptitle("Zwei wichtige Aktivierungsfunktionen", fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_aktivierungsfunktionen.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_gradient_descent():
    w = np.linspace(-3.2, 3.2, 300)
    loss = 0.5 * w**2 + 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=200)

    # Good learning rate
    steps_good = [2.6]
    lr_good = 0.35
    for _ in range(7):
        cur = steps_good[-1]
        steps_good.append(cur - lr_good * cur)
    steps_good = np.array(steps_good)

    ax1.plot(w, loss, color=C_DARK, linewidth=2.2)
    ax1.scatter(steps_good, 0.5 * steps_good**2 + 0.3, color=C_HID, s=55, zorder=5)
    for a, b in zip(steps_good[:-1], steps_good[1:]):
        ax1.annotate("", xy=(b, 0.5 * b**2 + 0.3), xytext=(a, 0.5 * a**2 + 0.3),
                     arrowprops=dict(arrowstyle="->", color=C_HID, lw=1.6))
    ax1.scatter([0], [0.3], color=C_OUT, s=90, marker="*", zorder=6, label="Minimum")
    ax1.set_title("Passende Lernrate", fontsize=12.5, fontweight="bold", color=C_DARK)
    ax1.set_xlabel("Gewicht w", fontsize=11)
    ax1.set_ylabel("Verlust (Fehler)", fontsize=11)
    ax1.set_ylim(0, 5.8)
    ax1.grid(alpha=0.25)
    ax1.text(2.38, 3.85, "Start", ha="right", fontsize=10, fontweight="bold", color=C_HID)
    ax1.legend(loc="upper left", frameon=True)

    # Too large learning rate
    steps_bad = [1.4]
    lr_bad = 2.18
    for _ in range(4):
        cur = steps_bad[-1]
        steps_bad.append(cur - lr_bad * cur)
    steps_bad = np.array(steps_bad)

    ax2.plot(w, loss, color=C_DARK, linewidth=2.2)
    ax2.scatter(steps_bad, 0.5 * steps_bad**2 + 0.3, color=C_RED, s=55, zorder=5)
    for a, b in zip(steps_bad[:-1], steps_bad[1:]):
        ax2.annotate("", xy=(b, 0.5 * b**2 + 0.3), xytext=(a, 0.5 * a**2 + 0.3),
                     arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5))
    ax2.scatter([0], [0.3], color=C_OUT, s=90, marker="*", zorder=6, label="Minimum")
    ax2.set_title("Zu grosse Lernrate", fontsize=12.5, fontweight="bold", color=C_DARK)
    ax2.set_xlabel("Gewicht w", fontsize=11)
    ax2.set_ylabel("Verlust (Fehler)", fontsize=11)
    ax2.set_ylim(0, 5.8)
    ax2.grid(alpha=0.25)
    ax2.text(1.62, 0.92, "Start", ha="left", fontsize=10, fontweight="bold", color=C_RED)
    ax2.text(0, 4.9, "Schiesst über das Ziel hinaus", ha="center", fontsize=10.5, fontweight="bold", color=C_RED)
    ax2.legend(loc="upper left", frameon=True)

    fig.suptitle("Gradientenabstieg: Schritt für Schritt bergab", fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_gradientenabstieg.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_single_neuron()
    make_mlp_architecture()
    make_activations()
    make_gradient_descent()
    print("Diagrams written to:", OUT_DIR)
