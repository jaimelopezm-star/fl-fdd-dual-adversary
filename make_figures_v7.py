"""
make_figures_v7.py — Figura del cruce Adv2(modelo,S3) vs Adv1(sensor,S4) en v7_blocked.

Genera en results/:
  fig__v7_blocked__cruce_s3_s4.png

Un panel por alpha (0.3 y 0.5). Un punto por defensa robusta (excluye FedAvg, que es el
control sin defensa). Eje X = ASR media bajo S3 (vector de modelo, dual clasico,
model_mode=constrained). Eje Y = ASR media bajo S4 (vector de sensor, el PEOR caso entre
sensor-solo y dual completo, igual que en report_v5_s4.py). Lineas de umbral en THR=0.15
dibujan cuatro cuadrantes; el cuadrante "sobrevive ambos" (abajo-izquierda) es el que
interesa que quede vacio. Barras de error = IC95 por bootstrap, tomado directamente de
10_auditoria/out/cell_summary_v7.csv (no se recalcula: se reutiliza el numero ya auditado).

Uso:
    py -3.12 make_figures_v7.py
"""
from __future__ import annotations
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
AUDIT_CSV = os.path.join(HERE, "..", "..", "10_auditoria", "out", "cell_summary_v7.csv")

THR = 0.15
DEFENSES = ["AutoGM", "AutoGM-full", "AutoGM-PFL", "D-WFA"]
COLORS = {
    "AutoGM": "#2CA6A0",
    "AutoGM-full": "#B23A3A",
    "AutoGM-PFL": "#2C4C5C",
    "D-WFA": "#8C6E28",
}


def _load_rows():
    with open(AUDIT_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find(rows, defense, alpha, scenario, model_mode):
    """model_mode: 'constrained' (S3), '' (S4 sensor-solo, columna vacia), 'constrained' (S4 dual)."""
    for r in rows:
        if (r["defense"] == defense and r["scenario"] == scenario
                and abs(float(r["alpha"]) - alpha) < 1e-9
                and (r["model_mode"] or None) == model_mode):
            return r
    return None


def _f(r, key):
    return float(r[key])


def build():
    rows = _load_rows()
    alphas = sorted({float(r["alpha"]) for r in rows if r["scenario"] in ("S3", "S4")})

    fig, axes = plt.subplots(1, len(alphas), figsize=(5.5 * len(alphas), 5), squeeze=False)

    for ax, alpha in zip(axes[0], alphas):
        # Cuadrantes de fondo — rectangulo en coordenadas de DATOS, no de ejes
        ax.add_patch(plt.Rectangle((0, 0), THR, THR, color="#2CA6A0", alpha=0.12, zorder=0))
        ax.axhline(THR, color="gray", linestyle="--", linewidth=1)
        ax.axvline(THR, color="gray", linestyle="--", linewidth=1)
        ax.text(THR / 2, THR / 2, "sobrevive\nAMBOS", ha="center", va="center",
                fontsize=9, color="#2CA6A0", fontweight="bold")

        for dfn in DEFENSES:
            s3 = _find(rows, dfn, alpha, "S3", "constrained")
            s4_solo = _find(rows, dfn, alpha, "S4", None)
            s4_dual = _find(rows, dfn, alpha, "S4", "constrained")
            if not s3 or not (s4_solo or s4_dual):
                continue
            # peor caso del sensor, igual criterio que report_v5_s4.py::_sel
            s4_candidates = [c for c in (s4_solo, s4_dual) if c]
            s4 = max(s4_candidates, key=lambda c: _f(c, "ASR_mean"))

            x, xlo, xhi = _f(s3, "ASR_mean"), _f(s3, "ASR_CI95_lo"), _f(s3, "ASR_CI95_hi")
            y, ylo, yhi = _f(s4, "ASR_mean"), _f(s4, "ASR_CI95_lo"), _f(s4, "ASR_CI95_hi")

            ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                        fmt="o", markersize=11, color=COLORS[dfn], ecolor=COLORS[dfn],
                        elinewidth=1.3, capsize=4, label=dfn, zorder=3)
            ax.annotate(dfn, (x, y), textcoords="offset points", xytext=(8, 6),
                        fontsize=9, color=COLORS[dfn], fontweight="bold")

        ax.set_xlim(-0.03, 1.0)
        ax.set_ylim(-0.03, 1.0)
        ax.set_xlabel("ASR bajo S3 (vector de MODELO, sigiloso)")
        ax.set_ylabel("ASR bajo S4 (vector de SENSOR, peor caso)")
        ax.set_title(f"alpha = {alpha}  ·  beta = 0.4  ·  v7_blocked (sin fuga)")
        ax.grid(alpha=0.25)

    fig.suptitle("Ninguna defensa de servidor (M2) sobrevive los dos vectores del adversario dual",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RESULTS, "fig__v7_blocked__cruce_s3_s4.png")
    fig.savefig(out, dpi=160)
    print(f"[make_figures_v7] escrito {out}")


if __name__ == "__main__":
    build()
