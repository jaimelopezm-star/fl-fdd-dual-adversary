"""
make_figures_f4.py — Figuras de la caracterización F4 a partir de results/F4_grid.json.

Genera en results/:
  fig_F4_asr_vs_beta.png      : ASR vs β por defensa (AutoGM aguanta hasta la frontera Byzantine).
  fig_F4_supraadditividad.png : ASR(S3) - [ASR(S1)+ASR(S2)] vs β  (efecto de coordinación).
  fig_F4_evasion.png          : peso a_malicioso vs a_honesto, overt vs sigiloso (prueba de evasión).
  fig_F4_clase_rara.png       : recall de la clase rara en partición concentrada, FedAvg vs AutoGM.

Promedia sobre semillas. Robusto al grid reducido (omite paneles sin datos).
"""
from __future__ import annotations
import os, sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TAG = "grid"   # prefijo de las figuras; se fija al cargar según la config


def load(path=None):
    global TAG
    path = path or os.path.join(RESULTS, "F4_grid.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    TAG = data.get("config", {}).get("name", "grid")
    return data["records"]


def _figpath(stem):
    return os.path.join(RESULTS, f"fig__{TAG}__{stem}.png")


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _agg(records, predicate, group_keys, value_key):
    """Promedia value_key sobre semillas, agrupando por group_keys (tupla). -> {grupo: media}."""
    buckets = defaultdict(list)
    for r in records:
        if predicate(r):
            buckets[tuple(r[k] for k in group_keys)].append(r)
    return {g: _mean(rows, value_key) for g, rows in buckets.items()}


def fig_asr_vs_beta(records):
    dyn = [r for r in records if r["partition"] == "dirichlet" and r["scenario"] == "S3"
           and r["model_mode"] == "constrained"]
    if not dyn:
        return
    alphas = sorted({r["alpha"] for r in dyn})
    defenses = sorted({r["defense"] for r in dyn})
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4), squeeze=False)
    for ax, alpha in zip(axes[0], alphas):
        for dfn in defenses:
            g = _agg(records, lambda r, a=alpha, d=dfn: (r["partition"] == "dirichlet"
                     and r["scenario"] == "S3" and r["model_mode"] == "constrained"
                     and r["alpha"] == a and r["defense"] == d), ["beta"], "ASR")
            xs = sorted(g)
            ax.plot([b[0] for b in xs], [g[b] for b in xs], marker="o", label=dfn)
        ax.set_title(f"S3 sigiloso  α={alpha}"); ax.set_xlabel("β (frac. maliciosa)")
        ax.set_ylabel("ASR"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Daño (ASR) del ataque coordinado sigiloso vs fracción maliciosa")
    fig.tight_layout(); fig.savefig(_figpath("asr_vs_beta"), dpi=130)
    plt.close(fig)


def fig_supraadditividad(records):
    def asr(scen, a, b, dfn, mode):
        g = _agg(records, lambda r: (r["partition"] == "dirichlet" and r["scenario"] == scen
                 and r["alpha"] == a and r["beta"] == b and r["defense"] == dfn
                 and (scen == "S1" or r["model_mode"] == mode)), ["scenario"], "ASR")
        return g.get((scen,))
    rows = [r for r in records if r["partition"] == "dirichlet" and r["defense"] == "AutoGM"
            and r["scenario"] == "S3" and r["model_mode"] == "constrained"]
    if not rows:
        return
    alphas = sorted({r["alpha"] for r in rows}); betas = sorted({r["beta"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in alphas:
        ys = []
        for b in betas:
            s1, s2, s3 = asr("S1", a, b, "AutoGM", "constrained"), \
                         asr("S2", a, b, "AutoGM", "constrained"), \
                         asr("S3", a, b, "AutoGM", "constrained")
            ys.append((s3 - (s1 + s2)) if None not in (s1, s2, s3) else np.nan)
        ax.plot(betas, ys, marker="s", label=f"α={a}")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("β"); ax.set_ylabel("ASR(S3) − [ASR(S1)+ASR(S2)]")
    ax.set_title("Supra-aditividad de la coordinación (AutoGM, sigiloso)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(_figpath("supraadditividad"), dpi=130)
    plt.close(fig)


def fig_evasion(records):
    rows = [r for r in records if r["partition"] == "dirichlet" and r["scenario"] == "S3"
            and r["defense"] == "AutoGM" and r["a_malicious"] is not None]
    if not rows:
        return
    betas = sorted({r["beta"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4)); w = 0.35; x = np.arange(len(betas))
    for off, mode, col in [(-w/2, "gaussian", "#c44"), (w/2, "constrained", "#48c")]:
        am = [_agg(records, lambda r, b=b, m=mode: (r["partition"] == "dirichlet"
              and r["scenario"] == "S3" and r["defense"] == "AutoGM" and r["beta"] == b
              and r["model_mode"] == m), ["beta"], "a_malicious").get((b,), np.nan) for b in betas]
        ax.bar(x + off, am, w, label=f"a_malicioso ({'overt' if mode=='gaussian' else 'sigiloso'})", color=col)
    ah = [_agg(records, lambda r, b=b: (r["partition"] == "dirichlet" and r["scenario"] == "S3"
          and r["defense"] == "AutoGM" and r["beta"] == b), ["beta"], "a_honest").get((b,), np.nan) for b in betas]
    ax.plot(x, ah, "k--o", label="a_honesto (ref.)")
    ax.set_xticks(x); ax.set_xticklabels([f"β={b}" for b in betas]); ax.set_ylabel("peso de agregación")
    ax.set_title("Evasión bajo AutoGM: overt es filtrado, sigiloso ≈ honesto"); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(_figpath("evasion"), dpi=130)
    plt.close(fig)


def fig_clase_rara(records):
    rows = [r for r in records if r["partition"] == "concentrated" and r["scenario"] == "S0"]
    if not rows:
        return
    fos = sorted({r["fault_owners"] for r in rows}); defenses = sorted({r["defense"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4)); w = 0.8 / max(1, len(defenses)); x = np.arange(len(fos))
    for i, dfn in enumerate(defenses):
        ys = []
        for fo in fos:
            g = [r for r in rows if r["defense"] == dfn and r["fault_owners"] == fo]
            # recall de la clase rara objetivo (clase 3) promediado sobre semillas
            ys.append(float(np.mean([r["recall_per_class"][3] for r in g])) if g else np.nan)
        ax.bar(x + i * w, ys, w, label=dfn)
    ax.set_xticks(x + w * (len(defenses) - 1) / 2); ax.set_xticklabels([f"{fo} dueños" for fo in fos])
    ax.set_ylabel("recall clase rara (limpio, S0)"); ax.set_ylim(0, 1.05)
    ax.set_title("Supresión de clase rara: la agregación robusta descarta a sus dueños")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(_figpath("clase_rara"), dpi=130)
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    records = load(path)
    fig_asr_vs_beta(records)
    fig_supraadditividad(records)
    fig_evasion(records)
    fig_clase_rara(records)
    print(f"[make_figures_f4] figuras escritas en {RESULTS}")


if __name__ == "__main__":
    main()
