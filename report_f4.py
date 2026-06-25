"""
report_f4.py — Reporte en Markdown de una corrida del grid F4 (results/F4_grid__<name>.json).

Produce results/report__<name>.md con tablas compactas (promediadas sobre semillas) y una lectura
automática del patrón. Pensado para compartir/pegar fácilmente o para comparar v1 vs v2.

Uso:
    py -3.12 report_f4.py results/F4_grid__v2_scaled.json
"""
from __future__ import annotations
import os, sys, json
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _avg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def _f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def _sel(recs, **cond):
    out = []
    for r in recs:
        if all(r.get(k) == v for k, v in cond.items()):
            out.append(r)
    return out


def build_report(path):
    data = json.load(open(path, encoding="utf-8"))
    recs = data["records"]
    cfg = data.get("config", {})
    name = cfg.get("name", "grid")
    d, fl = cfg.get("data", {}), cfg.get("fl", {})
    tgt = cfg.get("grid", {}).get("target_class", 3)

    L = []
    L.append(f"# Reporte F4 — `{name}`\n")
    L.append(f"> {cfg.get('description','').strip()}\n")
    L.append(f"**Escala:** ventana={d.get('window')} stride={d.get('stride')} · "
             f"clientes={fl.get('n_clients')} · rondas={fl.get('rounds')} · "
             f"celdas={data.get('meta',{}).get('n_cells')} · "
             f"tiempo={data.get('meta',{}).get('elapsed_s')}s · device={data.get('meta',{}).get('device')}\n")

    defenses = sorted({r["defense"] for r in recs})
    alphas = sorted({r["alpha"] for r in recs if r["partition"] == "dirichlet" and r["alpha"] is not None})
    betas = sorted({r["beta"] for r in recs if r["partition"] == "dirichlet" and r["beta"] > 0})

    # --- Tabla 1: AutoGM minoría — evasión y daño (overt vs sigiloso) ---
    L.append("\n## 1. AutoGM — evasión y daño (S3, por α y β)\n")
    L.append("`a_mal` = peso que AutoGM da al malicioso; ≈`a_hon` (evade) o ~0 (filtrado). ASR = daño.\n")
    if "AutoGM" in defenses:
        L.append("| α | β | ASR overt | ASR sigiloso | a_mal overt | a_mal sigiloso | a_hon |")
        L.append("|---|---|---|---|---|---|---|")
        for a in alphas:
            for b in betas:
                ov = _sel(recs, partition="dirichlet", defense="AutoGM", scenario="S3",
                          alpha=a, beta=b, model_mode="gaussian")
                st = _sel(recs, partition="dirichlet", defense="AutoGM", scenario="S3",
                          alpha=a, beta=b, model_mode="constrained")
                L.append(f"| {a} | {b} | {_f(_avg(ov,'ASR'))} | {_f(_avg(st,'ASR'))} | "
                         f"{_f(_avg(ov,'a_malicious'),3)} | {_f(_avg(st,'a_malicious'),3)} | "
                         f"{_f(_avg(st,'a_honest'),3)} |")

    # --- Tabla 2: ASR(S3 sigiloso) por defensa y β (dónde se rompe cada defensa) ---
    L.append("\n## 2. ASR del ataque sigiloso (S3) por defensa y β\n")
    L.append("| defensa | " + " | ".join(f"β={b}" for b in betas) + " |")
    L.append("|" + "---|" * (len(betas) + 1))
    for dfn in defenses:
        cells = []
        for b in betas:
            rows = _sel(recs, partition="dirichlet", defense=dfn, scenario="S3",
                        model_mode="constrained", beta=b)
            # promedia sobre α también (resumen)
            cells.append(_f(_avg(rows, "ASR")))
        L.append(f"| {dfn} | " + " | ".join(cells) + " |")

    # --- Tabla 3: supresión de clase rara (S0 limpio, concentrado) ---
    fos = sorted({r["fault_owners"] for r in recs if r["partition"] == "concentrated" and r["fault_owners"]})
    if fos:
        L.append("\n## 3. Supresión de clase rara — recall clase objetivo (S0 LIMPIO, concentrado)\n")
        L.append("Si AutoGM << FedAvg con pocos dueños → la agregación robusta sacrifica la falla rara.\n")
        L.append("| defensa | " + " | ".join(f"{fo} dueños" for fo in fos) + " |")
        L.append("|" + "---|" * (len(fos) + 1))
        for dfn in defenses:
            cells = []
            for fo in fos:
                rows = _sel(recs, partition="concentrated", defense=dfn, scenario="S0", fault_owners=fo)
                rec = _avg([{"v": r["recall_per_class"][tgt]} for r in rows], "v") if rows else None
                cells.append(_f(rec))
            L.append(f"| {dfn} | " + " | ".join(cells) + " |")

    # --- Lectura automática ---
    L.append("\n## 4. Lectura automática\n")
    notes = []
    if "AutoGM" in defenses and betas:
        bmin = min(betas)
        st_min = _avg(_sel_multi(recs, "AutoGM", "constrained", bmin), "ASR")
        ov_min = _avg(_sel_multi(recs, "AutoGM", "constrained", bmin, want_amal=True), "a_malicious")
        ah_min = _avg(_sel_multi(recs, "AutoGM", "constrained", bmin, want_amal=True), "a_honest")
        if st_min is not None:
            evade = (ov_min or 0) >= 0.8 * (ah_min or 1)
            notes.append(f"- En minoría (β={bmin}): ASR sigiloso ≈ **{_f(st_min)}** "
                         f"({'SIN daño → AutoGM aguanta' if st_min < 0.2 else 'CON daño → AutoGM cae'}); "
                         f"evasión {'SÍ' if evade else 'NO'} (a_mal {_f(ov_min,3)} vs a_hon {_f(ah_min,3)}).")
    if fos and "AutoGM" in defenses and "FedAvg" in defenses:
        fo0 = min(fos)
        ag = _avg([{"v": r["recall_per_class"][tgt]} for r in _sel(recs, partition="concentrated",
                   defense="AutoGM", scenario="S0", fault_owners=fo0)], "v")
        fa = _avg([{"v": r["recall_per_class"][tgt]} for r in _sel(recs, partition="concentrated",
                   defense="FedAvg", scenario="S0", fault_owners=fo0)], "v")
        if ag is not None and fa is not None:
            notes.append(f"- Clase rara ({fo0} dueños, limpio): AutoGM **{_f(ag)}** vs FedAvg **{_f(fa)}** → "
                         f"{'SUPRESIÓN confirmada' if fa - ag > 0.3 else 'sin supresión clara'}.")
    L.extend(notes or ["- (sin lectura automática: faltan celdas)"])

    out = os.path.join(RESULTS, f"report__{name}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"[report_f4] escrito {out}")
    return out


def _sel_multi(recs, defense, mode, beta, want_amal=False):
    rows = [r for r in recs if r["partition"] == "dirichlet" and r["defense"] == defense
            and r["scenario"] == "S3" and r["model_mode"] == mode and r["beta"] == beta]
    return rows


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "F4_grid__v1_local_small.json")
    build_report(p)
