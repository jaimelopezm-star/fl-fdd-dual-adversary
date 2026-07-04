"""
report_v5_s4.py — Reporte del contraste S3 (dual clasico) vs S4 (dual con sensor compartido).

Responde la pregunta central: ¿el sensor compartido (S4, Adv1 sobre clientes honestos) ROMPE a
AutoGM-PFL donde el dual clasico (S3) no? Descompone S4 en sensor-solo (aisla Adv1) y dual completo.
Por defensa y alpha reporta, sobre las semillas: ASR media, nº de COLAPSOS (ASR>0.5), a_mal/a_hon y
el recall de la clase objetivo. Genera results/report__<name>.md.

Uso:
    py -3.12 report_v5_s4.py results/F4_grid__v5_pfl_sensor.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def _sel(recs, **cond):
    return [r for r in recs if all(r.get(k) == v for k, v in cond.items())]


def _asr_list(rows):
    return [r["ASR"] for r in rows if r.get("ASR") is not None]


def _avg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def _recall_tgt(rows, tgt):
    v = [r["recall_per_class"][tgt] for r in rows if r.get("recall_per_class")]
    return float(np.mean(v)) if v else None


def _cell(recs, defense, alpha, scenario, mode, tgt):
    """Resumen (ASR media, colapsos/N, a_mal, a_hon, recall_tgt) de una celda (sobre semillas)."""
    rows = _sel(recs, partition="dirichlet", defense=defense, alpha=alpha,
                scenario=scenario, model_mode=mode)
    asr = _asr_list(rows)
    if not asr:
        return None
    collapses = sum(1 for a in asr if a > 0.5)
    return {"asr_mean": float(np.mean(asr)), "asr_list": asr, "n": len(asr),
            "collapses": collapses, "a_mal": _avg(rows, "a_malicious"),
            "a_hon": _avg(rows, "a_honest"), "recall": _recall_tgt(rows, tgt),
            "n_victims": _avg(rows, "n_victims")}


def build_report(path):
    data = json.load(open(path, encoding="utf-8"))
    recs = data["records"]
    cfg = data.get("config", {})
    name = cfg.get("name", "grid")
    d, fl = cfg.get("data", {}), cfg.get("fl", {})
    tgt = cfg.get("grid", {}).get("target_class", 3)
    meta = data.get("meta", {})

    defenses = ["FedAvg", "AutoGM", "AutoGM-full", "AutoGM-PFL", "D-WFA"]
    defenses = [x for x in defenses if x in {r["defense"] for r in recs}]
    alphas = sorted({r["alpha"] for r in recs if r["partition"] == "dirichlet" and r["alpha"] is not None})

    # variantes a contrastar: (etiqueta, scenario, model_mode)
    variants = [("S3 dual clasico", "S3", "constrained"),
                ("S4 sensor-solo", "S4", None),
                ("S4 dual completo", "S4", "constrained")]

    L = []
    L.append(f"# Reporte v5 — S3 (dual clasico) vs S4 (sensor compartido)\n")
    L.append(f"> {cfg.get('description','').strip()}\n")
    L.append(f"**Escala:** ventana={d.get('window')} stride={d.get('stride')} · "
             f"clientes={fl.get('n_clients')} · rondas={fl.get('rounds')} · theta={fl.get('theta')} · "
             f"celdas={meta.get('n_cells')} · tiempo={meta.get('elapsed_s')}s · device={meta.get('device')}\n")
    L.append("\nASR = fraccion de la falla objetivo clasificada como Normal (daño). En PFL se mide sobre el "
             "ENSEMBLE personalizado de honestos (incluidas las victimas del sensor). Colapsos = semillas con "
             "ASR>0.5. `a_mal`/`a_hon` = pesos de agregacion del GLOBAL (evasion).\n")

    for a in alphas:
        L.append(f"\n## alpha = {a}  (beta=0.4, clase objetivo={tgt})\n")
        L.append("| defensa | variante | ASR media | colapsos | recall_obj | a_mal | a_hon |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for dfn in defenses:
            for lbl, scen, mode in variants:
                c = _cell(recs, dfn, a, scen, mode, tgt)
                if c is None:
                    continue
                L.append(f"| {dfn} | {lbl} | {_f(c['asr_mean'])} | {c['collapses']}/{c['n']} | "
                         f"{_f(c['recall'])} | {_f(c['a_mal'],3)} | {_f(c['a_hon'],3)} |")

    # --- Lectura automatica: el veredicto PFL S3 vs S4 ---
    L.append("\n## Lectura automatica — ¿el sensor compartido rompe PFL?\n")
    notes = []
    a_lo = min(alphas)  # regimen mas heterogeneo (donde el dual muerde)
    pfl = "AutoGM-PFL"
    if pfl in defenses:
        s3 = _cell(recs, pfl, a_lo, "S3", "constrained", tgt)
        s4s = _cell(recs, pfl, a_lo, "S4", None, tgt)
        s4d = _cell(recs, pfl, a_lo, "S4", "constrained", tgt)
        if s3 and s4d:
            notes.append(
                f"- **AutoGM-PFL (alpha={a_lo}):** S3 dual clasico ASR={_f(s3['asr_mean'])} "
                f"({s3['collapses']}/{s3['n']} colapsos) vs **S4 dual completo ASR={_f(s4d['asr_mean'])} "
                f"({s4d['collapses']}/{s4d['n']})**"
                + (f"; S4 sensor-solo ASR={_f(s4s['asr_mean'])} ({s4s['collapses']}/{s4s['n']})." if s4s else "."))
            broke = s4d["asr_mean"] - s3["asr_mean"]
            if broke > 0.2:
                notes.append(
                    f"- **VEREDICTO: el sensor compartido SI quiebra la personalizacion** (Δ ASR = "
                    f"+{_f(broke)} sobre S3). La mitad-dispositivo llega a lo local, que es lo unico de lo "
                    f"que PFL depende -> gap REAFIRMADO contra el baseline mas fuerte (rompiendo, no argumentando).")
            else:
                notes.append(
                    f"- **VEREDICTO: PFL tambien aguanta S4** (Δ ASR = {_f(broke)}). El sensor compartido no "
                    f"basta con este `sensor_frac`; replantear (subir fraccion / atacar test por cliente).")
    # contraste con defensas de modelo global: ¿protegen a las victimas del sensor?
    for dfn in ("AutoGM", "AutoGM-full"):
        if dfn in defenses:
            g3 = _cell(recs, dfn, a_lo, "S3", "constrained", tgt)
            g4 = _cell(recs, dfn, a_lo, "S4", "constrained", tgt)
            if g3 and g4:
                notes.append(
                    f"- **{dfn} (modelo global, alpha={a_lo}):** S3 ASR={_f(g3['asr_mean'])} "
                    f"({g3['collapses']}/{g3['n']}) vs S4 ASR={_f(g4['asr_mean'])} ({g4['collapses']}/{g4['n']}) "
                    f"-> el modelo global {'PROTEGE' if g4['asr_mean'] < 0.2 else 'NO protege'} a las victimas del "
                    f"sensor (aprende la falla de los honestos no-victima).")
    L.extend(notes or ["- (sin lectura: faltan celdas)"])

    out = os.path.join(RESULTS, f"report__{name}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"[report_v5_s4] escrito {out}")
    return out


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "F4_grid__v5_pfl_sensor.json")
    build_report(p)
