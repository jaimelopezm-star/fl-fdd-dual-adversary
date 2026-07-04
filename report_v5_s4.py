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

    # --- Lectura automatica: cruce S3 (modelo) vs S4 (sensor) — ¿alguna M2 sobrevive AMBOS? ---
    THR = 0.15  # ASR < THR => la defensa "sobrevive" ese vector
    a_lo = min(alphas)  # regimen mas heterogeneo (donde el dual muerde)
    L.append(f"\n## Lectura automatica — cruce Adv2(modelo,S3) vs Adv1(sensor,S4), alpha={a_lo}\n")
    L.append(f"Una defensa 'sobrevive' un vector si ASR<{THR}. Interesa si EXISTE alguna que sobreviva LOS DOS.\n")
    L.append("| defensa | ASR S3 (modelo) | ASR S4 (sensor, peor) | sobrevive S3 | sobrevive S4 | sobrevive AMBOS |")
    L.append("|---|---:|---:|:--:|:--:|:--:|")
    robustos = [d for d in defenses if d != "FedAvg"]
    survive_both = []
    for dfn in robustos:
        s3 = _cell(recs, dfn, a_lo, "S3", "constrained", tgt)
        s4s = _cell(recs, dfn, a_lo, "S4", None, tgt)
        s4d = _cell(recs, dfn, a_lo, "S4", "constrained", tgt)
        if not s3 or not (s4s or s4d):
            continue
        asr_s3 = s3["asr_mean"]
        asr_s4 = max([c["asr_mean"] for c in (s4s, s4d) if c])  # peor caso del sensor
        ok3, ok4 = asr_s3 < THR, asr_s4 < THR
        both = "SI" if (ok3 and ok4) else "**NO**"
        if ok3 and ok4:
            survive_both.append(dfn)
        L.append(f"| {dfn} | {_f(asr_s3)} | {_f(asr_s4)} | {'si' if ok3 else 'CAE'} | "
                 f"{'si' if ok4 else 'CAE'} | {both} |")

    L.append("")
    if not survive_both:
        L.append("- **VEREDICTO: NINGUNA defensa de servidor (M2) sobrevive los DOS vectores.** El adversario "
                 "dual, al disponer de Adv2 (envenenamiento de modelo, S3) Y Adv1 (relabel de sensor, S4), "
                 "derrota CUALQUIER eleccion de defensa de agregacion: la unica que resiste el modelo (PFL) "
                 "cae ante el sensor, y las que resisten el sensor (AutoGM suave, D-WFA) caen ante el modelo. "
                 "Es el gap DEMOSTRADO por ruptura (no por argumento) -> motiva el acoplamiento M1(borde)+M2.")
    else:
        L.append(f"- **VEREDICTO: {', '.join(survive_both)} sobrevive(n) ambos vectores** -> el gap dual no "
                 f"se sostiene con este diseno; revisar.")
    # mecanismo del fallo del sensor: exclusion dura descarta a los portadores honestos de la falla
    for dfn in ("AutoGM-full", "AutoGM-PFL"):
        c = _cell(recs, dfn, a_lo, "S4", None, tgt)
        if c and c["recall"] is not None and c["recall"] < 0.3:
            L.append(f"- **{dfn}:** bajo S4 el recall de la falla objetivo cae a {_f(c['recall'])} (la victima "
                     f"honesta reetiquetada + exclusion dura η* que descarta a los portadores honestos de la "
                     f"falla) -> supresion de clase rara WEAPONIZADA por el sensor.")

    out = os.path.join(RESULTS, f"report__{name}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"[report_v5_s4] escrito {out}")
    return out


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "F4_grid__v5_pfl_sensor.json")
    build_report(p)
