"""
grid_f4.py — Barrido (grid) F4 VERSIONADO: caracterización de la vulnerabilidad del baseline.

Cada corrida se rige por un archivo de configuración YAML (config/*.yaml) y se guarda ETIQUETADA con el
nombre de la config, de modo que distintas escalas conviven sin pisarse y TODAS quedan reportadas:

    results/F4_grid__<name>.json

Esto permite comparar, p.ej., 'v1_local_small' (ya ejecutado) vs 'v2_scaled' (Colab) sin descartar
ninguno. Ejes: defensa{FedAvg,AutoGM,D-WFA} × escenario{S0,S1,S2,S3} × (α Dirichlet / concentrado)
× β × {overt 'gaussian', sigiloso 'constrained'}. Métricas DR/FAR/ASR + recall por clase + evasión a_i.

Uso:
    py -3.12 grid_f4.py --config config/v1_local_small.yaml
    # en Colab/pod (GPU):
    py -3.12 grid_f4.py --config config/v2_scaled.yaml --device cuda
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.load_cwru import load_dataset, load_dataset_split, FILE_MAP
from data.partition import dirichlet_partition, concentrated_partition
from models.cnn1d import build_model
from federated.fedavg import run_fl, evaluate, make_loader, fedavg_aggregate
from federated.pfl import run_fl_pfl
from defenses.aggregators import autogm_aggregate, autogm_full_aggregate, dwfa_aggregate
from attacks.coordinated import build_scenario
from metrics.detection import detection_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
# "AutoGM"      = GM suavizada (IRLS), legacy que reproduce v1-v3.
# "AutoGM-full" = AutoGM fiel a Li 2023 (regularización λ + recorte η* que excluye outliers).
AGGS = {"FedAvg": fedavg_aggregate, "AutoGM": autogm_aggregate,
        "AutoGM-full": autogm_full_aggregate, "D-WFA": dwfa_aggregate}

_CACHE = {}


def load_config(path):
    """Lee el YAML de configuración y rellena defaults."""
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cfg.setdefault("description", "")
    cfg.setdefault("data", {}); cfg["data"].setdefault("window", 2048); cfg["data"].setdefault("stride", 2048)
    cfg["data"].setdefault("split", "random")  # 'random' (v1-v6, split por ventanas) o 'blocked' (temporal, sin fuga)
    cfg.setdefault("fl", {}); cfg["fl"].setdefault("n_clients", 10); cfg["fl"].setdefault("rounds", 20)
    cfg["fl"].setdefault("local_epochs", 2); cfg["fl"].setdefault("lr", 1e-3); cfg["fl"].setdefault("batch_size", 64)
    cfg["fl"].setdefault("theta", 0.1)   # regularizacion Ditto para la defensa AutoGM-PFL (0 = local puro)
    g = cfg.setdefault("grid", {})
    g.setdefault("seeds", [0]); g.setdefault("alphas", [0.5, 0.3]); g.setdefault("betas", [0.2, 0.4])
    g.setdefault("defenses", ["FedAvg", "AutoGM"]); g.setdefault("fault_owners", [2, 4])
    g.setdefault("target_class", 3)
    # escenarios/modos a incluir (permite grids enfocados, p.ej. v3 = solo S0+S3 overt+sigiloso)
    g.setdefault("scenarios", ["S0", "S1", "S2", "S3"])
    g.setdefault("modes", ["gaussian", "constrained"])
    cfg.setdefault("attack", {}); cfg["attack"].setdefault("model_boost", 4.0); cfg["attack"].setdefault("model_tau", 1.0)
    cfg["attack"].setdefault("model_sigma", 0.5)
    cfg["attack"].setdefault("sensor_frac", 0.3)   # fraccion de clientes honestos cuyo sensor compromete Adv1 (S4)
    cfg["attack"].setdefault("sensor_gamma", 1.0)  # fraccion de ventanas de la clase objetivo afectadas (S4)
    cfg["attack"].setdefault("sensor_mode", "mask")  # 'mask' (borra senal, v5) o 'flip' (relabel a Normal, v6)
    return cfg


def load_split(window, stride, test_frac=0.2, split_seed=0, split="random"):
    """Carga CWRU (window/stride) y hace split train/test (cacheado por window/stride/split).

    split='random'  : split estratificado sobre ventanas ya generadas (v1-v6). FUGA si stride<window
                      (ventanas de test solapan con las de train).
    split='blocked' : split TEMPORAL por archivo (bloques contiguos train/test con guarda) -> SIN fuga.
    """
    key = ("split", window, stride, split)
    if key not in _CACHE:
        raw = os.path.join(HERE, "data", "raw")
        if split == "blocked":
            _CACHE[key] = load_dataset_split(raw, FILE_MAP, channels=("DE", "FE"),
                                             window=window, stride=stride, test_frac=test_frac)
        else:
            X, y, _ = load_dataset(raw, FILE_MAP, channels=("DE", "FE"), window=window, stride=stride)
            rng = np.random.default_rng(split_seed)
            tr, te = [], []
            for c in np.unique(y):
                idx = np.where(y == c)[0]; rng.shuffle(idx)
                k = int(test_frac * len(idx)); te += idx[:k].tolist(); tr += idx[k:].tolist()
            _CACHE[key] = (X[np.array(tr)], y[np.array(tr)], X[np.array(te)], y[np.array(te)])
    return _CACHE[key]


def _evasion(hist, last=5):
    wm, wh = [], []
    for h in hist[-last:]:
        for w, m in zip(h["agg_weights"], h["mal_mask"]):
            (wm if m else wh).append(w)
    return (float(np.mean(wm)) if wm else None, float(np.mean(wh)) if wh else None)


def run_cell(cfg, *, partition, defense, scenario, seed, beta=0.0, alpha=None,
             fault_owners=2, model_mode="gaussian", device="cpu"):
    """Corre UNA celda del grid y devuelve su registro."""
    d, fl, at = cfg["data"], cfg["fl"], cfg["attack"]
    target = cfg["grid"]["target_class"]
    Xtr, ytr, Xte, yte = load_split(d["window"], d["stride"], split=d.get("split", "random"))
    model_fn = lambda: build_model(in_channels=2, n_classes=4)

    if partition == "dirichlet":
        cidx = dirichlet_partition(ytr, n_clients=fl["n_clients"], alpha=alpha, seed=seed)
        fault_count = [int(np.sum(ytr[i] != 0)) for i in cidx]
        order = list(np.argsort(fault_count)[::-1])
        n_mal = max(0, int(round(beta * fl["n_clients"])))
        mal = [int(c) for c in order[:n_mal]]
    else:  # concentrated
        cidx, owners = concentrated_partition(ytr, n_clients=fl["n_clients"],
                                              fault_owners=fault_owners, seed=seed)
        mal = list(owners.get(target, [])) if beta > 0 else []

    client_data = [(Xtr[i], ytr[i]) for i in cidx]

    # S4 (sensor compartido): el sensor comprometido alimenta a clientes VICTIMA honestos — los honestos
    # que MAS muestras de la clase objetivo tienen (los que de otro modo la detectarian). Enmascara solo
    # esa clase en su entrenamiento local -> rompe la personalizacion (PFL), que asume honestos limpios.
    victim_ids = None
    sensor_only = None
    if scenario == "S4":
        mal_set = set(mal)
        tcount = [int(np.sum(yc == target)) for (_, yc) in client_data]
        honest = [c for c in range(len(client_data)) if c not in mal_set]
        honest.sort(key=lambda c: tcount[c], reverse=True)
        n_v = max(1, int(round(at["sensor_frac"] * len(client_data))))
        victim_ids = honest[:n_v]
        sensor_only = target

    cd, atk = build_scenario(scenario, client_data, mal, seed=seed,
                             fdi_mode=(at["sensor_mode"] if scenario == "S4" else "mask"),
                             fdi_gamma=at["sensor_gamma"] if scenario == "S4" else 1.0,
                             model_mode=model_mode, model_sigma=at["model_sigma"],
                             model_boost=at["model_boost"], model_tau=at["model_tau"],
                             victim_ids=victim_ids, sensor_only_class=sensor_only)

    if defense == "AutoGM-PFL":
        # Variante personalizada (Ditto theta) sobre agregacion robusta AutoGM-full.
        # ASR/DR/FAR se miden sobre el ENSEMBLE PERSONALIZADO (modelos v_k por cliente); la evasion
        # (a_mal/a_hon) sobre la agregacion robusta del global. Ver nota de semantica en federated/pfl.py.
        _, _, hist = run_fl_pfl(model_fn, cd, (Xte, yte), rounds=fl["rounds"],
                                local_epochs=fl["local_epochs"], lr=fl["lr"],
                                batch_size=fl["batch_size"], aggregate=autogm_full_aggregate,
                                theta=fl["theta"], malicious_ids=mal, model_attack_fn=atk,
                                device=device, seed=seed, verbose=False)
        cm = np.asarray(hist[-1]["cm_personalized"])
        met = detection_metrics(cm)
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(4), where=cm.sum(axis=1) > 0)
        ev = {"recall": recall}
        am, ah = _evasion(hist)
    else:
        model, hist = run_fl(model_fn, cd, (Xte, yte), rounds=fl["rounds"], local_epochs=fl["local_epochs"],
                             lr=fl["lr"], batch_size=fl["batch_size"], aggregate=AGGS[defense],
                             malicious_ids=mal, model_attack_fn=atk, device=device, seed=seed, verbose=False)
        ev = evaluate(model, make_loader(Xte, yte, shuffle=False), device, n_classes=4)
        cm = ev["cm"]
        met = detection_metrics(cm)
        am, ah = _evasion(hist)

    # Persiste la matriz de confusion (filas=clase real, columnas=clase predicha) para poder recalcular
    # post-hoc metricas que el grid no guardaba (p.ej. ASR restringido a UNA clase, en vez de "ASR" =
    # todas las fallas -> Normal). ASR_target = fraccion de la clase objetivo (target_class) clasificada
    # como Normal; es la lectura "clase objetivo" que ya narraban los reportes pero que ASR no medía.
    cm_arr = np.asarray(cm)
    normal_class = 0
    tgt_total = float(cm_arr[target].sum())
    asr_target = float(cm_arr[target, normal_class] / tgt_total) if tgt_total else None

    return {
        "partition": partition, "defense": defense, "scenario": scenario, "seed": seed,
        "alpha": alpha, "fault_owners": fault_owners if partition == "concentrated" else None,
        "beta": beta, "n_malicious": len(mal), "malicious_ids": mal, "model_mode": model_mode,
        "n_victims": (len(victim_ids) if victim_ids is not None else 0),
        "victim_ids": victim_ids if victim_ids is not None else [],
        "acc": met["acc"], "DR": met["DR"], "FAR": met["FAR"], "ASR": met["ASR"],
        "ASR_target": asr_target,
        "recall_per_class": [float(r) for r in ev["recall"]],
        "a_malicious": am, "a_honest": ah,
        "cm": [[int(x) for x in row] for row in cm_arr],
    }


def build_grid(cfg):
    """Genera la lista de especificaciones (kwargs de run_cell) a partir de la config.
    Respeta grid.scenarios y grid.modes (para grids enfocados) y omite el eje concentrado si
    grid.fault_owners está vacío."""
    g = cfg["grid"]
    scen_on = set(g["scenarios"]); modes = g["modes"]
    specs = []
    for seed in g["seeds"]:
        for defense in g["defenses"]:
            for alpha in g["alphas"]:
                if "S0" in scen_on:
                    specs.append(dict(partition="dirichlet", defense=defense, scenario="S0",
                                      seed=seed, alpha=alpha, beta=0.0, model_mode=None))
                for beta in g["betas"]:
                    if "S1" in scen_on:
                        specs.append(dict(partition="dirichlet", defense=defense, scenario="S1",
                                          seed=seed, alpha=alpha, beta=beta, model_mode=None))
                    for scen in ("S2", "S3"):
                        if scen not in scen_on:
                            continue
                        for mode in modes:
                            specs.append(dict(partition="dirichlet", defense=defense, scenario=scen,
                                              seed=seed, alpha=alpha, beta=beta, model_mode=mode))
                    if "S4" in scen_on:
                        # sensor-solo (aisla Adv1 sobre honestos) + dual completo (Adv1 sensor + Adv2 modelo)
                        specs.append(dict(partition="dirichlet", defense=defense, scenario="S4",
                                          seed=seed, alpha=alpha, beta=beta, model_mode=None))
                        for mode in modes:
                            specs.append(dict(partition="dirichlet", defense=defense, scenario="S4",
                                              seed=seed, alpha=alpha, beta=beta, model_mode=mode))
    for seed in g["seeds"]:
        for defense in g["defenses"]:
            for fo in g["fault_owners"]:
                specs.append(dict(partition="concentrated", defense=defense, scenario="S0",
                                  seed=seed, fault_owners=fo, beta=0.0, model_mode=None))
                specs.append(dict(partition="concentrated", defense=defense, scenario="S3",
                                  seed=seed, fault_owners=fo, beta=1.0,
                                  model_mode="constrained"))
    return specs


def _fmt(rec, i, n):
    ev = "" if rec["a_malicious"] is None else f"a_mal={rec['a_malicious']:.3f} a_hon={rec['a_honest']:.3f}"
    return (f"  [{i:4}/{n}] {rec['partition'][:4]} {rec['defense']:7} {rec['scenario']} "
            f"a={rec['alpha']} fo={rec['fault_owners']} b={rec['beta']} {str(rec['model_mode']):11} "
            f"ASR={rec['ASR']:.2f} DR={rec['DR']:.2f} {ev}")


def _run_spec(payload):
    """Worker para ejecución en paralelo. Cada celda es INDEPENDIENTE y con su propia semilla, así que
    el resultado es idéntico al secuencial (la paralelización solo cambia el orden, no el cómputo)."""
    idx, cfg, device, sp = payload
    try:
        import torch
        torch.set_num_threads(1)   # evita sobre-suscripción de hilos entre procesos
    except Exception:
        pass
    return idx, run_cell(cfg, device=device, **sp)


def _cell_key(d):
    """Identidad canónica de una celda (para checkpoint/resume). Vale tanto para una spec como para un
    registro ya calculado: usa los mismos campos que definen unívocamente la celda."""
    partition = d.get("partition")
    return (partition, d.get("defense"), d.get("scenario"), d.get("seed"),
            d.get("alpha"), d.get("beta", 0.0),
            d.get("fault_owners") if partition == "concentrated" else None,
            d.get("model_mode"))


def _append_ckpt(ckpt_path, rec):
    """Anexa un registro al checkpoint JSONL (una línea por celda) y hace flush — la corrida larga
    queda protegida: si el proceso muere, los resultados ya calculados sobreviven."""
    if not ckpt_path:
        return
    with open(ckpt_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def run_grid(cfg, specs, device="cpu", jobs=1, ckpt_path=None, done_keys=None):
    """Ejecuta el grid (secuencial si jobs<=1; en paralelo por procesos si jobs>1).
    Devuelve la lista de registros NUEVOS. Si `ckpt_path`, anexa cada celda al JSONL según se completa.
    Si `done_keys`, omite las celdas ya presentes en el checkpoint (resume)."""
    done_keys = done_keys or set()
    pending = [sp for sp in specs if _cell_key(sp) not in done_keys]
    n = len(pending)
    skipped = len(specs) - n
    if skipped:
        print(f"[grid_f4] resume: {skipped} celdas ya en checkpoint -> se omiten; quedan {n}.", flush=True)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results = [None] * n
        payloads = [(i, cfg, device, sp) for i, sp in enumerate(pending)]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_run_spec, p) for p in payloads]
            done = 0
            for fut in tqdm(as_completed(futs), total=n, desc=f"grid {cfg['name']} x{jobs}", unit="celda"):
                idx, rec = fut.result()
                results[idx] = rec
                _append_ckpt(ckpt_path, rec)
                done += 1
                print(_fmt(rec, done, n), flush=True)
        return results

    records = []
    for i, sp in tqdm(list(enumerate(pending, 1)), total=n, desc=f"grid {cfg['name']}", unit="celda"):
        rec = run_cell(cfg, device=device, **sp)
        records.append(rec)
        _append_ckpt(ckpt_path, rec)
        print(_fmt(rec, i, n), flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="ruta al YAML de configuración")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jobs", type=int, default=1,
                    help="nº de procesos en paralelo (cada celda es independiente; no altera resultados)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="reanuda: omite celdas ya presentes en el checkpoint JSONL y las conserva")
    ap.add_argument("--limit", type=int, default=0,
                    help="si >0, corre solo las primeras N celdas (para benchmark/smoke)")
    args = ap.parse_args()

    if args.jobs > 1 and args.device == "cuda":
        print("[grid_f4] NOTA: --jobs>1 con --device cuda -> varios procesos comparten la GPU. En una GPU "
              "grande (p.ej. RTX 4090 24GB) con modelo diminuto es seguro y mejora el aprovechamiento.")

    cfg = load_config(args.config)
    name = cfg["name"]
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, f"F4_grid__{name}.json")
    ckpt = out + ".jsonl"                 # checkpoint incremental (una línea por celda)
    specs = build_grid(cfg)
    if args.limit and args.limit > 0:
        specs = specs[:args.limit]

    # resume: carga registros ya calculados del checkpoint
    prior = []
    done_keys = set()
    if args.resume and os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prior.append(rec)
                done_keys.add(_cell_key(rec))
    elif not args.resume and os.path.exists(ckpt):
        os.remove(ckpt)                   # corrida limpia: descarta checkpoint viejo

    print(f"[grid_f4] config='{name}' -> {len(specs)} celdas | "
          f"win={cfg['data']['window']} stride={cfg['data']['stride']} "
          f"n_clients={cfg['fl']['n_clients']} rounds={cfg['fl']['rounds']} "
          f"defenses={cfg['grid']['defenses']} device={args.device} jobs={args.jobs} "
          f"resume={args.resume}({len(done_keys)} hechas)")

    t0 = time.time()
    new_records = run_grid(cfg, specs, device=args.device, jobs=args.jobs,
                           ckpt_path=ckpt, done_keys=done_keys)

    # combina prior (resume) + nuevos y reordena según build_grid para salida estable
    by_key = {_cell_key(r): r for r in prior}
    by_key.update({_cell_key(r): r for r in new_records})
    records = [by_key[_cell_key(sp)] for sp in specs if _cell_key(sp) in by_key]

    payload = {"config": cfg, "meta": {"n_cells": len(records), "device": args.device, "jobs": args.jobs,
                                       "elapsed_s": round(time.time() - t0, 1)},
               "records": records}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[grid_f4] guardado {out}  ({payload['meta']['elapsed_s']}s)  checkpoint={ckpt}")


if __name__ == "__main__":
    main()
