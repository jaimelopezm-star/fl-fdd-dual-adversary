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
from data.load_cwru import load_dataset, FILE_MAP
from data.partition import dirichlet_partition, concentrated_partition
from models.cnn1d import build_model
from federated.fedavg import run_fl, evaluate, make_loader, fedavg_aggregate
from defenses.aggregators import autogm_aggregate, dwfa_aggregate
from attacks.coordinated import build_scenario
from metrics.detection import detection_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
AGGS = {"FedAvg": fedavg_aggregate, "AutoGM": autogm_aggregate, "D-WFA": dwfa_aggregate}

_CACHE = {}


def load_config(path):
    """Lee el YAML de configuración y rellena defaults."""
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cfg.setdefault("description", "")
    cfg.setdefault("data", {}); cfg["data"].setdefault("window", 2048); cfg["data"].setdefault("stride", 2048)
    cfg.setdefault("fl", {}); cfg["fl"].setdefault("n_clients", 10); cfg["fl"].setdefault("rounds", 20)
    cfg["fl"].setdefault("local_epochs", 2); cfg["fl"].setdefault("lr", 1e-3); cfg["fl"].setdefault("batch_size", 64)
    g = cfg.setdefault("grid", {})
    g.setdefault("seeds", [0]); g.setdefault("alphas", [0.5, 0.3]); g.setdefault("betas", [0.2, 0.4])
    g.setdefault("defenses", ["FedAvg", "AutoGM"]); g.setdefault("fault_owners", [2, 4])
    g.setdefault("target_class", 3)
    cfg.setdefault("attack", {}); cfg["attack"].setdefault("model_boost", 4.0); cfg["attack"].setdefault("model_tau", 1.0)
    cfg["attack"].setdefault("model_sigma", 0.5)
    return cfg


def load_split(window, stride, test_frac=0.2, split_seed=0):
    """Carga CWRU (window/stride) y hace split estratificado train/test (cacheado por window/stride)."""
    key = ("split", window, stride)
    if key not in _CACHE:
        X, y, _ = load_dataset(os.path.join(HERE, "data", "raw"), FILE_MAP,
                               channels=("DE", "FE"), window=window, stride=stride)
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
    Xtr, ytr, Xte, yte = load_split(d["window"], d["stride"])
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
    cd, atk = build_scenario(scenario, client_data, mal, seed=seed, fdi_mode="mask",
                             model_mode=model_mode, model_sigma=at["model_sigma"],
                             model_boost=at["model_boost"], model_tau=at["model_tau"])
    model, hist = run_fl(model_fn, cd, (Xte, yte), rounds=fl["rounds"], local_epochs=fl["local_epochs"],
                         lr=fl["lr"], batch_size=fl["batch_size"], aggregate=AGGS[defense],
                         malicious_ids=mal, model_attack_fn=atk, device=device, seed=seed, verbose=False)
    ev = evaluate(model, make_loader(Xte, yte, shuffle=False), device, n_classes=4)
    met = detection_metrics(ev["cm"])
    am, ah = _evasion(hist)
    return {
        "partition": partition, "defense": defense, "scenario": scenario, "seed": seed,
        "alpha": alpha, "fault_owners": fault_owners if partition == "concentrated" else None,
        "beta": beta, "n_malicious": len(mal), "malicious_ids": mal, "model_mode": model_mode,
        "acc": met["acc"], "DR": met["DR"], "FAR": met["FAR"], "ASR": met["ASR"],
        "recall_per_class": [float(r) for r in ev["recall"]],
        "a_malicious": am, "a_honest": ah,
    }


def build_grid(cfg):
    """Genera la lista de especificaciones (kwargs de run_cell) a partir de la config."""
    g = cfg["grid"]
    specs = []
    for seed in g["seeds"]:
        for defense in g["defenses"]:
            for alpha in g["alphas"]:
                specs.append(dict(partition="dirichlet", defense=defense, scenario="S0",
                                  seed=seed, alpha=alpha, beta=0.0, model_mode=None))
                for beta in g["betas"]:
                    specs.append(dict(partition="dirichlet", defense=defense, scenario="S1",
                                      seed=seed, alpha=alpha, beta=beta, model_mode=None))
                    for scen in ("S2", "S3"):
                        for mode in ("gaussian", "constrained"):
                            specs.append(dict(partition="dirichlet", defense=defense, scenario=scen,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="ruta al YAML de configuración")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["name"]
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, f"F4_grid__{name}.json")
    specs = build_grid(cfg)
    print(f"[grid_f4] config='{name}' -> {len(specs)} celdas | "
          f"win={cfg['data']['window']} stride={cfg['data']['stride']} "
          f"n_clients={cfg['fl']['n_clients']} rounds={cfg['fl']['rounds']} device={args.device}")

    try:
        from tqdm import tqdm
        iterator = tqdm(list(enumerate(specs, 1)), total=len(specs), desc=f"grid {name}", unit="celda")
    except ImportError:
        iterator = enumerate(specs, 1)

    records, t0 = [], time.time()
    for i, sp in iterator:
        rec = run_cell(cfg, device=args.device, **sp)
        records.append(rec)
        ev = "" if rec["a_malicious"] is None else f"a_mal={rec['a_malicious']:.3f} a_hon={rec['a_honest']:.3f}"
        print(f"  [{i:4}/{len(specs)}] {rec['partition'][:4]} {rec['defense']:7} {rec['scenario']} "
              f"a={rec['alpha']} fo={rec['fault_owners']} b={rec['beta']} {str(rec['model_mode']):11} "
              f"ASR={rec['ASR']:.2f} DR={rec['DR']:.2f} {ev}", flush=True)

    payload = {"config": cfg, "meta": {"n_cells": len(records), "device": args.device,
                                       "elapsed_s": round(time.time() - t0, 1)},
               "records": records}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[grid_f4] guardado {out}  ({payload['meta']['elapsed_s']}s)")


if __name__ == "__main__":
    main()
