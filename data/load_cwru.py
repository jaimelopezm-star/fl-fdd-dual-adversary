"""
load_cwru.py — Cargador del CWRU Bearing Dataset para FL-FDD V2.

Maneja explícitamente los canales Drive-End (DE) y Fan-End (FE), porque el Mecanismo 1 (Opción C)
hace consenso multi-sensor DE vs FE. Verifica que ambos canales existan y segmenta en ventanas.

Uso típico:
    from load_cwru import load_dataset, FILE_MAP
    X, y, meta = load_dataset("raw", FILE_MAP, channels=("DE", "FE"),
                              window=2048, stride=2048)
    # X: (n_ventanas, n_canales, window) ; y: (n_ventanas,) ; meta: lista de dicts

Ejecutar directo para autodiagnóstico (usa datos sintéticos si no hay .mat en raw/):
    py -3.12 load_cwru.py
"""
from __future__ import annotations
import os
import glob
import re
import numpy as np

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None

# --- Mapeo archivo CWRU -> (clase, severidad_pulg, carga_HP) -------------------------------------
# Subconjunto 12 kHz Drive-End recomendado para el PoC. Ampliar según los .mat descargados.
# clase: 0=Normal, 1=InnerRace, 2=OuterRace, 3=Ball
FILE_MAP = {
    "97.mat":  {"clase": 0, "sev": 0.000, "carga": 0},  # Normal
    "98.mat":  {"clase": 0, "sev": 0.000, "carga": 1},
    "99.mat":  {"clase": 0, "sev": 0.000, "carga": 2},
    "100.mat": {"clase": 0, "sev": 0.000, "carga": 3},
    "105.mat": {"clase": 1, "sev": 0.007, "carga": 0},  # Inner Race 0.007"
    "106.mat": {"clase": 1, "sev": 0.007, "carga": 1},
    "107.mat": {"clase": 1, "sev": 0.007, "carga": 2},
    "108.mat": {"clase": 1, "sev": 0.007, "carga": 3},
    "130.mat": {"clase": 2, "sev": 0.007, "carga": 0},  # Outer Race 0.007" @6
    "131.mat": {"clase": 2, "sev": 0.007, "carga": 1},
    "132.mat": {"clase": 2, "sev": 0.007, "carga": 2},
    "133.mat": {"clase": 2, "sev": 0.007, "carga": 3},
    "118.mat": {"clase": 3, "sev": 0.007, "carga": 0},  # Ball 0.007"
    "119.mat": {"clase": 3, "sev": 0.007, "carga": 1},
    "120.mat": {"clase": 3, "sev": 0.007, "carga": 2},
    "121.mat": {"clase": 3, "sev": 0.007, "carga": 3},
}
CLASES = {0: "Normal", 1: "InnerRace", 2: "OuterRace", 3: "Ball"}


def find_signal_keys(mat: dict) -> dict:
    """Detecta las claves de canal en un .mat de CWRU (p.ej. 'X105_DE_time').

    Devuelve {'DE': key|None, 'FE': key|None, 'BA': key|None}.
    """
    found = {"DE": None, "FE": None, "BA": None}
    for key in mat:
        if key.startswith("__"):
            continue
        m = re.search(r"_(DE|FE|BA)_time$", key)
        if m:
            found[m.group(1)] = key
    return found


def load_mat_file(path: str) -> dict:
    """Carga un .mat de CWRU y devuelve {'DE': arr1d, 'FE': arr1d, 'BA': arr1d?, 'rpm': float?}.

    Lanza ValueError si falta DE o FE (requisito del Mecanismo 1 Opción C).
    """
    if loadmat is None:
        raise ImportError("scipy no disponible; usar 'py -3.12'.")
    mat = loadmat(path)
    keys = find_signal_keys(mat)
    if keys["DE"] is None or keys["FE"] is None:
        raise ValueError(
            f"{os.path.basename(path)}: falta canal "
            f"{'DE' if keys['DE'] is None else ''}{'FE' if keys['FE'] is None else ''} "
            f"(canales hallados: {[k for k,v in keys.items() if v]}). "
            f"Opción C (consenso DE-FE) requiere ambos."
        )
    out = {ch: np.asarray(mat[k]).ravel().astype(np.float64)
           for ch, k in keys.items() if k is not None}
    rpm_key = next((k for k in mat if k.endswith("RPM")), None)
    out["rpm"] = float(np.asarray(mat[rpm_key]).ravel()[0]) if rpm_key else None
    return out


def segment_signal(signal: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Segmenta una señal 1D en ventanas (n_ventanas, window) con paso 'stride'."""
    n = (len(signal) - window) // stride + 1
    if n <= 0:
        return np.empty((0, window), dtype=signal.dtype)
    idx = np.arange(window)[None, :] + stride * np.arange(n)[:, None]
    return signal[idx]


def load_dataset(raw_dir: str, file_map: dict, channels=("DE", "FE"),
                 window: int = 2048, stride: int = 2048, normalize: bool = True):
    """Carga y segmenta CWRU en (X, y, meta).

    X: (n_ventanas, n_canales, window) — canales en el orden de 'channels'.
    y: (n_ventanas,) etiqueta de clase.
    meta: lista de dicts por ventana (archivo, clase, sev, carga, rpm).
    Los archivos sin DE+FE se SALTAN con aviso (quedan registrados en el retorno meta_skipped).
    """
    Xs, ys, meta = [], [], []
    skipped = []
    for fname, info in file_map.items():
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            skipped.append((fname, "no encontrado"))
            continue
        try:
            sig = load_mat_file(path)
        except ValueError as e:
            skipped.append((fname, str(e)))
            continue
        # Segmenta cada canal y los apila como dimensiones de canal
        segs = [segment_signal(sig[ch], window, stride) for ch in channels]
        n = min(len(s) for s in segs)
        if n == 0:
            skipped.append((fname, "señal más corta que la ventana"))
            continue
        stacked = np.stack([s[:n] for s in segs], axis=1)  # (n, n_canales, window)
        if normalize:  # z-score por ventana y canal
            mu = stacked.mean(axis=2, keepdims=True)
            sd = stacked.std(axis=2, keepdims=True) + 1e-8
            stacked = (stacked - mu) / sd
        Xs.append(stacked)
        ys.append(np.full(n, info["clase"], dtype=np.int64))
        meta.extend([{"archivo": fname, **info, "rpm": sig.get("rpm")} for _ in range(n)])
    if not Xs:
        return (np.empty((0, len(channels), window)), np.empty((0,), dtype=np.int64),
                {"meta": [], "skipped": skipped})
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, {"meta": meta, "skipped": skipped}


def _self_test():
    """Autodiagnóstico: usa raw/ si hay .mat; si no, valida el pipeline con señal sintética."""
    raw = os.path.join(os.path.dirname(__file__), "raw")
    mats = glob.glob(os.path.join(raw, "*.mat"))
    print(f"[load_cwru] raw/: {len(mats)} archivo(s) .mat encontrados.")
    if mats:
        X, y, info = load_dataset(raw, FILE_MAP, channels=("DE", "FE"), window=2048, stride=2048)
        print(f"  X={X.shape}  y={y.shape}  clases={np.unique(y, return_counts=True)}")
        if info["skipped"]:
            print("  SALTADOS:")
            for f, r in info["skipped"]:
                print(f"    - {f}: {r}")
    else:
        print("  No hay .mat aún -> validando pipeline con señal SINTÉTICA (2 canales).")
        de = np.sin(np.linspace(0, 200 * np.pi, 12000)) + 0.1 * np.random.randn(12000)
        fe = 0.5 * np.sin(np.linspace(0, 200 * np.pi, 12000)) + 0.1 * np.random.randn(12000)
        segs_de = segment_signal(de, 2048, 2048)
        segs_fe = segment_signal(fe, 2048, 2048)
        X = np.stack([segs_de, segs_fe], axis=1)
        print(f"  pipeline OK -> X sintético={X.shape} (n_ventanas, 2 canales DE/FE, 2048)")
        print("  Coloca los .mat de CWRU en data/raw/ y vuelve a correr para datos reales.")


if __name__ == "__main__":
    _self_test()
