"""
adv1_fdi.py — Adversario 1: False Data Injection (FDI) en el sensor.

Adv1 compromete el sensor físico y corrompe la SEÑAL (features) que el nodo edge usa para entrenar.
A diferencia del label-flipping (Adv2), aquí se altera la propia lectura del sensor.

Modos:
  'mask'  : ENMASCARA la falla. Toma ventanas de falla del cliente comprometido y reemplaza su señal
            por una ventana Normal (muestreada del pool global) y pone la etiqueta = Normal. Resultado:
            datos benignos y CONSISTENTES (señal normal + etiqueta normal). Es el ataque SIGILOSO:
            el update del cliente no se ve anómalo -> evade la agregación robusta; el modelo ve menos
            fallas -> fallas no detectadas (ASR alto). Es el modo clave para el escenario dual.
  'noise' : suma ruido gaussiano fuerte a la señal (spoof burdo / sensor dañado), etiqueta intacta.
  'scale' : escala la amplitud de la señal (deriva/saturación de sensor), etiqueta intacta.

No toca el conjunto de test. Devuelve un nuevo client_data (copias), nunca modifica el original.
"""
from __future__ import annotations
import numpy as np


def _normal_pool(client_data, normal_class):
    pools = [X[y == normal_class] for X, y in client_data]
    pools = [p for p in pools if len(p)]
    return np.concatenate(pools, axis=0) if pools else None


def apply_fdi(client_data, malicious_ids, gamma: float = 1.0, mode: str = "mask",
              normal_class: int = 0, noise_std: float = 3.0, scale: float = 0.2,
              seed: int = 0):
    """Aplica FDI a una fracción `gamma` de las ventanas de cada cliente comprometido."""
    rng = np.random.default_rng(seed)
    malicious_ids = set(malicious_ids)
    pool = _normal_pool(client_data, normal_class) if mode == "mask" else None
    out = []
    for cid, (X, y) in enumerate(client_data):
        if cid not in malicious_ids:
            out.append((X.copy(), y.copy()))
            continue
        Xc, yc = X.copy(), y.copy()
        if mode == "mask":
            tgt = np.where(yc != normal_class)[0]            # ventanas de falla
            k = int(round(gamma * len(tgt)))
            sel = rng.choice(tgt, size=k, replace=False) if k else np.array([], dtype=int)
            if len(sel) and pool is not None:
                rep = rng.integers(0, len(pool), size=len(sel))
                Xc[sel] = pool[rep]                           # señal -> Normal
                yc[sel] = normal_class                        # etiqueta -> Normal
        elif mode in ("noise", "scale"):
            n = len(yc)
            k = int(round(gamma * n))
            sel = rng.choice(n, size=k, replace=False) if k else np.array([], dtype=int)
            if mode == "noise":
                Xc[sel] = Xc[sel] + rng.normal(0.0, noise_std, size=Xc[sel].shape)
            else:
                Xc[sel] = Xc[sel] * scale
        else:
            raise ValueError(f"modo FDI desconocido: {mode}")
        out.append((Xc, yc))
    return out
