"""
adv2_data.py — Adversario 2 (parte de DATOS): label-flipping dirigido.

El cliente FL malicioso voltea a `target_class` (Normal) una fracción `frac` de sus etiquetas de
falla, dejando la SEÑAL intacta. Así el modelo aprende 'señal de falla -> Normal' (fallas no
detectadas). A diferencia del modo 'mask' de Adv1, aquí la señal sigue siendo de falla, por lo que
el update SÍ tiende a verse anómalo (más detectable por agregación robusta).
"""
from __future__ import annotations
import numpy as np


def apply_label_flip(client_data, malicious_ids, frac: float = 1.0, target_class: int = 0,
                     from_classes=None, seed: int = 0):
    """Voltea etiquetas de falla -> target_class en los clientes comprometidos."""
    rng = np.random.default_rng(seed)
    malicious_ids = set(malicious_ids)
    out = []
    for cid, (X, y) in enumerate(client_data):
        if cid not in malicious_ids:
            out.append((X.copy(), y.copy()))
            continue
        Xc, yc = X.copy(), y.copy()
        mask = (yc != target_class) if from_classes is None else np.isin(yc, from_classes)
        tgt = np.where(mask)[0]
        k = int(round(frac * len(tgt)))
        sel = rng.choice(tgt, size=k, replace=False) if k else np.array([], dtype=int)
        yc[sel] = target_class
        out.append((Xc, yc))
    return out
