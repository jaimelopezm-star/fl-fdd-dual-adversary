"""
detection.py — Métricas de detección/ataque para FL-FDD.

A partir de la matriz de confusión (filas=clase real, columnas=clase predicha):
  acc : exactitud global.
  DR  : Detection Rate = recall promedio sobre las clases de FALLA (qué tan bien se detectan fallas).
  FAR : False Alarm Rate = Normal clasificado como alguna falla (falsa alarma).
  ASR : Attack Success Rate = fracción de FALLAS reales clasificadas como Normal. Es el objetivo del
        ataque dirigido (una falla que pasa como "Normal" = detección perdida -> riesgo industrial).
"""
from __future__ import annotations
import numpy as np


def detection_metrics(cm, normal_class: int = 0) -> dict:
    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0

    fault = [c for c in range(n) if c != normal_class]
    fault_total = cm[fault].sum() if fault else 0.0

    # ASR: fallas reales predichas como Normal
    asr = float(cm[fault, normal_class].sum() / fault_total) if fault_total else 0.0

    # DR: recall promedio en clases de falla
    drs = [cm[c, c] / cm[c].sum() for c in fault if cm[c].sum() > 0]
    dr = float(np.mean(drs)) if drs else 0.0

    # FAR: Normal predicho como falla
    nrow = cm[normal_class].sum()
    far = float((nrow - cm[normal_class, normal_class]) / nrow) if nrow else 0.0

    return {"acc": acc, "DR": dr, "FAR": far, "ASR": asr}
