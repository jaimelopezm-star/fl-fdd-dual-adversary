"""
adv2_model.py — Adversario 2 (parte de MODELO): envenenamiento de la actualización.

Devuelve una función fn(state_dict, rng) que perturba el update del cliente malicioso DESPUÉS de su
entrenamiento local y ANTES de enviarlo al servidor (se engancha en `run_fl(model_attack_fn=...)`).

Modos:
  'gaussian' : suma ruido gaussiano (sigma) a los parámetros -> degrada el modelo global.
  'scale'    : amplifica el update (x scale) para dominar la media de la agregación.
  'sign'     : invierte el sentido del update (x -scale) -> ataque de signo.
"""
from __future__ import annotations
import torch


def make_model_poison(mode: str = "gaussian", sigma: float = 0.1, scale: float = 1.0):
    """Crea el hook de envenenamiento de modelo. Solo toca tensores float (no buffers enteros)."""
    def attack(state_dict, rng):  # rng (numpy) disponible por si se quiere aleatoriedad reproducible
        out = {}
        for k, v in state_dict.items():
            if torch.is_floating_point(v):
                if mode == "gaussian":
                    out[k] = v + torch.randn_like(v) * sigma
                elif mode == "scale":
                    out[k] = v * scale
                elif mode == "sign":
                    out[k] = -v * scale
                else:
                    raise ValueError(f"modo de envenenamiento de modelo desconocido: {mode}")
            else:
                out[k] = v.clone()
        return out
    return attack
