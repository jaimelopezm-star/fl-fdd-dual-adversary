"""
adv2_model.py — Adversario 2 (parte de MODELO): envenenamiento de la actualización.

Devuelve una función fn(state_dict, rng, ctx) que perturba el update del cliente malicioso DESPUÉS de
su entrenamiento local y ANTES de enviarlo al servidor (se engancha en `run_fl(model_attack_fn=...)`).
`ctx` = {gstate, ref_norm} aporta el estado global de la ronda y la norma honesta de referencia.

Modos:
  'gaussian'   : suma ruido gaussiano (sigma) a los parámetros -> degrada el modelo global. OVERT:
                 produce un outlier lejano que la agregación robusta (AutoGM) filtra. (baseline F3)
  'scale'      : amplifica el update (x scale) para dominar la media. OVERT, también detectable.
  'sign'       : invierte el sentido del update (x -scale). OVERT.
  'constrained': SIGILOSO (F4). Toma el update del cliente (entrenado sobre datos enmascarados por
                 Adv1), lo amplifica en su MISMA dirección (x boost) y luego CLIPA su norma L2 a
                 <= tau * ref_norm (la magnitud mediana de los updates honestos de la ronda). Así el
                 update malicioso permanece dentro de la dispersión legítima non-IID -> la mediana
                 geométrica de AutoGM no lo distingue (a_malicioso ~ a_honesto) y aun así arrastra al
                 modelo a llamar "Normal" a las fallas (ASR alto). Es el corazón del ataque coordinado.
"""
from __future__ import annotations
import torch


def make_model_poison(mode: str = "gaussian", sigma: float = 0.1, scale: float = 1.0,
                      boost: float = 1.0, tau: float = 1.0):
    """Crea el hook de envenenamiento de modelo. Solo toca tensores float (no buffers enteros).

    Parámetros (según modo):
      sigma : desviación del ruido ('gaussian').
      scale : factor de escala/inversión ('scale','sign').
      boost : amplificación del update en su dirección antes del clip ('constrained').
      tau   : presupuesto de norma relativo a la mediana honesta ('constrained'); <=1 = más sigiloso.
    """
    def attack(state_dict, rng, ctx=None):  # ctx = {gstate, ref_norm}
        if mode == "constrained":
            return _constrained(state_dict, ctx, boost=boost, tau=tau)
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


def _constrained(state_dict, ctx, *, boost: float, tau: float):
    """Update sigiloso: delta amplificado x boost y clipado a <= tau * ref_norm."""
    if ctx is None or ctx.get("gstate") is None:
        # sin contexto no hay referencia: degrada a passthrough (el masking de Adv1 ya envenena).
        return {k: v.clone() for k, v in state_dict.items()}
    gstate = ctx["gstate"]
    ref_norm = ctx.get("ref_norm")

    fkeys = [k for k, v in state_dict.items() if torch.is_floating_point(v)]
    # delta = update del cliente respecto al global, amplificado en su misma dirección.
    deltas = {k: (state_dict[k].float() - gstate[k].float()) * boost for k in fkeys}
    norm = float(sum(float(torch.sum(d ** 2)) for d in deltas.values())) ** 0.5

    if ref_norm is not None and norm > tau * ref_norm and norm > 0:
        factor = (tau * ref_norm) / norm        # proyecta dentro del presupuesto honesto
        deltas = {k: d * factor for k, d in deltas.items()}

    out = {}
    for k, v in state_dict.items():
        if torch.is_floating_point(v):
            out[k] = (gstate[k].float() + deltas[k]).to(v.dtype)
        else:
            out[k] = v.clone()
    return out
