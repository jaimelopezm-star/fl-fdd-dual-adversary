"""
coordinated.py — Composición de escenarios de amenaza (incluye el DUAL COORDINADO, novedad de la tesis).

`build_scenario` devuelve (client_data_envenenado, model_attack_fn) listos para `run_fl`:

  'S0' : limpio (sin ataque).
  'S1' : solo Adv1 — FDI en el sensor (modo 'mask' por defecto, sigiloso).
  'S2' : solo Adv2 — label-flip (datos) + envenenamiento del update (modelo).
  'S3' : DUAL COORDINADO — el MISMO actor compromete sensor (Adv1 FDI) y cliente (Adv2 modelo).
         La señal enmascarada por Adv1 hace que el update no se vea anómalo, mientras Adv2 empuja
         el modelo global -> la coordinación amplifica el daño y evade la defensa del servidor.

Todos usan el mismo conjunto `malicious_ids` (mismo actor controla ambas capas en S3).
"""
from __future__ import annotations
from .adv1_fdi import apply_fdi
from .adv2_data import apply_label_flip
from .adv2_model import make_model_poison


def build_scenario(name, client_data, malicious_ids, *, seed: int = 0,
                   fdi_gamma: float = 1.0, fdi_mode: str = "mask",
                   flip_frac: float = 1.0,
                   model_mode: str = "gaussian", model_sigma: float = 0.5,
                   model_scale: float = 1.0, model_boost: float = 1.0, model_tau: float = 1.0):
    """Construye el escenario `name`. Devuelve (client_data, model_attack_fn|None).

    `model_mode`:
      - 'gaussian'/'scale'/'sign' : Adv2 OVERT (baseline F3, detectable por AutoGM).
      - 'constrained'             : Adv2 SIGILOSO (F4), usa model_boost/model_tau.
      - None                      : S3 'puro masking' (sin poison de modelo) — aísla el efecto de Adv1.
    """
    cd = [(X.copy(), y.copy()) for X, y in client_data]
    atk = None

    def _poison():
        if model_mode is None:
            return None
        return make_model_poison(mode=model_mode, sigma=model_sigma, scale=model_scale,
                                 boost=model_boost, tau=model_tau)

    if name == "S0":
        pass
    elif name == "S1":
        cd = apply_fdi(cd, malicious_ids, gamma=fdi_gamma, mode=fdi_mode, seed=seed)
    elif name == "S2":
        cd = apply_label_flip(cd, malicious_ids, frac=flip_frac, seed=seed)
        atk = _poison()
    elif name == "S3":
        cd = apply_fdi(cd, malicious_ids, gamma=fdi_gamma, mode=fdi_mode, seed=seed)
        atk = _poison()
    else:
        raise ValueError(f"escenario desconocido: {name}")

    return cd, atk


SCENARIOS = {
    "S0": "limpio",
    "S1": "solo Adv1 (FDI sensor)",
    "S2": "solo Adv2 (label-flip + modelo)",
    "S3": "dual coordinado (Adv1+Adv2)",
}
