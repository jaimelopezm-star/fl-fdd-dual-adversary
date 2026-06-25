"""
aggregators.py — Reglas de agregación robustas del servidor (Mecanismo 2, baselines).

Todas operan sobre el vector aplanado de los parámetros float de cada cliente (los buffers enteros,
p.ej. num_batches_tracked de BatchNorm, se toman del primer cliente, igual que en FedAvg) y devuelven
un state_dict listo para `model.load_state_dict`. Firma compatible con FedAvg:

    aggregate(state_dicts, weights) -> state_dict

Implementadas:
- autogm_aggregate : AutoGM (Li 2021, IEEE TII) — mediana geométrica auto-ponderada (Weiszfeld).
                     Defensa Byzantine-robust: el óptimo de la mediana geométrica resiste outliers.
- dwfa_aggregate   : D-WFA (Chen 2022, IEEE TIM) — Dynamic Weighted Federated Averaging.
                     Pondera dinámicamente a cada cliente según la divergencia de su actualización
                     respecto al centro. NOTA: el D-WFA original pondera por MMD entre distribuciones
                     de características (pensado para non-IID, NO seguridad); como el servidor no tiene
                     datos, aquí usamos la divergencia en el espacio de parámetros como proxy tratable.
                     Por diseño NO es Byzantine-robust → baseline que esperamos ver caer bajo ataque.

Referencia de implementación de AutoGM: github.com/lishenghui/AutoGM (su código es para Bosch/FEMNIST
imágenes; aquí se reimplementa la regla de agregación para nuestro 1D-CNN, no se ejecuta su repo).
"""
from __future__ import annotations
import copy
import torch


# --------------------------------------------------------------------------------------------------
# Utilidades de aplanado (solo tensores float; buffers enteros se conservan del primer cliente)
# --------------------------------------------------------------------------------------------------
def _float_keys(state_dict):
    return [k for k in state_dict if torch.is_floating_point(state_dict[k])]


def _flatten(state_dict, keys):
    return torch.cat([state_dict[k].reshape(-1).float() for k in keys])


def _unflatten(vec, ref_state, keys):
    """Reconstruye un state_dict a partir del vector plano `vec`, copiando de `ref_state`
    las llaves no-float (buffers enteros) tal cual."""
    out = copy.deepcopy(ref_state)
    i = 0
    for k in keys:
        n = ref_state[k].numel()
        out[k] = vec[i:i + n].reshape(ref_state[k].shape).to(ref_state[k].dtype)
        i += n
    return out


def _norm_weights(weights):
    w = torch.as_tensor(weights, dtype=torch.float32)
    return w / w.sum()


# --------------------------------------------------------------------------------------------------
# AutoGM — mediana geométrica auto-ponderada (Li 2021)
# --------------------------------------------------------------------------------------------------
def autogm_aggregate(state_dicts, weights, iters: int = 100, eps: float = 1e-5,
                     return_weights: bool = False):
    """Mediana geométrica ponderada vía Weiszfeld.

    Minimiza  sum_i  w_i * ||z - v_i||  sobre z. El reponderado a_i = w_i / ||z - v_i|| es la
    parte "auto-weighted": las actualizaciones lejanas (posibles envenenadas) pesan menos.

    Si `return_weights=True` devuelve (state_dict, a) donde `a` es el peso efectivo normalizado
    por cliente en la última iteración — diagnóstico de evasión: a_malicioso ≈ a_honesto significa
    que la defensa NO distingue al adversario.
    """
    keys = _float_keys(state_dicts[0])
    V = torch.stack([_flatten(sd, keys) for sd in state_dicts])   # (m, d)
    w = _norm_weights(weights)

    z = (w[:, None] * V).sum(dim=0)        # init: media ponderada (FedAvg)
    a = w.clone()
    for _ in range(iters):
        dist = torch.norm(V - z, dim=1) + 1e-8
        a = w / dist                        # peso auto-ajustado (IRLS)
        z_new = (a[:, None] * V).sum(dim=0) / a.sum()
        if torch.norm(z_new - z) < eps:
            z = z_new
            break
        z = z_new
    out = _unflatten(z, state_dicts[0], keys)
    if return_weights:
        return out, (a / a.sum()).tolist()
    return out


# --------------------------------------------------------------------------------------------------
# D-WFA — Dynamic Weighted Federated Averaging (Chen 2022)
# --------------------------------------------------------------------------------------------------
def dwfa_aggregate(state_dicts, weights, temp: float = 1.0, return_weights: bool = False):
    """Promedio con ponderación dinámica por divergencia respecto al centro.

    Proxy de la ponderación dinámica de D-WFA: w_i ∝ (n_i) · softmax(-d_i / mediana(d)), donde d_i es
    la distancia de la actualización del cliente i al centro ponderado. Maneja heterogeneidad pero
    NO está diseñado contra adversarios coordinados (baseline de contraste).

    Con `return_weights=True` devuelve (state_dict, w) — peso efectivo normalizado por cliente.
    """
    keys = _float_keys(state_dicts[0])
    V = torch.stack([_flatten(sd, keys) for sd in state_dicts])   # (m, d)
    w0 = _norm_weights(weights)

    center = (w0[:, None] * V).sum(dim=0)
    d = torch.norm(V - center, dim=1)
    dn = d / (d.median() + 1e-8)
    dyn = torch.softmax(-dn / temp, dim=0)     # cercanos al centro pesan más
    w = w0 * dyn
    w = w / w.sum()

    z = (w[:, None] * V).sum(dim=0)
    out = _unflatten(z, state_dicts[0], keys)
    if return_weights:
        return out, w.tolist()
    return out


# Registro nombre -> función, para elegir la defensa desde config/notebook.
AGGREGATORS = {
    "autogm": autogm_aggregate,
    "dwfa": dwfa_aggregate,
}
