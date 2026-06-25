"""
fedavg.py — Loop de Federated Learning propio (FedAvg) en PyTorch.

Diseño modular pensado para enganchar después ataques y defensas:
- local_train(): entrenamiento local de un cliente (aquí se inyectará el Adv2 más adelante).
- aggregate: función de agregación intercambiable (FedAvg ahora; AutoGM / D-WFA en F2).
- run_fl(): orquesta rondas (selección de clientes -> entrenamiento local -> agregación -> eval).

F1 sólo usa FedAvg y clientes honestos (escenario S0 limpio).
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def local_train(model, loader, epochs, lr, device):
    """Entrena el modelo local in-place y devuelve su state_dict (en CPU)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def fedavg_aggregate(state_dicts, weights, return_weights: bool = False):
    """Promedio ponderado de state_dicts (FedAvg). Pesos = nº de muestras por cliente.

    Con `return_weights=True` devuelve (state_dict, w) — peso normalizado por cliente, para que la
    firma sea uniforme con AutoGM/D-WFA (sin defensa: el peso es simplemente n_i / sum n)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    agg = copy.deepcopy(state_dicts[0])
    for k in agg:
        if torch.is_floating_point(agg[k]):
            stacked = torch.stack(
                [sd[k].float() * float(w[i]) for i, sd in enumerate(state_dicts)], dim=0)
            agg[k] = stacked.sum(dim=0).to(state_dicts[0][k].dtype)
        else:
            # buffers enteros (p.ej. num_batches_tracked): se toma el del primer cliente
            agg[k] = state_dicts[0][k].clone()
    if return_weights:
        return agg, w.tolist()
    return agg


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int = 4):
    """Devuelve acc, matriz de confusión y recall por clase sobre el loader."""
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(1).cpu().numpy()
        yb = yb.numpy()
        for t, p in zip(yb, pred):
            cm[t, p] += 1
    total = cm.sum()
    acc = float(np.trace(cm)) / total if total else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(np.diag(cm), cm.sum(axis=1),
                           out=np.zeros(n_classes), where=cm.sum(axis=1) > 0)
    return {"acc": acc, "cm": cm, "recall": recall}


def _delta_norm(sd, gstate):
    """Norma L2 del update (sd - gstate) sobre tensores float — proxy de 'magnitud' del cliente."""
    sq = 0.0
    for k, v in sd.items():
        if torch.is_floating_point(v):
            sq += float(torch.sum((v.float() - gstate[k].float()) ** 2))
    return sq ** 0.5


def run_fl(model_fn, client_data, test_data, *, rounds=20, local_epochs=2, lr=1e-3,
           batch_size=64, client_frac=1.0, aggregate=fedavg_aggregate,
           malicious_ids=None, model_attack_fn=None,
           device="cpu", seed=0, n_classes=4, verbose=True):
    """Ejecuta FL y devuelve (modelo_global, history).

    Parámetros:
      model_fn        : callable -> nn.Module nuevo (cada cliente y el global lo reinstancian).
      client_data     : lista [(X_c, y_c), ...] por cliente (ya envenenado a nivel datos si aplica).
      test_data       : (X_test, y_test) global para evaluar el modelo agregado.
      aggregate       : función de agregación (FedAvg por defecto; intercambiable por AutoGM/D-WFA).
                        Debe aceptar `return_weights=True` y devolver (state_dict, pesos_por_cliente).
      malicious_ids   : ids de clientes comprometidos (para el ataque de modelo Adv2).
      model_attack_fn : fn(state_dict, rng, ctx)->state_dict aplicada al update de un cliente
                        malicioso DESPUÉS de su entrenamiento local (envenenamiento del modelo).
                        `ctx` = {gstate, ref_norm} (estado global de la ronda y norma honesta de
                        referencia) para ataques sigilosos con restricción de norma. None = sin ataque.
    history: lista de dicts {round, acc, recall, selected, agg_weights, mal_mask} por ronda.
             `agg_weights` es el peso efectivo que la defensa asignó a cada cliente seleccionado y
             `mal_mask` marca cuáles eran maliciosos -> permite graficar evasión (a_mal vs a_honesto).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    malicious_ids = set(malicious_ids) if malicious_ids is not None else set()

    global_model = model_fn().to(device)
    test_loader = make_loader(*test_data, batch_size=batch_size, shuffle=False)
    n_clients = len(client_data)
    history = []

    for rnd in range(1, rounds + 1):
        m = max(1, int(round(client_frac * n_clients)))
        selected = rng.choice(n_clients, size=m, replace=False)
        gstate = {k: v.cpu() for k, v in global_model.state_dict().items()}

        # --- Pasada 1: entrenamiento local honesto de todos los seleccionados ---
        states, weights, mal_mask = [], [], []
        for cid in selected:
            Xc, yc = client_data[cid]
            local = model_fn().to(device)
            local.load_state_dict(gstate)
            loader = make_loader(Xc, yc, batch_size=batch_size, shuffle=True)
            sd = local_train(local, loader, local_epochs, lr, device)
            states.append(sd)
            weights.append(len(yc))
            mal_mask.append(cid in malicious_ids)

        # Norma honesta de referencia (mediana de ||update|| de los clientes NO maliciosos):
        # es el "presupuesto de magnitud" que un update sigiloso no debe exceder para evadir la defensa.
        honest_norms = [_delta_norm(states[i], gstate) for i in range(len(selected)) if not mal_mask[i]]
        ref_norm = float(np.median(honest_norms)) if honest_norms else None

        # --- Pasada 2: envenenamiento de modelo (Adv2) con contexto, solo a maliciosos ---
        if model_attack_fn is not None:
            ctx = {"gstate": gstate, "ref_norm": ref_norm}
            for i in range(len(selected)):
                if mal_mask[i]:
                    states[i] = model_attack_fn(states[i], rng, ctx)

        agg_out = aggregate(states, weights, return_weights=True)
        new_state, agg_weights = agg_out
        global_model.load_state_dict(new_state)
        ev = evaluate(global_model, test_loader, device, n_classes=n_classes)
        history.append({"round": rnd, "acc": ev["acc"], "recall": ev["recall"].tolist(),
                        "selected": [int(c) for c in selected],
                        "agg_weights": agg_weights, "mal_mask": mal_mask})
        if verbose:
            print(f"  ronda {rnd:2d}/{rounds}  acc={ev['acc']:.4f}")

    return global_model, history
