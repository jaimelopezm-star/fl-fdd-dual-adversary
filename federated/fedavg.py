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


def fedavg_aggregate(state_dicts, weights):
    """Promedio ponderado de state_dicts (FedAvg). Pesos = nº de muestras por cliente."""
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
      malicious_ids   : ids de clientes comprometidos (para el ataque de modelo Adv2).
      model_attack_fn : fn(state_dict, rng)->state_dict aplicada al update de un cliente malicioso
                        DESPUÉS de su entrenamiento local (envenenamiento del modelo). None = sin ataque.
    history: lista de dicts {round, acc, recall} por ronda.
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

        states, weights = [], []
        for cid in selected:
            Xc, yc = client_data[cid]
            local = model_fn().to(device)
            local.load_state_dict(gstate)
            loader = make_loader(Xc, yc, batch_size=batch_size, shuffle=True)
            sd = local_train(local, loader, local_epochs, lr, device)
            if model_attack_fn is not None and cid in malicious_ids:
                sd = model_attack_fn(sd, rng)   # Adv2: envenena el update del cliente malicioso
            states.append(sd)
            weights.append(len(yc))

        global_model.load_state_dict(aggregate(states, weights))
        ev = evaluate(global_model, test_loader, device, n_classes=n_classes)
        history.append({"round": rnd, "acc": ev["acc"], "recall": ev["recall"].tolist()})
        if verbose:
            print(f"  ronda {rnd:2d}/{rounds}  acc={ev['acc']:.4f}")

    return global_model, history
