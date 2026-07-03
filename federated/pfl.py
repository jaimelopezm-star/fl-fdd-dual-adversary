"""
pfl.py — Variante PERSONALIZADA de AutoGM (AutoGM_PFL, estilo Ditto) sobre agregacion robusta.

Implementa la 2a variante de despliegue de AutoGM (Li 2023, Sec. V.C, Eq. 13): ademas del modelo global
agregado con la regla robusta (AutoGM / AutoGM-full), cada cliente mantiene un modelo PERSONALIZADO v_k
regularizado hacia el global con el termino Ditto  theta*(v_k - w_global)  en el gradiente local. Es la
variante pensada para heterogeneidad non-IID. Referencia de codigo oficial:
replicacion_autogm/AutoGM/trainers/{autogmpfl,ditto}.py (ditto.py:40-43).

NOTA DE SEMANTICA (importante):
  En PFL el objeto que "usa" cada planta es su modelo PERSONALIZADO, no el global. Por eso:
    - La senal de SEGURIDAD (evasion: a_malicioso vs a_honesto) se mide sobre la agregacion robusta del
      GLOBAL, que en PFL sigue estando bien definida (el servidor agrega el global con `aggregate`).
    - La metrica de DETECCION (ASR/DR/FAR) se evalua sobre los modelos personalizados por cliente. Si se
      pasa `client_test_data` (particion de test por cliente), se evalua cada v_k sobre su propia
      distribucion (lo correcto en non-IID). Si no, se usa el test global como diagnostico aproximado.
  La definicion final de ASR-personalizado (particion de test, ponderacion, trato de maliciosos) es una
  decision de diseno del experimento — ver 03_experimento/mecanismos_defensa_base_y_faltantes.md, Sec. 6.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn

from .fedavg import make_loader, evaluate, fedavg_aggregate, _delta_norm


def ditto_local_train(global_state, local_state, model_fn, loader, epochs, lr, theta, device):
    """Entrena (a) una copia del modelo global -> update para la agregacion robusta del global; y
    (b) el modelo personalizado v_k con la regularizacion Ditto  grad += theta*(v_k - w_global).

    Devuelve (global_update_state_dict, new_local_state_dict), ambos en CPU."""
    crit = nn.CrossEntropyLoss()

    # (a) update del global (como un cliente FedAvg normal)
    gnet = model_fn().to(device)
    gnet.load_state_dict(global_state)
    gnet.train()
    opt = torch.optim.Adam(gnet.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(gnet(xb), yb).backward()
            opt.step()
    global_update = {k: v.detach().cpu() for k, v in gnet.state_dict().items()}

    # (b) modelo personalizado v_k con regularizacion Ditto hacia el global (referencia FIJA esta ronda)
    gref = {k: v.to(device) for k, v in global_state.items()}
    lnet = model_fn().to(device)
    lnet.load_state_dict(local_state)
    lnet.train()
    opt = torch.optim.Adam(lnet.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(lnet(xb), yb).backward()
            if theta > 0:
                with torch.no_grad():
                    for name, p in lnet.named_parameters():
                        if p.grad is not None:
                            p.grad.add_(theta * (p.detach() - gref[name]))   # Eq. 13 (Ditto)
            opt.step()
    new_local = {k: v.detach().cpu() for k, v in lnet.state_dict().items()}
    return global_update, new_local


@torch.no_grad()
def _eval_personalized(model_fn, local_states, selected, mal_mask, test_loader, client_test_data,
                       device, n_classes, batch_size):
    """Suma las matrices de confusion de los modelos personalizados de los clientes HONESTOS
    seleccionados (los 'victimas'). Cada v_k se evalua sobre su propio test si `client_test_data` existe,
    o sobre el test global en caso contrario."""
    net = model_fn().to(device)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    used = 0
    for i, cid in enumerate(selected):
        if mal_mask[i]:
            continue
        net.load_state_dict(local_states[cid])
        loader = (make_loader(*client_test_data[cid], batch_size=batch_size, shuffle=False)
                  if client_test_data is not None else test_loader)
        cm += evaluate(net, loader, device, n_classes)["cm"]
        used += 1
    if used == 0:  # todos maliciosos: evalua el ensemble completo como fallback
        for cid in selected:
            net.load_state_dict(local_states[cid])
            cm += evaluate(net, test_loader, device, n_classes)["cm"]
    return cm


def run_fl_pfl(model_fn, client_data, test_data, *, rounds=20, local_epochs=2, lr=1e-3,
               batch_size=64, client_frac=1.0, aggregate=fedavg_aggregate, theta=0.1,
               malicious_ids=None, model_attack_fn=None, device="cpu", seed=0, n_classes=4,
               client_test_data=None, verbose=True):
    """FL con personalizacion Ditto (AutoGM_PFL). Devuelve (global_model, local_states, history).

    history por ronda: {round, acc_global, cm_personalized, selected, agg_weights, mal_mask}. Los campos
    agg_weights/mal_mask permiten el diagnostico de evasion sobre la agregacion robusta del GLOBAL, igual
    que en run_fl; cm_personalized da la deteccion del ensemble personalizado (ver nota de semantica)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    malicious_ids = set(malicious_ids) if malicious_ids is not None else set()

    global_model = model_fn().to(device)
    n_clients = len(client_data)
    g0 = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
    local_states = [copy.deepcopy(g0) for _ in range(n_clients)]   # v_k init = global
    test_loader = make_loader(*test_data, batch_size=batch_size, shuffle=False)
    history = []

    for rnd in range(1, rounds + 1):
        m = max(1, int(round(client_frac * n_clients)))
        selected = rng.choice(n_clients, size=m, replace=False)
        gstate = {k: v.cpu() for k, v in global_model.state_dict().items()}

        states, weights, mal_mask = [], [], []
        for cid in selected:
            Xc, yc = client_data[cid]
            loader = make_loader(Xc, yc, batch_size=batch_size, shuffle=True)
            gupd, newloc = ditto_local_train(gstate, local_states[cid], model_fn, loader,
                                             local_epochs, lr, theta, device)
            local_states[cid] = newloc
            states.append(gupd)
            weights.append(len(yc))
            mal_mask.append(cid in malicious_ids)

        honest_norms = [_delta_norm(states[i], gstate) for i in range(len(selected)) if not mal_mask[i]]
        ref_norm = float(np.median(honest_norms)) if honest_norms else None

        if model_attack_fn is not None:                 # Adv2 sobre el update del GLOBAL
            ctx = {"gstate": gstate, "ref_norm": ref_norm}
            for i in range(len(selected)):
                if mal_mask[i]:
                    states[i] = model_attack_fn(states[i], rng, ctx)

        new_state, agg_weights = aggregate(states, weights, return_weights=True)
        global_model.load_state_dict(new_state)

        pers_cm = _eval_personalized(model_fn, local_states, selected, mal_mask,
                                     test_loader, client_test_data, device, n_classes, batch_size)
        gev = evaluate(global_model, test_loader, device, n_classes)
        history.append({"round": rnd, "acc_global": gev["acc"],
                        "cm_personalized": pers_cm.tolist(),
                        "selected": [int(c) for c in selected],
                        "agg_weights": agg_weights, "mal_mask": mal_mask})
        if verbose:
            print(f"  ronda {rnd:2d}/{rounds}  acc_global={gev['acc']:.4f}")

    return global_model, local_states, history
