"""
partition.py — Partición non-IID por Dirichlet para simular "plantas" en FL.

Cada cliente (planta industrial) recibe una mezcla distinta de clases de falla, controlada por el
parámetro de concentración alpha del reparto Dirichlet:
    alpha -> 0    : muy non-IID (cada planta ve pocas clases)
    alpha grande  : casi IID (todas las plantas ven una mezcla parecida)

Uso:
    from partition import dirichlet_partition
    idx_por_cliente = dirichlet_partition(y, n_clients=5, alpha=0.5, seed=0)
"""
from __future__ import annotations
import numpy as np


def dirichlet_partition(y, n_clients: int, alpha: float, seed: int = 0,
                        min_size: int = 8):
    """Reparte los índices de y entre n_clients con heterogeneidad Dirichlet(alpha).

    Devuelve una lista de n_clients arrays de índices (enteros) sobre y.
    Reintenta el reparto hasta que el cliente más pequeño tenga >= min_size muestras.
    """
    y = np.asarray(y)
    n_classes = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    n = len(y)

    for _ in range(1000):
        client_idx = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.repeat(alpha, n_clients))
            # puntos de corte proporcionales
            cuts = (np.cumsum(props)[:-1] * len(idx_c)).astype(int)
            for i, part in enumerate(np.split(idx_c, cuts)):
                client_idx[i].extend(part.tolist())
        sizes = [len(c) for c in client_idx]
        if min(sizes) >= min_size:
            break
    else:  # pragma: no cover - sólo si alpha es absurdamente pequeño
        raise RuntimeError(
            f"No se logró un reparto con min_size={min_size} tras 1000 intentos "
            f"(alpha={alpha}, n={n}). Sube alpha o baja min_size/n_clients.")

    return [np.array(sorted(c), dtype=np.int64) for c in client_idx]


def partition_report(client_idx, y, n_classes=None):
    """Devuelve una matriz (n_clients x n_classes) con el conteo de cada clase por cliente."""
    y = np.asarray(y)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    table = np.zeros((len(client_idx), n_classes), dtype=int)
    for i, idx in enumerate(client_idx):
        vals, counts = np.unique(y[idx], return_counts=True)
        for v, ct in zip(vals, counts):
            table[i, int(v)] = ct
    return table
