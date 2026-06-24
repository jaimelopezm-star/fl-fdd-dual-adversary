"""Paquete de defensas: reglas de agregación robustas del lado servidor (Mecanismo 2).

Todas las funciones de agregación comparten la firma de FedAvg:
    aggregate(state_dicts, weights) -> state_dict
para poder enchufarse directamente en `federated.fedavg.run_fl(aggregate=...)`.
"""
from .aggregators import autogm_aggregate, dwfa_aggregate, AGGREGATORS  # noqa: F401
