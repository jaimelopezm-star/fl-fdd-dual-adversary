# Reporte F4 — `v3_seeds`

> Confirmación con 5 semillas del quiebre de la agregación robusta por el ataque sigiloso a α=0.3,β=0.4 (vs overt filtrado y vs FedAvg sin daño). Escala v2 (30 clientes, 40 rondas, solape). Enfocado en S0+S3 para correr rápido (~150 celdas).

**Escala:** ventana=2048 stride=512 · clientes=30 · rondas=40 · celdas=150 · tiempo=5374.1s · device=cpu


## 1. AutoGM — evasión y daño (S3, por α y β)

`a_mal` = peso que AutoGM da al malicioso; ≈`a_hon` (evade) o ~0 (filtrado). ASR = daño.

| α | β | ASR overt | ASR sigiloso | a_mal overt | a_mal sigiloso | a_hon |
|---|---|---|---|---|---|---|
| 0.3 | 0.3 | 0.00 | 0.00 | 0.001 | 0.063 | 0.021 |
| 0.3 | 0.4 | 0.00 | 0.60 | 0.001 | 0.068 | 0.010 |
| 0.5 | 0.3 | 0.00 | 0.00 | 0.001 | 0.054 | 0.025 |
| 0.5 | 0.4 | 0.00 | 0.00 | 0.001 | 0.056 | 0.018 |

## 2. ASR del ataque sigiloso (S3) por defensa y β

| defensa | β=0.3 | β=0.4 |
|---|---|---|
| AutoGM | 0.00 | 0.30 |
| D-WFA | 0.00 | 0.21 |
| FedAvg | 0.00 | 0.00 |

## 4. Lectura automática

- En minoría (β=0.3): ASR sigiloso ≈ **0.00** (SIN daño → AutoGM aguanta); evasión SÍ (a_mal 0.058 vs a_hon 0.023).
