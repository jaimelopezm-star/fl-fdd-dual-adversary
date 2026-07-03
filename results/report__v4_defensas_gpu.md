# Reporte F4 — `v4_defensas_gpu`

> Comparacion pareada de 5 defensas (FedAvg, AutoGM suave, AutoGM-full fiel, AutoGM-PFL, D-WFA) en el regimen titular alpha{0.3,0.5} x beta{0.3,0.4}, S0+S3, overt+sigiloso, 10 semillas. Escala v2/v3.

**Escala:** ventana=2048 stride=512 · clientes=30 · rondas=40 · celdas=500 · tiempo=8758.8s · device=cuda


## 1. AutoGM — evasión y daño (S3, por α y β)

`a_mal` = peso que AutoGM da al malicioso; ≈`a_hon` (evade) o ~0 (filtrado). ASR = daño.

| α | β | ASR overt | ASR sigiloso | a_mal overt | a_mal sigiloso | a_hon |
|---|---|---|---|---|---|---|
| 0.3 | 0.3 | 0.00 | 0.10 | 0.001 | 0.062 | 0.021 |
| 0.3 | 0.4 | 0.00 | 0.50 | 0.001 | 0.066 | 0.012 |
| 0.5 | 0.3 | 0.00 | 0.00 | 0.001 | 0.055 | 0.024 |
| 0.5 | 0.4 | 0.00 | 0.20 | 0.001 | 0.057 | 0.017 |

## 2. ASR del ataque sigiloso (S3) por defensa y β

| defensa | β=0.3 | β=0.4 |
|---|---|---|
| AutoGM | 0.05 | 0.35 |
| AutoGM-PFL | 0.03 | 0.05 |
| AutoGM-full | 0.14 | 0.37 |
| D-WFA | 0.05 | 0.30 |
| FedAvg | 0.00 | 0.05 |

## 4. Lectura automática

- En minoría (β=0.3): ASR sigiloso ≈ **0.05** (SIN daño → AutoGM aguanta); evasión SÍ (a_mal 0.059 vs a_hon 0.023).
