# Reporte F4 — `v1_local_small`

> CWRU sin solape (1537 ventanas), 10 clientes, 20 rondas. Escala pequeña: limitada por el tamaño del dataset. Sirve de referencia para contrastar contra v2_scaled.

**Escala:** ventana=2048 stride=2048 · clientes=10 · rondas=20 · celdas=52 · tiempo=708.9s · device=cpu


## 1. AutoGM — evasión y daño (S3, por α y β)

`a_mal` = peso que AutoGM da al malicioso; ≈`a_hon` (evade) o ~0 (filtrado). ASR = daño.

| α | β | ASR overt | ASR sigiloso | a_mal overt | a_mal sigiloso | a_hon |
|---|---|---|---|---|---|---|
| 0.3 | 0.2 | 0.00 | 0.01 | 0.000 | 0.105 | 0.099 |
| 0.3 | 0.4 | 0.94 | 1.00 | 0.162 | 0.247 | 0.002 |
| 0.5 | 0.2 | 0.00 | 0.00 | 0.001 | 0.179 | 0.080 |
| 0.5 | 0.4 | 0.99 | 1.00 | 0.098 | 0.210 | 0.027 |

## 2. ASR del ataque sigiloso (S3) por defensa y β

| defensa | β=0.2 | β=0.4 |
|---|---|---|
| AutoGM | 0.01 | 1.00 |
| FedAvg | 0.00 | 1.00 |

## 3. Supresión de clase rara — recall clase objetivo (S0 LIMPIO, concentrado)

Si AutoGM << FedAvg con pocos dueños → la agregación robusta sacrifica la falla rara.

| defensa | 2 dueños | 4 dueños |
|---|---|---|
| AutoGM | 0.00 | 1.00 |
| FedAvg | 0.91 | 1.00 |

## 4. Lectura automática

- En minoría (β=0.2): ASR sigiloso ≈ **0.01** (SIN daño → AutoGM aguanta); evasión SÍ (a_mal 0.142 vs a_hon 0.089).
- Clase rara (2 dueños, limpio): AutoGM **0.00** vs FedAvg **0.91** → SUPRESIÓN confirmada.
