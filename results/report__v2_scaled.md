# Reporte F4 — `v2_scaled`

> CWRU con solape (stride 512), 30 clientes, 40 rondas, 2 semillas, 3 defensas. Pensado para GPU. NOTA de validez: el solape puede solapar ventanas entre train/test (fuga optimista); el split es estratificado por clase. Reportar este caveat. Para rigor máximo, separar por segmento de señal.

**Escala:** ventana=2048 stride=512 · clientes=30 · rondas=40 · celdas=288 · tiempo=10094.9s · device=cpu


## 1. AutoGM — evasión y daño (S3, por α y β)

`a_mal` = peso que AutoGM da al malicioso; ≈`a_hon` (evade) o ~0 (filtrado). ASR = daño.

| α | β | ASR overt | ASR sigiloso | a_mal overt | a_mal sigiloso | a_hon |
|---|---|---|---|---|---|---|
| 0.3 | 0.1 | 0.00 | 0.00 | 0.001 | 0.084 | 0.028 |
| 0.3 | 0.2 | 0.00 | 0.00 | 0.001 | 0.072 | 0.024 |
| 0.3 | 0.3 | 0.00 | 0.01 | 0.000 | 0.070 | 0.018 |
| 0.3 | 0.4 | 0.00 | 1.00 | 0.001 | 0.069 | 0.009 |
| 0.5 | 0.1 | 0.00 | 0.00 | 0.001 | 0.070 | 0.029 |
| 0.5 | 0.2 | 0.00 | 0.00 | 0.001 | 0.055 | 0.028 |
| 0.5 | 0.3 | 0.00 | 0.00 | 0.001 | 0.058 | 0.023 |
| 0.5 | 0.4 | 0.00 | 0.00 | 0.001 | 0.060 | 0.015 |

## 2. ASR del ataque sigiloso (S3) por defensa y β

| defensa | β=0.1 | β=0.2 | β=0.3 | β=0.4 |
|---|---|---|---|---|
| AutoGM | 0.00 | 0.00 | 0.00 | 0.50 |
| D-WFA | 0.00 | 0.00 | 0.00 | 0.27 |
| FedAvg | 0.00 | 0.00 | 0.00 | 0.00 |

## 3. Supresión de clase rara — recall clase objetivo (S0 LIMPIO, concentrado)

Si AutoGM << FedAvg con pocos dueños → la agregación robusta sacrifica la falla rara.

| defensa | 2 dueños | 4 dueños | 6 dueños |
|---|---|---|---|
| AutoGM | 0.80 | 0.50 | 1.00 |
| D-WFA | 0.50 | 0.59 | 1.00 |
| FedAvg | 1.00 | 1.00 | 1.00 |

## 4. Lectura automática

- En minoría (β=0.1): ASR sigiloso ≈ **0.00** (SIN daño → AutoGM aguanta); evasión SÍ (a_mal 0.077 vs a_hon 0.028).
- Clase rara (2 dueños, limpio): AutoGM **0.80** vs FedAvg **1.00** → sin supresión clara.
