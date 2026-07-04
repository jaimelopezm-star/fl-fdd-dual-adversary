# Reporte v5 — S3 (dual clasico) vs S4 (sensor compartido)

> Re-verificacion del hallazgo v6 (ningun M2 sobrevive Adv2-modelo S3 + Adv1-sensor S4-flip) bajo split TEMPORAL sin fuga. 5 defensas x 10 semillas, alpha{0.3,0.5} x beta{0.4}, S0+S3+S4.

**Escala:** ventana=2048 stride=512 · clientes=30 · rondas=40 · theta=0.1 · celdas=400 · tiempo=7278.1s · device=cuda


ASR = fraccion de la falla objetivo clasificada como Normal (daño). En PFL se mide sobre el ENSEMBLE personalizado de honestos (incluidas las victimas del sensor). Colapsos = semillas con ASR>0.5. `a_mal`/`a_hon` = pesos de agregacion del GLOBAL (evasion).


## alpha = 0.3  (beta=0.4, clase objetivo=3)

| defensa | variante | ASR media | colapsos | recall_obj | a_mal | a_hon |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | S3 dual clasico | 0.00 | 0/10 | 0.68 | 0.042 | 0.028 |
| FedAvg | S4 sensor-solo | 0.08 | 0/10 | 0.75 | 0.042 | 0.028 |
| FedAvg | S4 dual completo | 0.15 | 0/10 | 0.54 | 0.042 | 0.028 |
| AutoGM | S3 dual clasico | 0.35 | 3/10 | 0.48 | 0.057 | 0.018 |
| AutoGM | S4 sensor-solo | 0.13 | 0/10 | 0.62 | 0.033 | 0.033 |
| AutoGM | S4 dual completo | 0.13 | 0/10 | 0.60 | 0.050 | 0.022 |
| AutoGM-full | S3 dual clasico | 0.55 | 6/10 | 0.10 | 0.046 | 0.025 |
| AutoGM-full | S4 sensor-solo | 0.31 | 0/10 | 0.05 | 0.017 | 0.044 |
| AutoGM-full | S4 dual completo | 0.18 | 0/10 | 0.46 | 0.032 | 0.035 |
| AutoGM-PFL | S3 dual clasico | 0.07 | 0/10 | 0.50 | 0.046 | 0.025 |
| AutoGM-PFL | S4 sensor-solo | 0.29 | 0/10 | 0.12 | 0.017 | 0.044 |
| AutoGM-PFL | S4 dual completo | 0.25 | 0/10 | 0.26 | 0.032 | 0.034 |
| D-WFA | S3 dual clasico | 0.23 | 2/10 | 0.50 | 0.052 | 0.021 |
| D-WFA | S4 sensor-solo | 0.13 | 0/10 | 0.61 | 0.034 | 0.033 |
| D-WFA | S4 dual completo | 0.13 | 0/10 | 0.60 | 0.048 | 0.023 |

## alpha = 0.5  (beta=0.4, clase objetivo=3)

| defensa | variante | ASR media | colapsos | recall_obj | a_mal | a_hon |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | S3 dual clasico | 0.00 | 0/10 | 0.88 | 0.044 | 0.026 |
| FedAvg | S4 sensor-solo | 0.04 | 0/10 | 0.89 | 0.044 | 0.026 |
| FedAvg | S4 dual completo | 0.09 | 0/10 | 0.72 | 0.044 | 0.026 |
| AutoGM | S3 dual clasico | 0.30 | 3/10 | 0.60 | 0.060 | 0.016 |
| AutoGM | S4 sensor-solo | 0.03 | 0/10 | 0.90 | 0.037 | 0.031 |
| AutoGM | S4 dual completo | 0.05 | 0/10 | 0.85 | 0.049 | 0.023 |
| AutoGM-full | S3 dual clasico | 0.07 | 0/10 | 0.74 | 0.042 | 0.027 |
| AutoGM-full | S4 sensor-solo | 0.18 | 0/10 | 0.47 | 0.018 | 0.044 |
| AutoGM-full | S4 dual completo | 0.01 | 0/10 | 0.97 | 0.030 | 0.035 |
| AutoGM-PFL | S3 dual clasico | 0.02 | 0/10 | 0.77 | 0.039 | 0.030 |
| AutoGM-PFL | S4 sensor-solo | 0.20 | 0/10 | 0.39 | 0.018 | 0.044 |
| AutoGM-PFL | S4 dual completo | 0.17 | 0/10 | 0.49 | 0.030 | 0.035 |
| D-WFA | S3 dual clasico | 0.15 | 2/10 | 0.59 | 0.052 | 0.021 |
| D-WFA | S4 sensor-solo | 0.02 | 0/10 | 0.93 | 0.038 | 0.030 |
| D-WFA | S4 dual completo | 0.07 | 0/10 | 0.80 | 0.049 | 0.023 |

## Lectura automatica — cruce Adv2(modelo,S3) vs Adv1(sensor,S4), alpha=0.3

Una defensa 'sobrevive' un vector si ASR<0.15. Interesa si EXISTE alguna que sobreviva LOS DOS.

| defensa | ASR S3 (modelo) | ASR S4 (sensor, peor) | sobrevive S3 | sobrevive S4 | sobrevive AMBOS |
|---|---:|---:|:--:|:--:|:--:|
| AutoGM | 0.35 | 0.13 | CAE | si | **NO** |
| AutoGM-full | 0.55 | 0.31 | CAE | CAE | **NO** |
| AutoGM-PFL | 0.07 | 0.29 | si | CAE | **NO** |
| D-WFA | 0.23 | 0.13 | CAE | si | **NO** |

- **VEREDICTO: NINGUNA defensa de servidor (M2) sobrevive los DOS vectores.** El adversario dual, al disponer de Adv2 (envenenamiento de modelo, S3) Y Adv1 (relabel de sensor, S4), derrota CUALQUIER eleccion de defensa de agregacion: la unica que resiste el modelo (PFL) cae ante el sensor, y las que resisten el sensor (AutoGM suave, D-WFA) caen ante el modelo. Es el gap DEMOSTRADO por ruptura (no por argumento) -> motiva el acoplamiento M1(borde)+M2.
- **AutoGM-full:** bajo S4 el recall de la falla objetivo cae a 0.05 (la victima honesta reetiquetada + exclusion dura η* que descarta a los portadores honestos de la falla) -> supresion de clase rara WEAPONIZADA por el sensor.
- **AutoGM-PFL:** bajo S4 el recall de la falla objetivo cae a 0.12 (la victima honesta reetiquetada + exclusion dura η* que descarta a los portadores honestos de la falla) -> supresion de clase rara WEAPONIZADA por el sensor.
