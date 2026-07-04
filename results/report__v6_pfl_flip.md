# Reporte v5 — S3 (dual clasico) vs S4 (sensor compartido)

> Relabel de sensor (flip) sobre victimas honestas vs 5 defensas M2. ¿Alguna capa de agregacion (robusta global o personalizada) detiene un update envenenado de apariencia honesta? Contraste S3 vs S4-flip, alpha{0.3,0.5} x beta{0.4}, 10 semillas. Escala v2/v4/v5.

**Escala:** ventana=2048 stride=512 · clientes=30 · rondas=40 · theta=0.1 · celdas=400 · tiempo=7370.8s · device=cuda


ASR = fraccion de la falla objetivo clasificada como Normal (daño). En PFL se mide sobre el ENSEMBLE personalizado de honestos (incluidas las victimas del sensor). Colapsos = semillas con ASR>0.5. `a_mal`/`a_hon` = pesos de agregacion del GLOBAL (evasion).


## alpha = 0.3  (beta=0.4, clase objetivo=3)

| defensa | variante | ASR media | colapsos | recall_obj | a_mal | a_hon |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | S3 dual clasico | 0.10 | 1/10 | 0.70 | 0.048 | 0.024 |
| FedAvg | S4 sensor-solo | 0.03 | 0/10 | 0.90 | 0.048 | 0.024 |
| FedAvg | S4 dual completo | 0.21 | 0/10 | 0.36 | 0.048 | 0.024 |
| AutoGM | S3 dual clasico | 0.50 | 5/10 | 0.30 | 0.066 | 0.012 |
| AutoGM | S4 sensor-solo | 0.06 | 0/10 | 0.83 | 0.038 | 0.030 |
| AutoGM | S4 dual completo | 0.09 | 0/10 | 0.74 | 0.056 | 0.018 |
| AutoGM-full | S3 dual clasico | 0.60 | 6/10 | 0.05 | 0.043 | 0.027 |
| AutoGM-full | S4 sensor-solo | 0.33 | 0/10 | 0.00 | 0.012 | 0.048 |
| AutoGM-full | S4 dual completo | 0.19 | 0/10 | 0.42 | 0.033 | 0.033 |
| AutoGM-PFL | S3 dual clasico | 0.08 | 0/10 | 0.51 | 0.042 | 0.027 |
| AutoGM-PFL | S4 sensor-solo | 0.29 | 0/10 | 0.10 | 0.012 | 0.048 |
| AutoGM-PFL | S4 dual completo | 0.25 | 0/10 | 0.27 | 0.033 | 0.033 |
| D-WFA | S3 dual clasico | 0.44 | 4/10 | 0.41 | 0.060 | 0.016 |
| D-WFA | S4 sensor-solo | 0.04 | 0/10 | 0.88 | 0.039 | 0.030 |
| D-WFA | S4 dual completo | 0.09 | 0/10 | 0.72 | 0.055 | 0.019 |

## alpha = 0.5  (beta=0.4, clase objetivo=3)

| defensa | variante | ASR media | colapsos | recall_obj | a_mal | a_hon |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | S3 dual clasico | 0.00 | 0/10 | 1.00 | 0.043 | 0.027 |
| FedAvg | S4 sensor-solo | 0.00 | 0/10 | 1.00 | 0.043 | 0.027 |
| FedAvg | S4 dual completo | 0.01 | 0/10 | 0.97 | 0.043 | 0.027 |
| AutoGM | S3 dual clasico | 0.20 | 2/10 | 0.64 | 0.058 | 0.017 |
| AutoGM | S4 sensor-solo | 0.01 | 0/10 | 0.98 | 0.036 | 0.032 |
| AutoGM | S4 dual completo | 0.01 | 0/10 | 0.98 | 0.048 | 0.023 |
| AutoGM-full | S3 dual clasico | 0.12 | 1/10 | 0.89 | 0.047 | 0.024 |
| AutoGM-full | S4 sensor-solo | 0.17 | 0/10 | 0.48 | 0.017 | 0.044 |
| AutoGM-full | S4 dual completo | 0.02 | 0/10 | 0.93 | 0.031 | 0.035 |
| AutoGM-PFL | S3 dual clasico | 0.02 | 0/10 | 0.88 | 0.049 | 0.023 |
| AutoGM-PFL | S4 sensor-solo | 0.19 | 0/10 | 0.42 | 0.017 | 0.044 |
| AutoGM-PFL | S4 dual completo | 0.17 | 0/10 | 0.48 | 0.031 | 0.035 |
| D-WFA | S3 dual clasico | 0.16 | 2/10 | 0.76 | 0.051 | 0.022 |
| D-WFA | S4 sensor-solo | 0.00 | 0/10 | 0.99 | 0.037 | 0.031 |
| D-WFA | S4 dual completo | 0.01 | 0/10 | 0.97 | 0.048 | 0.024 |

## Lectura automatica — cruce Adv2(modelo,S3) vs Adv1(sensor,S4), alpha=0.3

Una defensa 'sobrevive' un vector si ASR<0.15. Interesa si EXISTE alguna que sobreviva LOS DOS.

| defensa | ASR S3 (modelo) | ASR S4 (sensor, peor) | sobrevive S3 | sobrevive S4 | sobrevive AMBOS |
|---|---:|---:|:--:|:--:|:--:|
| AutoGM | 0.50 | 0.09 | CAE | si | **NO** |
| AutoGM-full | 0.60 | 0.33 | CAE | CAE | **NO** |
| AutoGM-PFL | 0.08 | 0.29 | si | CAE | **NO** |
| D-WFA | 0.44 | 0.09 | CAE | si | **NO** |

- **VEREDICTO: NINGUNA defensa de servidor (M2) sobrevive los DOS vectores.** El adversario dual, al disponer de Adv2 (envenenamiento de modelo, S3) Y Adv1 (relabel de sensor, S4), derrota CUALQUIER eleccion de defensa de agregacion: la unica que resiste el modelo (PFL) cae ante el sensor, y las que resisten el sensor (AutoGM suave, D-WFA) caen ante el modelo. Es el gap DEMOSTRADO por ruptura (no por argumento) -> motiva el acoplamiento M1(borde)+M2.
- **AutoGM-full:** bajo S4 el recall de la falla objetivo cae a 0.00 (la victima honesta reetiquetada + exclusion dura η* que descarta a los portadores honestos de la falla) -> supresion de clase rara WEAPONIZADA por el sensor.
- **AutoGM-PFL:** bajo S4 el recall de la falla objetivo cae a 0.10 (la victima honesta reetiquetada + exclusion dura η* que descarta a los portadores honestos de la falla) -> supresion de clase rara WEAPONIZADA por el sensor.
