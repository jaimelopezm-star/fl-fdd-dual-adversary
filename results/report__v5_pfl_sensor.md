# Reporte v5 — S3 (dual clasico) vs S4 (sensor compartido)

> ¿El sensor compartido (S4, Adv1 sobre clientes honestos) rompe AutoGM-PFL donde el dual clasico (S3) no? Decomposicion sensor-solo vs dual completo, 5 defensas, alpha{0.3,0.5} x beta{0.4}, 10 semillas. Escala v2/v4.

**Escala:** ventana=2048 stride=512 · clientes=30 · rondas=40 · theta=0.1 · celdas=400 · tiempo=7096.2s · device=cuda


ASR = fraccion de la falla objetivo clasificada como Normal (daño). En PFL se mide sobre el ENSEMBLE personalizado de honestos (incluidas las victimas del sensor). Colapsos = semillas con ASR>0.5. `a_mal`/`a_hon` = pesos de agregacion del GLOBAL (evasion).


## alpha = 0.3  (beta=0.4, clase objetivo=3)

| defensa | variante | ASR media | colapsos | recall_obj | a_mal | a_hon |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | S3 dual clasico | 0.10 | 1/10 | 0.70 | 0.048 | 0.024 |
| FedAvg | S4 sensor-solo | 0.00 | 0/10 | 1.00 | 0.048 | 0.024 |
| FedAvg | S4 dual completo | 0.00 | 0/10 | 0.72 | 0.048 | 0.024 |
| AutoGM | S3 dual clasico | 0.50 | 5/10 | 0.30 | 0.066 | 0.012 |
| AutoGM | S4 sensor-solo | 0.00 | 0/10 | 1.00 | 0.035 | 0.032 |
| AutoGM | S4 dual completo | 0.00 | 0/10 | 1.00 | 0.056 | 0.018 |
| AutoGM-full | S3 dual clasico | 0.60 | 6/10 | 0.05 | 0.043 | 0.027 |
| AutoGM-full | S4 sensor-solo | 0.00 | 0/10 | 0.29 | 0.011 | 0.048 |
| AutoGM-full | S4 dual completo | 0.00 | 0/10 | 0.98 | 0.033 | 0.033 |
| AutoGM-PFL | S3 dual clasico | 0.09 | 0/10 | 0.52 | 0.042 | 0.027 |
| AutoGM-PFL | S4 sensor-solo | 0.02 | 0/10 | 0.28 | 0.011 | 0.048 |
| AutoGM-PFL | S4 dual completo | 0.01 | 0/10 | 0.78 | 0.033 | 0.033 |
| D-WFA | S3 dual clasico | 0.44 | 4/10 | 0.42 | 0.060 | 0.016 |
| D-WFA | S4 sensor-solo | 0.00 | 0/10 | 1.00 | 0.037 | 0.031 |
| D-WFA | S4 dual completo | 0.00 | 0/10 | 0.99 | 0.055 | 0.019 |

## alpha = 0.5  (beta=0.4, clase objetivo=3)

| defensa | variante | ASR media | colapsos | recall_obj | a_mal | a_hon |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | S3 dual clasico | 0.00 | 0/10 | 1.00 | 0.043 | 0.027 |
| FedAvg | S4 sensor-solo | 0.00 | 0/10 | 1.00 | 0.043 | 0.027 |
| FedAvg | S4 dual completo | 0.00 | 0/10 | 1.00 | 0.043 | 0.027 |
| AutoGM | S3 dual clasico | 0.20 | 2/10 | 0.63 | 0.058 | 0.017 |
| AutoGM | S4 sensor-solo | 0.00 | 0/10 | 1.00 | 0.032 | 0.034 |
| AutoGM | S4 dual completo | 0.00 | 0/10 | 1.00 | 0.048 | 0.024 |
| AutoGM-full | S3 dual clasico | 0.13 | 1/10 | 0.89 | 0.045 | 0.025 |
| AutoGM-full | S4 sensor-solo | 0.00 | 0/10 | 1.00 | 0.016 | 0.045 |
| AutoGM-full | S4 dual completo | 0.00 | 0/10 | 1.00 | 0.031 | 0.035 |
| AutoGM-PFL | S3 dual clasico | 0.02 | 0/10 | 0.88 | 0.046 | 0.025 |
| AutoGM-PFL | S4 sensor-solo | 0.00 | 0/10 | 0.96 | 0.016 | 0.045 |
| AutoGM-PFL | S4 dual completo | 0.00 | 0/10 | 0.99 | 0.031 | 0.035 |
| D-WFA | S3 dual clasico | 0.16 | 2/10 | 0.71 | 0.051 | 0.022 |
| D-WFA | S4 sensor-solo | 0.00 | 0/10 | 1.00 | 0.035 | 0.032 |
| D-WFA | S4 dual completo | 0.00 | 0/10 | 1.00 | 0.047 | 0.024 |

## Lectura automatica — ¿el sensor compartido rompe PFL?

- **AutoGM-PFL (alpha=0.3):** S3 dual clasico ASR=0.09 (0/10 colapsos) vs **S4 dual completo ASR=0.01 (0/10)**; S4 sensor-solo ASR=0.02 (0/10).
- **VEREDICTO: PFL tambien aguanta S4** (Δ ASR = -0.07). El sensor compartido no basta con este `sensor_frac`; replantear (subir fraccion / atacar test por cliente).
- **AutoGM (modelo global, alpha=0.3):** S3 ASR=0.50 (5/10) vs S4 ASR=0.00 (0/10) -> el modelo global PROTEGE a las victimas del sensor (aprende la falla de los honestos no-victima).
- **AutoGM-full (modelo global, alpha=0.3):** S3 ASR=0.60 (6/10) vs S4 ASR=0.00 (0/10) -> el modelo global PROTEGE a las victimas del sensor (aprende la falla de los honestos no-victima).
