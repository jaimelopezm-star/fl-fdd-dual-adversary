# Reporte v4 — Contraste de defensas (RTX 4090, 10 semillas)

> Comparación **pareada** de 5 defensas en el régimen titular, con el AutoGM **fiel** (λ + recorte η\*) y la
> variante **personalizada** AutoGM-PFL recién integradas. 500 celdas, RTX 4090, `cuda jobs=8`, 2 h 26 min.
> Config: `config/v4_defensas_gpu.yaml`. Datos: `F4_grid__v4_defensas_gpu.json` (+ `.jsonl` checkpoint).
>
> CAVEAT de protocolo: mismo split por ventanas con solape (stride 512) que v2/v3 → posible fuga optimista.
> La comparación **entre defensas** es válida (todas comparten split y semillas); la generalización absoluta
> queda para la corrida con split bloqueado (ver `03_experimento/evaluacion_del_experimento.md`).

---

## 1. Resultado titular — α=0.3, β=0.4, S3 (dual coordinado)

### Ataque SIGILOSO (`constrained`)

| Defensa | ASR por semilla | media | colapsos (ASR>0.5) | a_mal | a_hon |
|---|---|---:|---:|---:|---:|
| FedAvg | [0.01,0,0,0,0,0,0,0,0,1.0] | 0.10 | 1/10 | 0.048 | 0.024 |
| AutoGM (suave, legacy) | [1,1,1,0,0,0,0,0,1,1] | 0.50 | **5/10** | 0.066 | 0.012 |
| **AutoGM-full (FIEL, λ+η\*)** | [1,1,0.16,0,0,0.05,1,0.84,1,1] | **0.61** | **6/10** | 0.043 | 0.027 |
| **AutoGM-PFL (personalizado)** | [0.11,0.33,0.04,0.04,0.02,0.01,0.18,0.03,0.06,0.01] | **0.08** | **0/10** | 0.042 | 0.027 |
| D-WFA | [1,0.16,1,0,0,0,0.32,0,1,1] | 0.45 | 4/10 | 0.060 | 0.016 |

### Ataque OVERT (`gaussian`) — control positivo

| Defensa | ASR media | colapsos | a_mal | a_hon | Lectura |
|---|---:|---:|---:|---:|---|
| FedAvg | 1.00 | 10/10 | 0.048 | 0.024 | sin defensa, colapsa |
| AutoGM (suave) | 0.00 | 0/10 | 0.001 | 0.055 | filtra |
| AutoGM-full (fiel) | 0.00 | 0/10 | **0.000** | 0.056 | **excluye** al atacante (recorte η\*) |
| AutoGM-PFL | 0.01 | 0/10 | 0.000 | 0.056 | filtra |
| D-WFA | 0.13 | 1/10 | 0.005 | 0.053 | filtra casi siempre |

---

## 2. Los dos hallazgos definitivos

### Hallazgo A — El ataque sigiloso evade INCLUSO el AutoGM fiel (contribución central, blindada)

El `constrained` colapsa de forma **bimodal** a los agregadores robustos de **modelo global compartido**:
AutoGM suave **5/10**, **AutoGM-full (fiel, con recorte duro de outliers) 6/10**, D-WFA 4/10. La evasión está
confirmada por los pesos: en el sigiloso `a_mal ≈ a_hon` (AutoGM-full 0.043 vs 0.027), mientras que en el
overt `a_mal ≈ 0.000` (lo **excluye**). Es decir: **la misma defensa que excluye al atacante ruidoso es
engañada por el sigiloso**, porque el clip a `tau·ref_norm` lo mantiene por debajo del umbral η\*.

→ Esto **mata la objeción** "rompiste una versión debilitada de AutoGM": el AutoGM del paper (Teorema 1,
exclusión dura) es, si acaso, **igual o más vulnerable** (6/10 vs 5/10). El resultado v1–v3 se **confirma y
fortalece** con 10 semillas y con la defensa fiel.

### Hallazgo B — La personalización (AutoGM-PFL) mitiga de forma robusta (0/10)

AutoGM-PFL **nunca colapsa** (0/10, ASR media 0.08) en el mismo régimen. No es suerte de una semilla: aguanta
en las 10. La personalización aísla a los clientes honestos, cuyos modelos `v_k` siguen entrenando sobre sus
datos limpios y no heredan el envenenamiento del global.

---

## 3. ¿Esto tira la tesis? No — la afina, pero exige apoyarse en Adv1/Mecanismo 1

AutoGM-PFL mitiga **este** ataque, pero con límites que son, de hecho, tu diferenciador:

1. **No defiende el sensor (Adv1).** PFL es personalización de la agregación; no detecta un sensor
   físicamente falsificado. Tu **Mecanismo 1** (consenso DE-FE) ataca justamente lo que PFL no puede tocar.
   PFL sortea la mitad Adv2 del ataque, no la mitad Adv1.
2. **Mitiga renunciando al modelo global compartido.** Aísla a cada planta en su modelo local. En FDD eso
   probablemente **sacrifica la detección de fallas raras/no vistas localmente** (el fenómeno de supresión de
   clase rara de v1/v2): una planta que nunca vio "Ball" depende del global para detectarla; su modelo
   personalizado, entrenado sin "Ball", no la vería. → Verificable reactivando el eje concentrado con PFL.
3. **Atacante no adaptativo + métrica proxy.** El Adv2 actual envenena el *global*; no fue adaptado a PFL. Y
   el ASR de PFL se mide sobre el ensemble personalizado evaluado en el test global (aproximación, no
   partición de test por cliente).

**Lectura para la tesis:** el espectro queda así —

| Enfoque | Sigiloso dual (α=0.3,β=0.4) | Punto débil |
|---|---|---|
| Agregación robusta global (AutoGM suave/fiel, D-WFA) | **se rompe** (4–6/10) | asume que el update sigiloso se distingue → no |
| Personalización (AutoGM-PFL) | mitiga (0/10) | no ve el sensor; sacrifica el global/clase rara; no adaptativo |
| **Tu M1 (integridad de señal) + M2 (agregación)** | objetivo | — |

PFL se vuelve un **tercer baseline fuerte** que hace tu evaluación más convincente. Tu novedad migra su peso
—como ya venía— a **(1) el adversario dual coordinado, (2) la integridad de sensor (Adv1/M1) que ninguna
defensa de agregación resuelve, y (3) preservar un modelo compartido robusto** (donde PFL renuncia).

---

## 4. Otros regímenes

- **α=0.5 (menos heterogeneidad):** los colapsos del sigiloso caen (AutoGM 2/10, AutoGM-full 1/10, D-WFA
  2/10; PFL 0/10) → confirma que la vulnerabilidad es **peor bajo alta heterogeneidad** (α=0.3), consistente
  con v2/v3.
- **S0 limpio:** todas las defensas dan DR≈1.00, FAR≈0.00 (PFL DR 0.99) → ninguna daña la detección en
  ausencia de ataque.

---

## 5. Próximos pasos sugeridos

1. **Ataque adaptativo vs PFL:** que Adv2 apunte a la vía de personalización / entrenamiento local. Si PFL
   aguanta aún así, refuerza que el diferenciador real es Adv1/M1.
2. **PFL vs clase rara:** reactivar el eje concentrado (`fault_owners`) con AutoGM-PFL para medir si la
   personalización sacrifica la falla rara.
3. **ASR-PFL fiel:** partición de test por cliente (`client_test_data`) en lugar del proxy global.
4. **Split bloqueado sin solape** (v4-no-leak) para cerrar el caveat de fuga, ya con las 5 defensas.

---

## 6. Artefactos

- Datos: `results/F4_grid__v4_defensas_gpu.json` (+ `.jsonl` checkpoint, 500 celdas), `results/v4_run.log`.
- Reporte genérico: `results/report__v4_defensas_gpu.md`.
- Figuras: `results/fig__v4_defensas_gpu__{asr_vs_beta,evasion,supraadditividad}.png`.
- Config: `config/v4_defensas_gpu.yaml`. Corrida: RTX 4090, `cuda jobs=8`, 8758 s.

*Generado el 2026-07-03 a partir de la corrida v4 en la RTX 4090.*
