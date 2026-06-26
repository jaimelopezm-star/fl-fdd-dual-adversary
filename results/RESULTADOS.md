# Resultados del experimento — índice de corridas

Las **3 corridas** son configuraciones versionadas del **mismo** experimento (`grid_f4.py`), no
experimentos distintos. Cada una responde una pregunta y **ninguna se descarta**. Narrativa completa en
[`../INFORME_F4.md`](../INFORME_F4.md).

| Corrida | Config | Escala | Reporte | Datos |
|---|---|---|---|---|
| **v1** | [`v1_local_small.yaml`](../config/v1_local_small.yaml) | 10 clientes · 20 rondas · sin solape · 1 semilla (52 celdas) | [`report__v1_local_small.md`](report__v1_local_small.md) | [`F4_grid__v1_local_small.json`](F4_grid__v1_local_small.json) |
| **v2** | [`v2_scaled.yaml`](../config/v2_scaled.yaml) | 30 clientes · 40 rondas · solape · 2 semillas (288 celdas, 2h48m) | [`report__v2_scaled.md`](report__v2_scaled.md) | [`F4_grid__v2_scaled.json`](F4_grid__v2_scaled.json) |
| **v3** | [`v3_seeds.yaml`](../config/v3_seeds.yaml) | escala v2 · **5 semillas** · enfocado S0+S3 (150 celdas, 1h30m) | [`report__v3_seeds.md`](report__v3_seeds.md) | [`F4_grid__v3_seeds.json`](F4_grid__v3_seeds.json) |

---

## Qué representa y qué arrojó cada una

### v1 — exploración a escala pequeña
**Para qué:** construir y validar el pipeline a una escala que cabe en una laptop.
**Veredicto:** AutoGM aguanta el ataque sigiloso en minoría; a β=0.4 caen *tanto* overt como sigiloso (sin
diferenciación). La supresión de clase rara se veía dramática (AutoGM 0.00 vs FedAvg 0.91) — luego
resultó **parcialmente artefacto de datos chicos**. Señal de que el resultado dependía de la escala.

### v2 — escala realista (¿el resultado de v1 era artefacto de escala?)
**Para qué:** subir las 3 palancas (solape ≈4× datos, 30 clientes, 40 rondas) para descartar artefactos.
**Veredicto (cambia el cuadro):** AutoGM se vuelve **robusto al ataque overt incluso al 40 %**, pero el
**sigiloso lo rompe a α=0.3, β=0.4 (ASR=1.0)**. Ahora **el sigilo SÍ es el diferenciador**. Evasión más
fuerte (a_mal ≈ 2× a_hon). Supresión de clase rara se **debilita** (efecto secundario, no titular).

### v3 — confirmación con 5 semillas
**Para qué:** medir cuán reproducible es el quiebre de v2 (que se vio con solo 2 semillas).
**Veredicto (definitivo): el efecto es REAL pero BIMODAL.** A α=0.3, β=0.4:

| defensa | ASR sigiloso por semilla | media | ASR overt |
|---|---|---|---|
| **AutoGM** | [1.0, 1.0, 1.0, 0.0, 0.0] | **0.60 ± 0.49** | **0.00 (0/5)** |
| **D-WFA** | [1.0, 0.09, 1.0, 0.0, 0.0] | 0.42 ± 0.48 | 0.18 |
| **FedAvg** (sin defensa) | ~0.01 | 0.01 | 1.00 |

En **3 de 5 corridas el sigiloso colapsa AutoGM por completo (ASR=1.0)**, mientras el overt es filtrado en
**5/5** y FedAvg sin defensa **nunca** es afectado por el sigiloso. AutoGM/D-WFA aguantan en **5/5** a
α=0.5 y a β≤0.3 → el efecto es de **alta no-IID (α=0.3) + β=0.4** (< 50 %, dentro de la tolerancia
Byzantine declarada).

---

## Conclusión transversal (v1 → v2 → v3)

A escala pequeña el efecto se enmascaraba; **a escala realista emerge y se confirma con semillas**: el
ataque dual coordinado **sigiloso** evade la agregación robusta (la defensa le da más peso que a un
honesto) y, en régimen de alta heterogeneidad, **provoca el colapso de la detección donde el ataque
ruidoso es filtrado y donde no tener defensa no causa daño**. Es decir, **la defensa de servidor robusta
puede ser contraproducente** ante este ataque. → motiva el **Mecanismo 1 en el borde (F5)**.

## Figuras (por corrida, prefijo `fig__<corrida>__`)
- `*_asr_vs_beta.png` — daño (ASR) vs fracción maliciosa, por defensa.
- `*_evasion.png` — peso que la defensa da al malicioso vs al honesto (overt filtrado, sigiloso ≈/> honesto).
- `*_clase_rara.png` — supresión de clase rara (solo v1, v2; v3 no incluyó el eje concentrado).
- `*_supraadditividad.png` — métrica de coordinación (no concluyente; ver informe).

## Reproducir
```bash
py -3.12 grid_f4.py --config config/<corrida>.yaml --jobs N      # genera results/F4_grid__<corrida>.json
py -3.12 make_figures_f4.py results/F4_grid__<corrida>.json      # figuras
py -3.12 report_f4.py       results/F4_grid__<corrida>.json      # reporte md
```
