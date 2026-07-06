# Informe F4 — Caracterización de la vulnerabilidad del baseline de servidor frente al adversario dual coordinado

> FL-FDD robusto · Tesis INAOE MCTS · Jaime López · Fase F4 (campaña v1–v7)
> Artefactos: `results/F4_grid__{v1_local_small,v2_scaled,v3_seeds,v4_defensas_gpu,v5_pfl_sensor,v6_pfl_flip,v7_blocked}.json`
> + reportes `results/report__*.md` y figuras `results/fig__*.png`.

---

## 0. Resumen ejecutivo

Probamos si las **defensas de agregación robusta del servidor** (AutoGM en sus variantes suave, fiel y
personalizada; D-WFA), que son el estado del arte para resistir clientes maliciosos en aprendizaje
federado, **siguen protegiendo la detección de fallas cuando el adversario combina dos capas de ataque
coordinadas**: manipular la **señal del sensor** del dispositivo (Adv1) y envenenar el **update del modelo**
(Adv2).

**Resultado principal (un resultado de imposibilidad, demostrado por ruptura).** Al enfrentar cinco
defensas de servidor a los **dos vectores** del adversario dual —envenenamiento de modelo (S3) y
reetiquetado del sensor (S4)—, **ninguna sobrevive ambos**. La única que resiste el envenenamiento de
modelo (AutoGM personalizada, **PFL**) **cae ante el sensor**; las que resisten el sensor (AutoGM suave,
D-WFA) **caen ante el modelo**. El adversario dual, al disponer de las dos capas, **derrota cualquier
elección de defensa de agregación**.

**El camino hasta ahí (v1–v4)** establece el mecanismo: el ataque sigiloso **evade** la defensa robusta
(esta le asigna *más* peso que a un cliente honesto) y, en régimen de alta heterogeneidad (α=0.3) con una
fracción maliciosa del 40 % —**dentro de la tolerancia Byzantine que la defensa declara (< 50 %)**—
**colapsa la detección (ASR → 1.0) incluso contra el AutoGM fiel (6/10 semillas)**, mientras el ataque
"ruidoso" clásico (overt) es **filtrado en el 100 % de las corridas**. El resultado se **blinda contra la
fuga de datos** (corrida final v7, partición temporal sin solape): el cruce de vulnerabilidades es idéntico.

Este hallazgo justifica empíricamente la contribución de la tesis: como **ninguna capa de agregación sola
cubre ambos vectores**, la integridad de la señal **debe validarse en el borde** (Mecanismo 1, consenso
multi-sensor DE-FE) **acoplada** a la agregación del servidor (Mecanismo 2).

---

## 1. Qué estamos probando (y por qué)

### 1.1 El modelo de amenaza: adversario dual coordinado
En diagnóstico federado de fallas (FL-FDD), varias "plantas" entrenan un modelo común sin compartir datos.
El adversario que estudiamos controla, **con un mismo actor**, dos capas:

- **Adv1 — sensor / dispositivo (la contribución):** falsifica la **señal de vibración** *antes* del
  entrenamiento local (firmware comprometido, MITM en el bus de campo, spoofing). Dos modos:
  - **`mask`:** reemplaza ventanas de falla por ventanas "Normal" y reetiqueta a Normal — **borra** la
    firma de falla. Es el auto-enmascaramiento del atacante en S3.
  - **`flip`:** **conserva** la señal de falla pero cambia solo la etiqueta a Normal — un *backdoor* de
    sensor. Es el modo con el que Adv1 envenena a clientes víctima honestos en S4.
- **Adv2 — cliente / modelo:** perturba el update que el cliente envía al servidor. Dos variantes:
  - **overt (ruidoso):** ruido gaussiano fuerte sobre los parámetros (lo que la literatura de
    anti-poisoning, p. ej. AutoGM, ya cubre).
  - **sigiloso (`constrained`, la contribución):** amplifica el update en su misma dirección pero **acota
    su norma** a la de un cliente honesto, para **no parecer un outlier**.

### 1.2 Por qué importa (el vacío que llena)
La defensa de **doble capa** (validar en el borde + confiar en el servidor) **ya existe** (Hathout 2024,
MNIST i.i.d., label-flipping genérico). Por eso la novedad **no** es "la doble capa", sino:
1. el **adversario dual coordinado** (sensor físico + modelo, mismo actor), que nadie modela;
2. en **FL-FDD** (señal de vibración industrial, no imágenes genéricas);
3. mostrar **que ninguna defensa de servidor cubre ambos vectores** (el resultado de imposibilidad);
4. el **acoplamiento** del Mecanismo 1 (borde) con el Mecanismo 2 (servidor).

Las fases F1–F4 cubren los puntos 1–3 (caracterizar la vulnerabilidad). La fase F5 abordará el punto 4.

### 1.3 Hipótesis evaluadas
- **H-evasión:** un update sigiloso (señal enmascarada + norma acotada) produce una actualización **no
  anómala** → la agregación robusta **no la filtra**.
- **H-quiebre:** el ataque dual coordinado sigiloso **degrada la detección donde el overt no puede**,
  porque evade el filtro de la defensa.
- **H-contraste:** el efecto es máximo bajo **alta heterogeneidad non-IID** (α bajo).
- **H-dual (tesis central):** **no existe una defensa de agregación de servidor única que sobreviva
  simultáneamente el vector de modelo (Adv2, S3) y el vector de sensor (Adv1, S4).**

---

## 2. Por qué los experimentos están diseñados así (qué representa cada pieza)

### 2.1 Dataset — CWRU Bearing (vibración, dos sensores DE y FE)
Señal de vibración real de rodamientos con **dos acelerómetros (Drive-End y Fan-End)**. Se eligió porque:
(a) es señal cruda manipulable a nivel sensor (habilita Adv1), y (b) los **dos sensores** habilitan el
Mecanismo 1 de la fase siguiente (consenso DE-FE). Clases: Normal, Inner Race, Outer Race, Ball.
Ventaneo deslizante + normalización z-score por ventana.

### 2.2 Modelo — 1D-CNN (WDCNN, ~42 000 parámetros)
CNN 1D sobre la señal cruda; arquitectura estándar y dominante del campo. Modelo pequeño a propósito: el
dataset es pequeño y el foco es la **dinámica de la defensa**, no exprimir exactitud.

### 2.3 Partición non-IID — dos formas, cada una representa un escenario industrial
- **Dirichlet(α):** reparte las clases entre las plantas con heterogeneidad controlada por α (α→0 = cada
  planta ve pocas clases; α grande = casi homogéneo). Representa **heterogeneidad gradual** entre plantas.
- **Concentrada (k dueños por falla):** cada modo de falla vive en pocos clientes. Representa el caso
  realista de "**cada máquina/planta tiene sus propios modos de falla**" y sirve para revelar si la defensa
  **sacrifica las fallas raras** (mecanismo que S4 acabará *weaponizando*, ver §4.6).

### 2.4 Escenarios de amenaza — qué demuestra cada uno
| Escenario | Adv1 (sensor) | Adv2 (modelo) | Qué representa |
|---|:--:|:--:|---|
| **S0** | ✗ | ✗ | Línea base limpia (¿cuánto detecta sin ataque?) |
| **S1** | `mask` | ✗ | Solo el sensor falsificado (auto-enmascaramiento) |
| **S2** | ✗ | ✓ | Solo el envenenamiento de modelo (lo que el baseline dice manejar) |
| **S3** | `mask` | sigiloso | **Vector de modelo** — el malicioso enmascara su propia señal y envenena su update |
| **S4** | `flip` | (opc.) | **Vector de sensor** — Adv1 reetiqueta el sensor de **víctimas honestas** (variantes sensor-solo y dual completo) |

> **Ojo con la nomenclatura:** S3 y S4 *ambos* usan Adv1, pero por eso el nombre remite al **daño real**.
> En S3 el `mask` es auto-enmascaramiento (el malicioso oculta su propia falla; el daño lo hace su update)
> → **vector de modelo**. En S4 el `flip` ataca el sensor de **otros clientes honestos** → **vector de
> sensor**. Son las dos mitades del adversario dual.

### 2.5 Defensas comparadas — qué representa cada una (5 configuraciones)
- **FedAvg:** promedio simple, **sin defensa** → control negativo (qué pasa si no hay protección).
- **AutoGM-suave** (Li 2023, IEEE TII): mediana geométrica auto-ponderada por Weiszfeld/IRLS; pondera a la
  baja pero **nunca excluye**. Es la forma evaluada en v1–v3.
- **AutoGM-full (fiel):** la forma **completa** del paper, con regularización λ y **recorte duro** de
  outliers (α=0 si la distancia supera el umbral η\*; Teorema 1). Integrada en v4 para cerrar la objeción
  "rompieron una versión debilitada".
- **AutoGM-PFL (personalizada):** un modelo local por planta (estilo Ditto), pensada para non-IID.
  Integrada en v4; es la única que resiste el vector de modelo.
- **D-WFA** (Chen 2022, IEEE TIM): ponderación dinámica por divergencia; baseline **on-topic** de FDD
  non-IID, no diseñado para seguridad → punto de contraste.

### 2.6 Factores barridos — qué controla cada uno
- **α (Dirichlet):** heterogeneidad non-IID. Eje de H-contraste.
- **β:** fracción de clientes maliciosos. Se prueba **minoría** (10–30 %) y el límite Byzantine (40 %,
  todavía < 50 %).
- **modo de Adv2:** *overt* vs *sigiloso* → el contraste que aísla el efecto del sigilo.
- **vector:** S3 (modelo) vs S4 (sensor) → **el contraste decisivo** del resultado de imposibilidad.
- **k dueños** (concentrada): cuán rara es la falla.

### 2.7 Métricas — qué mide cada una y por qué
- **DR** (Detection Rate): fallas detectadas. Lo que la planta quiere alto.
- **FAR** (False Alarm Rate): falsas alarmas.
- **ASR** (Attack Success Rate): **fallas reales clasificadas como Normal** — una falla que pasa
  desapercibida = riesgo industrial. Es el objetivo del ataque; **la métrica central**. (Colapso = ASR > 0.5;
  una defensa "sobrevive" un vector si ASR < 0.15.)
- **Peso `a_i`** que la defensa asigna a cada cliente: mide la **evasión**. Si `a_malicioso ≈ a_honesto`
  (o mayor), la defensa **no distingue** al adversario. Es la evidencia directa del sigilo.

---

## 3. Por qué hay siete corridas versionadas y qué representa cada una

Los experimentos se versionan por archivo de configuración (`config/*.yaml`) y **ninguno se descarta**:
cada uno responde una pregunta distinta y juntos forman la cadena de evidencia. El tramo v1–v3 establece el
*mecanismo* a tres escalas; v4–v7 lo *blindan* (fidelidad, segundo vector y sin fuga) hasta el resultado
de imposibilidad.

| Versión | Escala / novedad | Qué representa / por qué se hizo |
|---|---|---|
| **v1_local_small** | 10 clientes · 20 rondas · 1 537 ventanas (sin solape) · 1 semilla | **Exploración inicial** a escala pequeña. Validó el pipeline y reveló que el resultado dependía de la escala. |
| **v2_scaled** | 30 clientes · 40 rondas · ~6 000 ventanas (solape) · 2 semillas · 3 defensas | **Escala realista.** ¿El resultado de v1 es real o artefacto de tener pocas instancias? Aquí emerge el quiebre del sigiloso. |
| **v3_seeds** | como v2 · **5 semillas** · S0+S3 | **Confirmación estadística.** El quiebre resulta **bimodal** (3/5 colapso total). |
| **v4_defensas_gpu** | **5 defensas** (AutoGM fiel + PFL) · **10 semillas** · RTX 4090 · 500 celdas | **Auditoría de fidelidad.** ¿Sobrevive el hallazgo al AutoGM *fiel* (η\*)? Sí (6/10). Pero **PFL resiste (0/10)** → abre la objeción metodológica. |
| **v5_pfl_sensor** | escenario **S4**, FDI `mask` · RTX 4000 Ada · 400 celdas | **Primer intento de romper PFL** por el sensor. *Resultado negativo honesto:* `mask` borra la señal, no induce detección-perdida. Instrumento equivocado. |
| **v6_pfl_flip** | S4 con FDI `flip` · 5 defensas · 10 semillas · 400 celdas | **El gap dual, por ruptura.** `flip` conserva la señal y reetiqueta → rompe PFL por el sensor. **Ninguna defensa sobrevive S3 y S4.** |
| **v7_blocked** | como v6 · **split temporal sin solape** · 400 celdas | **Blindaje contra la fuga.** Repite v6 sin solape train/test. El cruce es **idéntico** → números defendibles ante revisores. |

**Por qué en ese orden:** primero se construye y se entiende a escala chica (barato); se sube a escala
realista para descartar artefactos; se confirma con repeticiones; se audita la *fidelidad* de la defensa;
se añade el segundo vector de ataque; y por último se elimina la fuga de datos. Cada corrida cierra una
objeción concreta que la anterior deja abierta.

> **La bisagra metodológica.** Tras v4, AutoGM-PFL resistía el ataque de modelo (0/10). Aquí aplica el
> principio *"un vacío se afirma ROMPIENDO, no argumentando"*: contra AutoGM el gap estaba **demostrado**
> (se rompía), pero contra PFL solo había un **argumento** ("no ve el sensor"). Eso motivó el segundo
> vector S4 (v5–v6): pasar de argumentar a **romper** también PFL.

---

## 4. Resultados obtenidos

### 4.1 Evasión — confirmada en todas las escalas y semillas
Bajo AutoGM, en S3:
- **overt:** `a_malicioso ≈ 0.001` → la defensa lo **aplasta** (lo trata como outlier).
- **sigiloso:** `a_malicioso ≈ 0.06` vs `a_honesto ≈ 0.02` → la defensa le da **~2–6× el peso de un cliente
  honesto**. No solo no lo detecta: **lo premia**. (Figura `fig__*__evasion.png`.)

### 4.2 Daño (ASR) — el contraste overt vs sigiloso, por escala (v1–v3)

**v1 (escala pequeña):** a β=0.4 **tanto** overt como sigiloso rompían AutoGM (con 10 clientes el 40 % está
cerca del límite Byzantine efectivo) → el sigilo **no se distinguía**. Conclusión engañosa contaminada por
la escala chica.

**v2 (escala realista):** AutoGM se vuelve **robusto al overt incluso al 40 %** (lo filtra), pero el
**sigiloso lo rompe a α=0.3, β=0.4 (ASR=1.0)**. Ahora **el sigilo SÍ es el diferenciador**.

**v3 (5 semillas) — S3 a α=0.3, β=0.4:**

| defensa | ASR sigiloso por semilla | media | ASR overt |
|---|---|---|---|
| **AutoGM** | [1.0, 1.0, 1.0, 0.0, 0.0] | **0.60 ± 0.49** | **0.00 (0/5)** |
| **D-WFA** | [1.0, 0.09, 1.0, 0.0, 0.0] | 0.42 ± 0.48 | 0.18 |
| **FedAvg** (sin defensa) | [0.01, 0, 0.02, 0, 0] | 0.01 | 1.00 |

El efecto es **bimodal**: 3/5 colapso total, 2/5 nulo. (Figura `fig__v3_seeds__asr_vs_beta.png`.)

### 4.3 Auditoría de fidelidad — 5 defensas, 10 semillas (v4, S3, α=0.3, β=0.4)

| defensa | ASR sig. (media) | colapsos | ASR overt | Lectura |
|---|---:|---:|---:|---|
| FedAvg (sin defensa) | 0.10 | 1/10 | 1.00 | overt lo destruye |
| AutoGM-suave | 0.50 | 5/10 | 0.00 | sigiloso lo rompe |
| **AutoGM-full (fiel)** | **0.61** | **6/10** | 0.00 | el η\* no salva |
| **AutoGM-PFL** | **0.08** | **0/10** | 0.01 | **resiste el modelo** |
| D-WFA | 0.45 | 4/10 | 0.13 | sigiloso lo rompe |

- **Hallazgo A:** el sigiloso evade **incluso el AutoGM fiel** (6/10). En overt, el full **excluye** al
  atacante (`a_mal=0.000` por el recorte η\*); en sigiloso no lo distingue. → muere la objeción "rompieron
  una versión debilitada".
- **Hallazgo B:** **AutoGM-PFL nunca colapsa** (0/10) — la personalización aísla a los honestos, que
  entrenan sobre datos limpios y no heredan el envenenamiento del global.

### 4.4 S4 por el sensor: el instrumento importa (v5, resultado negativo)
S4 con `mask` **no rompió PFL** (ASR ≈ 0.02, 0/10). El diagnóstico es del *instrumento*: `mask` borra la
señal, así que la víctima confunde la falla con otra (baja recall) pero **no la llama "Normal"**. Para
inducir detección-perdida hace falta **conservar la señal y solo reetiquetar** (`flip`).
- **Hallazgo colateral sólido:** las defensas de **modelo global** (AutoGM suave/full, D-WFA) **absorben**
  un sensor comprometido en minoría de honestos (ASR→0, recall→1.0), porque el global aprende la falla de
  los honestos no-víctima. Contraste limpio con la personalización, que aísla a cada planta.

### 4.5 El gap dual demostrado por ruptura + blindaje sin fuga (v6 → v7)
Con `flip` (v6) el ataque de sensor sí induce detección-perdida. La corrida v7 lo repite con **partición
temporal bloqueada** (cero ventanas compartidas). Cruce de los dos vectores, α=0.3:

| defensa | ASR S3 (modelo) v6→v7 | ASR S4 (sensor) v6→v7 | sobrevive S3 | sobrevive S4 |
|---|---:|---:|:--:|:--:|
| AutoGM-suave | 0.50 → 0.35 | 0.09 → 0.13 | CAE | sí |
| AutoGM-full | 0.60 → 0.55 | 0.33 → 0.31 | CAE | CAE |
| **AutoGM-PFL** | **0.08 → 0.07** | **0.29 → 0.29** | **sí** | **CAE** |
| D-WFA | 0.44 → 0.23 | 0.09 → 0.13 | CAE | sí |

**Veredicto: ninguna defensa de servidor (M2) sobrevive los dos vectores.** El cruce es **idéntico sin
fuga**: la fuga solo inflaba un poco el ataque de modelo (S3 baja al quitarla), pero todas siguen cayendo;
el ataque de sensor (S4) es prácticamente inmune a la fuga. La clasificación sobrevive/CAE no cambia.

### 4.6 Mecanismo del fallo del sensor (y la supresión de clase rara, ahora weaponizada)
En AutoGM-full y PFL el recall de la falla objetivo cae a 0.05–0.12 bajo S4 porque el **recorte duro η\***
descarta a los pocos honestos que aún portan la falla (los ve como outliers frente a las víctimas
reetiquetadas). Es la **supresión de clase rara** —que en v1/v2 aparecía en limpio con partición
concentrada (v1: AutoGM recall 0.00 vs FedAvg 0.91; v2: leve, en parte artefacto de datos chicos)— ahora
**weaponizada por el sensor**. Las defensas sin exclusión dura (AutoGM suave, D-WFA) no descartan a los
portadores → sobreviven S4.

### 4.7 Dónde la defensa SÍ aguanta (límites del efecto)
- **α = 0.5** (heterogeneidad moderada): el efecto se atenúa en todas las defensas (PFL S4 ≈ 0.19,
  full ≈ 0.17) → la vulnerabilidad es **peor bajo alta no-IID**.
- **β ≤ 0.3:** las defensas robustas aguantan el vector de modelo. El daño de S3 requiere β=0.4.
- **S0 limpio:** todas dan DR ≈ 1.0, FAR ≈ 0.0 (PFL DR ≈ 0.95) → ninguna daña la detección sin ataque.

---

## 5. Interpretación

1. **Ninguna capa de agregación sola cubre ambos vectores (la tesis de la fase).** Endurecer la defensa
   contra el modelo (PFL) la abre al sensor; las que toleran el sensor ceden al modelo. No es que "falte
   afinar" una regla de agregación: el punto de observación (el update en el servidor) no basta para un
   adversario que actúa también en la señal. De ahí el **acoplamiento M1↔M2**.
2. **El ataque sigiloso es un "asesino de agregación robusta".** Contra FedAvg no hace nada (su norma
   acotada se diluye en el promedio); contra AutoGM/D-WFA, que confían en los updates "no anómalos",
   obtiene peso alto y arrastra el modelo. **La robustez del servidor se vuelve en su contra.**
3. **El quiebre ocurre dentro de la garantía de la defensa.** β=0.4 < 50 % es minoría Byzantine; AutoGM
   *debería* tolerarlo (y tolera el overt). Que el sigiloso lo rompa ahí es un fallo **dentro de su zona
   declarada de seguridad**.
4. **El efecto de modelo (S3) es bimodal; el de sensor (S4) es consistente.** En S3, 3/5 colapso total;
   la varianza es parte del hallazgo (cuando la defensa es explotable, el fallo es catastrófico). En S4 la
   degradación es consistente (recall ~0.10). En producción no se puede apostar a una partición afortunada.
5. **La escala y la fidelidad importaban.** A escala chica el efecto se enmascaraba (v1); a escala realista
   emerge (v2–v3); con el AutoGM fiel se confirma (v4); sin fuga se blinda (v7). Por eso las siete corridas
   son necesarias y ninguna se descarta.

---

## 6. Amenazas a la validez (honestas)

- **Reimplementación propia** de AutoGM/D-WFA (no el código oficial). Mitigación: la réplica oficial sobre
  Bosch confirmó el comportamiento de AutoGM, el overt se filtra al 100 %, y en v4 se integró el AutoGM
  **fiel** (η\*) que reproduce la exclusión del atacante overt; queda pendiente portar CWRU al repo oficial.
- **Fuga por solape de ventanas** (stride < ventana). Mitigación **ya resuelta:** la corrida **v7** rehízo
  el experimento con partición temporal bloqueada (cero ventanas compartidas) y el cruce resultó idéntico.
- **Atacante no adaptativo.** Adv2 envenena el modelo global; no fue adaptado a la vía de personalización
  de PFL, y el ASR de PFL se mide sobre el ensemble personalizado (proxy). Pendiente: ataque adaptativo y
  partición de test por cliente.
- **Efecto de S3 acotado** a α=0.3 / β=0.4. La vía de sensor (S4) es más consistente. El resultado *de
  imposibilidad* (ninguna defensa cubre ambos) se sostiene en todo el régimen probado.

---

## 7. Conclusión y siguiente paso

La fase F1–F4, a lo largo de siete corridas (v1–v7), queda **caracterizada, auditada y blindada**. El
camino v1–v4 estableció el mecanismo (el ataque dual sigiloso evade la agregación robusta y colapsa la
detección incluso contra el AutoGM fiel, dentro de la tolerancia Byzantine), y v5–v7 lo elevaron a un
**resultado de imposibilidad, demostrado por ruptura y sin fuga**: frente al adversario dual (Adv2 sobre el
modelo y Adv1 sobre el sensor), **ninguna defensa de agregación de servidor sobrevive los dos vectores**.
Endurecer una capa del servidor solo mueve el punto ciego —no lo cierra.

**Siguiente:** **F5 — Mecanismo 1 (consenso multi-sensor DE-FE) acoplado a M2.** La predicción es que M1
detecta la incoherencia de señal que Adv1 introduce (la firma de falla que `flip` reetiqueta en un sensor
sigue presente en el otro) y, acoplado a M2, **recupera la detección en los DOS vectores** (S3 *y* S4),
justo donde toda defensa de servidor por sí sola falla — cerrando el ciclo ataque → vulnerabilidad →
defensa y validando H3 (recuperación) y H4 (el acoplamiento supera a M1 y M2 por separado).

---

*Datos y figuras: `results/`. Configuraciones: `config/v{1..7}_*.yaml`. Reproducible con
`grid_f4.py --config <cfg> [--jobs N]`. Reporte formal: `../../07_reporte_experimentos/reporte_experimentos_F1F4.tex`.
Bitácora: `../../BITACORA_V2.md`.*
