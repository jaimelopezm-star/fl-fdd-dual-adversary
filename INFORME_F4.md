# Informe F4 — Caracterización de la vulnerabilidad del baseline de servidor frente al adversario dual coordinado sigiloso

> FL-FDD robusto · Tesis INAOE MCTS · Jaime López · Fase F4
> Artefactos: `results/F4_grid__{v1_local_small,v2_scaled,v3_seeds}.json` + reportes y figuras `__*`.

---

## 0. Resumen ejecutivo

Probamos si las **defensas de agregación robusta del servidor** (AutoGM, D-WFA), que son el estado del
arte para resistir clientes maliciosos en aprendizaje federado, **siguen protegiendo la detección de
fallas cuando el adversario combina dos capas de ataque coordinadas**: manipular la **señal del sensor**
del dispositivo (Adv1) y envenenar el **update del modelo** (Adv2), de forma **sigilosa** (que parezca un
cliente honesto).

**Resultado principal.** El ataque dual coordinado sigiloso **evade** la defensa robusta (esta le asigna
*más* peso que a un cliente honesto) y, en régimen de alta heterogeneidad (non-IID extremo) con una
fracción maliciosa del 40 % —**dentro de la tolerancia Byzantine que la defensa declara (< 50 %)**—
**provoca el colapso total de la detección (ASR → 1.0) en 3 de 5 corridas contra AutoGM y 2 de 5 contra
D-WFA**, mientras el ataque "ruidoso" clásico (overt) es **filtrado en el 100 % de las corridas** y la
línea sin defensa (FedAvg) **nunca** es afectada por el ataque sigiloso. Es decir: **la agregación robusta
puede ser peor que no tener defensa** ante este ataque, porque está diseñado para parecer confiable y la
defensa lo recompensa.

Este hallazgo justifica empíricamente la contribución de la tesis: la integridad de la señal **debe
validarse en el borde** (Mecanismo 1, consenso multi-sensor DE-FE), porque el servidor o no ve la
manipulación o incluso la premia.

---

## 1. Qué estamos probando (y por qué)

### 1.1 El modelo de amenaza: adversario dual coordinado
En diagnóstico federado de fallas (FL-FDD), varias "plantas" entrenan un modelo común sin compartir datos.
El adversario que estudiamos controla, **con un mismo actor**, dos capas:

- **Adv1 — sensor / dispositivo (la contribución):** falsifica la **señal de vibración** *antes* del
  entrenamiento local (firmware comprometido, MITM en el bus de campo, spoofing). En este experimento se
  implementa como *masking*: reemplaza ventanas de falla por ventanas "Normal" y reetiqueta a Normal, de
  modo que los datos quedan **consistentes** (señal normal + etiqueta normal) y el cliente entrena de buena
  fe sobre datos envenenados.
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
3. mostrar **cómo y cuándo** la defensa de servidor SOTA falla ante él;
4. el **acoplamiento** del Mecanismo 1 (borde) con el Mecanismo 2 (servidor).

Las fases F1–F4 cubren los puntos 1–3 (caracterizar la vulnerabilidad). La fase F5 abordará el punto 4.

### 1.3 Hipótesis evaluadas
- **H-evasión:** un update sigiloso (señal enmascarada + norma acotada) produce una actualización **no
  anómala** → la agregación robusta **no la filtra**.
- **H-quiebre:** el ataque dual coordinado sigiloso **degrada la detección donde el overt no puede**,
  porque evade el filtro de la defensa.
- **H-contraste:** el efecto es máximo bajo **alta heterogeneidad non-IID** (α bajo).

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
  **sacrifica las fallas raras**.

### 2.4 Escenarios de amenaza — qué demuestra cada uno
| Escenario | Adv1 (sensor) | Adv2 (modelo) | Qué representa |
|---|:--:|:--:|---|
| **S0** | ✗ | ✗ | Línea base limpia (¿cuánto detecta sin ataque?) |
| **S1** | ✓ | ✗ | Solo el sensor falsificado |
| **S2** | ✗ | ✓ | Solo el envenenamiento de modelo (lo que el baseline dice manejar) |
| **S3** | ✓ | ✓ | **Dual coordinado** — el corazón de la contribución |

### 2.5 Defensas comparadas — qué representa cada una
- **FedAvg:** promedio simple, **sin defensa** → control negativo (qué pasa si no hay protección).
- **AutoGM** (Li 2021, IEEE TII): mediana geométrica auto-ponderada; **baseline SOTA** anti-poisoning
  (Mecanismo 2). Es la defensa que el ataque debe vencer.
- **D-WFA** (Chen 2022, IEEE TIM): ponderación dinámica por divergencia; baseline **on-topic** de FDD
  non-IID, no diseñado para seguridad → segundo punto de contraste.

### 2.6 Factores barridos — qué controla cada uno
- **α (Dirichlet):** heterogeneidad non-IID. Eje de H-contraste.
- **β:** fracción de clientes maliciosos. Crítico: se prueba **minoría** (10–30 %) y el límite Byzantine
  (40 %, todavía < 50 %).
- **modo de Adv2:** *overt* vs *sigiloso* → **el contraste decisivo** (¿el sigilo aporta algo?).
- **k dueños** (concentrada): cuán rara es la falla.

### 2.7 Métricas — qué mide cada una y por qué
- **DR** (Detection Rate): fallas detectadas. Lo que la planta quiere alto.
- **FAR** (False Alarm Rate): falsas alarmas.
- **ASR** (Attack Success Rate): **fallas reales clasificadas como Normal** — una falla que pasa
  desapercibida = riesgo industrial. Es el objetivo del ataque; **la métrica central**.
- **Peso `a_i`** que la defensa asigna a cada cliente: mide la **evasión**. Si `a_malicioso ≈ a_honesto`
  (o mayor), la defensa **no distingue** al adversario. Es la evidencia directa del sigilo.

---

## 3. Por qué hay tres corridas versionadas (v1, v2, v3) y qué representa cada una

Los experimentos se versionan por archivo de configuración (`config/*.yaml`) y **ninguno se descarta**:
cada uno responde una pregunta distinta y juntos forman la cadena de evidencia.

| Versión | Escala | Qué representa / por qué se hizo |
|---|---|---|
| **v1_local_small** | 10 clientes · 20 rondas · 1 537 ventanas (sin solape) · 1 semilla | **Exploración inicial** a escala pequeña (la que cabe en una laptop). Sirvió para construir y validar el pipeline y para detectar que el resultado dependía de la escala. |
| **v2_scaled** | 30 clientes · 40 rondas · ~6 000 ventanas (solape) · 2 semillas · 3 defensas | **Escala realista.** Responde a la pregunta de validez "¿el resultado de v1 es real o artefacto de tener pocas instancias/dispositivos?". Sube las 3 palancas de escala (más datos por solape, más clientes, más rondas). |
| **v3_seeds** | igual que v2 · **5 semillas** · enfocado en S0+S3, α∈{0.3,0.5}, β∈{0.3,0.4} | **Confirmación estadística.** Repite el régimen donde v2 mostró el quiebre, con más semillas, para medir **cuán reproducible** es (barras de error). |

**Por qué en ese orden:** primero se construye y se entiende a escala chica (barato), luego se sube a
escala realista para descartar artefactos, y por último se confirma con repeticiones. Es el camino estándar
para no sobre-afirmar con poca evidencia.

---

## 4. Resultados obtenidos

### 4.1 Evasión — confirmada en todas las escalas y semillas
Bajo AutoGM, en S3:
- **overt:** `a_malicioso ≈ 0.001` → la defensa lo **aplasta** (lo trata como outlier).
- **sigiloso:** `a_malicioso ≈ 0.06` vs `a_honesto ≈ 0.02` → la defensa le da **~2–6× el peso de un cliente
  honesto**. No solo no lo detecta: **lo premia**. (Figura `fig__*__evasion.png`.)

### 4.2 Daño (ASR) — el contraste overt vs sigiloso, por escala

**v1 (escala pequeña):** a β=0.4 **tanto** overt como sigiloso rompían AutoGM (porque con 10 clientes el
40 % está cerca del límite Byzantine efectivo) → el sigilo **no se distinguía**. Conclusión engañosa de
"AutoGM aguanta en minoría" contaminada por la escala chica.

**v2 (escala realista):** AutoGM se vuelve **robusto al overt incluso al 40 %** (lo filtra), pero el
**sigiloso lo rompe a α=0.3, β=0.4 (ASR=1.0)**. Ahora **el sigilo SÍ es el diferenciador**.

**v3 (5 semillas) — el resultado definitivo, S3 a α=0.3, β=0.4:**

| defensa | ASR sigiloso por semilla | media | ASR overt |
|---|---|---|---|
| **AutoGM** | [1.0, 1.0, 1.0, 0.0, 0.0] | **0.60 ± 0.49** | **0.00 (0/5)** |
| **D-WFA** | [1.0, 0.09, 1.0, 0.0, 0.0] | 0.42 ± 0.48 | 0.18 |
| **FedAvg** (sin defensa) | [0.01, 0, 0.02, 0, 0] | 0.01 | 1.00 |

(Figura `fig__v3_seeds__asr_vs_beta.png`.)

### 4.3 Dónde la defensa SÍ aguanta (límites del efecto, en las 5 semillas)
- **α = 0.5** (heterogeneidad moderada): AutoGM y D-WFA aguantan el sigiloso en **5/5** (ASR=0).
- **β ≤ 0.3:** aguantan en **5/5**. El daño requiere β=0.4.
→ El efecto es de **alta no-IID (α=0.3) + fracción alta evasiva (β=0.4)**.

### 4.4 Supresión de clase rara (hallazgo secundario, partición concentrada, S0 limpio)
Con la falla concentrada en pocos dueños, **sin ningún ataque**, la agregación robusta descarta a esos
pocos clientes como si fueran outliers:
- **v1 (datos chicos):** efecto dramático — AutoGM recall 0.00 vs FedAvg 0.91 con 2 dueños.
- **v2 (escala):** efecto **leve y ruidoso** — AutoGM 0.80, D-WFA 0.50 vs FedAvg 1.00.
→ Era en parte **artefacto de datos chicos**; se reporta como efecto secundario, no como titular.

---

## 5. Interpretación

1. **El ataque sigiloso es específicamente un "asesino de agregación robusta".** Contra FedAvg (sin
   defensa) no hace nada (su norma acotada se diluye en el promedio); contra AutoGM/D-WFA, que confían en
   los updates "no anómalos", obtiene peso alto y, cuando la partición lo favorece, arrastra el modelo a
   etiquetar las fallas como Normal. **La robustez del servidor se vuelve en su contra.**
2. **El quiebre ocurre dentro de la garantía de la defensa.** β=0.4 < 50 % es minoría Byzantine; AutoGM
   *debería* tolerarlo (y de hecho tolera el overt). Que el sigiloso lo rompa ahí es un fallo **dentro de
   su zona declarada de seguridad**, no un caso trivial de mayoría maliciosa.
3. **El efecto es bimodal, no determinista.** En 3/5 semillas el colapso es total (ASR=1.0); en 2/5 no
   ocurre. La varianza **es** parte del hallazgo: la defensa robusta no siempre es explotable, **pero
   cuando lo es, el fallo es catastrófico** — y en producción no se puede apostar a una partición
   "afortunada". Eso es exactamente lo que motiva una defensa de borde.
4. **La escala importaba (la intuición de "pocas instancias" era correcta).** A escala chica el efecto se
   enmascaraba; a escala realista emerge limpio. Por eso los tres experimentos son necesarios.

---

## 6. Amenazas a la validez (honestas)

- **Reimplementación propia** de AutoGM/D-WFA (no el código oficial). Mitigación: la réplica oficial sobre
  Bosch ya confirmó el comportamiento de AutoGM; queda pendiente portar CWRU al repo oficial (Track 2).
- **Solape de ventanas** (stride < ventana) puede filtrar información entre train/test → exactitud
  optimista; afecta por igual a todas las condiciones, así que no cambia el *contraste*.
- **Efecto acotado** a α=0.3 / β=0.4; no es un ataque universal, sino un fallo condicionado al régimen.
- **5 semillas** dan barras de error amplias (±0.49). Para publicación conviene subir a 10 semillas y
  mapear el umbral β con más finura.

---

## 7. Conclusión y siguiente paso

Las fases F1–F4 quedan **caracterizadas y confirmadas**: la agregación robusta de servidor (M2) es
**insuficiente y hasta contraproducente** frente al adversario dual coordinado sigiloso en FL-FDD bajo
alta heterogeneidad, mientras es robusta al ataque overt. Esto sostiene empíricamente la necesidad de
validar la integridad de señal en el **borde**.

**Siguiente:** **F5 — Mecanismo 1 (consenso multi-sensor DE-FE) acoplado a M2.** La predicción es que M1
detecta la incoherencia de señal que Adv1 introduce (la firma de falla desaparece de un sensor pero no del
otro) y **recupera la detección justo en el régimen (α=0.3, β=0.4) donde AutoGM colapsa** — cerrando el
ciclo ataque → vulnerabilidad → defensa.

---

*Datos y figuras: `results/`. Configuraciones: `config/v{1,2,3}_*.yaml`. Reproducible con
`grid_f4.py --config <cfg> [--jobs N]`. Bitácora: `../BITACORA_V2.md`.*
