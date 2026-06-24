# Experimento V2 — FL-FDD Robusto a Adversario Dual Coordinado

Código del experimento de tesis. Diseño completo en `../planteamiento_experimento_V2.md`.

## Decisiones (2026-06-23)
- **Framework FL:** loop propio en PyTorch (control total de agregación e inyección de ataques).
- **Modelo base:** 1D-CNN sobre señal de vibración cruda.
- **Mecanismo 1 (PoC):** Opción C — consenso multi-sensor **Drive-End (DE) vs Fan-End (FE)** de CWRU.
- **Mecanismo 2 (baseline):** AutoGM (Li 2021); comparar contra FedAvg, D-WFA (Chen 2022), Hathout (2024).

## Entorno
- **Python 3.12** — usar SIEMPRE `py -3.12` (el `python` del PATH es MSYS2 3.14 sin pip, NO usar).
- Paquetes ya instalados: torch, numpy, scipy, scikit-learn, matplotlib, PyYAML.
- 100% simulación sobre CPU (CWRU es pequeño); GPU opcional.

## Cómo correr (cuando estén las fases listas)
```
py -3.12 run.py --config config/exp_template.yaml
```

## Estructura
```
config/      YAMLs por experimento (γ, β, α, κ, defensa)
data/        carga y partición de CWRU (raw/ = .mat originales)
models/      1D-CNN
attacks/     adv1_fdi.py, adv2_model.py, adv2_data.py, coordinated.py
defenses/    fedavg.py, autogm.py, dwfa.py, hathout.py, mecanismo1.py
federated/   loop FL (cliente/servidor/ronda)
metrics/     DR, FAR, ASR, overhead
results/     logs, CSVs, figuras
run.py       orquestador
```

## Estado de fases
- [x] **F0** — entorno + estructura + cargador CWRU (en curso)
- [ ] **F1** — baseline limpio (1D-CNN federado, FedAvg)
- [ ] **F2** — reproducir M2 (AutoGM, D-WFA)
- [ ] **F3** — ataques (Adv1 FDI, Adv2 model/data, coordinado)
- [ ] **F4** — caracterización de vulnerabilidad ★
- [ ] **F5** — Mecanismo 1 + acoplamiento M1↔M2
- [ ] **F6** — evaluación dual + figuras
