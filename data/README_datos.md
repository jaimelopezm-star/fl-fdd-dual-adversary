# Datos — CWRU Bearing Dataset

## Descarga
Fuente oficial: https://engineering.case.edu/bearingdatacenter/download-data-file
Los archivos vienen como `.mat` individuales (uno por condición). Colocarlos en `raw/`.

## ⚠️ Requisito CRÍTICO para el Mecanismo 1 Opción C (consenso DE-FE)
El Mecanismo 1 elegido compara los acelerómetros **Drive-End (DE)** y **Fan-End (FE)**. Por eso
necesitamos archivos que contengan **ambos canales simultáneamente**.

- Cada `.mat` de CWRU contiene variables tipo `XnnnDE_time`, `XnnnFE_time`, `XnnnBA_time` (no todos
  tienen los tres). El cargador (`load_cwru.py`) detecta y verifica DE+FE.
- **A verificar en F0:** confirmar que los archivos elegidos tienen DE **y** FE. Los datos de falla a
  **12 kHz Drive-End** típicamente traen DE+FE → priorizar esos. Si algún archivo solo trae DE, queda
  fuera del experimento de Opción C (se registra en el log).

## Subconjunto recomendado para el PoC (12 kHz Drive-End)
- **Normal** (sin falla): baseline (cargas 0–3 HP).
- **Inner Race (IR)**, **Outer Race (OR)**, **Ball (B)**: severidades 0.007", 0.014", 0.021".
- 4 clases × (severidades) × 4 cargas → suficiente para particionar en "plantas" non-IID.

## Convención de nombres CWRU (referencia)
| Archivo (ejemplo) | Condición | Carga |
|---|---|---|
| 97.mat–100.mat | Normal | 0–3 HP |
| 105.mat–108.mat | IR 0.007" | 0–3 HP |
| 130.mat–133.mat | OR 0.007" (@6) | 0–3 HP |
| 118.mat–121.mat | Ball 0.007" | 0–3 HP |
(La tabla completa está en la página de CWRU; `load_cwru.py` usa un mapeo configurable.)

## Estado
- [x] `.mat` descargados en `raw/`
- [x] Verificada disponibilidad de canales DE+FE
- [x] Mapeo archivo→(clase, severidad, carga) confirmado

## Verificación de integridad (`raw_manifest.md5`)
Los 16 `.mat` de `FILE_MAP` (`load_cwru.py`) NO se versionan (ver `.gitignore`), pero su md5 sí
queda registrado en `raw_manifest.md5` — generado sobre la copia usada en la campaña v1-v7
(pod RunPod, Aug 2026). Para regenerar y validar `raw/`:

```bash
python download_cwru.py          # baja los 16 .mat de FILE_MAP a raw/
cd raw && md5sum -c ../raw_manifest.md5   # compara contra el manifiesto versionado
```

Si algún hash no coincide, el archivo de CWRU cambió respecto al usado en la campaña original;
investigar antes de reusar resultados anteriores como referencia.
