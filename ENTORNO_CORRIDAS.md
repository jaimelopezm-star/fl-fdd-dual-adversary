# Entorno de ejecución (pod RunPod, campaña v4-v7)

Registro del hardware/software usado en las corridas de la campaña F4 (v1-v7), antes de
liberar el almacenamiento del pod. Complementa `requirements.txt` (raíz del repo).

## Hardware

- GPU: `NVIDIA RTX 4000 Ada Generation` (20475 MiB)
- Driver NVIDIA: `550.127.08`, CUDA `12.4`
- Usado para las corridas v4, v5, v6 y v7 (la campaña v1-v3 corrió en un pod previo, RTX 4090
  para v4 según el mensaje de commit `32fe9ba`; v6/v7 se re-corrieron en este pod RTX 4000 Ada
  con el parche de matriz de confusión / ASR_target — commit `c66d0b8`).

## Software

- Python `3.11.10`
- Paquetes clave (ver `requirements.txt` completo, `pip freeze` del venv):
  - `torch==2.6.0+cu124`
  - `numpy==2.4.6`
  - `scipy==1.17.1`
  - `PyYAML==6.0.3`

## Cómo se lanzó la corrida v6/v7

Script `run_v6v7.sh` (raíz del repo):

```bash
cd /workspace/fl-fdd
source /workspace/venv/bin/activate
python grid_f4.py --config config/v6_pfl_flip.yaml --device cuda --jobs 8 --resume
python grid_f4.py --config config/v7_blocked.yaml --device cuda --jobs 8
```

## Duración medida

- v7_blocked: 400 celdas en 1h 56min 59s (~17.55 s/celda promedio, `--jobs 8`).
- Log completo del driver (timestamps de inicio/fin de cada etapa, progreso por celda):
  `results/run_v6v7_driver.log`.

## Reproducibilidad

1. Crear un venv con Python 3.11 e instalar `pip install -r requirements.txt`.
2. Regenerar `data/raw/` con `data/download_cwru.py` (descarga oficial de CWRU) y verificar
   contra `data/raw_manifest.md5`.
3. Correr `python grid_f4.py --config config/<version>.yaml --device cuda --jobs 8` (usar
   `--resume` si se retoma un checkpoint `.jsonl` existente).

Sin GPU, los resultados ya versionados en `results/` (JSON + checkpoints `.jsonl`) bastan para
regenerar las figuras y reportes (`make_figures_f4.py`, `make_figures_v7.py`, `report_f4.py`)
sin volver a correr el experimento.
