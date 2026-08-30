#!/bin/bash
set -e
cd /workspace/fl-fdd
source /workspace/venv/bin/activate
echo "=== v6 start $(date) ==="
python grid_f4.py --config config/v6_pfl_flip.yaml --device cuda --jobs 8 --resume
echo "=== v6 done $(date) ==="
echo "=== v7 start (SIN resume, corrida fresca) $(date) ==="
python grid_f4.py --config config/v7_blocked.yaml --device cuda --jobs 8
echo "=== v7 done $(date) ==="
touch /workspace/fl-fdd/RUN_DONE
