#!/bin/bash
# Tüm hattı sırayla çalıştırır. 01 (çıkarım) zaten yapıldıysa atlanır.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY scripts/01_extract.py
$PY scripts/02_cka_scan.py
MCT_DEVICE=cpu $PY scripts/03_train_ladder.py
MCT_DEVICE=cpu $PY scripts/04_semantic_tests.py
MCT_DEVICE=cpu $PY scripts/05_neutral_joint.py
MCT_DEVICE=cpu $PY scripts/06_depth_translation.py
$PY scripts/07_sae_bridge.py
$PY scripts/08_patching.py
$PY scripts/09_report.py
echo "ALL DONE"
