#!/bin/bash
# Full appearance-eval pipeline: image similarity (N-CLIP + PL + LPIPS),
# CSV export, and per-scene side-by-side comparison figures.
#
# Run from anywhere. Outputs in evaluation/.
#
# Set the dataset / baseline paths via env vars (or edit the placeholders below):
#
#   export REF_DIR=/path/to/testing_first1
#   export REGEN_DIR=/path/to/3D-RE-GEN/results_batch
#   export GEN3DSR_DIR=/path/to/Gen3DSR/out/testing_first1
#   export HARMONY_ROOTS=/path/to/harmony/outputs/Demo/Artifact,/path/to/harmony/outputs/Demo/_processed
#   bash evaluation.sh

set -e

# ─── Config ───────────────────────────────────────────────────────────────────

PY=${PY:-python}                                   # python with torch + transformers
SCENEWEAVE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repo root (parent of evaluation/)

# Input photos (flat dir of {scene}.{jpg,png,avif,...})
REF_DIR=${REF_DIR:-/path/to/testing_first1}
# Baseline outputs (see loaders.py for the expected per-method layout)
REGEN_DIR=${REGEN_DIR:-/path/to/3D-RE-GEN/results_batch}
GEN3DSR_DIR=${GEN3DSR_DIR:-/path/to/Gen3DSR/out/testing_first1}
# HARMONY outputs: comma-separated roots, earlier roots win
ARTIFACT_DIR=${ARTIFACT_DIR:-/path/to/harmony/outputs/Demo/Artifact}
PROCESSED_DIR=${PROCESSED_DIR:-/path/to/harmony/outputs/Demo/_processed}
HARMONY_ROOTS=${HARMONY_ROOTS:-"$ARTIFACT_DIR,$PROCESSED_DIR"}
export REF_DIR REGEN_DIR GEN3DSR_DIR HARMONY_ROOTS

# Eval outputs (CLIP + photometric)
PIXEL_JSON=$SCENEWEAVE/evaluation/results_walltex.json
PIXEL_CSV=$SCENEWEAVE/evaluation/results_walltex.csv


# GPU to use (eval_runner is small)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Skip user-site so torch/CLIP load from the intended env cleanly
export PYTHONNOUSERSITE=1


for d in "$REF_DIR" "$REGEN_DIR" "$GEN3DSR_DIR"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: directory not found: $d  (set REF_DIR / REGEN_DIR / GEN3DSR_DIR / HARMONY_ROOTS)" >&2
        exit 1
    fi
done

cd "$SCENEWEAVE"

# ─── 1. Image-similarity eval (CLIP + photometric MSE) ────────────────────────

echo "===== 1. eval_runner.py — CLIP + photometric =====               $(date +%H:%M:%S)"
$PY evaluation/eval_runner.py \
    --ref  "flat:$REF_DIR" \
    --pred "Gen3DSR=gen3dsr:$GEN3DSR_DIR" \
    --pred "3DREGEN=3dregen:$REGEN_DIR" \
    --pred "Harmony=harmony:$HARMONY_ROOTS" \
    --out  "$PIXEL_JSON" 2>&1 | grep -vE "size mismatch|^  [A-Z]" | tail -12

echo
echo "===== 2. json_to_csv (pixel metrics) =====                       $(date +%H:%M:%S)"
$PY evaluation/json_to_csv.py "$PIXEL_JSON" "$PIXEL_CSV"


# ─── 3. Per-scene comparison figures ──────────────────────────────────────────

echo
echo "===== 3. make_comparison_figures.py — side-by-side panels =====  $(date +%H:%M:%S)"
$PY evaluation/make_comparison_figures.py 2>&1 | tail -5

# ─── Summary ──────────────────────────────────────────────────────────────────

echo
echo "===== DONE $(date) ====="
echo
echo "Outputs:"
echo "  - $PIXEL_JSON / $PIXEL_CSV       (CLIP + photometric)"
echo "  - $SCENEWEAVE/evaluation/comparison_figures/      (per-scene PNGs + index.html)"

# Print key averages
$PY -c "
import json
d = json.load(open('$PIXEL_JSON'))
methods = list(d['methods'].keys())
def valid(m): return {k for k,v in d['methods'][m]['per_scene'].items() if 'n_clip' in v}
inter = set.intersection(*(valid(m) for m in methods))
print()
print(f'=== INTERSECT ({len(inter)} scenes) ===')
for m in methods:
    per = d['methods'][m]['per_scene']
    nc = [per[k]['n_clip'] for k in inter]
    pl = [per[k]['pl'] for k in inter]
    print(f'  {m:<10}  n_clip={sum(nc)/len(nc):.4f}  pl={sum(pl)/len(pl):.4f}')
"
