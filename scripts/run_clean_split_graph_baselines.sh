#!/usr/bin/env bash
set -euo pipefail

TENSOR="${TENSOR:-data/processed/beijing_3x4_hourly_historical_era5.npz}"
SEEDS="${SEEDS:-13 42 2026}"
MODELS="${MODELS:-stgcn graph_wavenet agcrn}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-remote_results/graph_baselines_clean_split}"
MODEL_DIR="${MODEL_DIR:-models/graph_baselines_clean_split}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.1}"
EPOCHS="${EPOCHS:-20}"
FORCE="${FORCE:-0}"

mkdir -p "$OUT_DIR" "$MODEL_DIR"

run_if_missing() {
  local marker="$1"
  shift
  if [[ "$FORCE" != "1" && -s "$marker" ]]; then
    echo "[skip] $marker"
    return
  fi
  echo "[run] $*"
  "$@"
}

for model in $MODELS; do
  for seed in $SEEDS; do
    run_if_missing \
      "$OUT_DIR/${model}_seed${seed}_aggregate_metrics.csv" \
      env PYTHONPATH=src "$PYTHON_BIN" scripts/train_clean_split_graph_baselines.py \
        --tensor "$TENSOR" \
        --model "$model" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch-size 512 \
        --hidden-size 64 \
        --learning-rate 1e-3 \
        --train-ratio "$TRAIN_RATIO" \
        --val-ratio "$VAL_RATIO" \
        --device "$DEVICE" \
        --model-output "$MODEL_DIR/${model}_seed${seed}.pt" \
        --station-output "$OUT_DIR/${model}_seed${seed}_station_metrics.csv" \
        --aggregate-output "$OUT_DIR/${model}_seed${seed}_aggregate_metrics.csv" \
        --prediction-output "$OUT_DIR/${model}_seed${seed}_predictions.csv"
  done
done

echo "Clean-split graph baselines completed"
