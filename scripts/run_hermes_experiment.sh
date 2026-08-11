#!/usr/bin/env bash
set -euo pipefail

TENSOR="${TENSOR:-data/processed/beijing_3x4_hourly_historical_era5.npz}"
SEEDS="${SEEDS:-13 42 2026}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-remote_results/hermes_clean_split}"
MODEL_DIR="${MODEL_DIR:-models/hermes_clean_split}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.1}"
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

train_component() {
  local variant="$1"
  local seed="$2"
  local epochs="$3"
  local lr="$4"
  local era5_weight="$5"
  local horizon_power="$6"
  local h1_weight="$7"
  local mid_weight="$8"
  local model_path="$MODEL_DIR/${variant}_seed${seed}.pt"
  run_if_missing \
    "$model_path" \
    env PYTHONPATH=src "$PYTHON_BIN" scripts/train_hermes_component.py \
      --tensor "$TENSOR" \
      --seed "$seed" \
      --epochs "$epochs" \
      --batch-size 256 \
      --hidden-size 64 \
      --train-ratio "$TRAIN_RATIO" \
      --val-ratio "$VAL_RATIO" \
      --learning-rate "$lr" \
      --era5-aux-weight "$era5_weight" \
      --route-entropy-weight 0.001 \
      --horizon-loss-power "$horizon_power" \
      --h1-focus-weight "$h1_weight" \
      --mid-focus-weight "$mid_weight" \
      --focus-start-hour 3 \
      --focus-end-hour 10 \
      --device "$DEVICE" \
      --model-output "$model_path" \
      --station-output "$OUT_DIR/${variant}_seed${seed}_station_metrics.csv" \
      --aggregate-output "$OUT_DIR/${variant}_seed${seed}_aggregate_metrics.csv" \
      --era5-output "$OUT_DIR/${variant}_seed${seed}_era5_metrics.csv" \
      --routes-output "$OUT_DIR/${variant}_seed${seed}_routes.csv" \
      --prediction-output "$OUT_DIR/${variant}_seed${seed}_predictions.csv"
}

for seed in $SEEDS; do
  train_component h1mid "$seed" 20 1e-3 0.08 0.5 0.25 0.9
  train_component hardmid "$seed" 30 9e-4 0.06 0.45 0.15 1.5
  train_component balanced "$seed" 24 8e-4 0.08 0.5 0.2 1.2
done

WEIGHT_TABLE="$OUT_DIR/hermes_validation_weight_grid.csv"
run_if_missing \
  "$WEIGHT_TABLE" \
  env PYTHONPATH=src "$PYTHON_BIN" scripts/select_hermes_weights.py \
    --tensor "$TENSOR" \
    --model-dir "$MODEL_DIR" \
    --seeds "${SEEDS// /,}" \
    --train-ratio "$TRAIN_RATIO" \
    --val-ratio "$VAL_RATIO" \
    --device "$DEVICE" \
    --output "$WEIGHT_TABLE"

for seed in $SEEDS; do
  run_if_missing \
    "$OUT_DIR/hermes_seed${seed}_aggregate_metrics.csv" \
    env PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_hermes.py \
      --tensor "$TENSOR" \
      --train-ratio "$TRAIN_RATIO" \
      --val-ratio "$VAL_RATIO" \
      --checkpoint "$MODEL_DIR/h1mid_seed${seed}.pt" \
      --checkpoint "$MODEL_DIR/hardmid_seed${seed}.pt" \
      --checkpoint "$MODEL_DIR/balanced_seed${seed}.pt" \
      --weight-table "$WEIGHT_TABLE" \
      --seed "$seed" \
      --split test \
      --device "$DEVICE" \
      --station-output "$OUT_DIR/hermes_seed${seed}_station_metrics.csv" \
      --aggregate-output "$OUT_DIR/hermes_seed${seed}_aggregate_metrics.csv" \
      --prediction-output "$OUT_DIR/hermes_seed${seed}_predictions.csv"
done

echo "HERMES experiment completed"
