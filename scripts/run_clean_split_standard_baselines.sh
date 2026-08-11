#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results/standard_baselines_clean_split}"
MODEL_DIR="${MODEL_DIR:-models/standard_baselines_clean_split}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.1}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-512}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
FORCE="${FORCE:-0}"

mkdir -p "$OUT_DIR" "$MODEL_DIR"

for model in gru_rnn bilstm_ma spatial_cnn; do
  for seed in 13 42 2026; do
    aggregate="$OUT_DIR/${model}_seed${seed}_aggregate_metrics.csv"
    if [[ "$FORCE" != "1" && -s "$aggregate" ]]; then
      echo "Skip existing $aggregate"
      continue
    fi
    echo "Training $model seed $seed"
    env PYTHONPATH=src "$PYTHON_BIN" scripts/train_clean_split_standard_baselines.py \
      --model "$model" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --hidden-size "$HIDDEN_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --train-ratio "$TRAIN_RATIO" \
      --val-ratio "$VAL_RATIO" \
      --device "$DEVICE" \
      --model-output "$MODEL_DIR/${model}_seed${seed}.pt" \
      --station-output "$OUT_DIR/${model}_seed${seed}_station_metrics.csv" \
      --aggregate-output "$aggregate" \
      --prediction-output "$OUT_DIR/${model}_seed${seed}_predictions.csv"
  done
done

echo "Clean-split standard baselines completed"
