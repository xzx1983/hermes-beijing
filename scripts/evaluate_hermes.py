from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from pm25pred.hermes import (
    aggregate_pm_metrics,
    load_component_checkpoint,
    load_sequence_data,
    pm_metric_rows,
    pm_metric_rows_for_target,
    prediction_rows,
    predict_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fixed-weight HERMES ensemble.")
    parser.add_argument("--tensor", type=Path, default=Path("data/processed/beijing_3x4_hourly_historical_era5.npz"))
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--forecast-hours", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--no-era5-input", action="store_true")
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--weight", action="append", type=float)
    parser.add_argument("--weight-table", type=Path, help="Validation-selected weight table produced by select_hermes_weights.py.")
    parser.add_argument("--seed", type=int, help="Seed row to read from --weight-table.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--station-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = _resolve_weights(args)
    if len(args.checkpoint) != len(weights):
        raise ValueError("--checkpoint and --weight must have the same length")
    data, grid_shape = load_sequence_data(
        args.tensor,
        lookback=args.lookback,
        forecast_hours=args.forecast_hours,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        include_era5_input=not args.no_era5_input,
    )
    models = [
        load_component_checkpoint(
            checkpoint_path=path,
            input_size=data.x_train.shape[-1],
            station_count=data.x_train.shape[2],
            grid_shape=grid_shape,
            era5_size=len(data.era5_feature_names),
            forecast_hours=args.forecast_hours,
            device=args.device,
        )
        for path in args.checkpoint
    ]
    x_eval = data.x_val if args.split == "val" else data.x_test
    y_eval = data.pm_val if args.split == "val" else data.pm_test
    pm_norm = predict_ensemble(models, weights, x_eval, batch_size=args.batch_size, device=args.device)
    if args.split == "test":
        station_rows = pm_metric_rows(pm_norm, data)
    else:
        station_rows = pm_metric_rows_for_target(pm_norm, y_eval.numpy(), data.pm_min, data.pm_max, data.station_names)
    aggregate_rows = aggregate_pm_metrics(station_rows)
    _write_csv(args.station_output, station_rows)
    _write_csv(args.aggregate_output, aggregate_rows)
    if args.prediction_output:
        _write_csv(args.prediction_output, prediction_rows(pm_norm, data))
    print(
        json.dumps(
            {
                "checkpoints": [str(path) for path in args.checkpoint],
                "weights": _normalize(weights),
                "split": args.split,
                "station_output": str(args.station_output),
                "aggregate_output": str(args.aggregate_output),
                "prediction_output": str(args.prediction_output) if args.prediction_output else None,
                "train_sequences": int(data.x_train.shape[0]),
                "val_sequences": int(data.x_val.shape[0]),
                "test_sequences": int(data.x_test.shape[0]),
                "train_boundary": int(data.train_boundary),
                "val_boundary": int(data.val_boundary),
                "aggregate_metrics": aggregate_rows,
            },
            indent=2,
        )
    )


def _normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return [weight / total for weight in weights]


def _resolve_weights(args: argparse.Namespace) -> list[float]:
    if args.weight_table:
        if args.seed is None:
            raise ValueError("--seed is required with --weight-table")
        return _weights_from_table(args.weight_table, args.seed, args.checkpoint)
    if args.weight is None:
        raise ValueError("Provide --weight values or --weight-table")
    return args.weight


def _weights_from_table(path: Path, seed: int, checkpoints: list[Path]) -> list[float]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    matches = [row for row in rows if int(row["seed"]) == seed and row.get("selected", "").lower() == "true"]
    if not matches:
        raise ValueError(f"No selected weight row found for seed {seed} in {path}")
    row = matches[0]
    weights = []
    for checkpoint in checkpoints:
        variant = checkpoint.stem.split("_seed", 1)[0]
        column = f"weight_{variant}"
        if column not in row:
            raise ValueError(f"Missing {column} in {path}")
        weights.append(float(row[column]))
    return weights


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
