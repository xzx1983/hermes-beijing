from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np
import torch

from pm25pred.hermes import (
    aggregate_pm_metrics,
    load_component_checkpoint,
    load_sequence_data,
    pm_metric_rows_for_target,
    predict_ensemble,
)


COMPONENTS = ["h1mid", "hardmid", "balanced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select fixed HERMES ensemble weights on the validation split.")
    parser.add_argument("--tensor", type=Path, default=Path("data/processed/beijing_3x4_hourly_historical_era5.npz"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/hermes"))
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.replace(" ", ",").split(",") if item]
    grid = [float(item) for item in args.grid.split(",") if item]
    data, grid_shape = load_sequence_data(args.tensor, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        models = [
            load_component_checkpoint(
                checkpoint_path=args.model_dir / f"{component}_seed{seed}.pt",
                input_size=data.x_train.shape[-1],
                station_count=data.x_train.shape[2],
                grid_shape=grid_shape,
                era5_size=len(data.era5_feature_names),
                forecast_hours=data.forecast_hours,
                device=args.device,
            )
            for component in COMPONENTS
        ]
        seed_rows = _score_seed(seed, models, grid, data, args.batch_size, args.device)
        best_index = min(range(len(seed_rows)), key=lambda index: float(seed_rows[index]["val_mean_rmse"]))
        for index, row in enumerate(seed_rows):
            row["selected"] = str(index == best_index).lower()
            rows.append(row)
    _write_csv(args.output, rows)
    selected = [row for row in rows if row["selected"] == "true"]
    print(f"wrote {args.output}")
    for row in selected:
        print(
            "seed={seed} weights=({weight_h1mid:.3f}, {weight_hardmid:.3f}, {weight_balanced:.3f}) "
            "val_mean_rmse={val_mean_rmse:.4f}".format(**row)
        )


def _score_seed(seed: int, models, grid: list[float], data, batch_size: int, device: str) -> list[dict[str, object]]:
    rows = []
    seen: set[tuple[float, ...]] = set()
    for raw_weights in product(grid, repeat=len(COMPONENTS)):
        if sum(raw_weights) <= 0:
            continue
        weights = [weight / sum(raw_weights) for weight in raw_weights]
        key = tuple(round(weight, 10) for weight in weights)
        if key in seen:
            continue
        seen.add(key)
        pm_norm = predict_ensemble(models, weights, data.x_val, batch_size=batch_size, device=device)
        station_rows = pm_metric_rows_for_target(pm_norm, data.pm_val.numpy(), data.pm_min, data.pm_max, data.station_names)
        aggregate_rows = aggregate_pm_metrics(station_rows)
        rows.append(
            {
                "seed": seed,
                "weight_h1mid": weights[0],
                "weight_hardmid": weights[1],
                "weight_balanced": weights[2],
                "val_mean_rmse": float(np.mean([float(row["rmse"]) for row in aggregate_rows])),
                "val_mean_mae": float(np.mean([float(row["mae"]) for row in aggregate_rows])),
                "selected": "false",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
