from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from pm25pred.hermes import (
    HermesComponent,
    aggregate_pm_metrics,
    evaluate_component,
    load_sequence_data,
    prediction_rows,
    train_component,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one HERMES component checkpoint.")
    parser.add_argument("--tensor", type=Path, default=Path("data/processed/beijing_3x4_hourly_historical_era5.npz"))
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--forecast-hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--no-era5-input", action="store_true")
    parser.add_argument("--era5-aux-weight", type=float, default=0.1)
    parser.add_argument("--route-entropy-weight", type=float, default=0.001)
    parser.add_argument("--horizon-loss-power", type=float, default=0.5)
    parser.add_argument("--h1-focus-weight", type=float, default=0.5)
    parser.add_argument("--mid-focus-weight", type=float, default=0.35)
    parser.add_argument("--focus-start-hour", type=int, default=3)
    parser.add_argument("--focus-end-hour", type=int, default=10)
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--no-routing", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-output", type=Path, default=Path("models/hermes_component.pt"))
    parser.add_argument("--station-output", type=Path, default=Path("models/hermes_component_station_metrics.csv"))
    parser.add_argument("--aggregate-output", type=Path, default=Path("models/hermes_component_aggregate_metrics.csv"))
    parser.add_argument("--era5-output", type=Path, default=Path("models/hermes_component_era5_metrics.csv"))
    parser.add_argument("--routes-output", type=Path, default=Path("models/hermes_component_routes.csv"))
    parser.add_argument("--prediction-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    data, grid_shape = load_sequence_data(
        args.tensor,
        lookback=args.lookback,
        forecast_hours=args.forecast_hours,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        include_era5_input=not args.no_era5_input,
    )
    model = HermesComponent(
        input_size=data.x_train.shape[-1],
        station_count=data.x_train.shape[2],
        grid_shape=grid_shape,
        forecast_hours=args.forecast_hours,
        era5_size=len(data.era5_feature_names),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        use_temporal=not args.no_temporal,
        use_spatial=not args.no_spatial,
        use_routing=not args.no_routing,
        use_residual=not args.no_residual,
    )
    args.use_temporal = not args.no_temporal
    args.use_spatial = not args.no_spatial
    args.use_routing = not args.no_routing
    args.use_residual = not args.no_residual
    history = train_component(
        model,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        era5_aux_weight=args.era5_aux_weight,
        route_entropy_weight=args.route_entropy_weight,
        horizon_loss_power=args.horizon_loss_power,
        h1_focus_weight=args.h1_focus_weight,
        mid_focus_weight=args.mid_focus_weight,
        focus_start_hour=args.focus_start_hour,
        focus_end_hour=args.focus_end_hour,
        device=args.device,
    )
    station_rows, era5_rows, route_rows = evaluate_component(model, data, batch_size=args.batch_size, device=args.device)
    aggregate_rows = aggregate_pm_metrics(station_rows)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "input_feature_names": data.input_feature_names,
            "era5_feature_names": data.era5_feature_names,
            "station_grid_shape": grid_shape,
            "history": history,
            "aggregate_metrics": aggregate_rows,
            "station_metrics": station_rows,
            "era5_metrics": era5_rows,
            "route_metrics": route_rows,
        },
        args.model_output,
    )
    _write_csv(args.station_output, station_rows)
    _write_csv(args.aggregate_output, aggregate_rows)
    _write_csv(args.era5_output, era5_rows)
    _write_csv(args.routes_output, route_rows)
    if args.prediction_output:
        pm_norm = _predict_component(model, data, batch_size=args.batch_size, device=args.device)
        _write_csv(args.prediction_output, prediction_rows(pm_norm, data))
    print(
        json.dumps(
            {
                "model": str(args.model_output),
                "aggregate_output": str(args.aggregate_output),
                "train_sequences": int(data.x_train.shape[0]),
                "val_sequences": int(data.x_val.shape[0]),
                "test_sequences": int(data.x_test.shape[0]),
                "train_boundary": int(data.train_boundary),
                "val_boundary": int(data.val_boundary),
                "aggregate_metrics": aggregate_rows,
                "last_epoch": history[-1] if history else None,
            },
            indent=2,
        )
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _predict_component(
    model: HermesComponent,
    data,
    batch_size: int,
    device: str,
) -> np.ndarray:
    chunks = []
    loader = torch.utils.data.DataLoader(data.x_test, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for x_batch in loader:
            pm_pred, _ = model(x_batch.to(device))
            chunks.append(pm_pred.cpu())
    return torch.cat(chunks, dim=0).numpy()


if __name__ == "__main__":
    main()
