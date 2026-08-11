from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pm25pred.hermes import SequenceData, aggregate_pm_metrics, load_sequence_data, pm_metric_rows, prediction_rows


class StationGRURNN(nn.Module):
    def __init__(self, input_size: int, station_count: int, forecast_hours: int, hidden_size: int) -> None:
        super().__init__()
        self.station_count = station_count
        self.station_embedding = nn.Embedding(station_count, hidden_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size * 2, forecast_hours)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, station_count, _ = x.shape
        h = torch.relu(self.input_proj(x))
        series = h.permute(0, 2, 1, 3).reshape(batch_size * station_count, time_steps, -1)
        _, gru_state = self.gru(series)
        _, rnn_state = self.rnn(series)
        state = torch.cat([gru_state[-1], rnn_state[-1]], dim=-1).reshape(batch_size, station_count, -1)
        station_ids = torch.arange(station_count, device=x.device)
        station_bias = self.station_embedding(station_ids).unsqueeze(0)
        pred = self.head(state + torch.cat([station_bias, station_bias], dim=-1))
        return pred.permute(0, 2, 1)


class StationBiLSTMMA(nn.Module):
    def __init__(self, input_size: int, station_count: int, forecast_hours: int, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.station_count = station_count
        self.station_embedding = nn.Embedding(station_count, hidden_size * 2)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Linear(hidden_size * 2, forecast_hours)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, station_count, _ = x.shape
        h = torch.relu(self.input_proj(x))
        series = h.permute(0, 2, 1, 3).reshape(batch_size * station_count, time_steps, -1)
        encoded, _ = self.lstm(series)
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        state = self.norm(encoded[:, -1] + attended[:, -1]).reshape(batch_size, station_count, -1)
        station_ids = torch.arange(station_count, device=x.device)
        state = state + self.station_embedding(station_ids).unsqueeze(0)
        return self.head(state).permute(0, 2, 1)


class SpatialCNNBaseline(nn.Module):
    def __init__(
        self,
        input_size: int,
        station_count: int,
        grid_shape: tuple[int, int],
        lookback: int,
        forecast_hours: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        rows, cols = grid_shape
        if rows * cols != station_count:
            raise ValueError(f"grid_shape {grid_shape} does not match station_count {station_count}")
        self.rows = rows
        self.cols = cols
        self.lookback = lookback
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size * lookback, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(hidden_size, forecast_hours, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, station_count, channels = x.shape
        grid = x.reshape(batch_size, time_steps, self.rows, self.cols, channels)
        grid = grid.permute(0, 1, 4, 2, 3).reshape(batch_size, time_steps * channels, self.rows, self.cols)
        pred = self.head(self.encoder(grid))
        return pred.reshape(batch_size, pred.shape[1], station_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clean-split station-wise and Spatial CNN baselines.")
    parser.add_argument("--tensor", type=Path, default=Path("data/processed/beijing_3x4_hourly_historical_era5.npz"))
    parser.add_argument("--model", choices=["gru_rnn", "bilstm_ma", "spatial_cnn"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--station-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    data, grid_shape = load_sequence_data(args.tensor, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    model = _build_model(args.model, data, grid_shape, args.hidden_size, args.attention_heads).to(args.device)
    history = _train(model, data, args.epochs, args.batch_size, args.learning_rate, args.device)
    pm_norm = _predict(model, data.x_test, args.batch_size, args.device)
    station_rows = pm_metric_rows(pm_norm, data)
    aggregate_rows = aggregate_pm_metrics(station_rows)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "input_feature_names": data.input_feature_names,
            "history": history,
            "aggregate_metrics": aggregate_rows,
            "station_metrics": station_rows,
        },
        args.model_output,
    )
    _write_csv(args.station_output, station_rows)
    _write_csv(args.aggregate_output, aggregate_rows)
    _write_csv(args.prediction_output, prediction_rows(pm_norm, data))
    print(
        json.dumps(
            {
                "model": args.model,
                "seed": args.seed,
                "train_sequences": int(data.x_train.shape[0]),
                "val_sequences": int(data.x_val.shape[0]),
                "test_sequences": int(data.x_test.shape[0]),
                "aggregate_output": str(args.aggregate_output),
                "mean_rmse": float(np.mean([row["rmse"] for row in aggregate_rows])),
                "last_epoch": history[-1],
            },
            indent=2,
        )
    )


def _build_model(
    name: str,
    data: SequenceData,
    grid_shape: tuple[int, int],
    hidden_size: int,
    attention_heads: int,
) -> nn.Module:
    input_size = data.x_train.shape[-1]
    station_count = data.x_train.shape[2]
    lookback = data.x_train.shape[1]
    if name == "gru_rnn":
        return StationGRURNN(input_size, station_count, data.forecast_hours, hidden_size)
    if name == "bilstm_ma":
        return StationBiLSTMMA(input_size, station_count, data.forecast_hours, hidden_size, attention_heads)
    if name == "spatial_cnn":
        return SpatialCNNBaseline(input_size, station_count, grid_shape, lookback, data.forecast_hours, hidden_size)
    raise ValueError(name)


def _train(
    model: nn.Module,
    data: SequenceData,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data.x_train, data.pm_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data.x_val, data.pm_val),
        batch_size=batch_size,
        shuffle=False,
    )
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = torch.mean(torch.abs(model(x_batch) - y_batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                val_loss += torch.mean(torch.abs(model(x_batch) - y_batch)).item() * len(x_batch)
        mean_val_loss = val_loss / len(data.x_val)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": float(epoch),
                "train_mae_normalized": train_loss / len(data.x_train),
                "val_mae_normalized": mean_val_loss,
                "is_best_validation": float(epoch == best_epoch),
            }
        )
    model.load_state_dict(best_state)
    return history


def _predict(model: nn.Module, x_test: torch.Tensor, batch_size: int, device: str) -> np.ndarray:
    loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)
    chunks = []
    model.eval()
    with torch.no_grad():
        for x_batch in loader:
            chunks.append(model(x_batch.to(device)).cpu())
    return torch.cat(chunks, dim=0).numpy()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
