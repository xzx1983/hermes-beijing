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


class STGCNBaseline(nn.Module):
    def __init__(
        self,
        input_size: int,
        station_count: int,
        forecast_hours: int,
        hidden_size: int,
        grid_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        self.register_buffer("supports", _chebyshev_supports(_grid_adjacency(grid_shape), order=3))
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.block1 = STConvBlock(hidden_size, hidden_size, support_count=3)
        self.block2 = STConvBlock(hidden_size, hidden_size, support_count=3)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, forecast_hours)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(x))
        h = self.block1(h, self.supports)
        h = self.block2(h, self.supports)
        return self.head(self.norm(h[:, -1])).permute(0, 2, 1)


class GraphWaveNetBaseline(nn.Module):
    def __init__(
        self,
        input_size: int,
        station_count: int,
        forecast_hours: int,
        hidden_size: int,
        grid_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        self.node_source = nn.Parameter(torch.randn(station_count, hidden_size) * 0.05)
        self.node_target = nn.Parameter(torch.randn(station_count, hidden_size) * 0.05)
        self.register_buffer("fixed_support", _grid_adjacency(grid_shape))
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [GraphWaveNetBlock(hidden_size, hidden_size, dilation=dilation, support_count=2) for dilation in [1, 2, 4, 8]]
        )
        self.skip_proj = nn.Linear(hidden_size * len(self.blocks), hidden_size)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, forecast_hours))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adaptive = torch.softmax(torch.relu(self.node_source @ self.node_target.T), dim=-1)
        supports = [self.fixed_support, adaptive]
        h = torch.relu(self.input_proj(x))
        skips = []
        for block in self.blocks:
            h, skip = block(h, supports)
            skips.append(skip[:, -1])
        state = self.skip_proj(torch.cat(skips, dim=-1))
        return self.head(state).permute(0, 2, 1)


class AGCRNBaseline(nn.Module):
    def __init__(
        self,
        input_size: int,
        station_count: int,
        forecast_hours: int,
        hidden_size: int,
        grid_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        del grid_shape
        self.station_count = station_count
        self.hidden_size = hidden_size
        self.node_embeddings = nn.Parameter(torch.randn(station_count, hidden_size) * 0.05)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.cell = AGCRNCell(hidden_size, hidden_size, embedding_size=hidden_size)
        self.head = nn.Linear(hidden_size, forecast_hours)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(x))
        state = torch.zeros(h.shape[0], self.station_count, self.hidden_size, dtype=h.dtype, device=h.device)
        adjacency = torch.softmax(torch.relu(self.node_embeddings @ self.node_embeddings.T), dim=-1)
        for step in range(h.shape[1]):
            state = self.cell(h[:, step], state, adjacency, self.node_embeddings)
        return self.head(state).permute(0, 2, 1)


class GraphConv(nn.Module):
    def __init__(self, input_size: int, output_size: int, support_count: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size * support_count, output_size)

    def forward(self, x: torch.Tensor, supports: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        support_list = list(supports) if isinstance(supports, torch.Tensor) else supports
        propagated = [torch.einsum("btnd,nm->btmd", x, support) for support in support_list]
        return self.proj(torch.cat(propagated, dim=-1))


class STConvBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, support_count: int) -> None:
        super().__init__()
        self.temporal_in = GatedTemporalConv(input_size, hidden_size, dilation=1)
        self.graph = GraphConv(hidden_size, hidden_size, support_count=support_count)
        self.temporal_out = GatedTemporalConv(hidden_size, hidden_size, dilation=1)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.temporal_in(x)
        h = torch.relu(self.graph(h, supports))
        h = self.temporal_out(h)
        if residual.shape[-1] == h.shape[-1]:
            h = h + residual[:, -h.shape[1] :]
        return self.norm(h)


class GatedTemporalConv(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.filter = nn.Conv1d(input_size, hidden_size, kernel_size=2, dilation=dilation, padding=padding)
        self.gate = nn.Conv1d(input_size, hidden_size, kernel_size=2, dilation=dilation, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, stations, channels = x.shape
        series = x.permute(0, 2, 3, 1).reshape(batch * stations, channels, time_steps)
        filtered = torch.tanh(self.filter(series))[..., :time_steps]
        gated = torch.sigmoid(self.gate(series))[..., :time_steps]
        out = filtered * gated
        return out.reshape(batch, stations, -1, time_steps).permute(0, 3, 1, 2)


class GraphWaveNetBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dilation: int, support_count: int) -> None:
        super().__init__()
        self.temporal = GatedTemporalConv(input_size, hidden_size, dilation=dilation)
        self.graph = GraphConv(hidden_size, hidden_size, support_count=support_count)
        self.residual = nn.Linear(input_size, hidden_size)
        self.skip = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.temporal(x)
        h = torch.relu(self.graph(h, supports))
        residual = self.residual(x[:, -h.shape[1] :])
        h = self.norm(h + residual)
        return h, self.skip(h)


class AGCRNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, embedding_size: int) -> None:
        super().__init__()
        self.gate = AdaptiveGraphConv(input_size + hidden_size, 2 * hidden_size, embedding_size)
        self.update = AdaptiveGraphConv(input_size + hidden_size, hidden_size, embedding_size)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        adjacency: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([x, state], dim=-1)
        gates = torch.sigmoid(self.gate(combined, adjacency, node_embeddings))
        reset, update = torch.chunk(gates, 2, dim=-1)
        candidate = torch.tanh(self.update(torch.cat([x, reset * state], dim=-1), adjacency, node_embeddings))
        return update * state + (1.0 - update) * candidate


class AdaptiveGraphConv(nn.Module):
    def __init__(self, input_size: int, output_size: int, embedding_size: int) -> None:
        super().__init__()
        self.weight_pool = nn.Parameter(torch.randn(embedding_size, input_size, output_size) * 0.05)
        self.bias_pool = nn.Parameter(torch.zeros(embedding_size, output_size))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("bnc,nm->bmc", x, adjacency)
        weights = torch.einsum("ne,eio->nio", node_embeddings, self.weight_pool)
        bias = torch.einsum("ne,eo->no", node_embeddings, self.bias_pool)
        return torch.einsum("bni,nio->bno", propagated, weights) + bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clean-split graph baselines for the HERMES manuscript.")
    parser.add_argument("--tensor", type=Path, default=Path("data/processed/beijing_3x4_hourly_historical_era5.npz"))
    parser.add_argument("--model", choices=["stgcn", "graph_wavenet", "agcrn"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
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
    model = _build_model(args.model, data, grid_shape, args.hidden_size).to(args.device)
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


def _build_model(name: str, data: SequenceData, grid_shape: tuple[int, int], hidden_size: int) -> nn.Module:
    kwargs = {
        "input_size": data.x_train.shape[-1],
        "station_count": data.x_train.shape[2],
        "forecast_hours": data.forecast_hours,
        "hidden_size": hidden_size,
        "grid_shape": grid_shape,
    }
    if name == "stgcn":
        return STGCNBaseline(**kwargs)
    if name == "graph_wavenet":
        return GraphWaveNetBaseline(**kwargs)
    if name == "agcrn":
        return AGCRNBaseline(**kwargs)
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


def _grid_adjacency(grid_shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = grid_shape
    station_count = rows * cols
    adjacency = torch.eye(station_count, dtype=torch.float32)
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            for next_row, next_col in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                if 0 <= next_row < rows and 0 <= next_col < cols:
                    adjacency[index, next_row * cols + next_col] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def _chebyshev_supports(adjacency: torch.Tensor, order: int) -> torch.Tensor:
    supports = [torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)]
    if order > 1:
        supports.append(adjacency)
    for _ in range(2, order):
        supports.append(2 * adjacency @ supports[-1] - supports[-2])
    return torch.stack(supports[:order], dim=0)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
