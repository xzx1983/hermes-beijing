from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pm25pred.beijing_data import load_tensor


MODEL_NAME = "HERMES"
MODEL_FULL_NAME = "Horizon-specialized Ensemble Residual Meteorology-Enhanced System"

DEFAULT_ERA5_FEATURES = [
    "u10",
    "v10",
    "t2m",
    "d2m",
    "sp",
    "tp",
    "blh",
    "era5_wind_speed",
    "era5_flow_sin",
    "era5_flow_cos",
]


@dataclass(frozen=True)
class SequenceData:
    x_train: torch.Tensor
    pm_train: torch.Tensor
    era5_train: torch.Tensor
    x_val: torch.Tensor
    pm_val: torch.Tensor
    era5_val: torch.Tensor
    x_test: torch.Tensor
    pm_test: torch.Tensor
    era5_test: torch.Tensor
    train_target_start: np.ndarray
    train_target_end: np.ndarray
    val_target_start: np.ndarray
    val_target_end: np.ndarray
    test_target_start: np.ndarray
    test_target_end: np.ndarray
    train_boundary: int
    val_boundary: int
    input_min: np.ndarray
    input_max: np.ndarray
    pm_min: np.ndarray
    pm_max: np.ndarray
    era5_min: np.ndarray
    era5_max: np.ndarray
    forecast_hours: int
    station_names: list[str]
    input_feature_names: list[str]
    era5_feature_names: list[str]


@dataclass(frozen=True)
class HermesDiagnostics:
    route_weights: torch.Tensor
    era5_prediction: torch.Tensor


class HermesComponent(nn.Module):
    """One horizon-specialized residual component used inside HERMES."""

    def __init__(
        self,
        input_size: int,
        station_count: int,
        grid_shape: tuple[int, int],
        forecast_hours: int = 24,
        era5_size: int = 10,
        hidden_size: int = 64,
        dropout: float = 0.1,
        pm25_index: int = 0,
        use_temporal: bool = True,
        use_spatial: bool = True,
        use_routing: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        rows, cols = grid_shape
        if rows * cols != station_count:
            raise ValueError(f"grid_shape {grid_shape} does not match station_count {station_count}")
        self.rows = rows
        self.cols = cols
        self.station_count = station_count
        self.forecast_hours = forecast_hours
        self.era5_size = era5_size
        self.hidden_size = hidden_size
        self.pm25_index = pm25_index
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.use_routing = use_routing
        self.use_residual = use_residual

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.local_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.local_head = nn.Linear(hidden_size, forecast_hours)

        self.temporal_conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2)
        self.temporal_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=4, dilation=4)
        self.temporal_norm = nn.LayerNorm(hidden_size)
        self.temporal_head = nn.Linear(hidden_size, forecast_hours)

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.spatial_norm = nn.LayerNorm(hidden_size)
        self.spatial_head = nn.Linear(hidden_size, forecast_hours)

        self.era5_decoder = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, era5_size),
        )
        self.era5_context = nn.Sequential(
            nn.Linear(era5_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        self.lead_embedding = nn.Embedding(forecast_hours, hidden_size)
        self.route = nn.Sequential(
            nn.LayerNorm(hidden_size * 5),
            nn.Linear(hidden_size * 5, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4),
        )
        self.route_prior = nn.Parameter(self._initial_route_prior(forecast_hours), requires_grad=True)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 5),
            nn.Linear(hidden_size * 5, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.h1_residual = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.mid_residual = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        h1_mask = torch.zeros(forecast_hours, dtype=torch.float32)
        h1_mask[0] = 1.0
        mid_mask = torch.zeros(forecast_hours, dtype=torch.float32)
        mid_mask[2 : min(10, forecast_hours)] = 1.0
        self.register_buffer("h1_mask", h1_mask)
        self.register_buffer("mid_mask", mid_mask)
        self.last_diagnostics: HermesDiagnostics | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, station_count, _ = x.shape
        if station_count != self.station_count:
            raise ValueError(f"Expected {self.station_count} stations, got {station_count}")

        local_state = self._local_state(x)
        temporal_state = self._temporal_state(x) if self.use_temporal else torch.zeros_like(local_state)
        spatial_state = self._spatial_state(x) if self.use_spatial else torch.zeros_like(local_state)
        lead_ids = torch.arange(self.forecast_hours, device=x.device)
        lead = self.lead_embedding(lead_ids).view(1, self.forecast_hours, 1, -1)

        local_raw = self.local_head(local_state).permute(0, 2, 1)
        persistence_raw = x[:, -1, :, self.pm25_index].unsqueeze(1).expand(-1, self.forecast_hours, -1)
        temporal_raw = self.temporal_head(temporal_state).permute(0, 2, 1) if self.use_temporal else persistence_raw
        spatial_raw = self.spatial_head(spatial_state).permute(0, 2, 1) if self.use_spatial else persistence_raw
        expert_stack = torch.stack([persistence_raw, local_raw, temporal_raw, spatial_raw], dim=-1)

        local_by_lead = local_state.unsqueeze(1).expand(-1, self.forecast_hours, -1, -1)
        era5_input = torch.cat([local_by_lead, lead.expand(batch_size, -1, self.station_count, -1)], dim=-1)
        era5_pred = self.era5_decoder(era5_input)

        outputs = []
        route_weights = []
        for lead_index in range(self.forecast_hours):
            lead_context = lead[:, lead_index].expand(batch_size, self.station_count, -1)
            era5_context = self.era5_context(era5_pred[:, lead_index])
            context = torch.cat([local_state, temporal_state, spatial_state, era5_context, lead_context], dim=-1)
            if self.use_routing:
                route_logits = self.route(context) + self.route_prior[lead_index].view(1, 1, 4)
                weights = torch.softmax(route_logits, dim=-1)
            else:
                weights = torch.full(
                    (batch_size, self.station_count, 4),
                    0.25,
                    dtype=expert_stack.dtype,
                    device=expert_stack.device,
                )
            routed = torch.sum(weights * expert_stack[:, lead_index], dim=-1)
            residual = self.residual_head(context).squeeze(-1) if self.use_residual else torch.zeros_like(routed)
            h1_delta = (
                self.h1_residual(torch.cat([local_state, lead_context], dim=-1)).squeeze(-1)
                if self.use_residual
                else torch.zeros_like(routed)
            )
            mid_delta = (
                self.mid_residual(torch.cat([local_state, temporal_state, spatial_state, lead_context], dim=-1)).squeeze(-1)
                if self.use_residual
                else torch.zeros_like(routed)
            )
            output = (
                routed
                + residual
                + self.h1_mask[lead_index] * h1_delta
                + self.mid_mask[lead_index] * mid_delta
            )
            outputs.append(output)
            route_weights.append(weights)

        pm_pred = torch.stack(outputs, dim=1)
        routes = torch.stack(route_weights, dim=1)
        self.last_diagnostics = HermesDiagnostics(route_weights=routes, era5_prediction=era5_pred)
        return pm_pred, era5_pred

    def _local_state(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        batch_size, time_steps, station_count, hidden = h.shape
        series = h.permute(0, 2, 1, 3).reshape(batch_size * station_count, time_steps, hidden)
        _, state = self.local_gru(series)
        return state[-1].reshape(batch_size, station_count, hidden)

    def _temporal_state(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        batch_size, time_steps, station_count, hidden = h.shape
        series = h.permute(0, 2, 3, 1).reshape(batch_size * station_count, hidden, time_steps)
        h = torch.relu(self.temporal_conv1(series))[..., :time_steps]
        h = torch.relu(self.temporal_conv2(h))[..., :time_steps]
        last = h[:, :, -1].reshape(batch_size, station_count, hidden)
        return self.temporal_norm(last)

    def _spatial_state(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, station_count, channels = x.shape
        grid = x[:, -1].reshape(batch_size, self.rows, self.cols, channels).permute(0, 3, 1, 2)
        h = self.spatial_encoder(grid)
        h = h.permute(0, 2, 3, 1).reshape(batch_size, station_count, self.hidden_size)
        return self.spatial_norm(h)

    @staticmethod
    def _initial_route_prior(forecast_hours: int) -> torch.Tensor:
        prior = torch.zeros(forecast_hours, 4, dtype=torch.float32)
        for lead in range(1, forecast_hours + 1):
            if lead == 1:
                prior[lead - 1] = torch.tensor([0.2, 1.2, 0.0, 0.0])
            elif lead <= 2:
                prior[lead - 1] = torch.tensor([0.1, 0.8, 0.3, 0.0])
            elif lead <= 10:
                prior[lead - 1] = torch.tensor([0.0, 0.2, 0.8, 0.6])
            else:
                prior[lead - 1] = torch.tensor([0.5, 0.7, 0.2, 0.1])
        return prior


def make_sequences(
    values: np.ndarray,
    feature_names: list[str],
    station_grid: np.ndarray,
    lookback: int = 24,
    forecast_hours: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    era5_features: list[str] | None = None,
    include_era5_input: bool = True,
) -> SequenceData:
    if values.ndim != 4:
        raise ValueError(f"Expected values [time, rows, cols, features], got {values.shape}")
    if "PM2.5" not in feature_names:
        raise ValueError("feature_names must include PM2.5")
    era5_feature_names = era5_features or DEFAULT_ERA5_FEATURES
    missing_era5 = sorted(set(era5_feature_names) - set(feature_names))
    if missing_era5:
        raise ValueError(f"Missing ERA5 features: {missing_era5}")

    original_count = 13
    input_feature_names = feature_names if include_era5_input else feature_names[:original_count]
    input_indices = [feature_names.index(name) for name in input_feature_names]
    era5_indices = [feature_names.index(name) for name in era5_feature_names]
    pm_index = feature_names.index("PM2.5")

    time_steps, rows, cols, _ = values.shape
    station_count = rows * cols
    flat_inputs = values[..., input_indices].reshape(time_steps, station_count, len(input_indices))
    pm_values = values[..., pm_index].reshape(time_steps, station_count)
    era5_values = values[..., era5_indices].reshape(time_steps, station_count, len(era5_indices))
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    train_boundary = int(time_steps * train_ratio)
    val_boundary = int(time_steps * (train_ratio + val_ratio))

    input_min = flat_inputs[:train_boundary].min(axis=(0, 1), keepdims=True)
    input_max = flat_inputs[:train_boundary].max(axis=(0, 1), keepdims=True)
    pm_min = pm_values[:train_boundary].min(axis=0)
    pm_max = pm_values[:train_boundary].max(axis=0)
    era5_min = era5_values[:train_boundary].min(axis=(0, 1), keepdims=True)
    era5_max = era5_values[:train_boundary].max(axis=(0, 1), keepdims=True)

    x_all = _minmax(flat_inputs, input_min, input_max)
    pm_all = _minmax(pm_values, pm_min, pm_max)
    era5_all = _minmax(era5_values, era5_min, era5_max)

    x_seq = []
    pm_seq = []
    era5_seq = []
    target_start_times = []
    target_end_times = []
    for end in range(lookback - 1, time_steps - forecast_hours):
        x_seq.append(x_all[end - lookback + 1 : end + 1])
        target_slice = slice(end + 1, end + forecast_hours + 1)
        pm_seq.append(pm_all[target_slice])
        era5_seq.append(era5_all[target_slice])
        target_start_times.append(end + 1)
        target_end_times.append(end + forecast_hours)

    x = np.stack(x_seq).astype(np.float32)
    pm = np.stack(pm_seq).astype(np.float32)
    era5 = np.stack(era5_seq).astype(np.float32)
    target_start = np.array(target_start_times)
    target_end = np.array(target_end_times)
    train_mask = target_end < train_boundary
    val_mask = (target_start >= train_boundary) & (target_end < val_boundary)
    test_mask = target_start >= val_boundary
    station_names = [str(name) for name in station_grid.reshape(-1)]
    return SequenceData(
        x_train=torch.from_numpy(x[train_mask]),
        pm_train=torch.from_numpy(pm[train_mask]),
        era5_train=torch.from_numpy(era5[train_mask]),
        x_val=torch.from_numpy(x[val_mask]),
        pm_val=torch.from_numpy(pm[val_mask]),
        era5_val=torch.from_numpy(era5[val_mask]),
        x_test=torch.from_numpy(x[test_mask]),
        pm_test=torch.from_numpy(pm[test_mask]),
        era5_test=torch.from_numpy(era5[test_mask]),
        train_target_start=target_start[train_mask].astype(np.int64),
        train_target_end=target_end[train_mask].astype(np.int64),
        val_target_start=target_start[val_mask].astype(np.int64),
        val_target_end=target_end[val_mask].astype(np.int64),
        test_target_start=target_start[test_mask].astype(np.int64),
        test_target_end=target_end[test_mask].astype(np.int64),
        train_boundary=train_boundary,
        val_boundary=val_boundary,
        input_min=input_min.astype(np.float32),
        input_max=input_max.astype(np.float32),
        pm_min=pm_min.astype(np.float32),
        pm_max=pm_max.astype(np.float32),
        era5_min=era5_min.astype(np.float32),
        era5_max=era5_max.astype(np.float32),
        forecast_hours=forecast_hours,
        station_names=station_names,
        input_feature_names=input_feature_names,
        era5_feature_names=era5_feature_names,
    )


def load_sequence_data(
    tensor_path: Path,
    lookback: int = 24,
    forecast_hours: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    include_era5_input: bool = True,
) -> tuple[SequenceData, tuple[int, int]]:
    tensor = load_tensor(tensor_path)
    data = make_sequences(
        tensor.values,
        tensor.feature_names,
        tensor.station_grid,
        lookback=lookback,
        forecast_hours=forecast_hours,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        era5_features=DEFAULT_ERA5_FEATURES,
        include_era5_input=include_era5_input,
    )
    return data, tuple(int(v) for v in tensor.station_grid.shape)


def train_component(
    model: HermesComponent,
    data: SequenceData,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    era5_aux_weight: float = 0.1,
    route_entropy_weight: float = 0.001,
    horizon_loss_power: float = 0.5,
    h1_focus_weight: float = 0.5,
    mid_focus_weight: float = 0.35,
    focus_start_hour: int = 3,
    focus_end_hour: int = 10,
    device: str = "cpu",
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(data.x_train, data.pm_train, data.era5_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(data.x_val, data.pm_val, data.era5_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    horizon_weights = torch.arange(1, data.forecast_hours + 1, dtype=torch.float32)
    horizon_weights = horizon_weights.pow(horizon_loss_power)
    horizon_weights = horizon_weights / horizon_weights.mean()
    start_index = max(0, focus_start_hour - 1)
    end_index = min(data.forecast_hours, focus_end_hour)
    if start_index >= end_index:
        raise ValueError(f"Invalid focus window H{focus_start_hour}-H{focus_end_hour}")
    focus_slice = slice(start_index, end_index)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_pm_total = 0.0
        train_era5_total = 0.0
        train_entropy_total = 0.0
        for x_batch, pm_batch, era5_batch in loader:
            x_batch = x_batch.to(device)
            pm_batch = pm_batch.to(device)
            era5_batch = era5_batch.to(device)
            weights = horizon_weights.to(device).view(1, -1, 1)
            optimizer.zero_grad()
            pm_pred, era5_pred = model(x_batch)
            base_loss = torch.mean(torch.abs(pm_pred - pm_batch) * weights)
            h1_loss = torch.mean(torch.abs(pm_pred[:, 0] - pm_batch[:, 0]))
            focus_loss = torch.mean(torch.abs(pm_pred[:, focus_slice] - pm_batch[:, focus_slice]))
            era5_loss = torch.mean(torch.abs(era5_pred - era5_batch))
            entropy_loss = _negative_route_entropy(model)
            pm_loss = base_loss + h1_focus_weight * h1_loss + mid_focus_weight * focus_loss
            loss = pm_loss + era5_aux_weight * era5_loss + route_entropy_weight * entropy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_pm_total += pm_loss.item() * len(x_batch)
            train_era5_total += era5_loss.item() * len(x_batch)
            train_entropy_total += entropy_loss.item() * len(x_batch)
        model.eval()
        val_pm_total = 0.0
        val_era5_total = 0.0
        with torch.no_grad():
            for x_batch, pm_batch, era5_batch in val_loader:
                x_batch = x_batch.to(device)
                pm_batch = pm_batch.to(device)
                era5_batch = era5_batch.to(device)
                weights = horizon_weights.to(device).view(1, -1, 1)
                pm_pred, era5_pred = model(x_batch)
                base_loss = torch.mean(torch.abs(pm_pred - pm_batch) * weights)
                h1_loss = torch.mean(torch.abs(pm_pred[:, 0] - pm_batch[:, 0]))
                focus_loss = torch.mean(torch.abs(pm_pred[:, focus_slice] - pm_batch[:, focus_slice]))
                pm_loss = base_loss + h1_focus_weight * h1_loss + mid_focus_weight * focus_loss
                val_pm_total += pm_loss.item() * len(x_batch)
                val_era5_total += torch.mean(torch.abs(era5_pred - era5_batch)).item() * len(x_batch)
        mean_val_pm_loss = val_pm_total / len(val_dataset)
        mean_val_era5_loss = val_era5_total / len(val_dataset)
        if mean_val_pm_loss < best_val_loss:
            best_val_loss = mean_val_pm_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": float(epoch),
                "train_pm_specialist_loss": train_pm_total / len(dataset),
                "train_era5_mae_normalized": train_era5_total / len(dataset),
                "train_negative_route_entropy": train_entropy_total / len(dataset),
                "val_pm_specialist_loss": mean_val_pm_loss,
                "val_era5_mae_normalized": mean_val_era5_loss,
                "is_best_validation": float(epoch == best_epoch),
            }
        )
    model.load_state_dict(best_state)
    return history


def evaluate_component(
    model: HermesComponent,
    data: SequenceData,
    batch_size: int = 256,
    device: str = "cpu",
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    model.eval()
    pm_predictions = []
    era5_predictions = []
    route_chunks = []
    loader = torch.utils.data.DataLoader(data.x_test, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x_batch in loader:
            pm_pred, era5_pred = model(x_batch.to(device))
            pm_predictions.append(pm_pred.cpu())
            era5_predictions.append(era5_pred.cpu())
            if model.last_diagnostics is None:
                raise RuntimeError("HERMES diagnostics were not populated")
            route_chunks.append(model.last_diagnostics.route_weights.cpu())
    pm_norm = torch.cat(pm_predictions, dim=0).numpy()
    era5_norm = torch.cat(era5_predictions, dim=0).numpy()
    routes = torch.cat(route_chunks, dim=0).numpy()
    return pm_metric_rows(pm_norm, data), era5_metric_rows(era5_norm, data), route_rows(routes)


def predict_ensemble(
    models: list[HermesComponent],
    weights: list[float],
    x_test: torch.Tensor,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    if len(models) != len(weights):
        raise ValueError("models and weights must have the same length")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Ensemble weights must have positive sum")
    normalized = [weight / weight_sum for weight in weights]
    chunks = []
    loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x_batch in loader:
            x_batch = x_batch.to(device)
            ensemble = None
            for model, weight in zip(models, normalized):
                pm_pred, _ = model(x_batch)
                weighted = pm_pred * weight
                ensemble = weighted if ensemble is None else ensemble + weighted
            if ensemble is None:
                raise RuntimeError("No ensemble models were provided")
            chunks.append(ensemble.cpu())
    return torch.cat(chunks, dim=0).numpy()


def prediction_rows(pm_norm: np.ndarray, data: SequenceData) -> list[dict[str, float | int | str]]:
    pred = _inverse_minmax(pm_norm, data.pm_min[None, None, :], data.pm_max[None, None, :])
    actual = _inverse_minmax(data.pm_test.numpy(), data.pm_min[None, None, :], data.pm_max[None, None, :])
    rows = []
    for sample_index, (target_start, target_end) in enumerate(zip(data.test_target_start, data.test_target_end)):
        for lead_index in range(data.forecast_hours):
            target_time = int(target_start + lead_index)
            for station_index, station in enumerate(data.station_names):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "target_start_index": int(target_start),
                        "target_end_index": int(target_end),
                        "target_time_index": target_time,
                        "lead_hour": lead_index + 1,
                        "station": station,
                        "actual": float(actual[sample_index, lead_index, station_index]),
                        "prediction": float(pred[sample_index, lead_index, station_index]),
                        "error": float(pred[sample_index, lead_index, station_index] - actual[sample_index, lead_index, station_index]),
                    }
                )
    return rows


def load_component_checkpoint(
    checkpoint_path: Path,
    input_size: int,
    station_count: int,
    grid_shape: tuple[int, int],
    era5_size: int,
    forecast_hours: int,
    device: str,
) -> HermesComponent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    model = HermesComponent(
        input_size=input_size,
        station_count=station_count,
        grid_shape=grid_shape,
        forecast_hours=forecast_hours,
        era5_size=era5_size,
        hidden_size=int(args.get("hidden_size", 64)),
        dropout=float(args.get("dropout", 0.1)),
        use_temporal=bool(args.get("use_temporal", True)),
        use_spatial=bool(args.get("use_spatial", True)),
        use_routing=bool(args.get("use_routing", True)),
        use_residual=bool(args.get("use_residual", True)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def aggregate_pm_metrics(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    aggregate_rows = []
    for lead in sorted({int(row["lead_hour"]) for row in rows}):
        lead_rows = [row for row in rows if int(row["lead_hour"]) == lead]
        aggregate_rows.append(
            {
                "lead_hour": lead,
                "station": "mean",
                "rmse": float(np.mean([float(row["rmse"]) for row in lead_rows])),
                "mae": float(np.mean([float(row["mae"]) for row in lead_rows])),
                "r2": float(np.mean([float(row["r2"]) for row in lead_rows])),
                "rmsle": float(np.mean([float(row["rmsle"]) for row in lead_rows])),
            }
        )
    return aggregate_rows


def pm_metric_rows(pm_norm: np.ndarray, data: SequenceData) -> list[dict[str, float | int | str]]:
    return pm_metric_rows_for_target(pm_norm, data.pm_test.numpy(), data.pm_min, data.pm_max, data.station_names)


def pm_metric_rows_for_target(
    pm_norm: np.ndarray,
    target_norm: np.ndarray,
    pm_min: np.ndarray,
    pm_max: np.ndarray,
    station_names: list[str],
) -> list[dict[str, float | int | str]]:
    pred = _inverse_minmax(pm_norm, pm_min[None, None, :], pm_max[None, None, :])
    actual = _inverse_minmax(target_norm, pm_min[None, None, :], pm_max[None, None, :])
    rows = []
    for lead_index in range(pred.shape[1]):
        for station_index, station in enumerate(station_names):
            rows.append(
                {
                    "lead_hour": lead_index + 1,
                    "station": station,
                    **_regression_metrics(pred[:, lead_index, station_index], actual[:, lead_index, station_index]),
                }
            )
    return rows


def era5_metric_rows(era5_norm: np.ndarray, data: SequenceData) -> list[dict[str, float | int | str]]:
    pred = _inverse_minmax(era5_norm, data.era5_min[None, None, :, :], data.era5_max[None, None, :, :])
    actual = _inverse_minmax(data.era5_test.numpy(), data.era5_min[None, None, :, :], data.era5_max[None, None, :, :])
    rows = []
    for lead_index in range(data.forecast_hours):
        for feature_index, feature in enumerate(data.era5_feature_names):
            errors = pred[:, lead_index, :, feature_index] - actual[:, lead_index, :, feature_index]
            rows.append(
                {
                    "lead_hour": lead_index + 1,
                    "feature": feature,
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "mae": float(np.mean(np.abs(errors))),
                }
            )
    return rows


def route_rows(routes: np.ndarray) -> list[dict[str, float | int | str]]:
    rows = []
    names = ["persistence", "local_gru", "dilated_tcn", "spatial_cnn"]
    for lead_index in range(routes.shape[1]):
        lead_routes = routes[:, lead_index]
        for expert_index, name in enumerate(names):
            values = lead_routes[..., expert_index]
            rows.append({"lead_hour": lead_index + 1, "expert": name, "mean_weight": float(values.mean()), "std_weight": float(values.std())})
    return rows


def _negative_route_entropy(model: HermesComponent) -> torch.Tensor:
    if model.last_diagnostics is None:
        return torch.zeros((), device=next(model.parameters()).device)
    routes = model.last_diagnostics.route_weights
    entropy = -torch.sum(routes * torch.log(routes.clamp_min(1e-8)), dim=-1).mean()
    return -entropy


def _regression_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    errors = pred - actual
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmsle = float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(actual)) ** 2)))
    return {"rmse": rmse, "mae": mae, "r2": r2, "rmsle": rmsle}


def _minmax(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    denominator = np.where(max_values > min_values, max_values - min_values, 1.0)
    return (values - min_values) / denominator


def _inverse_minmax(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    denominator = np.where(max_values > min_values, max_values - min_values, 1.0)
    return values * denominator + min_values
