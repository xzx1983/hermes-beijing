from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pm25pred.beijing_data import BeijingTensor, load_tensor, save_tensor


DEFAULT_PROXY_FEATURES = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd_sin", "wd_cos"]


@dataclass(frozen=True)
class FutureMetSource:
    values: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]
    station_grid: np.ndarray


def build_station_proxy_future_source(
    tensor: BeijingTensor,
    feature_names: list[str] | None = None,
) -> FutureMetSource:
    """Use station-observed meteorology as a deterministic smoke-test source.

    This is not a publication data source. It exists so the future-meteorology
    model path can be tested before ERA5 station-interpolated files are present.
    """
    selected = feature_names or DEFAULT_PROXY_FEATURES
    missing = sorted(set(selected) - set(tensor.feature_names))
    if missing:
        raise ValueError(f"Proxy source features are missing from tensor: {missing}")
    indices = [tensor.feature_names.index(name) for name in selected]
    return FutureMetSource(
        values=tensor.values[..., indices],
        timestamps=tensor.timestamps,
        feature_names=selected,
        station_grid=tensor.station_grid,
    )


def load_station_future_met_csv(path: Path, station_grid: np.ndarray) -> FutureMetSource:
    """Load station-interpolated future meteorology from a long CSV file.

    Required columns are ``timestamp`` and ``station``. All remaining numeric
    columns are treated as meteorological predictors. The expected row coverage
    is one row per timestamp and station.
    """
    df = pd.read_csv(path)
    required = {"timestamp", "station"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(f"Future-met CSV is missing required columns: {missing_required}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    stations = [str(name) for name in station_grid.reshape(-1)]
    feature_names = [
        column
        for column in df.columns
        if column not in {"timestamp", "station"} and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not feature_names:
        raise ValueError("Future-met CSV must contain at least one numeric feature column")

    timestamps = np.array(sorted(df["timestamp"].drop_duplicates()), dtype="datetime64[ns]")
    lookup = (
        df.set_index(["timestamp", "station"])
        .sort_index()[feature_names]
        .apply(pd.to_numeric, errors="coerce")
    )

    values = np.empty((len(timestamps), station_grid.shape[0], station_grid.shape[1], len(feature_names)), dtype=np.float32)
    for time_index, timestamp in enumerate(pd.to_datetime(timestamps)):
        for station_index, station in enumerate(stations):
            row = station_index // station_grid.shape[1]
            col = station_index % station_grid.shape[1]
            try:
                values[time_index, row, col, :] = lookup.loc[(timestamp, station)].to_numpy(dtype=np.float32)
            except KeyError as exc:
                raise ValueError(f"Missing future-met row for timestamp={timestamp}, station={station}") from exc

    if np.isnan(values).any():
        values = _fill_nan_by_feature(values)

    return FutureMetSource(
        values=values,
        timestamps=pd.to_datetime(timestamps).astype(str).to_numpy(),
        feature_names=feature_names,
        station_grid=station_grid,
    )


def append_future_met_features(
    tensor: BeijingTensor,
    future_source: FutureMetSource,
    horizons: list[int] | tuple[int, ...],
    prefix: str = "future",
) -> BeijingTensor:
    """Append shifted target-horizon meteorology to a Beijing tensor.

    For a horizon ``h``, each row at timestamp ``t`` receives meteorological
    features from ``t + h`` named ``{prefix}_h{h}_{feature}``. Samples whose
    shifted values exceed the available range are filled by edge values; the
    sequence builder still drops rows that cannot form a target at max horizon.
    """
    horizon_steps = sorted(int(horizon) for horizon in horizons)
    if not horizon_steps or horizon_steps[0] < 1:
        raise ValueError(f"Horizons must be positive integers, got {horizon_steps}")
    _check_compatible(tensor, future_source)

    future_blocks = []
    future_names = []
    for horizon in horizon_steps:
        shifted = _shift_backward(future_source.values, horizon)
        future_blocks.append(shifted)
        future_names.extend([f"{prefix}_h{horizon}_{name}" for name in future_source.feature_names])

    values = np.concatenate([tensor.values, *future_blocks], axis=-1).astype(np.float32)
    return BeijingTensor(
        values=values,
        timestamps=tensor.timestamps,
        feature_names=[*tensor.feature_names, *future_names],
        station_grid=tensor.station_grid,
        longitude_grid=tensor.longitude_grid,
        latitude_grid=tensor.latitude_grid,
    )


def build_augmented_tensor_from_station_proxy(
    tensor_path: Path,
    output_path: Path,
    horizons: list[int] | tuple[int, ...],
    feature_names: list[str] | None = None,
) -> BeijingTensor:
    tensor = load_tensor(tensor_path)
    source = build_station_proxy_future_source(tensor, feature_names=feature_names)
    augmented = append_future_met_features(tensor, source, horizons=horizons)
    save_tensor(augmented, output_path)
    return augmented


def _check_compatible(tensor: BeijingTensor, source: FutureMetSource) -> None:
    if tensor.values.shape[:3] != source.values.shape[:3]:
        raise ValueError(
            "Future-met source must match tensor time and grid dimensions, "
            f"got {source.values.shape[:3]} and {tensor.values.shape[:3]}"
        )
    if list(tensor.timestamps) != list(source.timestamps):
        raise ValueError("Future-met source timestamps do not match tensor timestamps")
    if list(tensor.station_grid.reshape(-1)) != list(source.station_grid.reshape(-1)):
        raise ValueError("Future-met source station grid does not match tensor station grid")


def _shift_backward(values: np.ndarray, horizon: int) -> np.ndarray:
    shifted = np.empty_like(values)
    shifted[:-horizon] = values[horizon:]
    shifted[-horizon:] = values[-1:]
    return shifted


def _fill_nan_by_feature(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1, values.shape[-1])
    for feature_index in range(values.shape[-1]):
        feature = flat[:, :, feature_index]
        frame = pd.DataFrame(feature)
        frame = frame.interpolate(method="linear", limit_direction="both").ffill().bfill()
        flat[:, :, feature_index] = frame.to_numpy(dtype=np.float32)
    return flat.reshape(values.shape)
