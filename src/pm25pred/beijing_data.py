from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]

WIND_DIRECTIONS_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

STATION_METADATA_ALIASES = {
    "Nongzhanguan": "Nongzhangguan",
}


@dataclass(frozen=True)
class BeijingTensor:
    values: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]
    station_grid: np.ndarray
    longitude_grid: np.ndarray
    latitude_grid: np.ndarray


def load_station_grid(raw_dir: Path, rows: int = 3, cols: int = 4) -> pd.DataFrame:
    """Return raw-data stations assigned to a deterministic latitude/longitude grid."""
    station_names = []
    for csv_path in sorted(raw_dir.glob("PRSA_Data_*.csv")):
        station = pd.read_csv(csv_path, usecols=["station"], nrows=1)["station"].iloc[0]
        station_names.append(station)

    expected = rows * cols
    if len(station_names) != expected:
        raise ValueError(f"Expected {expected} station CSV files, found {len(station_names)}")

    meta_path = raw_dir / "BJ_sites_meta(1).csv"
    meta = pd.read_csv(meta_path, encoding="utf-8-sig")
    lookup_names = [STATION_METADATA_ALIASES.get(station, station) for station in station_names]
    grid = meta.loc[
        meta["Name_EN"].isin(lookup_names),
        ["Name_EN", "Longitude", "Latitude", "Type"],
    ].copy()
    reverse_aliases = {metadata: station for station, metadata in STATION_METADATA_ALIASES.items()}
    grid["Name_EN"] = grid["Name_EN"].replace(reverse_aliases)

    missing = sorted(set(station_names) - set(grid["Name_EN"]))
    if missing:
        raise ValueError(f"Stations missing from metadata: {missing}")

    grid = grid.sort_values(["Latitude", "Longitude"], ascending=[False, True]).reset_index(drop=True)
    row_ids = np.repeat(np.arange(rows), cols)
    grid["grid_row"] = row_ids
    grid = grid.sort_values(["grid_row", "Longitude"], ascending=[True, True]).reset_index(drop=True)
    grid["grid_col"] = grid.groupby("grid_row").cumcount()
    return grid.sort_values(["grid_row", "grid_col"]).reset_index(drop=True)


def _load_station_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("timestamp").sort_index()

    for feature in NUMERIC_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[feature] = df[feature].interpolate(method="time").ffill().bfill()

    degrees = df["wd"].map(WIND_DIRECTIONS_DEGREES)
    degrees = degrees.interpolate().ffill().bfill()
    radians = np.deg2rad(degrees.to_numpy(dtype=np.float32))
    df["wd_sin"] = np.sin(radians)
    df["wd_cos"] = np.cos(radians)
    return df[NUMERIC_FEATURES + ["wd_sin", "wd_cos"]]


def build_beijing_tensor(raw_dir: Path, rows: int = 3, cols: int = 4) -> BeijingTensor:
    grid = load_station_grid(raw_dir, rows=rows, cols=cols)
    feature_names = NUMERIC_FEATURES + ["wd_sin", "wd_cos"]
    frames: dict[str, pd.DataFrame] = {}

    for station in grid["Name_EN"]:
        csv_path = raw_dir / f"PRSA_Data_{station}_20130301-20170228.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        frames[station] = _load_station_frame(csv_path)

    reference_index = next(iter(frames.values())).index
    for station, frame in frames.items():
        if not frame.index.equals(reference_index):
            raise ValueError(f"Timestamp index differs for station {station}")

    values = np.empty((len(reference_index), rows, cols, len(feature_names)), dtype=np.float32)
    station_grid = np.empty((rows, cols), dtype=object)
    lon_grid = np.empty((rows, cols), dtype=np.float32)
    lat_grid = np.empty((rows, cols), dtype=np.float32)

    for item in grid.itertuples(index=False):
        row = int(item.grid_row)
        col = int(item.grid_col)
        station_grid[row, col] = item.Name_EN
        lon_grid[row, col] = item.Longitude
        lat_grid[row, col] = item.Latitude
        values[:, row, col, :] = frames[item.Name_EN][feature_names].to_numpy(dtype=np.float32)

    return BeijingTensor(
        values=values,
        timestamps=reference_index.astype(str).to_numpy(),
        feature_names=feature_names,
        station_grid=station_grid,
        longitude_grid=lon_grid,
        latitude_grid=lat_grid,
    )


def save_tensor(tensor: BeijingTensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        values=tensor.values,
        timestamps=tensor.timestamps,
        feature_names=np.array(tensor.feature_names),
        station_grid=tensor.station_grid,
        longitude_grid=tensor.longitude_grid,
        latitude_grid=tensor.latitude_grid,
    )


def load_tensor(path: Path) -> BeijingTensor:
    data = np.load(path, allow_pickle=True)
    return BeijingTensor(
        values=data["values"],
        timestamps=data["timestamps"],
        feature_names=data["feature_names"].astype(str).tolist(),
        station_grid=data["station_grid"],
        longitude_grid=data["longitude_grid"],
        latitude_grid=data["latitude_grid"],
    )
