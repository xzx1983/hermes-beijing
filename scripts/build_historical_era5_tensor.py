from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pm25pred.beijing_data import BeijingTensor, load_tensor, save_tensor
from pm25pred.future_met import load_station_future_met_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append same-hour station-interpolated ERA5 variables as operational historical inputs."
    )
    parser.add_argument("--tensor", type=Path, default=Path("data/processed/beijing_3x4_hourly.npz"))
    parser.add_argument("--met-csv", type=Path, default=Path("data/processed/era5_beijing_station_hourly.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/beijing_3x4_hourly_historical_era5.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor = load_tensor(args.tensor)
    source = load_station_future_met_csv(args.met_csv, tensor.station_grid)
    if list(tensor.timestamps) != list(source.timestamps):
        raise ValueError("ERA5 timestamps do not match the base tensor timestamps")

    duplicate = sorted(set(tensor.feature_names) & set(source.feature_names))
    if duplicate:
        raise ValueError(f"ERA5 features duplicate existing tensor feature names: {duplicate}")

    augmented = BeijingTensor(
        values=np.concatenate([tensor.values, source.values], axis=-1).astype(np.float32),
        timestamps=tensor.timestamps,
        feature_names=[*tensor.feature_names, *source.feature_names],
        station_grid=tensor.station_grid,
        longitude_grid=tensor.longitude_grid,
        latitude_grid=tensor.latitude_grid,
    )
    save_tensor(augmented, args.output)
    print(
        json.dumps(
            {
                "input": str(args.tensor),
                "met_csv": str(args.met_csv),
                "output": str(args.output),
                "shape": list(augmented.values.shape),
                "feature_count": len(augmented.feature_names),
                "era5_features": source.feature_names,
                "start": str(augmented.timestamps[0]),
                "end": str(augmented.timestamps[-1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
