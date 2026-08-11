from __future__ import annotations

import argparse
import json
from pathlib import Path

from pm25pred.beijing_data import build_beijing_tensor, load_station_grid, save_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3x4 Beijing PM2.5 station tensors.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/beijing_3x4_hourly.npz"))
    parser.add_argument("--grid-output", type=Path, default=Path("data/processed/beijing_3x4_station_grid.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor = build_beijing_tensor(args.raw_dir)
    save_tensor(tensor, args.output)

    grid = load_station_grid(args.raw_dir)
    args.grid_output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_output, index=False)

    summary = {
        "tensor": str(args.output),
        "grid": str(args.grid_output),
        "shape": list(tensor.values.shape),
        "feature_names": tensor.feature_names,
        "start": str(tensor.timestamps[0]),
        "end": str(tensor.timestamps[-1]),
        "station_grid": tensor.station_grid.tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

