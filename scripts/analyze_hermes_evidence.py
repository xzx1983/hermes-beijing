from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pm25pred.beijing_data import load_tensor


START_TIME = datetime(2013, 3, 1)
SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HERMES reviewer-block evidence tables from prediction files.")
    parser.add_argument("--tensor", type=Path, default=Path("data/processed/beijing_3x4_hourly_historical_era5.npz"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/hermes_clean_split"))
    parser.add_argument("--graph-baseline-dir", type=Path, default=Path("results/graph_baselines_clean_split"))
    parser.add_argument("--standard-baseline-dir", type=Path, default=Path("results/standard_baselines_clean_split"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("results/hermes_ablation_clean_split"))
    parser.add_argument("--output-dir", type=Path, default=Path("evidence"))
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--block-hours", type=int, default=24)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(args.result_dir, seeds, "hermes", "HERMES")
    persistence = _persistence_rows(args.tensor, predictions)
    baseline_frames = [persistence]
    for prefix, label in [("stgcn", "STGCN"), ("graph_wavenet", "Graph WaveNet"), ("agcrn", "AGCRN")]:
        try:
            baseline_frames.append(_load_predictions(args.graph_baseline_dir, seeds, prefix, label))
        except FileNotFoundError:
            pass
    for prefix, label in [("gru_rnn", "GRU+RNN"), ("bilstm_ma", "BiLSTM-MA"), ("spatial_cnn", "Spatial CNN")]:
        try:
            baseline_frames.append(_load_predictions(args.standard_baseline_dir, seeds, prefix, label))
        except FileNotFoundError:
            pass
    prediction_frame = pd.concat([predictions, *baseline_frames], ignore_index=True)

    split = _split_summary(predictions)
    block_stats = _block_bootstrap(prediction_frame, args.block_hours, args.bootstrap_draws)
    regimes = _regime_diagnostics(args.tensor, prediction_frame)
    ablations = _ablation_summary(args.result_dir) + _ablation_summary(args.ablation_dir)

    _write_csv(args.output_dir / "split_summary.csv", split)
    _write_csv(args.output_dir / "block_bootstrap.csv", block_stats)
    _write_csv(args.output_dir / "regime_diagnostics.csv", regimes)
    _write_csv(args.output_dir / "ablation_summary.csv", ablations)
    report = {
        "split_summary": split,
        "block_bootstrap": block_stats,
        "regime_diagnostics": regimes,
        "ablation_summary": ablations,
        "limits": _limits(prediction_frame),
    }
    (args.output_dir / "evidence_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(args.output_dir / "analysis-report.md", report)
    print(json.dumps(report, indent=2))


def _limits(frame: pd.DataFrame) -> list[str]:
    methods = sorted(set(frame["method"]))
    limits = [f"Timestamp-level prediction files were analyzed for: {', '.join(methods)}."]
    if {"STGCN", "Graph WaveNet", "AGCRN"}.issubset(methods):
        limits.append("Clean-split graph baselines are available for the same test samples.")
    if {"GRU+RNN", "BiLSTM-MA", "Spatial CNN"}.issubset(methods):
        limits.append("Clean-split station-wise and Spatial CNN baselines are available for the same test samples.")
    else:
        limits.append("Some station-wise or Spatial CNN clean-split prediction files are not generated in this pass.")
    return limits


def _load_predictions(result_dir: Path, seeds: list[int], prefix: str, label: str) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = result_dir / f"{prefix}_seed{seed}_predictions.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["method"] = label
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction files found for {prefix} in {result_dir}")
    return pd.concat(frames, ignore_index=True)


def _persistence_rows(tensor_path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    tensor = load_tensor(tensor_path)
    pm_index = tensor.feature_names.index("PM2.5")
    flat_pm = tensor.values[..., pm_index].reshape(tensor.values.shape[0], -1)
    station_to_index = {str(name): index for index, name in enumerate(tensor.station_grid.reshape(-1))}
    frame = predictions.copy()
    frame["method"] = "Persistence"
    frame["prediction"] = [
        float(flat_pm[int(start) - 1, station_to_index[str(station)]])
        for start, station in zip(frame["target_start_index"], frame["station"])
    ]
    frame["error"] = frame["prediction"] - frame["actual"]
    return frame


def _split_summary(predictions: pd.DataFrame) -> list[dict[str, int]]:
    rows = []
    for seed, frame in predictions.groupby("seed"):
        rows.append(
            {
                "seed": int(seed),
                "test_samples": int(frame["sample_index"].nunique()),
                "test_target_start_min": int(frame["target_start_index"].min()),
                "test_target_end_max": int(frame["target_end_index"].max()),
                "test_target_time_min": int(frame["target_time_index"].min()),
                "test_target_time_max": int(frame["target_time_index"].max()),
            }
        )
    return rows


def _block_bootstrap(frame: pd.DataFrame, block_hours: int, draws: int) -> list[dict[str, float | int | str]]:
    rows = []
    horizons = [1, 6, 12, 24]
    for horizon in horizons:
        sub = frame[frame["lead_hour"] == horizon].copy()
        sub["block"] = sub["target_time_index"] // block_hours
        for metric in ["rmse", "mae"]:
            hermes = _block_metric(sub[sub["method"] == "HERMES"], metric)
            for baseline in sorted(set(sub["method"]) - {"HERMES"}):
                base = _block_metric(sub[sub["method"] == baseline], metric)
                common = sorted(set(hermes) & set(base))
                if not common:
                    continue
                diffs = np.array([hermes[key] - base[key] for key in common], dtype=float)
                ci_low, ci_high, p_value = _bootstrap_and_sign_flip(diffs, draws)
                rows.append(
                    {
                        "comparison": f"HERMES_minus_{baseline}",
                        "lead_hour": horizon,
                        "metric": metric,
                        "n_blocks": len(common),
                        "mean_delta": float(diffs.mean()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_value,
                        "wins": int(np.sum(diffs < 0)),
                    }
                )
    return rows


def _block_metric(frame: pd.DataFrame, metric: str) -> dict[tuple[int, int], float]:
    out = {}
    for (seed, block), group in frame.groupby(["seed", "block"]):
        errors = group["error"].to_numpy(dtype=float)
        value = math.sqrt(float(np.mean(errors**2))) if metric == "rmse" else float(np.mean(np.abs(errors)))
        out[(int(seed), int(block))] = value
    return out


def _bootstrap_and_sign_flip(diffs: np.ndarray, draws: int) -> tuple[float, float, float]:
    rng = random.Random(0)
    boot = []
    for _ in range(draws):
        boot.append(float(np.mean([diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))])))
    boot.sort()
    observed = abs(float(diffs.mean()))
    count = 0
    for _ in range(draws):
        signed = [value * (1 if rng.random() < 0.5 else -1) for value in diffs]
        if abs(float(np.mean(signed))) >= observed:
            count += 1
    return boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1], float((count + 1) / (draws + 1))


def _regime_diagnostics(tensor_path: Path, frame: pd.DataFrame) -> list[dict[str, float | int | str]]:
    tensor = load_tensor(tensor_path)
    feature_map = {name: index for index, name in enumerate(tensor.feature_names)}
    pm_index = feature_map["PM2.5"]
    wind_index = feature_map.get("era5_wind_speed")
    blh_index = feature_map.get("blh")
    flat = tensor.values.reshape(tensor.values.shape[0], -1, tensor.values.shape[-1])
    rows = []
    enriched = frame.copy()
    enriched["datetime"] = [START_TIME + timedelta(hours=int(idx)) for idx in enriched["target_time_index"]]
    enriched["season"] = [SEASONS[item.month] for item in enriched["datetime"]]
    enriched["abs_error"] = enriched["error"].abs()
    enriched["pm_bin"] = pd.cut(enriched["actual"], [-np.inf, 35, 75, 150, np.inf], labels=["low", "moderate", "high", "severe"])
    if wind_index is not None:
        wind_by_time = {int(t): float(np.nanmean(flat[int(t), :, wind_index])) for t in enriched["target_time_index"].unique()}
        enriched["wind_speed"] = [wind_by_time[int(t)] for t in enriched["target_time_index"]]
        enriched["wind_regime"] = pd.qcut(enriched["wind_speed"], q=3, labels=["low_wind", "mid_wind", "high_wind"], duplicates="drop")
    if blh_index is not None:
        blh_by_time = {int(t): float(np.nanmean(flat[int(t), :, blh_index])) for t in enriched["target_time_index"].unique()}
        enriched["blh"] = [blh_by_time[int(t)] for t in enriched["target_time_index"]]
        enriched["blh_regime"] = pd.qcut(enriched["blh"], q=3, labels=["low_blh", "mid_blh", "high_blh"], duplicates="drop")
    prev_pm = []
    station_to_index = {str(name): index for index, name in enumerate(tensor.station_grid.reshape(-1))}
    for t, station in zip(enriched["target_time_index"], enriched["station"]):
        previous = flat[max(0, int(t) - 1), station_to_index[str(station)], pm_index]
        prev_pm.append(float(previous))
    enriched["rapid_change"] = np.where((enriched["actual"] - prev_pm).abs() >= 30, "rapid_change", "stable_or_gradual")
    for group_col in ["season", "pm_bin", "rapid_change", "wind_regime", "blh_regime"]:
        if group_col not in enriched:
            continue
        for (method, horizon, group), sub in enriched.groupby(["method", "lead_hour", group_col], observed=False):
            if int(horizon) not in {1, 6, 12, 24} or len(sub) == 0:
                continue
            rows.append(
                {
                    "factor": group_col,
                    "level": str(group),
                    "method": str(method),
                    "lead_hour": int(horizon),
                    "n": int(len(sub)),
                    "rmse": float(math.sqrt(np.mean(sub["error"].to_numpy(dtype=float) ** 2))),
                    "mae": float(sub["abs_error"].mean()),
                }
            )
    return rows


def _ablation_summary(result_dir: Path) -> list[dict[str, float | int | str]]:
    rows = []
    for path in sorted(result_dir.glob("*_aggregate_metrics.csv")):
        name = path.name.removesuffix("_aggregate_metrics.csv")
        if "_seed" not in name:
            continue
        variant, seed_text = name.rsplit("_seed", 1)
        frame = pd.read_csv(path)
        rows.append(
            {
                "variant": variant,
                "seed": int(seed_text),
                "mean_rmse_h1_h24": float(frame["rmse"].mean()),
                "mean_mae_h1_h24": float(frame["mae"].mean()),
                "h1_rmse": float(frame.loc[frame["lead_hour"] == 1, "rmse"].iloc[0]),
                "h6_rmse": float(frame.loc[frame["lead_hour"] == 6, "rmse"].iloc[0]),
                "h12_rmse": float(frame.loc[frame["lead_hour"] == 12, "rmse"].iloc[0]),
                "h24_rmse": float(frame.loc[frame["lead_hour"] == 24, "rmse"].iloc[0]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# HERMES Reviewer Evidence Analysis",
        "",
        "This file is generated from timestamp-level HERMES prediction rows.",
        "",
        "## Current Limits",
    ]
    lines.extend(f"- {item}" for item in report["limits"])
    lines.append("")
    lines.append("## Output Tables")
    lines.append("- `split_summary.csv`: clean split target-index audit.")
    lines.append("- `block_bootstrap.csv`: daily-block bootstrap comparisons.")
    lines.append("- `regime_diagnostics.csv`: season, pollution-level, change, wind, and boundary-layer breakdowns.")
    lines.append("- `ablation_summary.csv`: aggregate metrics for all available HERMES variants in the result directory.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
