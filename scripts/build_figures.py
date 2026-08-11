from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
HERMES_RESULTS = RESULTS / "hermes_clean_split"
GRAPH_RESULTS = RESULTS / "graph_baselines_clean_split"
STANDARD_RESULTS = RESULTS / "standard_baselines_clean_split"
SEEDS = [13, 42, 2026]

FULL_BASELINES = {
    "GRU+RNN": {"kind": "aggregate", "folder": STANDARD_RESULTS, "pattern": "gru_rnn_seed{seed}_aggregate_metrics.csv"},
    "BiLSTM-MA": {"kind": "aggregate", "folder": STANDARD_RESULTS, "pattern": "bilstm_ma_seed{seed}_aggregate_metrics.csv"},
    "Spatial CNN": {"kind": "aggregate", "folder": STANDARD_RESULTS, "pattern": "spatial_cnn_seed{seed}_aggregate_metrics.csv"},
    "STGCN": {"kind": "aggregate", "folder": GRAPH_RESULTS, "pattern": "stgcn_seed{seed}_aggregate_metrics.csv"},
    "Graph WaveNet": {"kind": "aggregate", "folder": GRAPH_RESULTS, "pattern": "graph_wavenet_seed{seed}_aggregate_metrics.csv"},
    "AGCRN": {"kind": "aggregate", "folder": GRAPH_RESULTS, "pattern": "agcrn_seed{seed}_aggregate_metrics.csv"},
    "HERMES": {"kind": "aggregate", "folder": HERMES_RESULTS, "pattern": "hermes_seed{seed}_aggregate_metrics.csv"},
}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    full = _full_baseline_frame()
    summary = {
        "full_baseline_group_summary": _group_summary(full),
        "representative_horizons": _representative_summary(full),
        "component_summary": _component_summary(),
        "paired_tests": _paired_tests(),
    }
    (OUT / "asset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_metric(full, "rmse", "rmse_by_horizon", "RMSE (micrograms m$^{-3}$)")
    _plot_metric(full, "mae", "mae_by_horizon", "MAE (micrograms m$^{-3}$)")
    _plot_components(summary["component_summary"])
    _plot_routes()
    _plot_architecture()
    print(json.dumps(summary, indent=2))


def _full_baseline_frame() -> pd.DataFrame:
    frames = []
    for method, spec in FULL_BASELINES.items():
        if spec["kind"] == "stationwise":
            frames.append(_load_stationwise(method, spec["pattern"]))
        else:
            frames.append(_load_aggregate(method, spec["folder"], spec["pattern"]))
    return pd.concat(frames, ignore_index=True)


def _load_stationwise(method: str, pattern: str) -> pd.DataFrame:
    rows = []
    base = RESULTS / "stationwise_h1_h24"
    for horizon in range(1, 25):
        for seed in SEEDS:
            path = base / pattern.format(horizon=horizon, seed=seed)
            values = _read_csv(path)
            rows.append(
                {
                    "method": method,
                    "lead_hour": horizon,
                    "seed": seed,
                    "rmse": statistics.mean(float(row["rmse"]) for row in values),
                    "mae": statistics.mean(float(row["mae"]) for row in values),
                }
            )
    return pd.DataFrame(rows)


def _load_aggregate(method: str, folder: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        frame = pd.read_csv(folder / pattern.format(seed=seed))
        for _, row in frame.iterrows():
            rows.append(
                {
                    "method": method,
                    "lead_hour": int(row["lead_hour"]),
                    "seed": seed,
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                }
            )
    return pd.DataFrame(rows)


def _group_summary(full: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    groups = {
        "H1": (1, 1),
        "H2-H6": (2, 6),
        "H7-H12": (7, 12),
        "H13-H24": (13, 24),
        "H1-H24": (1, 24),
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for group, (lo, hi) in groups.items():
        out[group] = {}
        for method, frame in full.groupby("method"):
            sub = frame[(frame["lead_hour"] >= lo) & (frame["lead_hour"] <= hi)]
            seed_means = sub.groupby("seed")[["rmse", "mae"]].mean()
            out[group][method] = {
                "rmse_mean": float(seed_means["rmse"].mean()),
                "rmse_std": float(seed_means["rmse"].std(ddof=0)),
                "mae_mean": float(seed_means["mae"].mean()),
                "mae_std": float(seed_means["mae"].std(ddof=0)),
            }
    return out


def _representative_summary(full: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for horizon in [1, 6, 12, 24]:
        key = f"H{horizon}"
        out[key] = {}
        for method, frame in full[full["lead_hour"] == horizon].groupby("method"):
            out[key][method] = {
                "rmse_mean": float(frame["rmse"].mean()),
                "rmse_std": float(frame["rmse"].std(ddof=0)),
                "mae_mean": float(frame["mae"].mean()),
                "mae_std": float(frame["mae"].std(ddof=0)),
            }
    return out


def _component_summary() -> dict[str, dict[str, float]]:
    out = {}
    for component in ["h1mid", "hardmid", "balanced", "HERMES"]:
        seed_means = []
        prefix = "hermes" if component == "HERMES" else component
        for seed in SEEDS:
            frame = pd.read_csv(HERMES_RESULTS / f"{prefix}_seed{seed}_aggregate_metrics.csv")
            seed_means.append(float(frame["rmse"].mean()))
        out[component] = {
            "mean_rmse": float(statistics.mean(seed_means)),
            "std_rmse": float(statistics.pstdev(seed_means)),
        }
    return out


def _plot_metric(full: pd.DataFrame, metric: str, stem: str, ylabel: str) -> None:
    plt.figure(figsize=(7.2, 4.2))
    colors = {
        "GRU+RNN": "#4C78A8",
        "BiLSTM-MA": "#F58518",
        "Spatial CNN": "#9D755D",
        "STGCN": "#72B7B2",
        "Graph WaveNet": "#54A24B",
        "AGCRN": "#B279A2",
        "HERMES": "#E45756",
    }
    markers = {
        "GRU+RNN": "o",
        "BiLSTM-MA": "s",
        "Spatial CNN": "X",
        "STGCN": "^",
        "Graph WaveNet": "D",
        "AGCRN": "v",
        "HERMES": "P",
    }
    for method in ["GRU+RNN", "BiLSTM-MA", "Spatial CNN", "STGCN", "Graph WaveNet", "AGCRN", "HERMES"]:
        frame = full[full["method"] == method]
        grouped = frame.groupby("lead_hour")[metric].agg(["mean", "std"]).reset_index()
        linewidth = 2.4 if method == "HERMES" else 1.4
        alpha = 1.0 if method == "HERMES" else 0.78
        plt.plot(
            grouped["lead_hour"],
            grouped["mean"],
            marker=markers[method],
            linewidth=linewidth,
            markersize=3.8,
            alpha=alpha,
            color=colors[method],
            label=method,
        )
        if method == "HERMES":
            std = grouped["std"].fillna(0)
            plt.fill_between(
                grouped["lead_hour"],
                grouped["mean"] - std,
                grouped["mean"] + std,
                color=colors[method],
                alpha=0.14,
                linewidth=0,
            )
    plt.xlabel("Forecast horizon (h)")
    plt.ylabel(ylabel)
    plt.xticks(range(1, 25, 2))
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    _save(stem)


def _plot_components(component_summary: dict[str, dict[str, float]]) -> None:
    labels = ["h1mid", "hardmid", "balanced", "HERMES"]
    means = [component_summary[label]["mean_rmse"] for label in labels]
    stds = [component_summary[label]["std_rmse"] for label in labels]
    plt.figure(figsize=(5.8, 3.5))
    colors = ["#72B7B2", "#F58518", "#54A24B", "#E45756"]
    plt.bar(labels, means, yerr=stds, capsize=3, color=colors)
    plt.ylabel("Mean H1-H24 RMSE")
    plt.ylim(min(means) - 0.8, max(means) + 0.8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    _save("component_ensemble_rmse")


def _plot_routes() -> None:
    frames = []
    for component in ["h1mid", "hardmid", "balanced"]:
        for seed in SEEDS:
            path = HERMES_RESULTS / f"{component}_seed{seed}_routes.csv"
            frame = pd.read_csv(path)
            frame["component"] = component
            frame["seed"] = seed
            frames.append(frame)
    routes = pd.concat(frames, ignore_index=True)
    mean_routes = routes.groupby(["lead_hour", "expert"], as_index=False)["mean_weight"].mean()
    plt.figure(figsize=(7.0, 3.8))
    palette = {
        "persistence": "#4C78A8",
        "local_gru": "#F58518",
        "dilated_tcn": "#54A24B",
        "spatial_cnn": "#B279A2",
    }
    for expert, group in mean_routes.groupby("expert"):
        plt.plot(group["lead_hour"], group["mean_weight"], marker="o", linewidth=1.6, label=expert.replace("_", " "), color=palette[expert])
    plt.xlabel("Forecast horizon (h)")
    plt.ylabel("Mean route weight")
    plt.ylim(0, 1.0)
    plt.xticks(range(1, 25, 2))
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    _save("route_weights")


def _plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.axis("off")
    boxes = [
        ((0.04, 0.66), "Past 24 h inputs\\nPollution + local met.\\nHistorical ERA5"),
        ((0.30, 0.80), "Station GRU\\nlocal persistence"),
        ((0.30, 0.58), "Dilated TCN\\ntemporal change"),
        ((0.30, 0.36), "Grid CNN\\nspatial field"),
        ((0.30, 0.14), "Persistence\\nlast PM$_{2.5}$"),
        ((0.56, 0.62), "Lead-specific routing\\n+ residual heads"),
        ((0.56, 0.25), "Auxiliary future ERA5\\nprediction loss"),
        ((0.79, 0.62), "One HERMES component\\nH1/mid/balanced focus"),
        ((0.79, 0.24), "Validation-selected\\nfixed ensemble"),
        ((0.79, 0.04), "PM$_{2.5}$ forecasts\\nH1-H24, 12 stations"),
    ]
    for (x, y), text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#444444", linewidth=0.9),
        )
    arrows = [
        ((0.16, 0.66), (0.24, 0.80)),
        ((0.16, 0.66), (0.24, 0.58)),
        ((0.16, 0.66), (0.24, 0.36)),
        ((0.16, 0.66), (0.24, 0.14)),
        ((0.40, 0.80), (0.49, 0.62)),
        ((0.40, 0.58), (0.49, 0.62)),
        ((0.40, 0.36), (0.49, 0.62)),
        ((0.40, 0.14), (0.49, 0.62)),
        ((0.55, 0.25), (0.56, 0.49)),
        ((0.65, 0.62), (0.71, 0.62)),
        ((0.79, 0.52), (0.79, 0.34)),
        ((0.79, 0.15), (0.79, 0.09)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.0))
    plt.tight_layout()
    _save("architecture")


def _paired_tests() -> dict[str, dict[str, float]]:
    tests = {
        "H1_vs_Graph_WaveNet": (1, "graph_wavenet", "graph"),
        "H6_vs_Graph_WaveNet": (6, "graph_wavenet", "graph"),
        "H12_vs_Graph_WaveNet": (12, "graph_wavenet", "graph"),
        "H24_vs_Graph_WaveNet": (24, "graph_wavenet", "graph"),
    }
    return {name: _paired_test(*spec) for name, spec in tests.items()}


def _paired_test(horizon: int, baseline: str, kind: str) -> dict[str, float]:
    ours = _hermes_station_rmse(horizon)
    base = _stationwise_rmse(baseline, horizon) if kind == "station" else _graph_station_rmse(baseline, horizon)
    keys = sorted(set(ours) & set(base))
    diffs = [ours[key] - base[key] for key in keys]
    random.seed(0)
    boot = []
    for _ in range(10000):
        boot.append(statistics.mean(diffs[random.randrange(len(diffs))] for _ in diffs))
    boot.sort()
    observed = abs(statistics.mean(diffs))
    count = 0
    trials = 20000
    for _ in range(trials):
        mean = statistics.mean(diff * (1 if random.random() < 0.5 else -1) for diff in diffs)
        if abs(mean) >= observed:
            count += 1
    return {
        "n_pairs": len(diffs),
        "mean_delta": float(statistics.mean(diffs)),
        "ci_low": float(boot[int(0.025 * len(boot))]),
        "ci_high": float(boot[int(0.975 * len(boot)) - 1]),
        "wins": int(sum(diff < 0 for diff in diffs)),
        "p_value": float((count + 1) / (trials + 1)),
    }


def _hermes_station_rmse(horizon: int) -> dict[tuple[int, str], float]:
    out = {}
    for seed in SEEDS:
        rows = _read_csv(HERMES_RESULTS / f"hermes_seed{seed}_station_metrics.csv")
        for row in rows:
            if int(row["lead_hour"]) == horizon:
                out[(seed, row["station"])] = float(row["rmse"])
    return out


def _graph_station_rmse(model: str, horizon: int) -> dict[tuple[int, str], float]:
    out = {}
    for seed in SEEDS:
        rows = _read_csv(GRAPH_RESULTS / f"{model}_seed{seed}_station_metrics.csv")
        for row in rows:
            if int(row["lead_hour"]) == horizon:
                out[(seed, row["station"])] = float(row["rmse"])
    return out


def _stationwise_rmse(model: str, horizon: int) -> dict[tuple[int, str], float]:
    out = {}
    for seed in SEEDS:
        rows = _read_csv(RESULTS / "stationwise_h1_h24" / f"{model}_h{horizon}_seed{seed}_results.csv")
        for row in rows:
            out[(seed, row["station"])] = float(row["rmse"])
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(stem: str) -> None:
    plt.savefig(FIGURES / f"{stem}.pdf")
    plt.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
