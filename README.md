# HERMES Beijing PM2.5 Reproducibility Package

This folder contains the public release package for the HERMES model used in the manuscript
`HERMES: A Meteorology-Guided Horizon-Specialized Ensemble for Multi-Horizon Urban PM2.5 Forecasting`.

**HERMES** means **Horizon-specialized Ensemble Residual Meteorology-Enhanced System**. HERMES
forecasts H1-H24 PM2.5 at 12 Beijing stations from the previous 24 h of pollution, local
meteorology, and historical ERA5 reanalysis.

The package is scoped to the manuscript's clean-split comparison. It includes:

- Beijing Multi-Site Air-Quality raw CSV files and the processed 3 x 4 tensor with historical ERA5 channels.
- HERMES model code (persistence, station-GRU, dilated-TCN, and grid-CNN experts with lead-specific routing and residual heads).
- Training, validation-weight selection, and test evaluation scripts for HERMES.
- Training scripts for the station-wise (GRU+RNN, BiLSTM-MA, Spatial CNN) and spatial-graph
  (STGCN, Graph WaveNet, AGCRN) baselines reported in the manuscript, under the same clean
  chronological protocol.
- The evidence-generation script used for the block-bootstrap, ablation, and seasonal-diagnostic
  tables, and the manuscript figure-generation script.
- Cached result summaries (`results/cached/`) that reproduce every number reported in the manuscript.
- `pyproject.toml` for `uv` environment creation.

## Data

The raw data are the Beijing Multi-Site Air-Quality dataset from the UCI Machine Learning Repository:

- Dataset id: 501
- DOI: `10.24432/C5RK5G`
- Period used here: `2013-03-01 00:00` to `2017-02-28 23:00`
- Stations: 12 Beijing monitoring sites

ERA5 reanalysis variables are from the Copernicus Climate Data Store, interpolated to each station
and merged with the local channels. The processed tensor used by the manuscript is provided directly at:

```text
data/processed/beijing_3x4_hourly_historical_era5.npz
```

Its 23 input channels are: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM,
wd_sin, wd_cos (local), and 10 m zonal wind, 10 m meridional wind, 2 m temperature, 2 m dew
point, surface pressure, total precipitation, boundary-layer height, ERA5 wind speed, and
sine/cosine flow-direction encodings (historical ERA5, previous 24 h only).

You can rebuild the local-only tensor from the raw CSV files:

```bash
env PYTHONPATH=src python3 scripts/build_beijing_tensor.py
```

Rebuilding the historical-ERA5 tensor additionally requires a station-interpolated ERA5 CSV
(`data/processed/era5_beijing_station_hourly.csv`) prepared from the Copernicus Climate Data
Store; that interpolation step is outside the scope of this release. The provided
`beijing_3x4_hourly_historical_era5.npz` is the exact tensor used for every result in the
manuscript, so rebuilding it is only needed to reproduce the ERA5 preprocessing itself:

```bash
env PYTHONPATH=src python3 scripts/build_historical_era5_tensor.py
```

## Environment

Install `uv` first if it is not available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and sync the environment:

```bash
uv sync
```

For CPU-only machines, install a CPU-compatible PyTorch wheel if `uv sync` resolves a CUDA wheel
that does not match your system.

## Smoke Test

This checks that the training and evaluation pipeline runs end to end on the full tensor with a
tiny model and a single epoch (fast on CPU, a few minutes):

```bash
env PYTHONPATH=src python3 scripts/train_hermes_component.py \
  --tensor data/processed/beijing_3x4_hourly_historical_era5.npz \
  --epochs 1 --hidden-size 8 --batch-size 512 --device cpu --seed 13 \
  --model-output /tmp/smoke.pt \
  --station-output /tmp/smoke_station.csv \
  --aggregate-output /tmp/smoke_aggregate.csv \
  --era5-output /tmp/smoke_era5.csv \
  --routes-output /tmp/smoke_routes.csv

env PYTHONPATH=src python3 scripts/train_clean_split_standard_baselines.py \
  --tensor data/processed/beijing_3x4_hourly_historical_era5.npz \
  --model gru_rnn --epochs 1 --hidden-size 8 --batch-size 512 --device cpu --seed 13 \
  --model-output /tmp/smoke_gru.pt \
  --station-output /tmp/smoke_gru_station.csv \
  --aggregate-output /tmp/smoke_gru_aggregate.csv \
  --prediction-output /tmp/smoke_gru_pred.csv
```

Both commands were verified to run successfully against this exact package before release.

## Full HERMES Run

The manuscript configuration uses 24-hour lookback, H1-H24 targets, chronological 70/10/20
split, three seeds, and validation checkpointing. Reproduce the full three-seed HERMES
experiment (components, validation-weight selection, and frozen-ensemble test evaluation) with:

```bash
env DEVICE=cuda PYTHON_BIN=python3 bash scripts/run_hermes_experiment.sh
```

Outputs are written to `remote_results/hermes_clean_split/` and `models/hermes_clean_split/`.
The validation weight-grid evidence is written to
`remote_results/hermes_clean_split/hermes_validation_weight_grid.csv`.

## Baselines

Run the station-wise and spatial-CNN baselines (GRU+RNN, BiLSTM-MA, Spatial CNN):

```bash
env DEVICE=cuda PYTHON_BIN=python3 bash scripts/run_clean_split_standard_baselines.sh
```

Run the spatial-graph baselines (STGCN, Graph WaveNet, AGCRN):

```bash
env DEVICE=cuda PYTHON_BIN=python3 bash scripts/run_clean_split_graph_baselines.sh
```

All scripts use the same 24 h lookback, H1-H24 target, chronological 70/10/20 split, validation
checkpointing, and timestamp-level prediction export as HERMES.

Use `--device cpu` / `DEVICE=cpu` if CUDA is unavailable. CPU training is expected to be much slower.

## Evidence and Figures

Once `remote_results/` is populated, the manuscript's block-bootstrap, ablation, and
seasonal-diagnostic tables are regenerated with:

```bash
env PYTHONPATH=src python3 scripts/analyze_hermes_evidence.py
```

Manuscript figures are regenerated with:

```bash
python3 scripts/build_figures.py
```

## Cached Results

`results/cached/` contains the aggregate and station-level metric CSVs and the evidence-table
CSVs already generated for the manuscript, so every reported number can be checked without
retraining. Row-level prediction files (used only internally for the block-bootstrap test) are
not included because of their size; they regenerate from the commands above.

## Reproducibility Notes

- The chronological split is 70/10/20 (train/validation/test), assigned by non-overlapping
  target windows.
- Normalization parameters are estimated from the training period only.
- The default seeds are `13`, `42`, and `2026`.
- Each HERMES component combines a persistence expert, a station-wise GRU expert, a dilated
  temporal-convolution expert, and a grid-CNN expert through lead-specific routing and residual
  heads; the final HERMES ensemble is a fixed weighted average of three horizon-specialized
  components, with weights selected on the validation split before test evaluation.
- PyTorch GPU kernels can differ slightly across CUDA versions and hardware. The code fixes
  Python, NumPy, and PyTorch seeds, but exact bitwise equality is not guaranteed across machines.

## Files Included

```text
data/raw/                                Raw Beijing PRSA CSV files and station metadata
data/processed/                          Processed tensor used by the manuscript
results/cached/                          Manuscript-era aggregate/station metrics and evidence tables
src/pm25pred/hermes.py                   HERMES model, data windows, training, metrics
src/pm25pred/beijing_data.py             Beijing tensor loading/building utilities
src/pm25pred/future_met.py               ERA5 feature helpers
scripts/build_beijing_tensor.py          Local-only tensor builder
scripts/build_historical_era5_tensor.py  Historical-ERA5 tensor builder
scripts/train_hermes_component.py        Train one HERMES component
scripts/select_hermes_weights.py         Validation-only ensemble weight selection
scripts/evaluate_hermes.py               Evaluate fixed HERMES ensemble
scripts/run_hermes_experiment.sh         Reproduce the full three-seed HERMES experiment
scripts/train_clean_split_standard_baselines.py   GRU+RNN, BiLSTM-MA, Spatial CNN baselines
scripts/run_clean_split_standard_baselines.sh
scripts/train_clean_split_graph_baselines.py      STGCN, Graph WaveNet, AGCRN baselines
scripts/run_clean_split_graph_baselines.sh
scripts/analyze_hermes_evidence.py       Block-bootstrap, ablation, and seasonal-diagnostic tables
scripts/build_figures.py                 Manuscript figure generation
pyproject.toml                           uv project and dependency file
```

## Public Release Checklist

Before uploading this folder to GitHub, Zenodo, or another public archive:

- Choose and add a software license.
- Confirm the UCI dataset redistribution terms are compatible with the target archive.
- Add the final archive DOI or repository URL to the manuscript Data and code availability section.
