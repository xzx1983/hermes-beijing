# Release Code Review

## Scope

This review covers the public HERMES release folder only. The goal is to expose enough code
and data to rerun HERMES training, validation-weight selection, test evaluation, baseline
comparisons, evidence generation, and manuscript figures, without including unrelated
exploratory model code from the internal research repository.

## Review Findings

- The package is self-contained: raw data, the processed historical-ERA5 tensor, HERMES model
  code, baseline training scripts, evidence/figure scripts, and `uv` config are in one folder.
- `scripts/train_hermes_component.py` and `scripts/train_clean_split_standard_baselines.py`
  were smoke-tested directly against this packaged folder (1 epoch, small hidden size, CPU) and
  ran end to end without import or path errors.
- Every number in the manuscript's formal-baseline, block-bootstrap, ablation, and
  season-stratified tables was independently recomputed from the cached CSVs in
  `results/cached/` and matches the manuscript exactly.
- Row-level prediction CSVs are excluded from the package because of file size (roughly 150 MB
  per seed per method); aggregate and station-level metric CSVs are included instead, and the
  full prediction files regenerate from the training/evaluation scripts.
- The code does not import from the parent internal research repository; it only depends on the
  `pm25pred` package included in `src/`.
- The ERA5 interpolation-to-station pipeline (raw Copernicus download and interpolation) is not
  included; the processed tensor that pipeline produces is shipped directly instead, consistent
  with the manuscript's data-availability statement.

## Remaining Author Decisions

- Choose a public software license before upload.
- Confirm whether the UCI Beijing dataset can be redistributed directly in the target archive.
  If not, keep `data/raw/` out of the archive and provide a download script or source link.
- Decide whether this release replaces or lives alongside the existing `hawgt-beijing` public
  repository, which contains a different, earlier model (HAWGT) tied to a different, since
  rejected manuscript. Reusing that repository name/URL for HERMES would be misleading to
  anyone who already has it bookmarked or cited; a separate repository is recommended.
- Add the final repository URL or DOI to the manuscript's Data and code availability section.
