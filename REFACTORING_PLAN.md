# REFACTORING_PLAN.md

## Purpose

This document orchestrates an AI-agent-driven refactor of the `bscarlos` package: extracting logic currently trapped in `notebooks/*.ipynb` into a modular, production-ready, config-driven Python package under `src/bscarlos/`, adding test coverage against synthetic (non-PHI) data, and adding Docker + CI so the pipeline is portable across clinical workstations.

**Hard constraint: `notebooks/` is never modified, deleted, or overwritten by any task in this plan.** Notebooks are read-only reference material — the exact source of truth for extracted logic — and remain a valuable exploratory record independent of this refactor.

The repo layout follows the [Future-Proof-DS MLOps project structure](https://github.com/Future-Proof-DS/mlops-project-structure-for-data-scientists) convention (`src/` for importable logic, `scripts/` for thin CLI entry points, top-level `config/` for hyperparameters, top-level `models/` for saved checkpoint artifacts, `pyproject.toml`+`uv.lock` for dependencies) adapted where the notebook-compatibility contract requires it — see "Deliberate deviations from the reference layout" below.

Each task (T1–T22) below is scoped to be handed to a stateless AI coding agent (Claude Code, Google Jules, etc.) with **zero prior conversation context**. Every task states its exact inputs, exact outputs, and a concrete verification step. Tasks are ordered so earlier tasks unblock later ones — most importantly, the synthetic data generator (T6–T9) lands early because nearly everything downstream is tested against it instead of real KISPI patient data.

---

## Context for any agent picking up a task

- Repo root contains an installable package at `src/bscarlos/` (`pip install -e .` or `uv sync`), already used by several notebooks via `from bscarlos... import ...`.
- `src/bscarlos/settings.py` defines a **byte-for-byte stable contract** of module-level constants (`PROJECT_FOLDER`, `DATA_FOLDER`, `RAW_DATA_FOLDER`, `PROCESSED_DATA_FOLDER`, `RAW_KISPI_DATA_FOLDER`, `PROCESSED_KISPI_DATA_FOLDER`, `REFERENCES_FOLDER`, `OUTPUT_FOLDER`) loaded from a gitignored `local_settings.yml`. Every notebook imports these by exact name — **never rename or change the type of an existing constant**.
- `src/bscarlos/architectures/__init__.py` is currently an empty file (renamed from `models/` to avoid colliding with the top-level `models/` artifacts directory — see below) — filling it with the extracted VAE/dVAE code is the central goal of this refactor.
- Real data lives under `data/raw_kispi/` (PHI — real patient EDFs + annotations), `data/processed_kispi/{6_channel,17_channel}/` (parquet, still identifiable), `data/raw/` (de-identified numbered patients), `output/` (per-patient result HDF5s), and `references/mrn_pseudonym_keys.csv` (PHI mapping). **All of these are gitignored and must never be committed, baked into a Docker image, or required by CI.** Every task's verification step must be runnable against synthetic data only.
- Dependencies are managed via `pyproject.toml` + `uv.lock` (see "Environment reproducibility" below) — do not reintroduce `requirements.txt`/`setup.py`, both were removed in favor of this.
- Per project-owner decision, `tensorflow`/`tf_keras` are retained in `pyproject.toml` even though currently unused (possible future comparison work) — do not remove them.
- The canonical VAE architecture (used as the extraction source for T16–T18) lives in `notebooks/03_phase2_vae.ipynb`: `Encoder` (fc 512→256→latent_dim mu/logvar), `Decoder` (latent→256→512→output, sigmoid), `VAE` (`forward` returns `(reconstructed_x, mu, logvar, z)`), `loss_function` = L1 reconstruction + β·KL, module-level `reparameterize(mu, logvar)`, `EEGDataset(Dataset)`. Baseline hyperparameters observed: `window_size=512` samples, 6 channels ⇒ `input_dim=3072`, `latent_dim=32`, `batch_size=64`, `lr=1e-5`, `beta=0.8`, `num_epochs=100`.
- The canonical ETL pipeline (extraction source for T12–T15) lives in `notebooks/00_etl-2.ipynb` (the most recent, ground-truth-corrected variant — prefer this over `00_etl.ipynb`/`00_etl-1.ipynb`, which are earlier iterations of the same logic): reads raw `.edf` + annotator `.txt` files from `RAW_KISPI_DATA_FOLDER`, drops non-relevant channels, derives 6 bipolar channels from 17 raw 10-20 channels via fixed pairs (`Fp1-F7, F7-T3, T3-T5, Fp2-F8, F8-T4, T4-T6`), builds a binary ground-truth column from `"Burst start(s)"`/`"Burst end"` annotation rows, and writes parquet to `PROCESSED_KISPI_DATA_FOLDER`. Per-patient sample rate is **not constant** (200 Hz vs 256 Hz observed) — always read it from `data_attributes_kispi.csv`, never hardcode it.

### Deliberate deviations from the reference layout

- **`data/` keeps its existing sub-folder names** (`raw`, `raw_kispi`, `processed`, `processed_kispi`) instead of the reference's numbered `01_raw/02_interim/03_processed/04_predictions` staging — `settings.py`'s constants point at these exact names and every notebook imports them; renaming would break the notebook-compatibility contract.
- **`src/bscarlos/architectures/`** holds VAE/dVAE *code* (Python import path `bscarlos.architectures`), distinct from the top-level **`models/`** directory, which holds saved checkpoint *artifacts* (gitignored). The reference uses `models/` only for artifacts; we renamed our code subpackage to avoid the name collision rather than eliminate the code subpackage.
- **`scripts/`** (new, thin CLI entry points) coexists with the pre-existing Click command group in `src/bscarlos/__main__.py` (`download-raw-data`, `preprocess-data`), which is left as-is. New entry points introduced by this refactor (`train_vae`, `predict_vae`) go in `scripts/` per the reference convention; the two already-built legacy commands aren't migrated, to avoid unrelated churn.

---

## Target package structure

```
src/bscarlos/
├── settings.py                 # UNCHANGED public contract; may gain optional device/n_jobs
├── data_processor_kispi.py     # unchanged logic; tests added
├── detector.py / detector_v2.py / detector_v3.py   # unchanged logic; tests added
├── eeg_plotters.py, roc_plotter.py                 # unchanged
├── config/                     # NEW — Python config-loading code (not to be confused with top-level config/)
│   ├── __init__.py
│   └── schema.py                # VAETrainingConfig dataclass + load_training_config()
├── data/                        # EXISTING subpackage, extended
│   ├── download_raw_data.py     # unchanged stub
│   ├── preprocess_data.py       # wired to real logic (currently a no-op stub)
│   ├── edf_ingest.py            # NEW — extracted from 00_etl-2.ipynb
│   ├── bipolar.py               # NEW — extracted from 00_etl-2.ipynb
│   └── ground_truth.py          # NEW — extracted from 00_etl-2.ipynb
├── architectures/                # FILLS THE EMPTY STUB — central goal of this refactor (renamed from models/)
│   ├── __init__.py              # exports Encoder, Decoder, VAE, reparameterize, EEGDataset
│   ├── vae.py                   # NEW — extracted from 03_phase2_vae.ipynb
│   └── datasets.py              # NEW — extracted from 03_phase2_vae.ipynb
├── training/                    # NEW
│   └── train_vae.py             # extracted training loop, config-driven
├── inference/                   # NEW
│   └── predict_vae.py           # new synthesis: load checkpoint, reconstruction error, labeling
└── testing/                     # NEW — synthetic data generators, ships inside the package
    └── synthetic_data.py

config/                          # NEW top-level, tracked in git (hyperparameters only, no PHI)
├── vae_6ch.yml
└── vae_17ch.yml

models/                          # NEW top-level, gitignored — saved checkpoint artifacts (see models/.gitkeep)

scripts/                         # NEW top-level — thin CLI entry points calling into src/bscarlos
├── train_vae.py                 # NEW (T20)
└── predict_vae.py                # NEW (T20)

tests/                           # NEW top-level, mirrors src/bscarlos
├── conftest.py
└── unit/test_*.py               # one file per extracted module

Dockerfile                       # NEW — CPU image
Dockerfile.cuda                  # NEW — GPU image
docker-compose.yml               # NEW — local orchestration; bind-mounts notebooks/, data/, models/
.dockerignore                    # NEW
.github/workflows/ci.yml         # NEW
pyproject.toml                   # dependencies + metadata (uv/setuptools)
uv.lock                          # locked dependency versions
```

**Repo hygiene cleanup** (no new code, just removing cruft): committed `__pycache__/` and `.ipynb_checkpoints/` directories under `src/bscarlos/`, the root-level `test.py` (a 4-line non-test smoke script), and the root-level orphaned empty `__init__.py`.

---

## Configuration design

Two-tier YAML (no Hydra — chosen to minimize risk to the `settings.py` notebook-compatibility contract and avoid a new dependency at this project's scale):

1. **`local_settings.yml`** (gitignored, per-workstation, mechanism unchanged) — existing `data_folder` key, plus new *optional* keys `device` (`"cpu"`/`"cuda"`/`"mps"`, default `"cpu"`) and `n_jobs` (default `-1`), so existing workstation copies keep working unmodified.
2. **`config/vae_6ch.yml` / `config/vae_17ch.yml`** (tracked in git, no PHI — pure hyperparameters): `channel_set, n_channels, window_size, input_dim, latent_dim, beta, learning_rate, batch_size, num_epochs, train_val_test_split, random_seed, annotator, checkpoint_dir`. Defaults mirror the `03_phase2_vae.ipynb` baseline exactly.
3. **Loader**: `bscarlos/config/schema.py` — a `VAETrainingConfig` dataclass plus `load_training_config(path=None, **overrides)`, which validates `input_dim == window_size * n_channels` at load time and raises `ValueError` on mismatch.
4. **Portability**: moving to a new hospital workstation requires editing only `local_settings.yml` (paths, device); `config/*.yml` is identical everywhere and versioned; the CLI takes `--config config/vae_6ch.yml` so no source code edits are ever required.

---

## Environment reproducibility

- **Dependency management: `pyproject.toml` + `uv` + `uv.lock`** (migrated from `requirements.txt`/`setup.py`, both removed). `pyproject.toml`'s `[project.dependencies]` holds the full pinned dependency list (including `click==8.1.7`); `[project.optional-dependencies].dev` holds `pytest`/`pytest-cov`.
  - `edf2parquet` is **not on PyPI** — it must be installed from its exact git commit: `edf2parquet @ git+https://github.com/NarayanSchuetz/edf2parquet.git@e919f7bffce8256703d9b6974c8e3aaac636fd56`. The old `requirements.txt`'s plain `edf2parquet==0.1.2` pin never actually worked for a from-scratch install — do not revert to that form.
  - `requires-python = ">=3.11,<3.12"` — narrowed because `numpy==1.24.4` (pinned) is incompatible with Python ≥3.12 per `pandas==2.2.3`'s own constraints; this matches the actually-tested environment anyway.
  - `[tool.uv].required-environments` is pinned to Windows x86_64 + Linux x86_64 (`sys_platform == 'win32'`/`'linux' and platform_machine == 'AMD64'`/`'x86_64'`) because `tensorflow-io-gcs-filesystem` (a transitive `tensorflow` dependency) has no Windows wheel and isn't actually needed there; without this, `uv lock`/`uv sync` fails to resolve on Windows. Any Dockerfile task must keep the Linux target resolvable too.
  - Because `edf2parquet` is a VCS (git) dependency, any Docker build stage running `uv sync` needs `git` installed in the image, not just `uv`/Python.
  - `mne==1.9.0` and `edfio==0.4.16` were **not** originally declared anywhere (not in the old `requirements.txt` either) despite several notebooks importing them directly — they were only ever present because someone `pip install`ed them by hand into a local venv. Added as proper pinned dependencies during T6, since the synthetic EDF generator needs them functionally.
- **Two Dockerfiles**, both multi-stage, both using `uv sync --frozen` (reading `uv.lock` directly) instead of `pip install -r requirements.txt`:
  - `Dockerfile` — `python:3.11-slim` base, CPU-only.
  - `Dockerfile.cuda` — an `nvidia/cuda` runtime base, for GPU-equipped workstations. GitHub-hosted CI runners have no GPU, so CI only *builds* this image, not runs it — the CPU image is built and smoke-run.
- `.dockerignore` excludes `notebooks/, data/, output/, models/, references/, local_settings.yml, .git, .venv, __pycache__, .ipynb_checkpoints`.
- **PHI safety**: nothing under `data/`, `output/`, or `references/mrn_pseudonym_keys.csv` is ever copied into an image layer. Real data reaches a container only via `docker run -v` bind mounts at runtime. CI and test images never see real data — only synthetic fixtures generated in-process.
- **`docker-compose.yml`** (local dev orchestration, not used in CI): declares the same bind-mount pattern instead of ad-hoc `docker run -v` flags — mounts the repo root (so `notebooks/` is live-editable inside the container even though `.dockerignore` keeps it out of the built image itself) plus `data/`/`models/`/`local_settings.yml` read-write, and runs `jupyter notebook --ip=0.0.0.0` by default so notebooks stay usable in a containerized environment. A `gpu` compose profile builds `Dockerfile.cuda` instead. This is the resolution to "won't `notebooks/` be invisible in Docker?" below — the image never contains it, but the running container sees it via the mount.

---

## Synthetic/mock data strategy

Implemented at `src/bscarlos/testing/synthetic_data.py` (ships inside the installable package — any coding agent with zero PHI access can import and use it) with thin pytest fixture wrappers in `tests/conftest.py`. **No binary fixture files are committed to git** — everything is generated deterministically via a seeded RNG at test time into `tmp_path`.

Design principle: synthetic "burst" segments must carry genuinely higher signal energy than "suppression" segments (not just uniform random noise), because `detector_v3.ClusteringDetector_v3` clusters on per-window covariance-matrix energy — a vacuous synthetic signal would let broken detector logic pass tests trivially.

Functions to implement:
- `make_synthetic_eeg_signal(n_channels, n_seconds, sample_rate, seed)` — alternating burst/suppression segments with known boundaries.
- `make_synthetic_mne_raw(...)` + `export_synthetic_edf(raw, path)` — builds a real, parseable minimal EDF using the real 10-20 channel-name subset, via the `edfio` export backend already in the dependency set.
- `make_synthetic_annotations_txt(burst_intervals, seed)` — writes a fake annotator `.txt` matching the real `Onset`/`Duration`/`Annotation` schema with `"Burst starts"`/`"Burst end"` rows at known times.
- `make_synthetic_processed_parquet(channel_set, n_windows, seed)` — a DataFrame matching the `processed_kispi` schema (6 or 17 signal columns + `ground_truth`).
- `make_synthetic_data_attributes(patient_ids, sample_rates)` — mimics `data_attributes_kispi.csv`, deliberately using non-constant sample rates.
- `make_synthetic_windows(n_windows, window_size, n_channels, seed)` — an in-memory array bypassing parquet entirely, for fast model/training unit tests.

---

## Task list

**T1 — Repo hygiene cleanup**
- Input: repo tree (read-only inspection of `src/bscarlos/__pycache__/`, `src/bscarlos/data/__pycache__/`, `src/bscarlos/data/.ipynb_checkpoints/`, root `test.py`, root `__init__.py`).
- Output: delete all of the above.
- Verify: `git ls-files | grep -E '__pycache__|ipynb_checkpoints'` returns nothing.
- **Status: DONE.**

**T2 — Dependency & package-layout migration to the reference MLOps structure**
- Input: former `requirements.txt`/`setup.py`, former `src/bscarlos/models/` stub.
- Output: `pyproject.toml` with full `[project.dependencies]` + `dev` extra, `uv.lock` generated and verified via `uv sync --extra dev`; `requirements.txt`/`requirements-dev.txt`/`setup.py` deleted; `src/bscarlos/models/` renamed to `src/bscarlos/architectures/`; new top-level `models/` (gitignored, `.gitkeep`) and `scripts/` (with `README.md`) directories added; `.gitignore` updated with `models/*`/`!models/.gitkeep`.
- Verify: `uv sync --extra dev` succeeds from a clean checkout; `python -c "import bscarlos, bscarlos.architectures"` succeeds; `git ls-files | grep bscarlos/models` returns nothing.
- **Status: DONE.**

**T3 — pytest scaffolding**
- Input: none.
- Output: `tests/__init__.py`, `tests/conftest.py` (empty placeholder, filled in T9), `tests/unit/__init__.py`, `[tool.pytest.ini_options]` in `pyproject.toml` (`testpaths = ["tests"]`, `addopts = "-ra"`).
- Verify: `pytest --collect-only` runs with zero errors and zero collected tests.
- **Status: DONE.**

**T4 — Config schema module**
- Input: `src/bscarlos/settings.py` (read-only, for the exact existing constant names/types).
- Output: `src/bscarlos/config/__init__.py`, `src/bscarlos/config/schema.py` — `VAETrainingConfig` dataclass (fields: `channel_set, n_channels, window_size, input_dim, latent_dim, beta, learning_rate, batch_size, num_epochs, train_val_test_split, random_seed, annotator, checkpoint_dir`) and `load_training_config(path: Path | None = None, **overrides) -> VAETrainingConfig`, validating `input_dim == window_size * n_channels`. Do not modify `settings.py`.
- Verify: `tests/unit/test_settings.py` — defaults match the notebook baseline (`window_size=512, latent_dim=32, beta=0.8, learning_rate=1e-5, batch_size=64, num_epochs=100`); a mismatched `input_dim` raises `ValueError`.
- **Status: DONE.**

**T5 — Tracked default configs + extended local_settings template**
- Input: `config/schema.py` (T4), `local_settings_template.yml`.
- Output: `config/vae_6ch.yml`, `config/vae_17ch.yml`; update `local_settings_template.yml` with optional `device`/`n_jobs` keys, commented as optional.
- Verify: both YAML configs load via `load_training_config` without error.
- **Status: DONE.**

**T6 — Synthetic EEG signal + EDF generator**
- Input: none new (reference `notebooks/00_etl-2.ipynb` channel-name cells, read-only).
- Output: `src/bscarlos/testing/__init__.py`, `src/bscarlos/testing/synthetic_data.py` with `make_synthetic_eeg_signal(n_channels, n_seconds, sample_rate, seed) -> tuple[np.ndarray, list[tuple[float,float]]]`, `make_synthetic_mne_raw(...) -> mne.io.RawArray`, `export_synthetic_edf(raw, path) -> Path`.
- Verify: burst-interval mean squared amplitude exceeds suppression-interval amplitude; EDF round-trips through `mne.io.read_raw_edf` with matching channel count/duration.
- **Status: DONE.**

**T7 — Synthetic annotations + ground-truth generator**
- Input: `notebooks/00_etl-2.ipynb` annotation-parsing cells (read-only), T6 output.
- Output: add `make_synthetic_annotations_txt(burst_intervals, seed) -> pd.DataFrame` to `synthetic_data.py`.
- Verify: write-then-read round trip reproduces `2 * len(burst_intervals)` start/end rows.
- **Status: DONE.**

**T8 — Synthetic processed-parquet + data_attributes generator**
- Input: `notebooks/01_bsupp_all_kispi_a1_6ch.ipynb` (read-only, for `data_attributes_kispi.csv` schema), `data_processor_kispi.py` (read-only, for expected dict shape).
- Output: add `make_synthetic_processed_parquet(channel_set, n_windows, seed) -> pd.DataFrame` and `make_synthetic_data_attributes(patient_ids, sample_rates) -> pd.DataFrame` to `synthetic_data.py`.
- Verify: correct column count (6 or 17 signal columns + `ground_truth`) and dtypes.
- Also added `make_synthetic_windows(n_windows, window_size, n_channels, seed)` here — it was listed in the strategy overview's function list but not explicitly assigned to any single task, and T9's `synthetic_windows` fixture needs it.
- **Status: DONE.**

**T9 — pytest fixtures wiring synthetic generators**
- Input: T6–T8 functions.
- Output: fill `tests/conftest.py` with `synthetic_edf_file`, `synthetic_annotations_file`, `synthetic_parquet_6ch`, `synthetic_parquet_17ch`, `synthetic_windows` fixtures.
- Verify: `pytest tests/unit/test_synthetic_data.py -v` passes using the fixtures.
- **Status: DONE.**

**T10 — Unit tests for `PatientDataProcessor`**
- Input: `src/bscarlos/data_processor_kispi.py` (read-only), T8/T9 fixtures.
- Output: `tests/unit/test_data_processor_kispi.py` covering `create_patient_data_dict`, `apply_timedelta_index`, `apply_bandpass_filter`, `split_data_for_5fold_cv`, `remove_ground_truth_column`.
- Verify: filter output preserves shape/columns and doesn't mutate `ground_truth`.
- **Status: DONE.**

**T11 — Unit tests for classical detectors**
- Input: `src/bscarlos/detector.py`/`detector_v2.py`/`detector_v3.py` (read-only), synthetic fixtures.
- Output: `tests/unit/test_detectors.py` — fit/predict `ClusteringDetector_v3` on synthetic data, assert accuracy above chance (e.g. `> 0.6`) vs known synthetic ground truth.
- Verify: `pytest tests/unit/test_detectors.py -v` passes; no changes to detector source files.
- **Status: DONE.** Only `ClusteringDetector_v3` was tested (per this spec); `detector.py`/`detector_v2.py` remain untested read-only references, superseded by v3.

**Post-plan addition — `detector_v4.py`**: adds an opt-in `n_jobs` parameter for `compute_cov_distances` (the O(n^2) pairwise covariance-distance step that dominates `fit()`/`predict()` on long recordings), plus a symmetric-matrix optimization (self-distance matrices only need their upper triangle computed). `detector_v3.py` is left byte-for-byte unchanged as a preserved research artifact, per this project's existing `detector.py` → `detector_v2.py` → `detector_v3.py` versioning convention. Default `n_jobs=1` (sequential) — benchmarking showed naive per-pair and even batched `joblib` parallel dispatch are *slower* than sequential at realistic short-recording scales (~450 windows / 15 min) because of fixed worker-pool startup cost; parallelization only pays off for genuinely long recordings (thousands of windows). See `detector_v4.py`'s class docstring for the full reasoning. Tested in `tests/unit/test_detector_v4.py`.

**T12 — Extract `data/edf_ingest.py`**
- Input: `notebooks/00_etl-2.ipynb` (read-only, channel-drop and file-discovery cells), T9's `synthetic_edf_file`.
- Output: `src/bscarlos/data/edf_ingest.py` — `read_edf_raw(path) -> mne.io.Raw`, `CHANNELS_TO_DROP: list[str]`, `drop_nonrelevant_channels(raw) -> mne.io.Raw`, `find_annotation_edf_files(raw_dir, annotator) -> list[Path]`.
- Verify: the synthetic EDF loses exactly the channels listed in `CHANNELS_TO_DROP`.
- **Status: DONE.**

**T13 — Extract `data/bipolar.py`**
- Input: `notebooks/00_etl-2.ipynb` bipolar-derivation cell (read-only), T12 output type.
- Output: `src/bscarlos/data/bipolar.py` — `BIPOLAR_PAIRS: list[tuple[str,str,str]]` (the 6 pairs), `create_bipolar_channels(df) -> pd.DataFrame`.
- Verify: output has exactly 6 correctly-named bipolar columns matching manual channel-pair subtraction on a synthetic 17-channel DataFrame.
- **Status: DONE.**

**T14 — Extract `data/ground_truth.py`**
- Input: `notebooks/00_etl-2.ipynb` annotation/ground-truth cells (read-only), T7's `synthetic_annotations_file`.
- Output: `src/bscarlos/data/ground_truth.py` — `parse_annotations(path) -> pd.DataFrame`, `build_ground_truth_labels(timestamps, annotations) -> pd.Series`.
- Verify: reproduces the known synthetic burst intervals exactly.
- **Status: DONE.** Handles both "Burst starts" and "Burst start" (real annotator files use both spellings inconsistently across patients).

**T15 — Wire `data/preprocess_data.py` CLI to real ETL logic**
- Input: `src/bscarlos/data/preprocess_data.py` (current no-op stub, existing Click command registered in `__main__.py` — left in place per the deviations note above), T12–T14 modules.
- Output: chain `edf_ingest → bipolar → ground_truth` inside the existing `@click.command()`, writing parquet to `PROCESSED_KISPI_DATA_FOLDER`.
- Verify: `CliRunner` test with `RAW_KISPI_DATA_FOLDER` monkeypatched to synthetic EDF+annotations produces a correctly-schemed parquet, exit code 0.
- **Status: DONE.** Real EDFs carry an absolute `meas_date`; ground-truth timestamps must be timezone-naive to compare against annotation onsets, so the timezone is stripped after `raw.to_data_frame(time_format="datetime")`.

**T16 — Extract `architectures/vae.py`**
- Input: `notebooks/03_phase2_vae.ipynb` model-definition cells (read-only), current empty `architectures/__init__.py`.
- Output: `src/bscarlos/architectures/vae.py` — `Encoder`, `Decoder`, `reparameterize(mu, logvar)`, `VAE` (`forward` returns `(reconstructed_x, mu, logvar, z)`, `loss_function(recon_x, x, mu, logvar, beta=0.1)`), signatures identical to the notebook. Update `architectures/__init__.py` to export all of these.
- Verify: forward pass on a random `(4, 3072)` tensor gives correctly-shaped outputs; `loss_function` returns a non-negative scalar.
- **Status: DONE** (by Google Jules, PR #2).

**T17 — Extract `architectures/datasets.py`**
- Input: `notebooks/03_phase2_vae.ipynb` dataset/windowing cells (read-only), T9's `synthetic_windows`.
- Output: `src/bscarlos/architectures/datasets.py` — `EEGDataset(Dataset)` (identical to notebook), `reshape_to_windows(eeg_array, window_size, n_channels)` (generalized off the hardcoded notebook values). Update `architectures/__init__.py` to also export `EEGDataset`.
- Verify: `reshape_to_windows` output shape matches `(num_windows, window_size*n_channels)`; `EEGDataset[i]` returns `(float32 tensor, 0)`.
- **Status: DONE** (by Google Jules, PR #2).

**T18 — Extract `training/train_vae.py`**
- Input: `notebooks/03_phase2_vae.ipynb` training-loop cells (read-only — use the more complete version with per-component loss tracking), T16/T17 (`bscarlos.architectures`), `config/schema.py` (T4).
- Output: `src/bscarlos/training/__init__.py`, `src/bscarlos/training/train_vae.py` — `train_vae(windows, config, checkpoint_path) -> dict` (loss histories), saving a checkpoint via `torch.save` into the top-level `models/` directory by default (`checkpoint_path` derived from `config.checkpoint_dir`, resolved against `models/`, not `OUTPUT_FOLDER`).
- Verify: smoke test with `config.num_epochs=2` on `synthetic_windows` completes in under 30s on CPU, checkpoint file exists, losses are finite.
- **Status: DONE** (by Google Jules, PR #2). `checkpoint_dir` is used directly as the save directory (not joined against a separate `models/` prefix elsewhere), so `config/schema.py`'s `checkpoint_dir` default was set to `"models"` to fulfill this task's stated intent.

**T19 — Build `inference/predict_vae.py`**
- Input: `architectures/vae.py` (T16), the checkpoint format from T18. No direct notebook source — this is new synthesis.
- Output: `src/bscarlos/inference/__init__.py`, `src/bscarlos/inference/predict_vae.py` — `load_model(checkpoint_path, config) -> VAE`, `reconstruction_error(model, windows) -> np.ndarray`, `predict_labels(errors, threshold=None) -> np.ndarray` (default threshold: 75th percentile).
- Verify: loads T18's checkpoint, output length matches input, errors are finite non-negative, labels are binary.
- **Status: DONE** (by Google Jules, PR #2).

**T20 — Add `scripts/train_vae.py`/`scripts/predict_vae.py` thin entry points**
- Input: `scripts/README.md` (convention), `training/train_vae.py` (T18), `inference/predict_vae.py` (T19), `config/schema.py` (T4).
- Output: `scripts/train_vae.py` and `scripts/predict_vae.py` — thin argument-parsing wrappers (`--config path/to/config.yml` plus override flags) that import from `bscarlos.training`/`bscarlos.inference` and call straight through; no business logic lives in these files.
- Verify: `python scripts/train_vae.py --config config/vae_6ch.yml --num-epochs 1` (with `PROCESSED_KISPI_DATA_FOLDER` monkeypatched to synthetic parquet via a test harness invoking the script's `main()`) exits 0.
- **Status: DONE** (by Google Jules, PR #2).

**T21 — Dockerfiles (CPU + CUDA) + docker-compose.yml**
- Input: final `pyproject.toml`/`uv.lock` (T2), `src/bscarlos/__main__.py`, `scripts/`.
- Output: `Dockerfile` (python:3.11-slim, two-stage, installs `git` + `uv`, runs `uv sync --frozen --no-dev`) and `Dockerfile.cuda` (nvidia/cuda runtime base, same pattern), `.dockerignore`, and `docker-compose.yml` — a `bscarlos` service building `Dockerfile`, bind-mounting the repo root (`.:/app`) plus `local_settings.yml`, running `jupyter notebook --ip=0.0.0.0 --no-browser`, exposing port 8888; a `bscarlos-cuda` service under a `gpu` compose profile building `Dockerfile.cuda` with the same mounts plus `deploy.resources.reservations.devices` GPU reservation.
- Verify: `docker build -f Dockerfile -t bscarlos:cpu .` and `docker build -f Dockerfile.cuda -t bscarlos:cuda .` both succeed; `docker run --rm bscarlos:cpu --help` prints CLI help; confirm no `data/`/`references/`/`models/` path appears in the build context; `docker compose config` validates without error; `docker compose up bscarlos` starts and `notebooks/*.ipynb` is visible/editable inside the running container despite being absent from the built image (bind mount, not baked in).

**T22 — GitHub Actions CI workflow**
- Input: `pyproject.toml`/`uv.lock` (T2), full `tests/` suite (T1–T21).
- Output: `.github/workflows/ci.yml` — checkout, install `uv` (e.g. `astral-sh/setup-uv`), `uv sync --extra dev`, `uv run pytest tests/ -v --cov=bscarlos`, plus a `docker build` step for both Dockerfiles (build-only for CUDA, build + `--help` run for CPU). Include a comment stating no real data is ever fetched or mounted in CI.
- Verify: CI run is green on push/PR; workflow log contains no reference to `raw_kispi` or `mrn_pseudonym_keys.csv`.

---

## Verification summary (run after all tasks complete)

1. `uv run pytest tests/ -v --cov=bscarlos` — full suite green, using only synthetic data.
2. `python -m bscarlos preprocess-data` and `python scripts/train_vae.py --config config/vae_6ch.yml` succeed against synthetic fixtures via CLI smoke tests.
3. `docker build` succeeds for both `Dockerfile` and `Dockerfile.cuda`; the CPU image runs `--help`; `docker compose up bscarlos` gives live access to `notebooks/` inside the container via bind mount.
4. CI workflow is green on GitHub Actions and never touches real KISPI data.
5. `git diff --stat notebooks/` shows no changes — the notebooks directory remains byte-identical throughout this refactor.
