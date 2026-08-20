"""Synthetic EEG data generators so tests never need real KISPI patient data.

Channel names and drop/keep lists mirror notebooks/00_etl-2.ipynb (cells 5/7 for
the drop list, cell 49 for the bipolar-pair electrodes) so generated fixtures
exercise the real ingestion/bipolar/ground-truth code paths faithfully.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# 21 raw 10-20 EEG channels (the 13 that get dropped + the 8 electrodes used to
# derive the 6 bipolar channels) plus the 5 non-EEG channels also dropped.
EEG_CHANNELS = [
    "EEG Fp1", "EEG F3", "EEG F7", "EEG Fz", "EEG F4", "EEG F8",
    "EEG T3", "EEG C3", "EEG Cz", "EEG C4", "EEG T4",
    "EEG T5", "EEG P3", "EEG Pz", "EEG P4", "EEG T6",
    "EEG Fp2", "EEG O1", "EEG O2", "EEG A1", "EEG A2",
]
NON_EEG_CHANNELS = ["NASE", "EKG", "PHO", "EOG", "EMG"]
ALL_CHANNELS = EEG_CHANNELS + NON_EEG_CHANNELS

# Exact bipolar-derived column names (notebooks/00_etl-2.ipynb cell 49).
CHANNELS_6CH = ["Fp1-F7", "F7-T3", "T3-T5", "Fp2-F8", "F8-T4", "T4-T6"]
# No 17-channel-variant notebook defines explicit column names; this is a
# representative subset of the real 10-20 montage for schema-shape testing.
CHANNELS_17CH = EEG_CHANNELS[:17]


def make_synthetic_eeg_signal(
    n_channels: int,
    n_seconds: float,
    sample_rate: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Alternating burst/suppression segments with known boundaries.

    Returns an (n_channels, n_samples) array in microvolt-scale units, plus the
    list of (start_s, end_s) offsets where "burst" segments occur.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(n_seconds * sample_rate)
    t = np.arange(n_samples) / sample_rate

    segment_seconds = 5.0
    n_segments = int(np.ceil(n_seconds / segment_seconds))
    signal = np.zeros((n_channels, n_samples))
    burst_intervals: list[tuple[float, float]] = []

    for seg in range(n_segments):
        start_s = seg * segment_seconds
        end_s = min((seg + 1) * segment_seconds, n_seconds)
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        seg_t = t[start_idx:end_idx]
        is_burst = seg % 2 == 0

        for ch in range(n_channels):
            if is_burst:
                freqs = rng.uniform(1, 15, size=3)
                amps = rng.uniform(50, 150, size=3)
                wave = sum(a * np.sin(2 * np.pi * f * seg_t) for f, a in zip(freqs, amps))
                noise = rng.normal(0, 10, size=seg_t.shape)
                signal[ch, start_idx:end_idx] = wave + noise
            else:
                signal[ch, start_idx:end_idx] = rng.normal(0, 3, size=seg_t.shape)

        if is_burst:
            burst_intervals.append((start_s, end_s))

    return signal, burst_intervals


def make_synthetic_mne_raw(
    n_seconds: float = 30.0,
    sample_rate: int = 256,
    seed: int = 0,
) -> mne.io.RawArray:
    """Build a synthetic mne.io.RawArray using the real 10-20 channel-name subset."""
    signal, _ = make_synthetic_eeg_signal(
        n_channels=len(ALL_CHANNELS), n_seconds=n_seconds, sample_rate=sample_rate, seed=seed
    )
    signal_volts = signal * 1e-6  # mne expects volts; our signal is microvolt-scale

    ch_types = ["eeg"] * len(EEG_CHANNELS) + ["misc"] * len(NON_EEG_CHANNELS)
    info = mne.create_info(ch_names=ALL_CHANNELS, sfreq=sample_rate, ch_types=ch_types)
    return mne.io.RawArray(signal_volts, info, verbose=False)


def export_synthetic_edf(raw: mne.io.RawArray, path: Path) -> Path:
    """Export a synthetic Raw object to a real, parseable EDF file via edfio."""
    path = Path(path)
    raw.export(path, fmt="edf", overwrite=True, verbose=False)
    return path


def make_synthetic_annotations_txt(
    burst_intervals: list[tuple[float, float]],
    seed: int = 0,
    start_datetime: datetime = datetime(2024, 1, 1),
) -> pd.DataFrame:
    """Fake annotator export matching the real Onset/Duration/Annotation schema
    (see notebooks/00_etl-2.ipynb cells 57-58) with "Burst starts"/"Burst end"
    rows at the given known burst offsets (seconds from start_datetime).
    """
    rows = []
    for start_s, end_s in burst_intervals:
        onset_start = start_datetime + timedelta(seconds=start_s)
        onset_end = start_datetime + timedelta(seconds=end_s)
        rows.append(
            {"Onset": onset_start.strftime("%Y-%m-%dT%H:%M:%S.%f"), "Duration": np.nan, "Annotation": "Burst starts"}
        )
        rows.append(
            {"Onset": onset_end.strftime("%Y-%m-%dT%H:%M:%S.%f"), "Duration": np.nan, "Annotation": "Burst end"}
        )
    return pd.DataFrame(rows, columns=["Onset", "Duration", "Annotation"])


def make_synthetic_processed_parquet(channel_set: str, n_windows: int, seed: int) -> pd.DataFrame:
    """A DataFrame matching the processed_kispi schema: signal columns (6ch or
    17ch) plus a binary ground_truth column, with alternating burst/suppression
    segments so downstream detector/model tests exercise real signal-energy logic.
    """
    if channel_set == "6ch":
        columns = CHANNELS_6CH
    elif channel_set == "17ch":
        columns = CHANNELS_17CH
    else:
        raise ValueError(f"Unknown channel_set: {channel_set!r}")

    rng = np.random.default_rng(seed)
    segment_len = 50
    signal = np.zeros((n_windows, len(columns)))
    ground_truth = np.zeros(n_windows, dtype=int)

    for start in range(0, n_windows, segment_len):
        end = min(start + segment_len, n_windows)
        is_burst = (start // segment_len) % 2 == 0
        scale = 50.0 if is_burst else 3.0
        signal[start:end, :] = rng.normal(0, scale, size=(end - start, len(columns)))
        if is_burst:
            ground_truth[start:end] = 1

    df = pd.DataFrame(signal, columns=columns)
    df["ground_truth"] = ground_truth
    return df


def make_synthetic_windows(n_windows: int, window_size: int, n_channels: int, seed: int) -> np.ndarray:
    """In-memory (n_windows, window_size * n_channels) array, bypassing parquet
    entirely, for fast architectures/training unit tests. Matches the flattened
    window layout EEGDataset/VAE expect (see notebooks/03_phase2_vae.ipynb).
    """
    rng = np.random.default_rng(seed)
    input_dim = window_size * n_channels
    is_burst = rng.integers(0, 2, size=n_windows).astype(bool)

    windows = np.empty((n_windows, input_dim), dtype=np.float32)
    n_burst = int(is_burst.sum())
    windows[is_burst] = rng.normal(0, 50, size=(n_burst, input_dim)).astype(np.float32)
    windows[~is_burst] = rng.normal(0, 3, size=(n_windows - n_burst, input_dim)).astype(np.float32)
    return windows


def make_synthetic_data_attributes(patient_ids: list[str], sample_rates: list[int]) -> pd.DataFrame:
    """Mimics data_attributes_kispi.csv (unique_patient_id index, sample_rate
    column) — see notebooks/01_bsupp_all_kispi_a1_6ch.ipynb cell 2/4. Sample
    rates are taken per-patient rather than assumed constant, since real
    patients are recorded at different rates (200 Hz vs 256 Hz observed).
    """
    if len(patient_ids) != len(sample_rates):
        raise ValueError("patient_ids and sample_rates must be the same length")
    return pd.DataFrame({"unique_patient_id": patient_ids, "sample_rate": sample_rates}).set_index(
        "unique_patient_id"
    )
