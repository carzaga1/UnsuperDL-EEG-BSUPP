import mne
import numpy as np
import pandas as pd
import pytest

from bscarlos.testing.synthetic_data import (
    ALL_CHANNELS,
    export_synthetic_edf,
    make_synthetic_annotations_txt,
    make_synthetic_data_attributes,
    make_synthetic_eeg_signal,
    make_synthetic_mne_raw,
    make_synthetic_processed_parquet,
)


def test_burst_windows_have_higher_energy():
    signal, burst_intervals = make_synthetic_eeg_signal(
        n_channels=4, n_seconds=20, sample_rate=256, seed=0
    )
    sample_rate = 256
    burst_start_idx = int(burst_intervals[0][0] * sample_rate)
    burst_end_idx = int(burst_intervals[0][1] * sample_rate)
    burst_energy = np.mean(signal[:, burst_start_idx:burst_end_idx] ** 2)

    suppression_start_idx = burst_end_idx
    suppression_end_idx = suppression_start_idx + (burst_end_idx - burst_start_idx)
    suppression_energy = np.mean(signal[:, suppression_start_idx:suppression_end_idx] ** 2)

    assert burst_energy > suppression_energy


def test_export_synthetic_edf_roundtrip(tmp_path):
    raw = make_synthetic_mne_raw(n_seconds=10, sample_rate=256, seed=1)
    edf_path = export_synthetic_edf(raw, tmp_path / "synthetic.edf")

    raw_roundtrip = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    assert len(raw_roundtrip.ch_names) == len(ALL_CHANNELS)
    assert raw_roundtrip.times[-1] == pytest.approx(raw.times[-1], abs=0.1)


def test_annotations_roundtrip(tmp_path):
    burst_intervals = [(0.0, 5.0), (10.0, 15.0), (20.0, 22.5)]
    annotations = make_synthetic_annotations_txt(burst_intervals, seed=0)

    txt_path = tmp_path / "annotations.txt"
    annotations.to_csv(txt_path, index=False)
    roundtrip = pd.read_csv(txt_path)

    assert len(roundtrip) == 2 * len(burst_intervals)
    assert (roundtrip["Annotation"] == "Burst starts").sum() == len(burst_intervals)
    assert (roundtrip["Annotation"] == "Burst end").sum() == len(burst_intervals)


@pytest.mark.parametrize("channel_set,n_columns", [("6ch", 6), ("17ch", 17)])
def test_processed_parquet_schema(channel_set, n_columns):
    df = make_synthetic_processed_parquet(channel_set=channel_set, n_windows=200, seed=0)

    signal_columns = [c for c in df.columns if c != "ground_truth"]
    assert len(signal_columns) == n_columns
    assert "ground_truth" in df.columns
    assert df[signal_columns].dtypes.eq(np.float64).all()
    assert np.issubdtype(df["ground_truth"].dtype, np.integer)
    assert len(df) == 200


def test_data_attributes_schema():
    attrs = make_synthetic_data_attributes(
        patient_ids=["SE008_a1", "SE021_a1"], sample_rates=[256, 200]
    )

    assert attrs.index.name == "unique_patient_id"
    assert attrs.loc["SE008_a1", "sample_rate"] == 256
    assert attrs.loc["SE021_a1", "sample_rate"] == 200


# --- Task 9: exercise the pytest fixtures themselves (tests/conftest.py), ---
# --- not the generator functions directly, to confirm the wiring works.   ---


def test_synthetic_edf_file_fixture(synthetic_edf_file):
    raw = mne.io.read_raw_edf(synthetic_edf_file, preload=True, verbose=False)
    assert len(raw.ch_names) == len(ALL_CHANNELS)


def test_synthetic_annotations_file_fixture(synthetic_annotations_file):
    annotations = pd.read_csv(synthetic_annotations_file)
    assert set(annotations["Annotation"]) <= {"Burst starts", "Burst end"}
    assert len(annotations) > 0


def test_synthetic_parquet_6ch_fixture(synthetic_parquet_6ch):
    df = pd.read_parquet(synthetic_parquet_6ch)
    assert len([c for c in df.columns if c != "ground_truth"]) == 6


def test_synthetic_parquet_17ch_fixture(synthetic_parquet_17ch):
    df = pd.read_parquet(synthetic_parquet_17ch)
    assert len([c for c in df.columns if c != "ground_truth"]) == 17


def test_synthetic_windows_fixture(synthetic_windows):
    assert synthetic_windows.shape == (128, 512 * 6)
    assert synthetic_windows.dtype == np.float32
