import pytest

from bscarlos.testing.synthetic_data import (
    export_synthetic_edf,
    make_synthetic_annotations_txt,
    make_synthetic_eeg_signal,
    make_synthetic_mne_raw,
    make_synthetic_processed_parquet,
    make_synthetic_windows,
)


@pytest.fixture
def synthetic_edf_file(tmp_path):
    raw = make_synthetic_mne_raw(n_seconds=30.0, sample_rate=256, seed=0)
    return export_synthetic_edf(raw, tmp_path / "synthetic.edf")


@pytest.fixture
def synthetic_annotations_file(tmp_path):
    _, burst_intervals = make_synthetic_eeg_signal(n_channels=1, n_seconds=30.0, sample_rate=256, seed=0)
    annotations = make_synthetic_annotations_txt(burst_intervals, seed=0)
    path = tmp_path / "annotations.txt"
    annotations.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_parquet_6ch(tmp_path):
    df = make_synthetic_processed_parquet(channel_set="6ch", n_windows=500, seed=0)
    path = tmp_path / "synthetic_6ch.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def synthetic_parquet_17ch(tmp_path):
    df = make_synthetic_processed_parquet(channel_set="17ch", n_windows=500, seed=0)
    path = tmp_path / "synthetic_17ch.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def synthetic_windows():
    return make_synthetic_windows(n_windows=128, window_size=512, n_channels=6, seed=0)
