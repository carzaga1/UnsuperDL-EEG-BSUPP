from datetime import datetime, timezone

import pandas as pd
from click.testing import CliRunner

from bscarlos.data import preprocess_data
from bscarlos.testing.synthetic_data import (
    export_synthetic_edf,
    make_synthetic_annotations_txt,
    make_synthetic_eeg_signal,
    make_synthetic_mne_raw,
)


def test_preprocess_data_cli(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw_kispi"
    processed_dir = tmp_path / "processed_kispi"
    raw_dir.mkdir()

    n_seconds = 30.0
    sample_rate = 256
    start_datetime = datetime(2024, 1, 1)

    # Burst intervals are deterministic given (n_seconds, sample_rate), independent
    # of seed/channels, so this matches what make_synthetic_mne_raw generates below.
    _, burst_intervals = make_synthetic_eeg_signal(
        n_channels=1, n_seconds=n_seconds, sample_rate=sample_rate, seed=0
    )

    raw = make_synthetic_mne_raw(n_seconds=n_seconds, sample_rate=sample_rate, seed=0)
    raw.set_meas_date(start_datetime.replace(tzinfo=timezone.utc))
    export_synthetic_edf(raw, raw_dir / "SE999_annotated1.edf")

    annotations = make_synthetic_annotations_txt(burst_intervals, seed=0, start_datetime=start_datetime)
    annotations.to_csv(raw_dir / "SE999_annotated1_annotations.txt", index=False)

    monkeypatch.setattr(preprocess_data, "RAW_KISPI_DATA_FOLDER", raw_dir)
    monkeypatch.setattr(preprocess_data, "PROCESSED_KISPI_DATA_FOLDER", processed_dir)

    runner = CliRunner()
    result = runner.invoke(preprocess_data.main)

    assert result.exit_code == 0, result.output

    output_path = processed_dir / "SE999_a1_filt_merged.parquet"
    assert output_path.exists()

    df = pd.read_parquet(output_path)
    signal_columns = [c for c in df.columns if c != "ground_truth"]
    assert len(signal_columns) == 6
    assert "ground_truth" in df.columns
    assert df["ground_truth"].sum() > 0
