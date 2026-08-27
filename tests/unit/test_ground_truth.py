from datetime import datetime, timedelta

import pandas as pd

from bscarlos.data.ground_truth import build_ground_truth_labels, parse_annotations
from bscarlos.testing.synthetic_data import make_synthetic_annotations_txt, make_synthetic_eeg_signal


def test_build_ground_truth_labels_reproduces_known_intervals(tmp_path):
    _, burst_intervals = make_synthetic_eeg_signal(n_channels=1, n_seconds=30.0, sample_rate=256, seed=0)

    start_datetime = datetime(2024, 1, 1)
    annotations_df = make_synthetic_annotations_txt(burst_intervals, seed=0, start_datetime=start_datetime)
    annotations_path = tmp_path / "annotations.txt"
    annotations_df.to_csv(annotations_path, index=False)

    annotations = parse_annotations(annotations_path)

    sample_rate = 256
    n_samples = int(30.0 * sample_rate)
    timestamps = pd.DatetimeIndex(
        [start_datetime + timedelta(seconds=i / sample_rate) for i in range(n_samples)]
    )

    ground_truth = build_ground_truth_labels(timestamps, annotations)

    for start_s, end_s in burst_intervals:
        mid_ts = start_datetime + timedelta(seconds=(start_s + end_s) / 2)
        idx = ground_truth.index.get_indexer([mid_ts], method="nearest")[0]
        assert ground_truth.iloc[idx] == 1

    # A point strictly between two burst intervals should not be labeled burst.
    if len(burst_intervals) >= 2:
        gap_ts = start_datetime + timedelta(
            seconds=(burst_intervals[0][1] + burst_intervals[1][0]) / 2
        )
        idx = ground_truth.index.get_indexer([gap_ts], method="nearest")[0]
        assert ground_truth.iloc[idx] == 0
