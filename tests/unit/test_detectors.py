import numpy as np
import pandas as pd

from bscarlos.detector_v3 import ClusteringDetector_v3
from bscarlos.testing.synthetic_data import CHANNELS_6CH, make_synthetic_eeg_signal


def _make_eeg_dataframe(n_seconds, sample_rate, seed):
    signal, burst_intervals = make_synthetic_eeg_signal(
        n_channels=6, n_seconds=n_seconds, sample_rate=sample_rate, seed=seed
    )
    index = pd.timedelta_range(start=0, periods=signal.shape[1], freq=pd.Timedelta(seconds=1 / sample_rate))
    df = pd.DataFrame(signal.T, columns=CHANNELS_6CH, index=index)
    return df, burst_intervals


def _window_ground_truth(burst_intervals, windows, window_seconds=2.0):
    labels = []
    for window in windows:
        start_s = window.index[0].total_seconds()
        mid_s = start_s + window_seconds / 2
        is_burst = any(b_start <= mid_s < b_end for b_start, b_end in burst_intervals)
        labels.append(1 if is_burst else 0)
    return np.array(labels)


def _best_orientation_accuracy(predicted, ground_truth):
    # SpectralClustering/KNN label integers (0 vs 1) are arbitrary, not tied to
    # "burst"/"suppression" semantics, so check both orientations.
    predicted = np.asarray(predicted)
    return max(
        np.mean(predicted == ground_truth),
        np.mean((1 - predicted) == ground_truth),
    )


def test_clustering_detector_v3_fit_separates_burst_from_suppression():
    df, burst_intervals = _make_eeg_dataframe(n_seconds=60, sample_rate=256, seed=1)

    detector = ClusteringDetector_v3()
    detector.fit(df)

    windows = detector.get_windows(df)
    ground_truth = _window_ground_truth(burst_intervals, windows)

    accuracy = _best_orientation_accuracy(detector.labels_, ground_truth)
    assert accuracy > 0.6


def test_clustering_detector_v3_predict_on_held_out_data():
    train_df, _ = _make_eeg_dataframe(n_seconds=60, sample_rate=256, seed=1)
    test_df, test_burst_intervals = _make_eeg_dataframe(n_seconds=30, sample_rate=256, seed=2)

    detector = ClusteringDetector_v3()
    detector.fit(train_df)
    predicted = detector.predict(test_df)

    test_windows = detector.get_windows(test_df)
    ground_truth = _window_ground_truth(test_burst_intervals, test_windows)

    accuracy = _best_orientation_accuracy(predicted, ground_truth)
    assert accuracy > 0.6
