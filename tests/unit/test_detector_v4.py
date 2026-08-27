import numpy as np
import pandas as pd

from bscarlos.detector_v4 import ClusteringDetector_v4
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
    predicted = np.asarray(predicted)
    return max(
        np.mean(predicted == ground_truth),
        np.mean((1 - predicted) == ground_truth),
    )


def test_clustering_detector_v4_fit_separates_burst_from_suppression():
    df, burst_intervals = _make_eeg_dataframe(n_seconds=60, sample_rate=256, seed=1)

    detector = ClusteringDetector_v4()
    detector.fit(df)

    windows = detector.get_windows(df)
    ground_truth = _window_ground_truth(burst_intervals, windows)

    accuracy = _best_orientation_accuracy(detector.labels_, ground_truth)
    assert accuracy > 0.6


def test_clustering_detector_v4_predict_on_held_out_data():
    train_df, _ = _make_eeg_dataframe(n_seconds=60, sample_rate=256, seed=1)
    test_df, test_burst_intervals = _make_eeg_dataframe(n_seconds=30, sample_rate=256, seed=2)

    detector = ClusteringDetector_v4()
    detector.fit(train_df)
    predicted = detector.predict(test_df)

    test_windows = detector.get_windows(test_df)
    ground_truth = _window_ground_truth(test_burst_intervals, test_windows)

    accuracy = _best_orientation_accuracy(predicted, ground_truth)
    assert accuracy > 0.6


def test_compute_cov_distances_parallel_matches_sequential():
    rng = np.random.default_rng(0)
    cov_matrices = []
    for _ in range(12):
        a = rng.normal(size=(6, 6))
        cov = a @ a.T + np.eye(6) * 1e-3  # positive semi-definite
        cov_matrices.append(cov)

    sequential = ClusteringDetector_v4(n_jobs=1).compute_cov_distances(cov_matrices)
    parallel = ClusteringDetector_v4(n_jobs=2).compute_cov_distances(cov_matrices)

    np.testing.assert_allclose(sequential, parallel, rtol=1e-8)
    # Self-distance matrix must be symmetric with a zero diagonal.
    np.testing.assert_allclose(np.diag(sequential), 0.0, atol=1e-8)
    np.testing.assert_allclose(sequential, sequential.T, rtol=1e-8)


def test_compute_cov_distances_asymmetric_case_still_correct():
    # predict()'s use of compute_cov_distances (cov_matrices_2 given explicitly)
    # is the non-square, non-symmetric case, bypassing the mirroring optimization.
    rng = np.random.default_rng(1)

    def make_cov():
        a = rng.normal(size=(6, 6))
        return a @ a.T + np.eye(6) * 1e-3

    train_cov = [make_cov() for _ in range(8)]
    test_cov = [make_cov() for _ in range(5)]

    detector = ClusteringDetector_v4(n_jobs=1)
    result = detector.compute_cov_distances(train_cov, test_cov)

    assert result.shape == (8, 5)
    expected = np.array([[detector.cov_distance(c1, c2) for c2 in test_cov] for c1 in train_cov])
    np.testing.assert_allclose(result, expected, rtol=1e-8)
