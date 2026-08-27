import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import linalg

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier


def _cov_distance(cov_1, cov_2):
    eig_values = linalg.eigh(cov_1, cov_2, eigvals_only=True)
    # distance between two positive semi-definite symmetric matrices
    return np.sqrt(np.sum(np.square(np.log(eig_values))))


def _cov_distance_batch(pairs_batch, cov_matrices_1, cov_matrices_2):
    return [_cov_distance(cov_matrices_1[i], cov_matrices_2[j]) for i, j in pairs_batch]


class ClusteringDetector_v4():
    """Same algorithm as ClusteringDetector_v3, with two performance changes to
    compute_cov_distances -- the O(n^2) pairwise covariance-distance computation
    that dominates fit()/predict() runtime on long recordings:

    1. Self-distance matrices (as used by fit()) are symmetric with a zero
       diagonal -- cov_distance(c_i, c_j) == cov_distance(c_j, c_i), since
       swapping the two matrices inverts the generalized eigenvalues and
       squaring their logs cancels the sign, and cov_distance(c_i, c_i) == 0.
       Only the upper triangle is computed and mirrored. Always a pure win,
       independent of n_jobs.

    2. An opt-in n_jobs parameter batches the (i, j) pairs into n_jobs chunks
       and dispatches one large chunk per worker via joblib, rather than one
       task per pair (benchmarked: dispatching ~100k individual tiny eigh
       calls as separate tasks is dominated by per-task overhead and ends up
       SLOWER than sequential).

    n_jobs default is 1 (sequential), because joblib's worker-pool startup has
    a fixed cost of a few seconds per compute_cov_distances() call, which only
    pays for itself once total sequential compute time is well beyond that --
    i.e. for long recordings (thousands of windows / many hours of data), not
    short ones. For a 15-minute recording (~450 windows), n_jobs=1 measured
    faster than any parallel n_jobs in benchmarking. Pass n_jobs=-1 for full
    multi-hour recordings, where the O(n^2) sequential cost would otherwise run
    into hours.
    """

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def get_windows(self, eeg):
        windows = []
        for _, window in eeg.resample('2s'):
            windows.append(window)
        return windows

    def clean_windows(self, windows):
        cleaned_windows = []
        for window in windows:
            window = window.copy()
            max_abs_values = window.abs().max()
            cols_to_fix = max_abs_values.loc[max_abs_values > 100].index.to_list()

            for col in cols_to_fix:
                white_noise_replacement = np.random.normal(0, 1, size=len(window[col]))
                window[col] = white_noise_replacement

            cleaned_windows.append(window)
        return cleaned_windows

    def compute_cov_matrices(self, windows):
        cov_matrices = []
        for window in windows:
            num_cols = window.shape[1]  # get number of columns dynamically
            cov_matrix = np.cov(window.to_numpy().T) + np.eye(num_cols) * 1e-5
            cov_matrices.append(cov_matrix)
        return cov_matrices

    def cov_distance(self, cov_1, cov_2):
        return _cov_distance(cov_1, cov_2)

    def compute_cov_distances(self, cov_matrices_1, cov_matrices_2=None):
        symmetric = cov_matrices_2 is None
        if cov_matrices_2 is None:
            cov_matrices_2 = cov_matrices_1
        n = len(cov_matrices_1)
        m = len(cov_matrices_2)
        metric_d = np.zeros((n, m))

        if symmetric:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, m)]
        else:
            pairs = [(i, j) for i in range(n) for j in range(m)]

        if not pairs:
            return metric_d

        if self.n_jobs == 1:
            distances = [_cov_distance(cov_matrices_1[i], cov_matrices_2[j]) for i, j in pairs]
        else:
            n_workers = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
            n_workers = max(1, min(n_workers, len(pairs)))
            chunk_size = (len(pairs) + n_workers - 1) // n_workers
            chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

            chunk_results = Parallel(n_jobs=self.n_jobs)(
                delayed(_cov_distance_batch)(chunk, cov_matrices_1, cov_matrices_2) for chunk in chunks
            )
            distances = [d for chunk in chunk_results for d in chunk]

        for (i, j), d in zip(pairs, distances):
            metric_d[i, j] = d
            if symmetric:
                metric_d[j, i] = d

        return metric_d

    def get_cluster_labels(self, metric_d, windows):
        # cluster into 2 clusters
        sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
        sc.fit(np.exp(- metric_d ** 2 / (2. * np.median(metric_d.ravel()) ** 2)))
        labels = sc.labels_

        # auto-label clusters
        energy_cluster_0 = []
        energy_cluster_1 = []
        for label, window in zip(labels, windows):
            energy = window.pow(2).sum().sum()
            if label == 0:
                energy_cluster_0.append(energy)
            else:
                energy_cluster_1.append(energy)

        Z_0 = np.sum(energy_cluster_0)
        Z_1 = np.sum(energy_cluster_1)
        p_cluster0_burst_1 = Z_0 / (Z_0 + Z_1)

        n = np.size(labels)
        N_1 = np.sum(labels)
        N_0 = n - N_1
        p_cluster0_burst_2 = 1 - N_0 / (N_0 + N_1)

        p_cluster0_burst = 0.5 * (p_cluster0_burst_1 + p_cluster0_burst_2)

        if p_cluster0_burst > 0.5:
            labels = 1 - labels

        return labels

    def classify_cov_matrices(self, metric_d_learning, labels_learning,
                               cov_matrices_learning, cov_matrices):
        # get KNN classifier
        kn_clf = KNeighborsClassifier(
            n_neighbors=5,
            metric='precomputed'
        )
        kn_clf.fit(metric_d_learning, labels_learning)

        # compute dictances
        X_pred = self.compute_cov_distances(cov_matrices_learning, cov_matrices)

        # run prediction
        return kn_clf.predict(X_pred.T)

    def fit(self, eeg):
        windows = self.get_windows(eeg)
        windows_clean = self.clean_windows(windows)
        cov_matrices = self.compute_cov_matrices(windows_clean)
        metric_d = self.compute_cov_distances(cov_matrices)
        labels = self.get_cluster_labels(metric_d, windows_clean)

        self.cov_matrices_ = cov_matrices
        self.metric_d_ = metric_d
        self.labels_ = labels

    def predict(self, eeg):
        windows = self.get_windows(eeg)
        windows_clean = self.clean_windows(windows)
        cov_matrices = self.compute_cov_matrices(windows_clean)
        return self.classify_cov_matrices(
            self.metric_d_,
            self.labels_,
            self.cov_matrices_,
            cov_matrices)
