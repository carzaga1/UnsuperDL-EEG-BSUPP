import numpy as np
import pandas as pd
from scipy import linalg

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier


class ClusteringDetector_1():
    """
    A class used to detect clusters in EEG data using a combination of windowing, 
    cleaning, covariance matrix computation, and clustering algorithms.

    Attributes:
        None

    Methods:
        get_windows_fast(eeg): Splits EEG data into 2-second windows.
        clean_windows_200(windows): Replaces values in each window that exceed a threshold with white noise.
        compute_cov_matrices_fast(windows): Computes the covariance matrix for each window.
        cov_distance_fast(cov_1, cov_2): Calculates the distance between two covariance matrices.
        compute_cov_distances_fast(cov_matrices_1, cov_matrices_2=None): Computes the pairwise distances between covariance matrices.
        get_cluster_labels_fast(metric_d, windows): Clusters the windows into two groups using spectral clustering and auto-labels the clusters based on energy.
        classify_cov_matrices_fast(metric_d_learning, labels_learning, cov_matrices_learning, cov_matrices): Trains a KNN classifier on labeled covariance matrices and predicts labels for new covariance matrices.
        fit_detector(eeg): Fits the detector to EEG data by computing windows, cleaning, computing covariance matrices, and clustering.
        predict_detector(eeg): Predicts labels for new EEG data using the trained detector.
    """
    def __init__(self):
        pass
    
    def get_windows_fast(self, eeg):
        return [window for _, window in eeg.resample('2.5s')]
    
    def clean_windows_200(self, windows):
        cleaned_windows = []
        for window in windows:
            window = window.copy()
            max_abs_values = window.abs().max()
            artifact_channels = max_abs_values[max_abs_values > 200].index.to_list()

            for channel in artifact_channels:
                white_noise_replacement = np.random.normal(0, 1, size=len(window[channel]))
                window[channel] = white_noise_replacement

            cleaned_windows.append(window)
        return cleaned_windows
    
    def compute_cov_matrices_fast(self, windows):
        return [np.cov(window.to_numpy().T) + np.eye(6)*1e-5 for window in windows]
    
    def cov_distance_fast(self, cov_1, cov_2):
        # distance between two positive semi-definite symmetric matrices
        return np.sqrt(np.trace(linalg.logm(cov_1 @ linalg.inv(cov_2)) ** 2))

    def compute_cov_distances_fast(self, cov_matrices_1, cov_matrices_2=None):
        if cov_matrices_2 is None:
            cov_matrices_2 = cov_matrices_1

        metric_d = np.zeros((len(cov_matrices_1), len(cov_matrices_2)))
        for i, c_i in enumerate(cov_matrices_1):
            for j, c_j in enumerate(cov_matrices_2):
                metric_d[i, j] = self.cov_distance(c_i, c_j)
        return metric_d
    
    def get_cluster_labels_fast(self, metric_d, windows):
        sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(np.exp(- metric_d ** 2 / (2. * np.median(metric_d.ravel()) ** 2)))
        energy = np.array([window.pow(2).sum().sum() for window in windows])
        p_cluster0_burst = np.sum(energy[labels == 0]) / energy.sum()
        return labels if p_cluster0_burst > 0.5 else 1 - labels
    
    def classify_cov_matrices_fast(self, metric_d_learning, labels_learning, 
                              cov_matrices_learning, cov_matrices):
        # get KNN classifier
        kn_clf = KNeighborsClassifier(
            n_neighbors=5, 
            metric='precomputed', 
            algorithm='ball_tree'
        )
        kn_clf.fit(metric_d_learning, labels_learning)
        
        # compute dictances
        X_pred = self.compute_cov_distances_fast(cov_matrices_learning, cov_matrices)
        
        # run prediction
        return kn_clf.predict(X_pred.T)
    
    
    def fit_detector(self, eeg):
        windows = self.get_windows(eeg)
        windows_clean = self.clean_windows(windows)
        cov_matrices = self.compute_cov_matrices(windows_clean)
        metric_d = self.compute_cov_distances(cov_matrices)
        labels = self.get_cluster_labels(metric_d, windows_clean)
        
        self.cov_matrices_ = cov_matrices
        self.metric_d_ = metric_d
        self.labels_ = labels
        
    def predict_detector(self, eeg):
        windows = self.get_windows(eeg)
        windows_clean = self.clean_windows(windows)
        cov_matrices = self.compute_cov_matrices(windows_clean)
        return self.classify_cov_matrices(
            self.metric_d_, 
            self.labels_, 
            self.cov_matrices_, 
            cov_matrices)