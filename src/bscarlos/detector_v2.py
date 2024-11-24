import numpy as np
import pandas as pd
from scipy import linalg

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier


class ClusteringDetector_v2():
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
        windows = []
        for _, window in eeg.resample('2s'):  
            windows.append(window)
        return windows 
    
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
        cov_matrices = []
        for window in windows:
            cov_matrix = np.cov(window.to_numpy().T) + np.eye(6)*1e-5
            cov_matrices.append(cov_matrix)
        return cov_matrices
    
    def cov_distance_fast(self, cov_1, cov_2):
        eig_values = linalg.eigh(cov_1, cov_2, eigvals_only=True)
        # distance between two positive semi-definite symmetric matrices
        return np.sqrt(np.sum(np.square(np.log(eig_values))))

    def compute_cov_distances_fast(self, cov_matrices_1, cov_matrices_2=None):
        if cov_matrices_2 is None:
            cov_matrices_2 = cov_matrices_1

        metric_d = np.zeros((len(cov_matrices_1), len(cov_matrices_2)))
        for i, c_i in enumerate(cov_matrices_1):
            for j, c_j in enumerate(cov_matrices_2):
                metric_d[i, j] = self.cov_distance_fast(c_i, c_j)
        return metric_d
    
    def get_cluster_labels_fast(self, metric_d, windows):
        """
        Clusters the EEG windows and assigns labels based on energy.
        Includes confidence score calculation for windows labeled as bursts.

        Args:
          metric_d: The distance matrix between covariance matrices.
          windows: A list of EEG windows.

        Returns:
          A tuple containing:
            - labels: A list of cluster labels for each window.
            - confidence_scores: A list of confidence scores for each window labeled 1.
        """
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

        # adjust labels if necessary to ensure 1 is the high-energy cluster
        if p_cluster0_burst_1 > 0.5:
            labels = 1 - labels  # Swap labels

        # calculate confidence scores for cluster 1 (burst cluster)
        confidence_scores = []
        for label, window in zip(labels, windows):
            if label == 1:
                energy = window.pow(2).sum().sum()
                confidence = energy / Z_1
                confidence_scores.append(confidence)
            else:
                confidence_scores.append(None)

        return labels, confidence_scores
    
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
        windows = self.get_windows_fast(eeg)
        windows_clean = self.clean_windows_200(windows)
        cov_matrices = self.compute_cov_matrices_fast(windows_clean)
        metric_d = self.compute_cov_distances_fast(cov_matrices)
        labels = self.get_cluster_labels_fast(metric_d, windows_clean)
        
        self.cov_matrices_ = cov_matrices
        self.metric_d_ = metric_d
        self.labels_ = labels
        
    def predict_detector(self, eeg):
        windows = self.get_windows_fast(eeg)
        windows_clean = self.clean_windows_200(windows)
        cov_matrices = self.compute_cov_matrices_fast(windows_clean)
        return self.classify_cov_matrices_fast(
            self.metric_d_, 
            self.labels_, 
            self.cov_matrices_, 
            cov_matrices)