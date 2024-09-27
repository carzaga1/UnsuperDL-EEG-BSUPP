import numpy as np
import pandas as pd
from scipy import linalg

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier


class ClusteringDetector():
    def __init__(self):
        pass
    
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
            cols_to_fix = max_abs_values.loc[max_abs_values>100].index.to_list()

            for col in cols_to_fix:
                white_noise_replacement = np.random.normal(0, 1, size=len(window[col]))
                window[col] = white_noise_replacement

            cleaned_windows.append(window)
        return cleaned_windows
    
    def compute_cov_matrices(self, windows):
        cov_matrices = []
        for window in windows:
            cov_matrix = np.cov(window.to_numpy().T) + np.eye(6)*1e-5
            cov_matrices.append(cov_matrix)
        return cov_matrices
    
    def cov_distance(self, cov_1, cov_2):
        eig_values = linalg.eigh(cov_1, cov_2, eigvals_only=True)
        # distance between two positive semi-definite symmetric matrices
        return np.sqrt(np.sum(np.square(np.log(eig_values)))) 

    def compute_cov_distances(self, cov_matrices_1, cov_matrices_2=None):
        if cov_matrices_2 is None:
            cov_matrices_2 = cov_matrices_1
        n = len(cov_matrices_1)
        m = len(cov_matrices_2)
        metric_d = np.zeros((n,m))
        for i, c_i in enumerate(cov_matrices_1):
            for j, c_j in enumerate(cov_matrices_2):
                metric_d[i, j] = self.cov_distance(c_i, c_j)
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

        p_cluster0_burst = 0.5*(p_cluster0_burst_1 + p_cluster0_burst_2)

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