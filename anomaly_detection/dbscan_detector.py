"""
DBSCAN-Based Anomaly Detection
===============================

Uses DBSCAN clustering to identify anomalies as noise points that don't belong 
to any cluster. DBSCAN finds core samples of high density and expands clusters 
from them. Points not belonging to any cluster are marked as anomalies.
"""

import numpy as np
from sklearn.cluster import DBSCAN


class DBSCANDetector:
    """
    DBSCAN-based anomaly detector.
    
    Uses DBSCAN clustering algorithm to detect anomalies. Points that are 
    classified as noise (not belonging to any cluster) are considered anomalies.
    
    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered 
        as in the same neighborhood
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered 
        as a core point
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point. -1 indicates anomalies (noise points)
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples
    n_clusters_ : int
        Number of clusters found
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None
        self._dbscan = None
        
    def fit(self, X):
        """
        Fit the DBSCAN model using X as training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X)
        
        # Fit DBSCAN
        self._dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )
        self._dbscan.fit(X)
        
        # Get cluster labels (-1 for noise/anomalies)
        self.labels_ = self._dbscan.labels_
        self.core_sample_indices_ = self._dbscan.core_sample_indices_
        
        # Count number of clusters (excluding noise points labeled as -1)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        
        return self
    
    def fit_predict(self, X):
        """
        Fit the model and predict labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point. -1 indicates anomalies
        """
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        """
        Predict if samples are anomalies.
        
        Note: DBSCAN doesn't naturally support prediction on new data.
        This method returns -1 (anomaly) for all new samples as a 
        conservative approach.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            -1 for all samples (conservative approach)
        """
        X = np.asarray(X)
        # DBSCAN doesn't support predict on new data naturally
        # Return -1 for all as a conservative approach
        return np.full(X.shape[0], -1)
    
    def get_anomalies(self):
        """
        Get boolean mask for anomaly samples.
        
        Returns
        -------
        mask : ndarray of shape (n_samples,)
            Boolean array where True indicates an anomaly
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
        return self.labels_ == -1
    
    def get_anomaly_indices(self):
        """
        Get indices of anomaly samples.
        
        Returns
        -------
        indices : ndarray
            Indices of anomaly samples
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
        return np.where(self.labels_ == -1)[0]
    
    def get_cluster_info(self):
        """
        Get information about clusters.
        
        Returns
        -------
        info : dict
            Dictionary containing cluster information:
            - n_clusters: number of clusters
            - n_noise: number of noise points (anomalies)
            - cluster_sizes: size of each cluster
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
        
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        n_noise = cluster_sizes.pop(-1, 0)
        
        return {
            'n_clusters': self.n_clusters_,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes
        }
