"""
Distance-Based Anomaly Detection
=================================

Detects anomalies by measuring the distance of each point to its k-nearest neighbors.
Points with larger average distances to their neighbors are considered anomalies.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class DistanceBasedDetector:
    """
    Distance-based anomaly detector.
    
    This method identifies anomalies by calculating the average distance 
    to k-nearest neighbors. Points with distances above a threshold are 
    marked as anomalies.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for distance calculation
    contamination : float, default=0.1
        Expected proportion of outliers in the dataset (0 < contamination < 0.5)
    metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'minkowski', etc.)
    
    Attributes
    ----------
    threshold_ : float
        Distance threshold above which points are considered anomalies
    distances_ : ndarray of shape (n_samples,)
        Average distance to k-nearest neighbors for each sample
    labels_ : ndarray of shape (n_samples,)
        Label for each sample: 1 for inliers, -1 for outliers
    """
    
    def __init__(self, n_neighbors=5, contamination=0.1, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.threshold_ = None
        self.distances_ = None
        self.labels_ = None
        self._nbrs = None
        
    def fit(self, X):
        """
        Fit the model using X as training data.
        
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
        
        # Fit k-nearest neighbors
        self._nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric=self.metric)
        self._nbrs.fit(X)
        
        # Calculate distances to k-nearest neighbors (excluding the point itself)
        distances, _ = self._nbrs.kneighbors(X)
        # Average distance to k-nearest neighbors (excluding self at index 0)
        self.distances_ = np.mean(distances[:, 1:], axis=1)
        
        # Set threshold based on contamination rate
        self.threshold_ = np.percentile(self.distances_, 100 * (1 - self.contamination))
        
        # Label points
        self.labels_ = np.where(self.distances_ > self.threshold_, -1, 1)
        
        return self
    
    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            For each observation, returns 1 for inliers and -1 for outliers
        """
        if self.threshold_ is None or self._nbrs is None:
            raise ValueError("Model must be fitted before calling predict()")
        
        X = np.asarray(X)
        
        # Calculate distances to k-nearest neighbors in training data
        distances, _ = self._nbrs.kneighbors(X, n_neighbors=self.n_neighbors)
        # Average distance to k-nearest neighbors
        avg_distances = np.mean(distances, axis=1)
        
        return np.where(avg_distances > self.threshold_, -1, 1)
    
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
            For each observation, returns 1 for inliers and -1 for outliers
        """
        self.fit(X)
        return self.labels_
    
    def decision_function(self, X=None):
        """
        Anomaly score for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Data to score. If None, returns scores for training data.
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly score of each sample. Higher scores indicate more anomalous points.
        """
        if X is None:
            if self.distances_ is None:
                raise ValueError("Model must be fitted first")
            return self.distances_
        
        if self._nbrs is None:
            raise ValueError("Model must be fitted before calling decision_function()")
        
        # For new data, calculate distances to k-nearest neighbors in training data
        X = np.asarray(X)
        distances, _ = self._nbrs.kneighbors(X, n_neighbors=self.n_neighbors)
        # Average distance to k-nearest neighbors
        return np.mean(distances, axis=1)
