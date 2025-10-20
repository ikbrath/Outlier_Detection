"""
Local Outlier Factor (LOF) Anomaly Detection
=============================================

LOF measures the local deviation of density of a given sample with respect to 
its neighbors. It considers as outliers the samples that have substantially 
lower density than their neighbors.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class LOFDetector:
    """
    Local Outlier Factor (LOF) anomaly detector.
    
    The LOF algorithm is an unsupervised outlier detection method which computes 
    the local density deviation of a given data point with respect to its neighbors.
    Points with substantially lower density than their neighbors are considered outliers.
    
    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use for kneighbors queries
    contamination : float or 'auto', default='auto'
        The amount of contamination of the data set, i.e., the proportion 
        of outliers in the data set. When fitting, this is used to define 
        the threshold on the scores of the samples.
        - if 'auto', the threshold is determined as in the original paper
        - if float, the contamination should be in (0, 0.5]
    metric : str, default='minkowski'
        Metric used for distance computation
    novelty : bool, default=False
        If True, LOF can be used for novelty detection (predict on new data)
        If False, LOF is used for outlier detection (fit_predict only)
    
    Attributes
    ----------
    negative_outlier_factor_ : ndarray of shape (n_samples,)
        The opposite of the Local Outlier Factor of each sample.
        The lower, the more abnormal. Negative scores represent outliers,
        positive scores represent inliers.
    labels_ : ndarray of shape (n_samples,)
        Label for each sample: 1 for inliers, -1 for outliers
    n_neighbors_ : int
        The actual number of neighbors used
    """
    
    def __init__(self, n_neighbors=20, contamination='auto', metric='minkowski', novelty=False):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.novelty = novelty
        self.negative_outlier_factor_ = None
        self.labels_ = None
        self.n_neighbors_ = None
        self._lof = None
        
    def fit(self, X):
        """
        Fit the LOF model using X as training data.
        
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
        
        # Create and fit LOF model
        self._lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            novelty=self.novelty
        )
        
        if self.novelty:
            # For novelty detection, use fit
            self._lof.fit(X)
            # Manually predict on training data to get labels
            self.labels_ = np.ones(X.shape[0])  # All training data are inliers in novelty mode
        else:
            # For outlier detection, use fit_predict
            self.labels_ = self._lof.fit_predict(X)
        
        # Get negative outlier factors
        self.negative_outlier_factor_ = self._lof.negative_outlier_factor_
        self.n_neighbors_ = self._lof.n_neighbors_
        
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
            For each observation, returns 1 for inliers and -1 for outliers
        """
        if self.novelty:
            raise ValueError("fit_predict is not available when novelty=True. "
                           "Use fit and then predict for novelty detection.")
        
        self.fit(X)
        return self.labels_
    
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
        if self._lof is None:
            raise ValueError("Model must be fitted before calling predict()")
        
        if not self.novelty:
            raise ValueError("predict is not available when novelty=False. "
                           "Use fit_predict for outlier detection.")
        
        X = np.asarray(X)
        return self._lof.predict(X)
    
    def decision_function(self, X):
        """
        Shifted opposite of the Local Outlier Factor of X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Shifted opposite of the Local Outlier Factor.
            The lower, the more abnormal.
        """
        if self._lof is None:
            raise ValueError("Model must be fitted first")
        
        if not self.novelty:
            raise ValueError("decision_function is not available when novelty=False. "
                           "Use negative_outlier_factor_ attribute instead.")
        
        X = np.asarray(X)
        return self._lof.decision_function(X)
    
    def score_samples(self, X):
        """
        Opposite of the Local Outlier Factor of X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Opposite of the Local Outlier Factor.
            The lower, the more abnormal.
        """
        if self._lof is None:
            raise ValueError("Model must be fitted first")
        
        if not self.novelty:
            raise ValueError("score_samples is not available when novelty=False. "
                           "Use negative_outlier_factor_ attribute instead.")
        
        X = np.asarray(X)
        return self._lof.score_samples(X)
    
    def get_anomaly_indices(self):
        """
        Get indices of anomaly samples from training data.
        
        Returns
        -------
        indices : ndarray
            Indices of anomaly samples
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
        return np.where(self.labels_ == -1)[0]
