"""
Isolation Forest Anomaly Detection
===================================

Isolation Forest detects anomalies by randomly selecting a feature and then 
randomly selecting a split value between the maximum and minimum values of 
the selected feature. Anomalies are isolated closer to the root of the tree.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector.
    
    The IsolationForest isolates observations by randomly selecting a feature 
    and then randomly selecting a split value between the maximum and minimum 
    values of the selected feature. Since anomalies are few and different, 
    they are isolated closer to the root of the tree.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators (trees) in the ensemble
    max_samples : int or float, default='auto'
        The number of samples to draw to train each base estimator
        - if int, draw max_samples samples
        - if float, draw max_samples * n_samples samples
        - if 'auto', max_samples=min(256, n_samples)
    contamination : float or 'auto', default='auto'
        The amount of contamination of the data set, i.e., the proportion 
        of outliers in the data set. Used to define the threshold.
        - if 'auto', the threshold is determined as in the original paper
        - if float, the contamination should be in (0, 0.5]
    max_features : int or float, default=1.0
        The number of features to draw to train each base estimator
        - if int, draw max_features features
        - if float, draw max_features * n_features features
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of features and samples
    
    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators
    labels_ : ndarray of shape (n_samples,)
        Label for each sample: 1 for inliers, -1 for outliers
    scores_ : ndarray of shape (n_samples,)
        Anomaly score for each sample
    threshold_ : float
        The threshold used to separate inliers from outliers
    """
    
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto',
                 max_features=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = None
        self.labels_ = None
        self.scores_ = None
        self.threshold_ = None
        self._iforest = None
        
    def fit(self, X):
        """
        Fit the Isolation Forest model using X as training data.
        
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
        
        # Create and fit Isolation Forest model
        self._iforest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state
        )
        self._iforest.fit(X)
        
        # Get predictions and scores for training data
        self.labels_ = self._iforest.predict(X)
        self.scores_ = self._iforest.score_samples(X)
        self.threshold_ = self._iforest.offset_
        self.estimators_ = self._iforest.estimators_
        
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
        if self._iforest is None:
            raise ValueError("Model must be fitted before calling predict()")
        
        X = np.asarray(X)
        return self._iforest.predict(X)
    
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
    
    def decision_function(self, X):
        """
        Average anomaly score of X.
        
        The anomaly score of an input sample is computed as the mean anomaly 
        score of the trees in the forest. The lower, the more abnormal.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly score. The lower, the more abnormal.
        """
        if self._iforest is None:
            raise ValueError("Model must be fitted first")
        
        X = np.asarray(X)
        return self._iforest.decision_function(X)
    
    def score_samples(self, X):
        """
        Opposite of the anomaly score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Opposite of the anomaly score. The lower, the more abnormal.
        """
        if self._iforest is None:
            raise ValueError("Model must be fitted first")
        
        X = np.asarray(X)
        return self._iforest.score_samples(X)
    
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
    
    def get_params(self):
        """
        Get parameters of the model.
        
        Returns
        -------
        params : dict
            Dictionary of model parameters
        """
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'max_features': self.max_features,
            'random_state': self.random_state
        }
