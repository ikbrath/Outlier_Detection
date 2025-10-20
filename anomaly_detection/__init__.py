"""
Anomaly Detection Library
=========================

A comprehensive library for detecting anomalies in data using various algorithms:
- Distance-based anomaly detection
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- LOF (Local Outlier Factor)
- Isolation Forest

Author: Ikbar Athallah Taufik
"""

from .distance_based import DistanceBasedDetector
from .dbscan_detector import DBSCANDetector
from .lof_detector import LOFDetector
from .isolation_forest_detector import IsolationForestDetector

__version__ = '1.0.0'
__all__ = [
    'DistanceBasedDetector',
    'DBSCANDetector',
    'LOFDetector',
    'IsolationForestDetector',
]
