"""
Test Script for Anomaly Detection Library
==========================================

Simple tests to verify all detectors work correctly.
"""

import numpy as np
from anomaly_detection import (
    DistanceBasedDetector,
    DBSCANDetector,
    LOFDetector,
    IsolationForestDetector
)


def test_distance_based():
    """Test Distance-Based Detector"""
    print("Testing Distance-Based Detector...")
    X = np.random.randn(100, 2)
    detector = DistanceBasedDetector(n_neighbors=5, contamination=0.1)
    labels = detector.fit_predict(X)
    assert labels.shape[0] == 100
    assert set(labels).issubset({-1, 1})
    assert np.sum(labels == -1) > 0  # Should detect some anomalies
    print("✓ Distance-Based Detector works correctly")


def test_dbscan():
    """Test DBSCAN Detector"""
    print("Testing DBSCAN Detector...")
    X = np.random.randn(100, 2)
    detector = DBSCANDetector(eps=0.5, min_samples=5)
    labels = detector.fit_predict(X)
    assert labels.shape[0] == 100
    info = detector.get_cluster_info()
    assert 'n_clusters' in info
    assert 'n_noise' in info
    print("✓ DBSCAN Detector works correctly")


def test_lof():
    """Test LOF Detector"""
    print("Testing LOF Detector...")
    X = np.random.randn(100, 2)
    
    # Test outlier detection mode
    detector = LOFDetector(n_neighbors=10, contamination=0.1, novelty=False)
    labels = detector.fit_predict(X)
    assert labels.shape[0] == 100
    assert set(labels).issubset({-1, 1})
    assert np.sum(labels == -1) > 0
    
    # Test novelty detection mode
    X_train = np.random.randn(80, 2)
    X_test = np.random.randn(20, 2)
    detector_nov = LOFDetector(n_neighbors=10, contamination=0.1, novelty=True)
    detector_nov.fit(X_train)
    labels_test = detector_nov.predict(X_test)
    assert labels_test.shape[0] == 20
    
    print("✓ LOF Detector works correctly")


def test_isolation_forest():
    """Test Isolation Forest Detector"""
    print("Testing Isolation Forest Detector...")
    X = np.random.randn(100, 2)
    detector = IsolationForestDetector(
        n_estimators=50,
        contamination=0.1,
        random_state=42
    )
    labels = detector.fit_predict(X)
    assert labels.shape[0] == 100
    assert set(labels).issubset({-1, 1})
    assert np.sum(labels == -1) > 0
    
    # Test predict on new data
    X_test = np.random.randn(20, 2)
    labels_test = detector.predict(X_test)
    assert labels_test.shape[0] == 20
    
    # Test decision function
    scores = detector.decision_function(X_test)
    assert scores.shape[0] == 20
    
    print("✓ Isolation Forest Detector works correctly")


def test_integration():
    """Test that all detectors work on the same dataset"""
    print("Testing integration with shared dataset...")
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    detectors = [
        DistanceBasedDetector(n_neighbors=10, contamination=0.1),
        DBSCANDetector(eps=0.5, min_samples=5),
        LOFDetector(n_neighbors=10, contamination=0.1),
        IsolationForestDetector(n_estimators=50, contamination=0.1, random_state=42)
    ]
    
    for detector in detectors:
        labels = detector.fit_predict(X)
        assert labels.shape[0] == 100
        n_anomalies = np.sum(labels == -1)
        print(f"  {detector.__class__.__name__}: detected {n_anomalies} anomalies")
    
    print("✓ Integration test passed")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Running Anomaly Detection Tests")
    print("=" * 70)
    print()
    
    try:
        test_distance_based()
        print()
        test_dbscan()
        print()
        test_lof()
        print()
        test_isolation_forest()
        print()
        test_integration()
        print()
        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        return 0
    except Exception as e:
        print()
        print("=" * 70)
        print(f"Test failed with error: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
