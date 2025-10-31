"""
Anomaly Detection Examples
===========================

This script demonstrates the usage of all four anomaly detection algorithms:
1. Distance-based anomaly detection
2. DBSCAN clustering-based anomaly detection
3. LOF (Local Outlier Factor)
4. Isolation Forest

The examples use synthetic data and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from anomaly_detection import (
    DistanceBasedDetector,
    DBSCANDetector,
    LOFDetector,
    IsolationForestDetector
)


def generate_sample_data(n_samples=300, n_outliers=30, random_state=42):
    """
    Generate sample data with outliers for demonstration.
    
    Parameters
    ----------
    n_samples : int
        Number of normal samples
    n_outliers : int
        Number of outlier samples
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    X : ndarray
        Combined normal and outlier data
    y_true : ndarray
        True labels (1 for normal, -1 for outlier)
    """
    np.random.seed(random_state)
    
    # Generate normal samples
    X_normal, _ = make_blobs(
        n_samples=n_samples,
        centers=2,
        cluster_std=0.5,
        random_state=random_state
    )
    
    # Generate outliers (uniformly distributed)
    X_outliers = np.random.uniform(
        low=X_normal.min(axis=0) - 2,
        high=X_normal.max(axis=0) + 2,
        size=(n_outliers, 2)
    )
    
    # Combine data
    X = np.vstack([X_normal, X_outliers])
    y_true = np.hstack([np.ones(n_samples), -np.ones(n_outliers)])
    
    return X, y_true


def visualize_results(X, y_pred, y_true, title):
    """
    Visualize anomaly detection results.
    
    Parameters
    ----------
    X : ndarray
        Data points
    y_pred : ndarray
        Predicted labels
    y_true : ndarray
        True labels
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 5))
    
    # Plot predictions
    plt.subplot(1, 2, 1)
    mask_inliers = y_pred == 1
    mask_outliers = y_pred == -1
    
    plt.scatter(X[mask_inliers, 0], X[mask_inliers, 1], 
               c='blue', label='Inliers', alpha=0.6, edgecolors='k')
    plt.scatter(X[mask_outliers, 0], X[mask_outliers, 1], 
               c='red', label='Outliers', alpha=0.8, marker='x', s=100)
    plt.title(f'{title} - Predictions')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot ground truth
    plt.subplot(1, 2, 2)
    mask_true_inliers = y_true == 1
    mask_true_outliers = y_true == -1
    
    plt.scatter(X[mask_true_inliers, 0], X[mask_true_inliers, 1], 
               c='blue', label='True Inliers', alpha=0.6, edgecolors='k')
    plt.scatter(X[mask_true_outliers, 0], X[mask_true_outliers, 1], 
               c='red', label='True Outliers', alpha=0.8, marker='x', s=100)
    plt.title('Ground Truth')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {title.lower().replace(' ', '_')}.png")


def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics.
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    
    Returns
    -------
    metrics : dict
        Dictionary of metrics
    """
    tp = np.sum((y_true == -1) & (y_pred == -1))  # True positives (correctly identified outliers)
    fp = np.sum((y_true == 1) & (y_pred == -1))   # False positives (normal labeled as outlier)
    tn = np.sum((y_true == 1) & (y_pred == 1))    # True negatives (correctly identified normal)
    fn = np.sum((y_true == -1) & (y_pred == 1))   # False negatives (outlier labeled as normal)
    
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    """Main function to demonstrate all anomaly detection methods."""
    
    print("=" * 80)
    print("Anomaly Detection Demonstration")
    print("=" * 80)
    print()
    
    # Generate sample data
    print("Generating sample data...")
    X, y_true = generate_sample_data(n_samples=300, n_outliers=30)
    print(f"Generated {len(X)} samples ({np.sum(y_true == 1)} normal, {np.sum(y_true == -1)} outliers)")
    print()
    
    # 1. Distance-Based Anomaly Detection
    print("-" * 80)
    print("1. Distance-Based Anomaly Detection")
    print("-" * 80)
    detector1 = DistanceBasedDetector(n_neighbors=10, contamination=0.1)
    y_pred1 = detector1.fit_predict(X)
    metrics1 = calculate_metrics(y_true, y_pred1)
    print(f"Detected {np.sum(y_pred1 == -1)} anomalies")
    print(f"Accuracy: {metrics1['accuracy']:.3f}")
    print(f"Precision: {metrics1['precision']:.3f}")
    print(f"Recall: {metrics1['recall']:.3f}")
    print(f"F1-Score: {metrics1['f1_score']:.3f}")
    visualize_results(X, y_pred1, y_true, "Distance-Based Detection")
    print()
    
    # 2. DBSCAN-Based Anomaly Detection
    print("-" * 80)
    print("2. DBSCAN-Based Anomaly Detection")
    print("-" * 80)
    detector2 = DBSCANDetector(eps=0.5, min_samples=5)
    y_pred2 = detector2.fit_predict(X)
    # Convert DBSCAN labels: noise points (-1) are outliers, others are inliers
    y_pred2_binary = np.where(y_pred2 == -1, -1, 1)
    metrics2 = calculate_metrics(y_true, y_pred2_binary)
    cluster_info = detector2.get_cluster_info()
    print(f"Found {cluster_info['n_clusters']} clusters")
    print(f"Detected {cluster_info['n_noise']} noise points (anomalies)")
    print(f"Accuracy: {metrics2['accuracy']:.3f}")
    print(f"Precision: {metrics2['precision']:.3f}")
    print(f"Recall: {metrics2['recall']:.3f}")
    print(f"F1-Score: {metrics2['f1_score']:.3f}")
    visualize_results(X, y_pred2_binary, y_true, "DBSCAN Detection")
    print()
    
    # 3. LOF (Local Outlier Factor)
    print("-" * 80)
    print("3. LOF (Local Outlier Factor)")
    print("-" * 80)
    detector3 = LOFDetector(n_neighbors=20, contamination=0.1)
    y_pred3 = detector3.fit_predict(X)
    metrics3 = calculate_metrics(y_true, y_pred3)
    print(f"Detected {np.sum(y_pred3 == -1)} anomalies")
    print(f"Accuracy: {metrics3['accuracy']:.3f}")
    print(f"Precision: {metrics3['precision']:.3f}")
    print(f"Recall: {metrics3['recall']:.3f}")
    print(f"F1-Score: {metrics3['f1_score']:.3f}")
    visualize_results(X, y_pred3, y_true, "LOF Detection")
    print()
    
    # 4. Isolation Forest
    print("-" * 80)
    print("4. Isolation Forest")
    print("-" * 80)
    detector4 = IsolationForestDetector(n_estimators=100, contamination=0.1, random_state=42)
    y_pred4 = detector4.fit_predict(X)
    metrics4 = calculate_metrics(y_true, y_pred4)
    print(f"Detected {np.sum(y_pred4 == -1)} anomalies")
    print(f"Accuracy: {metrics4['accuracy']:.3f}")
    print(f"Precision: {metrics4['precision']:.3f}")
    print(f"Recall: {metrics4['recall']:.3f}")
    print(f"F1-Score: {metrics4['f1_score']:.3f}")
    visualize_results(X, y_pred4, y_true, "Isolation Forest")
    print()
    
    # Summary comparison
    print("=" * 80)
    print("Summary Comparison")
    print("=" * 80)
    print(f"{'Method':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    print(f"{'Distance-Based':<30} {metrics1['accuracy']:<10.3f} {metrics1['precision']:<10.3f} {metrics1['recall']:<10.3f} {metrics1['f1_score']:<10.3f}")
    print(f"{'DBSCAN':<30} {metrics2['accuracy']:<10.3f} {metrics2['precision']:<10.3f} {metrics2['recall']:<10.3f} {metrics2['f1_score']:<10.3f}")
    print(f"{'LOF':<30} {metrics3['accuracy']:<10.3f} {metrics3['precision']:<10.3f} {metrics3['recall']:<10.3f} {metrics3['f1_score']:<10.3f}")
    print(f"{'Isolation Forest':<30} {metrics4['accuracy']:<10.3f} {metrics4['precision']:<10.3f} {metrics4['recall']:<10.3f} {metrics4['f1_score']:<10.3f}")
    print("=" * 80)
    print()
    print("All visualizations have been saved as PNG files.")


if __name__ == "__main__":
    main()
