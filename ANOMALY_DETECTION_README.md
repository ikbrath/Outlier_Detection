# Anomaly Detection Library

A comprehensive Python library for detecting anomalies in data using four different algorithms:

1. **Distance-Based Anomaly Detection**
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
3. **LOF (Local Outlier Factor)**
4. **Isolation Forest**

## Features

- Easy-to-use API consistent with scikit-learn
- Multiple algorithms for different use cases
- Comprehensive documentation and examples
- Visualization support
- Performance metrics calculation

## Installation

### Requirements

- Python 3.7 or higher
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0 (for visualization)
- pandas >= 1.3.0 (optional, for data manipulation)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from anomaly_detection import IsolationForestDetector
import numpy as np

# Generate sample data
X = np.random.randn(100, 2)

# Create and fit detector
detector = IsolationForestDetector(contamination=0.1)
labels = detector.fit_predict(X)

# -1 for outliers, 1 for inliers
outliers = X[labels == -1]
inliers = X[labels == 1]

print(f"Found {len(outliers)} anomalies")
```

## Algorithms

### 1. Distance-Based Anomaly Detection

Detects anomalies by calculating the average distance to k-nearest neighbors. Points with larger distances are considered anomalies.

**Use Case:** Simple and interpretable method for detecting outliers in low to medium-dimensional data.

**Example:**
```python
from anomaly_detection import DistanceBasedDetector

detector = DistanceBasedDetector(n_neighbors=10, contamination=0.1)
labels = detector.fit_predict(X)
```

**Parameters:**
- `n_neighbors`: Number of neighbors to use (default: 5)
- `contamination`: Expected proportion of outliers (default: 0.1)
- `metric`: Distance metric, e.g., 'euclidean', 'manhattan' (default: 'euclidean')

### 2. DBSCAN-Based Anomaly Detection

Uses density-based clustering to identify anomalies as noise points that don't belong to any cluster.

**Use Case:** Effective for data with clusters of varying densities; automatically determines the number of clusters.

**Example:**
```python
from anomaly_detection import DBSCANDetector

detector = DBSCANDetector(eps=0.5, min_samples=5)
labels = detector.fit_predict(X)

# Get cluster information
info = detector.get_cluster_info()
print(f"Found {info['n_clusters']} clusters")
print(f"Detected {info['n_noise']} noise points")
```

**Parameters:**
- `eps`: Maximum distance between neighbors (default: 0.5)
- `min_samples`: Minimum samples in a neighborhood (default: 5)
- `metric`: Distance metric (default: 'euclidean')

### 3. LOF (Local Outlier Factor)

Measures the local density deviation of each point relative to its neighbors. Points with substantially lower density are anomalies.

**Use Case:** Effective for detecting local outliers in data with varying densities.

**Example:**
```python
from anomaly_detection import LOFDetector

# For outlier detection (on training data)
detector = LOFDetector(n_neighbors=20, contamination=0.1, novelty=False)
labels = detector.fit_predict(X)

# For novelty detection (on new data)
detector = LOFDetector(n_neighbors=20, contamination=0.1, novelty=True)
detector.fit(X_train)
labels = detector.predict(X_test)
```

**Parameters:**
- `n_neighbors`: Number of neighbors (default: 20)
- `contamination`: Expected proportion of outliers (default: 'auto')
- `metric`: Distance metric (default: 'minkowski')
- `novelty`: Enable prediction on new data (default: False)

### 4. Isolation Forest

Isolates anomalies by randomly splitting features. Anomalies require fewer splits to isolate.

**Use Case:** Highly effective for high-dimensional data; fast and scalable.

**Example:**
```python
from anomaly_detection import IsolationForestDetector

detector = IsolationForestDetector(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
labels = detector.fit_predict(X)

# Get anomaly scores
scores = detector.decision_function(X)
```

**Parameters:**
- `n_estimators`: Number of trees (default: 100)
- `max_samples`: Samples per tree (default: 'auto')
- `contamination`: Expected proportion of outliers (default: 'auto')
- `max_features`: Features per split (default: 1.0)
- `random_state`: Random seed (default: None)

## Complete Example

Run the comprehensive example script:

```bash
python example_usage.py
```

This will:
1. Generate synthetic data with outliers
2. Apply all four anomaly detection algorithms
3. Calculate performance metrics
4. Generate visualizations comparing results
5. Print a summary comparison table

## API Reference

All detectors follow a consistent API:

### Methods

- `fit(X)`: Fit the model on training data
- `predict(X)`: Predict labels for new data (returns 1 for inliers, -1 for outliers)
- `fit_predict(X)`: Fit and predict in one step
- `decision_function(X)`: Get anomaly scores (where available)
- `get_anomaly_indices()`: Get indices of detected anomalies

### Return Values

- Labels: `1` for normal points (inliers), `-1` for anomalies (outliers)
- Scores: Lower scores indicate higher anomaly likelihood

## Choosing the Right Algorithm

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Distance-Based** | Simple datasets, interpretability | Easy to understand, fast | Sensitive to parameter choice |
| **DBSCAN** | Clustered data, varying densities | No assumptions on cluster shape | Requires parameter tuning |
| **LOF** | Local outliers, varying densities | Detects local anomalies well | Computationally expensive |
| **Isolation Forest** | High-dimensional data, large datasets | Fast, scalable, effective | Less interpretable |

## Performance Tips

1. **Normalize your data**: Most algorithms work better with normalized features
2. **Tune contamination**: Set based on expected outlier proportion
3. **Cross-validate**: Test different parameters to find optimal settings
4. **Combine methods**: Use ensemble of detectors for robust results

## Examples of Use Cases

### Credit Card Fraud Detection
```python
detector = IsolationForestDetector(contamination=0.01)  # Expect 1% fraud
labels = detector.fit_predict(transactions)
```

### Network Intrusion Detection
```python
detector = LOFDetector(n_neighbors=30, contamination=0.05)
labels = detector.fit_predict(network_traffic)
```

### Manufacturing Quality Control
```python
detector = DistanceBasedDetector(n_neighbors=10, contamination=0.02)
labels = detector.fit_predict(sensor_readings)
```

### Log Analysis
```python
detector = DBSCANDetector(eps=0.3, min_samples=10)
labels = detector.fit_predict(log_features)
```

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is created by Ikbar Athallah Taufik as part of learning and demonstrating data science skills.

## Contact

- **Email**: ikbaratallah@gmail.com
- **LinkedIn**: [Ikbar Athallah Taufik](https://linkedin.com/in/ikbar-athallah-taufik)
- **Kaggle**: [Ikbar Athallah](https://kaggle.com/ikbar-athallah)

## References

1. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers.
2. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters.
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
4. Knorr, E. M., & Ng, R. T. (1998). Algorithms for mining distance-based outliers.

---

*Built with ❤️ by Ikbar - A passionate data scientist from Indonesia*
