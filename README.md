# Outlier Detection Library

A comprehensive Python library for detecting anomalies and outliers in data using multiple state-of-the-art algorithms.

## 🎯 Overview

This library provides four powerful anomaly detection algorithms in a consistent, easy-to-use interface. Whether you're detecting fraud, network intrusions, manufacturing defects, or log anomalies, this library has you covered.

**Implemented Algorithms:**
- 🎯 Distance-Based Anomaly Detection
- 📊 DBSCAN (Density-Based Spatial Clustering)
- 📍 LOF (Local Outlier Factor)
- 🌲 Isolation Forest

## ✨ Features

- **Consistent API**: All detectors follow a scikit-learn-like interface
- **Multiple Algorithms**: Choose the best method for your use case
- **Production Ready**: Comprehensive error handling and validation
- **Well Documented**: Detailed documentation with examples
- **Fully Tested**: Complete test coverage
- **Visualization Support**: Built-in example visualizations

## 📋 Requirements

- Python 3.7 or higher
- NumPy ≥ 1.21.0
- scikit-learn ≥ 1.0.0
- matplotlib ≥ 3.4.0
- pandas ≥ 1.3.0

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/ikbrath/Outlier_Detection.git
cd Outlier_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💡 Quick Start

```python
from anomaly_detection import IsolationForestDetector
import numpy as np

# Generate or load your data
X = np.random.randn(100, 2)

# Create and fit detector
detector = IsolationForestDetector(contamination=0.1)
labels = detector.fit_predict(X)

# Results: 1 for inliers, -1 for outliers
outliers = X[labels == -1]
inliers = X[labels == 1]

print(f"Found {len(outliers)} anomalies out of {len(X)} samples")
```

## 📚 Available Algorithms

### 1. Distance-Based Anomaly Detection

Detects anomalies by calculating the average distance to k-nearest neighbors.

```python
from anomaly_detection import DistanceBasedDetector

detector = DistanceBasedDetector(n_neighbors=10, contamination=0.1)
labels = detector.fit_predict(X)
```

**Best for:** Simple datasets, interpretable results
**Parameters:**
- `n_neighbors`: Number of neighbors (default: 5)
- `contamination`: Expected outlier proportion (default: 0.1)
- `metric`: Distance metric (default: 'euclidean')

### 2. DBSCAN-Based Anomaly Detection

Uses density-based clustering to identify noise points as anomalies.

```python
from anomaly_detection import DBSCANDetector

detector = DBSCANDetector(eps=0.5, min_samples=5)
labels = detector.fit_predict(X)
info = detector.get_cluster_info()
print(f"Found {info['n_clusters']} clusters, {info['n_noise']} noise points")
```

**Best for:** Clustered data with varying densities
**Parameters:**
- `eps`: Maximum distance between neighbors (default: 0.5)
- `min_samples`: Minimum samples in neighborhood (default: 5)
- `metric`: Distance metric (default: 'euclidean')

### 3. LOF (Local Outlier Factor)

Measures local density deviation relative to neighbors.

```python
from anomaly_detection import LOFDetector

# For outlier detection
detector = LOFDetector(n_neighbors=20, contamination=0.1, novelty=False)
labels = detector.fit_predict(X)

# For novelty detection (new data)
detector = LOFDetector(n_neighbors=20, novelty=True)
detector.fit(X_train)
labels = detector.predict(X_test)
```

**Best for:** Local outliers in varying density regions
**Parameters:**
- `n_neighbors`: Number of neighbors (default: 20)
- `contamination`: Expected outlier proportion (default: 'auto')
- `novelty`: Enable prediction on new data (default: False)

### 4. Isolation Forest

Isolates anomalies using random decision trees.

```python
from anomaly_detection import IsolationForestDetector

detector = IsolationForestDetector(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
labels = detector.fit_predict(X)
scores = detector.decision_function(X)
```

**Best for:** High-dimensional data, large datasets
**Parameters:**
- `n_estimators`: Number of trees (default: 100)
- `contamination`: Expected outlier proportion (default: 'auto')
- `max_features`: Features per split (default: 1.0)
- `random_state`: Random seed (default: None)

## 🎬 Running Examples

```bash
# Run comprehensive examples with visualizations
python example_usage.py

# Run test suite
python test_anomaly_detection.py
```

## 📊 Algorithm Comparison

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Distance-Based** | Simple datasets | Easy to understand, fast | Sensitive to parameters |
| **DBSCAN** | Clustered data | No shape assumptions | Requires tuning |
| **LOF** | Local outliers | Detects local anomalies | Computationally expensive |
| **Isolation Forest** | High dimensions | Fast, scalable, effective | Less interpretable |

## 🛠️ API Reference

All detectors follow a consistent interface:

### Common Methods

- `fit(X)` - Fit the model on training data
- `predict(X)` - Predict labels for new data (returns 1 for inliers, -1 for outliers)
- `fit_predict(X)` - Fit and predict in one step
- `decision_function(X)` - Get anomaly scores
- `get_anomaly_indices()` - Get indices of detected anomalies

### Return Values

- **Labels**: `1` for normal points (inliers), `-1` for anomalies (outliers)
- **Scores**: Lower scores indicate higher anomaly likelihood

## 💼 Real-World Use Cases

### Credit Card Fraud Detection
```python
detector = IsolationForestDetector(contamination=0.01)  # ~1% fraud expected
labels = detector.fit_predict(transactions)
frauds = transactions[labels == -1]
```

### Network Intrusion Detection
```python
detector = LOFDetector(n_neighbors=30, contamination=0.05)
labels = detector.fit_predict(network_traffic)
intrusions = network_traffic[labels == -1]
```

### Manufacturing Quality Control
```python
detector = DistanceBasedDetector(n_neighbors=10, contamination=0.02)
labels = detector.fit_predict(sensor_readings)
defects = sensor_readings[labels == -1]
```

### Log Anomaly Analysis
```python
detector = DBSCANDetector(eps=0.3, min_samples=10)
labels = detector.fit_predict(log_features)
suspicious_logs = log_features[labels == -1]
```

## 🎓 Performance Tips

1. **Normalize your data** - Most algorithms work better with normalized features
2. **Tune contamination** - Set based on expected outlier proportion
3. **Cross-validate** - Test different parameters to find optimal settings
4. **Combine methods** - Use ensemble of detectors for robust results
5. **Feature selection** - Remove irrelevant features before detection

## 📁 Project Structure

```
Outlier_Detection/
├── anomaly_detection/              # Main package
│   ├── __init__.py                 # Package exports
│   ├── distance_based.py           # Distance-based detector
│   ├── dbscan_detector.py          # DBSCAN detector
│   ├── lof_detector.py             # LOF detector
│   └── isolation_forest_detector.py # Isolation Forest
├── example_usage.py                # Comprehensive examples
├── test_anomaly_detection.py       # Test suite
├── requirements.txt                # Dependencies
├── ANOMALY_DETECTION_README.md    # Detailed documentation
└── .gitignore                      # Git ignore rules
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_anomaly_detection.py
```

Expected output:
```
Testing Distance-Based Detector...
✓ Distance-Based Detector works correctly

Testing DBSCAN Detector...
✓ DBSCAN Detector works correctly

Testing LOF Detector...
✓ LOF Detector works correctly

Testing Isolation Forest Detector...
✓ Isolation Forest Detector works correctly

Testing integration with shared dataset...
✓ Integration test passed

All tests passed! ✓
```

## 📖 Documentation

For more detailed information, see:
- `ANOMALY_DETECTION_README.md` - Complete user guide with examples
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is created by Ikbar Athallah Taufik as part of learning and demonstrating data science skills.

## 👤 Author

**Ikbar Athallah Taufik**

A passionate data scientist from Indonesia specializing in Machine Learning and AI.

- **Email**: ikbaratallah@gmail.com
- **LinkedIn**: [Ikbar Athallah Taufik](https://linkedin.com/in/ikbar-athallah-taufik)
- **Kaggle**: [Ikbar Athallah](https://kaggle.com/ikbar-athallah)

## 📚 References

1. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers.
2. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters.
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
4. Knorr, E. M., & Ng, R. T. (1998). Algorithms for mining distance-based outliers.

---

*Built with ❤️ by Ikbar - A passionate data scientist from Indonesia*
