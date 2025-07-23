#  HAR Clustering & Optimization using Diffusion Maps and DFO Methods

This repository showcases two advanced projects that apply **Manifold Learning** and **Derivative-Free Optimization (DFO)** to solve real-world machine learning problems from clustering human activities to tuning SVM hyperparameters without gradients.

---

## Project Overview

| Task | Description |
|------|-------------|
| **Problem 1** | Dimensionality Reduction & Clustering of Human Activity Recognition (HAR) data using **Diffusion Maps** |
| **Problem 2** | Derivative-Free Optimization using **Nelder-Mead**, **Simulated Annealing**, and **CMA-ES** on benchmark functions and ML hyperparameters |

---

## Problem 1: Time-Series Clustering with Diffusion Maps

### Objective:
Cluster and visualize sensor-based human motion activities by reducing dimensionality using Diffusion Maps.

### Tasks Completed:
- Preprocessing UCI HAR time-series dataset
- Computing pairwise distances (DTW & Euclidean)
- Constructing similarity matrix and diffusion kernel
- Extracting low-dimensional embeddings via Diffusion Maps
- Clustering in embedded space (KMeans, DBSCAN)
- Evaluation using **ARI**, **Silhouette Score**
- Comparison with PCA & t-SNE

### Visualizations:
- Diffusion embeddings colored by activity
- Clustering results in different embedding spaces
- Activity separability comparison across methods

---

## Problem 2: Derivative-Free Optimization (DFO) Techniques

### Objective:
Benchmark DFO methods on test functions and apply them for ML hyperparameter tuning.

### Optimization Methods:
- **Nelder-Mead (Simplex)**
- **Simulated Annealing**
- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)

### Benchmark Functions:
- Rosenbrock
- Rastrigin
- Ackley

### ML Application:
- Hyperparameter tuning for **SVM on MNIST**
- Parameters tuned: `kernel`, `C`, `gamma`
- Evaluated based on accuracy, function calls, and robustness

---

## Libraries Used

```
numpy · pandas · matplotlib · seaborn · scikit-learn · dtw-python · cma · scipy
```

---

## 🔗 Useful Links

- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Diffusion Maps (Wikipedia)](https://en.wikipedia.org/wiki/Diffusion_map)

