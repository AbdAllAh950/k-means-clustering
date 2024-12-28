import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('data/dataset.csv')
X = data.to_numpy()

# K-Means Parameters
K = 5  # Number of clusters
max_iters = 100

# Randomly initialize centroids
np.random.seed(42)
centroids = X[np.random.choice(X.shape[0], K, replace=False)]

for iteration in range(max_iters):
    # Assign clusters
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    # Update centroids
    new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
    
    # Check for convergence
    if np.all(centroids == new_centroids):
        break
    
    centroids = new_centroids

# Output results
print(f"Centroids:\n{centroids}")
