import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import time
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

# Step 1: Generate Synthetic Dataset
X, y = make_blobs(n_samples=70000, centers=5, random_state=42)
dataset = pd.DataFrame(X, columns=['feature1', 'feature2'])
dataset.to_csv('data/dataset.csv', index=False)  # Save for reuse

# Step 2: Non-Parallel Implementation
def kmeans_non_parallel(X, K=5, max_iters=100):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]

    for iteration in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

# Measure runtime for non-parallel version
start_time = time.time()
centroids_np, labels_np = kmeans_non_parallel(X)
non_parallel_time = time.time() - start_time
print(f"Non-parallel runtime: {non_parallel_time:.2f} seconds")

# Step 3: Parallel Implementation with PySpark
# Initialize Spark Session
spark = SparkSession.builder.master("local[4]").appName("KMeans Clustering").getOrCreate()

# Load Dataset
data = spark.read.csv('data/dataset.csv', header=True, inferSchema=True)

# Transform to Vector
dataset_spark = VectorAssembler(inputCols=data.columns, outputCol="features").transform(data).select("features")

# Train K-Means Model
kmeans = KMeans(k=5, maxIter=100)
start_time = time.time()
model = kmeans.fit(dataset_spark)
parallel_time = time.time() - start_time

# Evaluate clustering
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(model.transform(dataset_spark))
print(f"Silhouette with squared euclidean distance: {silhouette}")

print(f"Parallel runtime: {parallel_time:.2f} seconds")

# Step 4: Speedup Calculation
speedup = non_parallel_time / parallel_time
print(f"Speedup: {speedup:.2f}")

# Step 5: Benchmark and Plot Results
threads = [1, 2, 4, 8]
speedups = []

for thread_count in threads:
    spark = SparkSession.builder.master(f"local[{thread_count}]").appName("KMeans Clustering").getOrCreate()
    start_time = time.time()
    kmeans.fit(dataset_spark)
    parallel_time = time.time() - start_time
    speedups.append(non_parallel_time / parallel_time)
    spark.stop()

plt.plot(threads, speedups, marker='o')
plt.title("Speedup vs Number of Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.grid()
plt.savefig("benchmarks/performance_plot.png")
plt.show()

spark.stop()
