# k-means-clustering
The Final Project for "Parallel Algorithms For The Analysis and Synthesis of Data" Course 

## 1. Algorithm and Parallelization Method
- **Algorithm**: K-Means Clustering
- **Parallelization Method**: PySpark for parallel implementation.

## 2. Instructions to Reproduce Results
### Prerequisites
- Install Python 3.x and PySpark.
- Install dependencies:
  ```bash
  pip install -r requirements.txt

## Running the Algorithm
### 1. Place your dataset in the data/ folder (use dataset.csv).
### 2. Run the non-parallel version:
# K-Means Clustering

## 1. Algorithm and Parallelization Method
- **Algorithm**: K-Means Clustering
- **Parallelization Method**: PySpark for parallel implementation.

## 2. Instructions to Reproduce Results
### Prerequisites
- Install Python 3.x and PySpark.
- Install dependencies:
  ```bash
  pip install -r requirements.txt

## Running the Algorithm
### 1. Place your dataset in the data/ folder (use dataset.csv).
### 2. Run the non-parallel version:
```bash
python non_parallel.py
```
### 3. Run the parallel version:
```bash
spark-submit parallel_pyspark.py
```
## 3. Parallelization Explanation 
The PySpark implementation parallelizes the cluster assignment and centroid update steps by distributing computations across multiple nodes using Spark's distributed DataFrame API.

## 4. Speedup Calculation
---
### **Benchmarking**
1. Generate a synthetic dataset with 70,000 samples using `sklearn.datasets.make_blobs`.
2. Measure the runtime for both implementations:
   - Use `time` or `timeit` for the non-parallel version.
   - Use Spark's built-in tools to measure execution time.
3. Calculate speedup: 
   \[
   \text{Speedup} = \frac{\text{Time (non-parallel)}}{\text{Time (parallel)}}
   \]
4. Plot results (e.g., using `matplotlib`).
---
python non_parallel.py

### 3. Run the parallel version:
spark-submit parallel_pyspark.py

## 3. Parallelization Explanation 
The PySpark implementation parallelizes the cluster assignment and centroid update steps by distributing computations across multiple nodes using Spark's distributed DataFrame API.

## 4. Speedup Calculation
---
### **Benchmarking**
1. Generate a synthetic dataset with 70,000 samples using `sklearn.datasets.make_blobs`.
2. Measure the runtime for both implementations:
   - Use `time` or `timeit` for the non-parallel version.
   - Use Spark's built-in tools to measure execution time.
3. Calculate speedup: 
   \[
   \text{Speedup} = \frac{\text{Time (non-parallel)}}{\text{Time (parallel)}}
   \]
4. Plot results (e.g., using `matplotlib`).
---
