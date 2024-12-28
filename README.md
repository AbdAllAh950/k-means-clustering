# K-Means Clustering Project

## Overview
This project implements the K-Means clustering algorithm in both non-parallel and parallel modes. The goal is to evaluate the performance and speedup achieved by parallelizing the algorithm. The implementation uses synthetic data with at least 70,000 samples, and the parallel version leverages PySpark for distributed computing.

## Implementation
The entire implementation, including dataset generation, non-parallel and parallel versions of K-Means clustering, benchmarking, and performance plotting, is provided in the `k_means_project.py` file.

## Instructions to Run

### Prerequisites
- **Python** (3.x)
- **PySpark**
- Required Python libraries (listed in `requirements.txt`):
  ```
  numpy
  pandas
  matplotlib
  pyspark
  scikit-learn
  ```
  Install all dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Code
1. Place the `k_means_project.py` file and the `requirements.txt` file in your working directory.
2. Generate and save the synthetic dataset:
   ```bash
   python k_means_project.py
   ```
   The dataset will be saved in the `data/` folder as `dataset.csv`.
3. The script will automatically execute the non-parallel and parallel versions of K-Means clustering, benchmark them, and generate the performance plot.
4. The performance plot will be saved in the `benchmarks/` folder as `performance_plot.png`.

## Key Features
- **Dataset Generation**: A synthetic dataset with 70,000 samples and 2 features is generated using `make_blobs` from Scikit-learn.
- **Non-Parallel K-Means**: Implements K-Means clustering in Python using NumPy.
- **Parallel K-Means**: Implements K-Means clustering in PySpark, leveraging Spark's distributed DataFrame API.
- **Benchmarking**: Measures runtime for both versions and calculates speedup.
- **Performance Plot**: Visualizes the relationship between the number of threads and speedup.

## Results
- **Speedup Calculation**: The script calculates the speedup achieved by the parallel implementation:
  
  \[ \text{Speedup} = \frac{\text{Time (non-parallel)}}{\text{Time (parallel)}} \]

- The generated plot (`performance_plot.png`) demonstrates how the speedup scales with the number of threads used.

## Repository Structure
```plaintext
kmeans-clustering/
|
├── README.md         # Project documentation
├── requirements.txt  # Python dependencies
├── k_means_project.py # Full implementation
├── data/             # Folder for benchmark datasets
│   └── dataset.csv   # Generated synthetic dataset
└── benchmarks/       # Folder for benchmarking results
    └── performance_plot.png
```

## Notes
- The code automatically generates the required dataset and plot.
- Make sure PySpark is properly configured in your environment.
