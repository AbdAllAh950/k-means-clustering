from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Initialize Spark Session
spark = SparkSession.builder.master("local[4]").appName("KMeans Clustering").getOrCreate()

# Load Dataset
data = spark.read.csv('data/dataset.csv', header=True, inferSchema=True)

# Transform to Vector
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=data.columns, outputCol="features")
dataset = assembler.transform(data).select("features")

# Train K-Means Model
kmeans = KMeans(k=5, maxIter=100)
model = kmeans.fit(dataset)

# Predict clusters
predictions = model.transform(dataset)

# Evaluate clustering
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette with squared euclidean distance: {silhouette}")

# Output results
model.clusterCenters()
