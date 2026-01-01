from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName("ml_kmeans").getOrCreate()

input_path  = "gs://cloud-data-service-yara-2026-01/input/yellow_tripdata_2023-01.parquet"
output_path = "gs://cloud-data-service-yara-2026-01/output/ml_kmeans_run2"

df = spark.read.parquet(input_path).select("trip_distance", "total_amount").dropna()

df = df.filter(
    (F.col("trip_distance") > 0) & (F.col("trip_distance") < 30) &
    (F.col("total_amount") > 0) & (F.col("total_amount") < 200)
)

data = VectorAssembler(
    inputCols=["trip_distance", "total_amount"],
    outputCol="features"
).transform(df).select("features")

model = KMeans(k=3, seed=42).fit(data)
pred  = model.transform(data)

sil = ClusteringEvaluator(metricName="silhouette").evaluate(pred)

counts = pred.groupBy("prediction").count().orderBy("prediction").collect()

lines = [f"k=3", f"silhouette={sil}", f"filtered_rows={df.count()}"]
for r in counts:
    lines.append(f"cluster_{r['prediction']}_count={r['count']}")

spark.sparkContext.parallelize(lines, 1).saveAsTextFile(output_path)
spark.stop()
