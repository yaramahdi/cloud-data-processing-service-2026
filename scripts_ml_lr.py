from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("ml_lr").getOrCreate()

input_path  = "gs://cloud-data-service-yara-2026-01/input/yellow_tripdata_2023-01.parquet"
output_path = "gs://cloud-data-service-yara-2026-01/output/ml_lr_run1"

df = spark.read.parquet(input_path).select("trip_distance", "total_amount").dropna()
df = df.filter((F.col("trip_distance") > 0) & (F.col("total_amount") > 0))

train, test = df.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(inputCols=["trip_distance"], outputCol="features")
train2 = assembler.transform(train).select("features", F.col("total_amount").alias("label"))
test2  = assembler.transform(test).select("features", F.col("total_amount").alias("label"))

model = LinearRegression(maxIter=10).fit(train2)
pred  = model.transform(test2)

rmse = RegressionEvaluator(metricName="rmse").evaluate(pred)
r2   = RegressionEvaluator(metricName="r2").evaluate(pred)

spark.sparkContext.parallelize([f"rmse={rmse}", f"r2={r2}"], 1).saveAsTextFile(output_path)
spark.stop()
