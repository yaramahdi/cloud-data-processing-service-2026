from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("ml_logreg").getOrCreate()

input_path  = "gs://cloud-data-service-yara-2026-01/input/yellow_tripdata_2023-01.parquet"
output_path = "gs://cloud-data-service-yara-2026-01/output/ml_logreg_run1"

df = spark.read.parquet(input_path).select("trip_distance", "total_amount").dropna()
df = df.filter((F.col("trip_distance") > 0) & (F.col("total_amount") > 0))

df = df.withColumn("label", F.when(F.col("total_amount") >= 20, 1.0).otherwise(0.0))

train, test = df.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(inputCols=["trip_distance"], outputCol="features")
train2 = assembler.transform(train).select("features", "label")
test2  = assembler.transform(test).select("features", "label")

model = LogisticRegression(maxIter=10).fit(train2)
pred  = model.transform(test2)

auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(pred)

spark.sparkContext.parallelize([f"auc={auc}"], 1).saveAsTextFile(output_path)
spark.stop()
