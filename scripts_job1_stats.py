from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("job1_stats").getOrCreate()

input_path = "gs://cloud-data-service-yara-2026-01/input/yellow_tripdata_2023-01.parquet"
output_path = "gs://cloud-data-service-yara-2026-01/output/job1_stats_run1"

df = spark.read.parquet(input_path)

rows = df.count()
cols = len(df.columns)

result = df.select(
    F.avg("trip_distance").alias("avg_trip_distance"),
    F.avg("fare_amount").alias("avg_fare_amount"),
    F.avg("total_amount").alias("avg_total_amount")
).collect()[0]

lines = [
    f"rows={rows}",
    f"columns={cols}",
    f"avg_trip_distance={result['avg_trip_distance']}",
    f"avg_fare_amount={result['avg_fare_amount']}",
    f"avg_total_amount={result['avg_total_amount']}"
]

spark.sparkContext.parallelize(lines, 1).saveAsTextFile(output_path)
spark.stop()
