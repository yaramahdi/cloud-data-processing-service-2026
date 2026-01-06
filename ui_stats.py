import sys, time
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ui_stats").getOrCreate()
t0 = time.time()

if len(sys.argv) < 3:
    raise Exception("Usage: ui_stats.py <input_path> <output_dir>")

input_path = sys.argv[1]
output_dir = sys.argv[2]

if input_path.lower().endswith(".parquet"):
    df = spark.read.parquet(input_path)
else:
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

rows = df.count()
cols = ", ".join(df.columns)
seconds = time.time() - t0

spark.sparkContext.parallelize(
    [f"rows={rows}", f"columns={cols}", f"seconds={seconds}"], 1
).saveAsTextFile(output_dir)

spark.stop()
