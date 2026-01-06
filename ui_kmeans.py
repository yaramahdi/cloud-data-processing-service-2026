import sys, time
from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName("ui_kmeans").getOrCreate()
t0 = time.time()

if len(sys.argv) < 3:
    raise Exception("Usage: ui_kmeans.py <input_path> <output_dir>")

input_path = sys.argv[1]
output_dir = sys.argv[2]

if input_path.lower().endswith(".parquet"):
    df = spark.read.parquet(input_path)
else:
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
if len(num_cols) < 1:
    raise Exception("Need at least 1 numeric column for KMeans.")

use_cols = []
for c in ["trip_distance", "total_amount"]:
    if c in df.columns and c in num_cols:
        use_cols.append(c)
for c in num_cols:
    if c not in use_cols:
        use_cols.append(c)
use_cols = use_cols[:2]

df2 = df.select(*use_cols).dropna().sample(False, 0.05, seed=42)

assembler = VectorAssembler(inputCols=use_cols, outputCol="raw_features")
vdf = assembler.transform(df2)

scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
smodel = scaler.fit(vdf)
sdf = smodel.transform(vdf)

k = 3
km = KMeans(k=k, seed=42, featuresCol="features")
model = km.fit(sdf)
pred = model.transform(sdf)

evaluator = ClusteringEvaluator(featuresCol="features")
silhouette = evaluator.evaluate(pred)

counts = pred.groupBy("prediction").count().orderBy("prediction").collect()

lines = [f"cols={', '.join(use_cols)}", f"k={k}", f"silhouette={silhouette}"]
for r in counts:
    lines.append(f"cluster_{r['prediction']}_count={r['count']}")

seconds = time.time() - t0
lines.append(f"seconds={seconds}")

spark.sparkContext.parallelize(lines, 1).saveAsTextFile(output_dir)
spark.stop()
