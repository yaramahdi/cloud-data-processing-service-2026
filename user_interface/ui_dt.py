import sys, time
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("ui_dt").getOrCreate()
t0 = time.time()

if len(sys.argv) < 3:
    raise Exception("Usage: ui_dt.py <input_path> <output_dir>")

input_path = sys.argv[1]
output_dir = sys.argv[2]

if input_path.lower().endswith(".parquet"):
    df = spark.read.parquet(input_path)
else:
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

feature_col = "trip_distance" if "trip_distance" in df.columns else (num_cols[0] if len(num_cols) > 0 else None)
label_col = "total_amount" if "total_amount" in df.columns else (num_cols[1] if len(num_cols) > 1 else None)

if feature_col is None or label_col is None:
    raise Exception("Need at least two numeric columns for DecisionTree regression.")

df = df.select(feature_col, label_col).dropna()
df = df.filter((F.col(feature_col) > 0) & (F.col(label_col) > 0))

df = df.sample(False, 0.10, seed=42)

train, test = df.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(inputCols=[feature_col], outputCol="features")
train2 = assembler.transform(train).select("features", F.col(label_col).alias("label"))
test2  = assembler.transform(test).select("features", F.col(label_col).alias("label"))

model = DecisionTreeRegressor(maxDepth=5).fit(train2)
pred = model.transform(test2)

rmse = RegressionEvaluator(metricName="rmse").evaluate(pred)
r2   = RegressionEvaluator(metricName="r2").evaluate(pred)

seconds = time.time() - t0

spark.sparkContext.parallelize(
    [
        f"feature={feature_col}",
        f"label={label_col}",
        f"rmse={rmse}",
        f"r2={r2}",
        f"seconds={seconds}",
    ],
    1
).saveAsTextFile(output_dir)

spark.stop()
