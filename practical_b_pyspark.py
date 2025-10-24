# practical_b_pyspark.py
# Requirements: pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("PracticalB_MapReduce") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# Example: compute counts by category from a CSV
df = spark.read.csv("data/big_events.csv", header=True, inferSchema=True)

# show schema and a sample
df.printSchema()
df.show(5)

# map-reduce pattern: groupBy (map -> key -> reduce)
counts = df.groupBy("category").agg(
    F.count("*").alias("n_events"),
    F.sum("value").alias("sum_value"),
    F.avg("value").alias("avg_value")
).orderBy(F.desc("n_events"))

counts.show(20, truncate=False)

# Example: wordcount on 'text' column (demonstrates flatMap / reduceByKey)
# (convert DataFrame column to RDD of tokens)
rdd = df.select("text").rdd.flatMap(lambda row: row[0].split() if row[0] else [])
word_counts = rdd.map(lambda w: (w.lower().strip(), 1)).reduceByKey(lambda a,b: a+b)
top = word_counts.takeOrdered(20, key=lambda x: -x[1])
print("Top words:", top)

spark.stop()
