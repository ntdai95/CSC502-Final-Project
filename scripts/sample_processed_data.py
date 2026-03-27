import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, count as spark_count


input_file = "data/ebird_data_processed.csv"
output_file = "data/ebird_data_processed_sampled.csv"
temp_output_dir = output_file + "_spark_temp"
if os.path.exists(temp_output_dir):
    shutil.rmtree(temp_output_dir, ignore_errors=True)

if os.path.exists(output_file):
    os.remove(output_file)

spark_temp_dir = "data/spark_temp_sample"
os.makedirs(spark_temp_dir, exist_ok=True)
sample_fraction = 0.05
num_quantile_bins = 8
random_seed = 42
spark = (
    SparkSession.builder
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "6g")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.default.parallelism", "16")
    .config("spark.local.dir", spark_temp_dir)
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.files.maxPartitionBytes", "134217728")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
try:
    df = spark.read.option("header", True).option("inferSchema", True).csv(input_file)
    needed_cols = ["TAXON CONCEPT ID", "species_frequency", "OBSERVATION COUNT", "LATITUDE", "LONGITUDE", "day_sin", "day_cos",
                   "REVIEWED"]

    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df.select(*existing_cols)
    if len(existing_cols) != len(needed_cols):
        missing_cols = [c for c in needed_cols if c not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = (
        df.withColumn("species_frequency", col("species_frequency").cast("double"))
        .withColumn("OBSERVATION COUNT", col("OBSERVATION COUNT").cast("double"))
        .withColumn("LATITUDE", col("LATITUDE").cast("double"))
        .withColumn("LONGITUDE", col("LONGITUDE").cast("double"))
        .withColumn("day_sin", col("day_sin").cast("double"))
        .withColumn("day_cos", col("day_cos").cast("double"))
        .withColumn("REVIEWED", col("REVIEWED").cast("int"))
    )

    df = df.dropna(subset=["TAXON CONCEPT ID", "species_frequency", "OBSERVATION COUNT", "LATITUDE", "LONGITUDE", "day_sin",
                           "day_cos", "REVIEWED"]).cache()

    quantile_probs = [i / num_quantile_bins for i in range(num_quantile_bins + 1)]
    boundaries = df.approxQuantile("species_frequency", quantile_probs, 0.001)
    unique_boundaries = sorted(set(boundaries))
    if len(unique_boundaries) < 2:
        raise ValueError("Could not compute valid quantile boundaries for species_frequency.")

    freq_bin = None
    for i in range(len(boundaries) - 1):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        label = f"Q{i + 1}"
        if i == 0:
            condition = (col("species_frequency") >= lower) & (col("species_frequency") <= upper)
        else:
            condition = (col("species_frequency") > lower) & (col("species_frequency") <= upper)

        if freq_bin is None:
            freq_bin = when(condition, label)
        else:
            freq_bin = freq_bin.when(condition, label)

    freq_bin = freq_bin.otherwise(f"Q{len(boundaries) - 1}")
    grouped = df.withColumn("freq_bin", freq_bin).cache()
    print("Original subgroup distribution:")
    grouped.groupBy("freq_bin").agg(spark_count("*").alias("count")).orderBy("freq_bin").show(truncate=False)
    sampled = grouped.withColumn("u", rand(random_seed)).filter(col("u") < sample_fraction).drop("u").cache()
    sampled_count = sampled.count()
    if sampled_count == 0:
        raise ValueError("Sampled dataset is empty. Increase sample_fraction.")

    print("Sampled subgroup distribution:")
    sampled.groupBy("freq_bin").agg(spark_count("*").alias("count")).orderBy("freq_bin").show(truncate=False)
    final_df = sampled.drop("freq_bin")
    final_df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_output_dir)
    part_file = None
    for filename in os.listdir(temp_output_dir):
        if filename.startswith("part-") and filename.endswith(".csv"):
            part_file = os.path.join(temp_output_dir, filename)
            break

    if part_file is None:
        raise FileNotFoundError("Could not find Spark output part CSV file.")

    shutil.move(part_file, output_file)
    shutil.rmtree(temp_output_dir)
    print(f"CSV saved to: {output_file}")
    sampled.unpersist()
    grouped.unpersist()
    df.unpersist()
finally:
    spark.stop()