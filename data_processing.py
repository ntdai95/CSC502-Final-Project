import os
import math
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, to_date, dayofyear, sin, cos, lit, when, count as spark_count, sum as spark_sum


spark_temp_dir = "data/spark-temp"
os.makedirs(spark_temp_dir, exist_ok=True)
spark = (
    SparkSession.builder
    .appName("eBird_Data_Processing_Single_CSV")
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

input_file = "data/ebird_data_raw.txt"
output_file = "data/ebird_data_processed.csv"
try:
    df = spark.read.option("header", True).option("sep", "\t").option("inferSchema", False).option("multiLine", False).csv(input_file)
    for c in df.columns:
        cleaned = c.strip()
        if cleaned != c:
            df = df.withColumnRenamed(c, cleaned)

    needed_cols = ["CATEGORY", "EXOTIC CODE", "OBSERVATION TYPE", "ALL SPECIES REPORTED", "DURATION MINUTES", "EFFORT DISTANCE KM",
                   "OBSERVATION DATE", "TAXON CONCEPT ID", "OBSERVATION COUNT", "LATITUDE", "LONGITUDE", "REVIEWED"]

    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df.select(*existing_cols)
    for c in ["CATEGORY", "EXOTIC CODE", "OBSERVATION TYPE", "OBSERVATION DATE", "TAXON CONCEPT ID", "OBSERVATION COUNT"]:
        df = df.withColumn(c, trim(col(c)))

    df = (
        df.withColumn("OBSERVATION COUNT",
                      when(col("OBSERVATION COUNT").isNull() | (col("OBSERVATION COUNT") == "") | (col("OBSERVATION COUNT") == "X"), 
                           None).otherwise(col("OBSERVATION COUNT")).cast("double")
        ).withColumn("LATITUDE",
                     when(col("LATITUDE").isNull() | (col("LATITUDE") == ""), 
                          None).otherwise(col("LATITUDE")).cast("double")
        ).withColumn("LONGITUDE",
                     when(col("LONGITUDE").isNull() | (col("LONGITUDE") == ""), 
                          None).otherwise(col("LONGITUDE")).cast("double")
        ).withColumn("DURATION MINUTES",
                     when(col("DURATION MINUTES").isNull() | (col("DURATION MINUTES") == ""),
                          None).otherwise(col("DURATION MINUTES")).cast("double")
        )
        .withColumn("EFFORT DISTANCE KM",
                    when(col("EFFORT DISTANCE KM").isNull() | (col("EFFORT DISTANCE KM") == ""),
                         None).otherwise(col("EFFORT DISTANCE KM")).cast("double")
        ).withColumn("ALL SPECIES REPORTED",
                     when(col("ALL SPECIES REPORTED").isNull() | (col("ALL SPECIES REPORTED") == ""),
                          None).otherwise(col("ALL SPECIES REPORTED")).cast("int")
        ).withColumn("REVIEWED",
                     when(col("REVIEWED").isNull() | (col("REVIEWED") == ""),
                          None).otherwise(col("REVIEWED")).cast("int")
        )
    )

    filtered = (
        df.filter(col("CATEGORY") == "species")
        .filter(col("EXOTIC CODE").isNull() | (col("EXOTIC CODE") == "") | (col("EXOTIC CODE") == "N"))
        .filter(col("OBSERVATION TYPE").isin("Traveling", "Stationary"))
        .filter(col("ALL SPECIES REPORTED") == 1)
        .filter(col("DURATION MINUTES") <= 300)
        .filter(col("EFFORT DISTANCE KM").isNull() | (col("EFFORT DISTANCE KM") <= 10))
        .filter(col("TAXON CONCEPT ID").isNotNull())
    )

    filtered = (
        filtered.withColumn("OBSERVATION DATE", to_date(col("OBSERVATION DATE"), "yyyy-MM-dd"))
        .filter(col("OBSERVATION DATE").isNotNull())
        .withColumn("day_of_year", dayofyear(col("OBSERVATION DATE")))
        .withColumn("day_sin", sin(col("day_of_year") * lit(2 * math.pi / 365.0)))
        .withColumn("day_cos", cos(col("day_of_year") * lit(2 * math.pi / 365.0)))
    )

    filtered = filtered.repartition(16, "TAXON CONCEPT ID").cache()
    species_counts = filtered.groupBy("TAXON CONCEPT ID").agg(spark_count("*").alias("species_count"))
    total_count = species_counts.agg(spark_sum("species_count").alias("total")).collect()[0]["total"]
    species_freq = species_counts.withColumn("species_frequency", 
                                             col("species_count") / lit(float(total_count))).select("TAXON CONCEPT ID", 
                                                                                                    "species_frequency")
    
    final_df = filtered.join(species_freq, on="TAXON CONCEPT ID", 
                             how="left").select("TAXON CONCEPT ID", "species_frequency", "OBSERVATION COUNT", "LATITUDE", 
                                                "LONGITUDE", "day_sin", "day_cos", "REVIEWED").dropna()

    temp_output_dir = output_file + "_spark_temp"
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

    if os.path.exists(output_file):
        os.remove(output_file)

    final_df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_output_dir)
    part_file = None
    for filename in os.listdir(temp_output_dir):
        if filename.startswith("part-") and filename.endswith(".csv"):
            part_file = os.path.join(temp_output_dir, filename)
            break

    shutil.move(part_file, output_file)
    shutil.rmtree(temp_output_dir)
    print(f"Done. Single CSV saved to: {output_file}")
    filtered.unpersist()
finally:
    spark.stop()