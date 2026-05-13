"""
Gold layer: silver tablosundan analitik/ML-hazir agregasyonlar.

Üç gold tablosu uretilir:
  - gold/type_stats     : suç tipine göre toplam, arrest_rate, domestic_rate
  - gold/district_stats : ilçeye göre toplam, en sık suç tipi, lat/lon merkezi
  - gold/hourly_stats   : saat × suç tipi heatmap için

Bu bir batch job - silver streaming ile birlikte periyodik calistirilir.

Calistirma:
  /opt/app/run.sh /opt/app/jobs/gold_features.py
"""
import sys
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from _session import (
    build_spark,
    gold_district_path,
    gold_hourly_path,
    gold_type_path,
    silver_path,
)


def main() -> int:
    spark = build_spark("gold-features")
    spark.sparkContext.setLogLevel("WARN")

    silver = spark.read.format("delta").load(silver_path())

    # --- Suç tipi başına istatistikler ---
    type_stats = (
        silver.groupBy("primary_type")
        .agg(
            F.count("*").alias("crime_count"),
            F.sum(F.when(F.col("arrest"), 1).otherwise(0)).alias("arrest_count"),
            F.sum(F.when(F.col("domestic"), 1).otherwise(0)).alias("domestic_count"),
            F.min("event_time").alias("first_seen"),
            F.max("event_time").alias("last_seen"),
        )
        .withColumn("arrest_rate", F.col("arrest_count") / F.col("crime_count"))
        .withColumn("domestic_rate", F.col("domestic_count") / F.col("crime_count"))
        .withColumn(
            "frequency_bucket",
            F.when(F.col("crime_count") >= 100000, "very_high")
             .when(F.col("crime_count") >= 10000, "high")
             .when(F.col("crime_count") >= 1000, "medium")
             .otherwise("low"),
        )
    )
    (type_stats.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(gold_type_path()))
    print(f"[gold] type_stats rows={type_stats.count()} -> {gold_type_path()}",
          flush=True)

    # --- İlçe (district) başına istatistikler ---
    district_base = (
        silver.where(F.col("district").isNotNull())
        .groupBy("district")
        .agg(
            F.count("*").alias("crime_count"),
            F.sum(F.when(F.col("arrest"), 1).otherwise(0)).alias("arrest_count"),
            F.avg("latitude").alias("avg_latitude"),
            F.avg("longitude").alias("avg_longitude"),
            F.countDistinct("primary_type").alias("unique_types"),
        )
        .withColumn("arrest_rate", F.col("arrest_count") / F.col("crime_count"))
    )

    # Her ilçenin en sık suç tipi (window function)
    type_per_district = (
        silver.where(F.col("district").isNotNull())
        .groupBy("district", "primary_type")
        .agg(F.count("*").alias("c"))
    )
    w = Window.partitionBy("district").orderBy(F.col("c").desc())
    top_type = (
        type_per_district.withColumn("rk", F.row_number().over(w))
        .where(F.col("rk") == 1)
        .select(
            F.col("district"),
            F.col("primary_type").alias("top_primary_type"),
            F.col("c").alias("top_primary_type_count"),
        )
    )

    district_stats = (
        district_base.join(top_type, on="district", how="left")
        .withColumn(
            "size_bucket",
            F.when(F.col("crime_count") >= 200000, "large")
             .when(F.col("crime_count") >= 50000, "medium")
             .otherwise("small"),
        )
    )
    (district_stats.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(gold_district_path()))
    print(f"[gold] district_stats rows={district_stats.count()} -> {gold_district_path()}",
          flush=True)

    # --- Saat × suç tipi heatmap için ---
    hourly_stats = (
        silver.groupBy("hour_of_day", "primary_type")
        .agg(F.count("*").alias("crime_count"))
        .where(F.col("hour_of_day").isNotNull())
    )
    (hourly_stats.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(gold_hourly_path()))
    print(f"[gold] hourly_stats rows={hourly_stats.count()} -> {gold_hourly_path()}",
          flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
