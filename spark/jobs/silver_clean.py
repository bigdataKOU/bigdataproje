"""
Silver layer: Bronze raw events -> temizlenmis, deduplicate, analitik/ML-hazir.

- id'ye gore dedup (ayni suc kaydi tekrar gelirse atilir)
- primary_type, district, latitude/longitude null'lari filtrele
- event_time'tan hour_of_day / day_of_week / month / year_derived turetilir
- Delta'ya partitionBy(event_year) ile yaz

Streaming MERGE icin foreachBatch kullaniyoruz (Delta upsert).
SILVER_BATCH_ONCE=1 ile tek seferlik batch mod (run_all.sh icin).

Calistirma:
  /opt/app/run.sh /opt/app/jobs/silver_clean.py
  SILVER_BATCH_ONCE=1 /opt/app/run.sh /opt/app/jobs/silver_clean.py
"""
import os
import sys

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from _session import (
    bronze_path,
    build_spark,
    CHECKPOINT_PATH,
    silver_path,
)


def bronze_to_cleaned_silver(bronze_df: DataFrame) -> DataFrame:
    """Bronze (batch veya stream micro-batch) satirlarini silver semasina."""
    return (
        bronze_df.where(
            F.col("id").isNotNull()
            & F.col("primary_type").isNotNull()
            & F.col("district").isNotNull()
            & F.col("latitude").isNotNull()
            & F.col("longitude").isNotNull()
        )
        # Aynı id birden cok kez gelirse en yenisi kalsin diye event_time DESC sort + dropDuplicates
        .dropDuplicates(["id"])
        .withColumn("hour_of_day", F.hour(F.col("event_time")))
        .withColumn("day_of_week", F.dayofweek(F.col("event_time")))
        .withColumn("month", F.month(F.col("event_time")))
        .withColumn("event_year", F.year(F.col("event_time")))
        .select(
            "id",
            "case_number",
            "primary_type",
            "description",
            "location_description",
            "arrest",
            "domestic",
            "beat",
            "district",
            "ward",
            "community_area",
            "fbi_code",
            "year",
            "latitude",
            "longitude",
            "event_time",
            "event_date",
            "event_year",
            "hour_of_day",
            "day_of_week",
            "month",
            "ingestedAt",
        )
    )


def upsert_to_silver(spark: SparkSession):
    target_path = silver_path()

    def _process(batch_df: DataFrame, batch_id: int) -> None:
        cleaned = bronze_to_cleaned_silver(batch_df).persist()
        try:
            n = cleaned.count()
            if n == 0:
                return

            if not DeltaTable.isDeltaTable(spark, target_path):
                (cleaned.write.format("delta")
                     .partitionBy("event_year")
                     .mode("overwrite")
                     .save(target_path))
                print(f"[silver] batch={batch_id} bootstrapped rows={n}",
                      flush=True)
                return

            target = DeltaTable.forPath(spark, target_path)
            (target.alias("t")
                  .merge(cleaned.alias("s"), "t.id = s.id")
                  .whenNotMatchedInsertAll()
                  .execute())
            print(f"[silver] batch={batch_id} merged rows={n}", flush=True)
        finally:
            cleaned.unpersist()

    return _process


def main_batch_once() -> int:
    """Bronze Delta'yi batch okuyup silver'i overwrite eder (run_all guvencesi)."""
    os.environ["SPARK_MASTER_URL"] = "local[*]"
    print("[silver-batch] Spark master=local[*]", flush=True)
    spark = build_spark("silver-batch-once")
    spark.sparkContext.setLogLevel("WARN")

    bronze_df = spark.read.format("delta").load(bronze_path())
    cleaned = bronze_to_cleaned_silver(bronze_df).repartition(32, "event_year")
    n = cleaned.count()
    if n == 0:
        print("[silver-batch] HATA: bronze sonrasi temiz satir yok", flush=True)
        spark.stop()
        return 1

    (
        cleaned.write.format("delta")
        .partitionBy("event_year")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(silver_path())
    )
    print(f"[silver-batch] yazildi rows={n} -> {silver_path()}", flush=True)
    spark.stop()
    return 0


def main_streaming() -> int:
    spark = build_spark("silver-clean")
    spark.sparkContext.setLogLevel("WARN")

    bronze = spark.readStream.format("delta").load(bronze_path())

    query = (
        bronze.writeStream
        .foreachBatch(upsert_to_silver(spark))
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/silver")
        .trigger(processingTime="20 seconds")
        .start()
    )

    print(f"[silver] streaming -> {silver_path()}", flush=True)
    query.awaitTermination()
    return 0


def main() -> int:
    if os.environ.get("SILVER_BATCH_ONCE", "").strip() in ("1", "true", "yes"):
        return main_batch_once()
    return main_streaming()


if __name__ == "__main__":
    sys.exit(main())
