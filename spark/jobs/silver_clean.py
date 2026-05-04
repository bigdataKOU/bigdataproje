"""
Silver layer: Bronze raw events -> temizlenmis, deduplicate, movies'le join'li
analitik-hazir tablo.

Streaming MERGE icin foreachBatch kullaniyoruz (Delta upsert).

Calistirma:
  /opt/app/run.sh /opt/app/jobs/silver_clean.py
"""
import sys
from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from _session import (
    bronze_path,
    build_spark,
    CHECKPOINT_PATH,
    movies_csv_path,
    silver_path,
)


MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), False),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True),
])


def load_movies(spark: SparkSession) -> DataFrame:
    return (
        spark.read.option("header", True)
        .schema(MOVIES_SCHEMA)
        .csv(movies_csv_path())
        .withColumn("genre_list", F.split(F.col("genres"), "\\|"))
    )


def upsert_to_silver(spark: SparkSession, movies: DataFrame):
    target_path = silver_path()

    def _process(batch_df: DataFrame, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return

        cleaned = (
            batch_df.where(
                F.col("userId").isNotNull()
                & F.col("movieId").isNotNull()
                & F.col("rating").between(0.5, 5.0)
            )
            .withColumn("rating_event_id",
                        F.concat_ws("_",
                                    F.col("userId").cast("string"),
                                    F.col("movieId").cast("string"),
                                    F.col("timestamp").cast("string")))
            .dropDuplicates(["rating_event_id"])
            .join(F.broadcast(movies), on="movieId", how="left")
            .select(
                "rating_event_id",
                "userId",
                "movieId",
                "rating",
                "timestamp",
                "event_time",
                "event_date",
                "title",
                "genres",
                "genre_list",
                "ingestedAt",
            )
        )

        if not DeltaTable.isDeltaTable(spark, target_path):
            (cleaned.write.format("delta")
                 .partitionBy("event_date")
                 .mode("overwrite")
                 .save(target_path))
            print(f"[silver] batch={batch_id} bootstrapped silver", flush=True)
            return

        target = DeltaTable.forPath(spark, target_path)
        (target.alias("t")
              .merge(cleaned.alias("s"),
                     "t.rating_event_id = s.rating_event_id")
              .whenNotMatchedInsertAll()
              .execute())
        print(f"[silver] batch={batch_id} merged rows={cleaned.count()}", flush=True)

    return _process


def main() -> int:
    spark = build_spark("silver-clean")
    spark.sparkContext.setLogLevel("WARN")

    movies = load_movies(spark).cache()
    movies.count()

    bronze = spark.readStream.format("delta").load(bronze_path())

    query = (
        bronze.writeStream
        .foreachBatch(upsert_to_silver(spark, movies))
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/silver")
        .trigger(processingTime="20 seconds")
        .start()
    )

    print(f"[silver] writing to {silver_path()}", flush=True)
    query.awaitTermination()
    return 0


if __name__ == "__main__":
    sys.exit(main())
