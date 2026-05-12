"""
Bronze layer: Kafka 'ratings' topic'inden ham JSON mesajlari okuyup
Delta tablosuna append eder. Hicbir donusum yok, raw event store.

Calistirma:
  /opt/app/run.sh /opt/app/jobs/bronze_ingest.py
"""
import sys
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    DoubleType,
    IntegerType,
)

from _session import (
    KAFKA_BROKER,
    KAFKA_TOPIC_RATINGS,
    bronze_path,
    build_spark,
    CHECKPOINT_PATH,
)


RATING_SCHEMA = StructType([
    StructField("userId", IntegerType(), False),
    StructField("movieId", IntegerType(), False),
    StructField("rating", DoubleType(), False),
    StructField("timestamp", LongType(), False),
    StructField("ingestedAt", LongType(), True),
])


def main() -> int:
    spark = build_spark("bronze-ingest")
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC_RATINGS)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw.select(
            F.col("key").cast("string").alias("kafka_key"),
            F.col("topic").alias("kafka_topic"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
            F.from_json(F.col("value").cast("string"), RATING_SCHEMA).alias("payload"),
        )
        .select(
            "kafka_key",
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
            "kafka_timestamp",
            "payload.userId",
            "payload.movieId",
            "payload.rating",
            "payload.timestamp",
            "payload.ingestedAt",
        )
        .withColumn("event_time", F.to_timestamp(F.col("timestamp")))
        .withColumn("event_date", F.to_date(F.col("event_time")))
    )

    query = (
        parsed.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/bronze")
        .option("mergeSchema", "true")
        .partitionBy("event_date")
        .trigger(processingTime="10 seconds")
        .start(bronze_path())
    )

    print(f"[bronze] writing to {bronze_path()}", flush=True)
    query.awaitTermination()
    return 0


if __name__ == "__main__":
    sys.exit(main())
